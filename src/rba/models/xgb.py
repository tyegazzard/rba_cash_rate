"""XGBoost gradient-boosted-trees wrapper for the §7 model ecosystem.

Holds the single ``xgboost_classifier`` model named in
``src/rba/config/models.yaml`` under ``rba.models.xgb``:
:class:`XGBClassifierWrapper`. Like every estimator in :mod:`rba.models`, it
implements the :class:`~rba.models.base.Model` contract (via
:class:`~rba.models.base.BaseModel`) so it is swappable into the evaluation
harness exactly like a baseline.

XGBoost is a strong non-linear, NaN-native learner, but its scikit-learn facade
has three quirks this wrapper hides so the caller sees the same clean interface
as every other model:

- **Labels must be contiguous ``0..K-1`` integers.** XGBoost rejects string
  labels and non-contiguous int encodings (e.g. the ``three_class`` ``{-1, 0,
  1}`` encoding) with ``"Invalid classes inferred ... Expected [0 1 2]"``. So
  :meth:`_fit` fits a :class:`~sklearn.preprocessing.LabelEncoder` on ``y``,
  trains on the ``0..K-1`` codes, and :meth:`predict` ``inverse_transform``\\ s
  back to the caller's original label values **and dtype** (strings return as
  strings, ints as ints).
- **No ``class_weight`` argument.** XGBoost has no ``class_weight`` (its
  ``scale_pos_weight`` is binary-only and unused for the 3-class target).
  ``class_weight='balanced'`` is instead realised as balanced *per-sample*
  weights via :func:`~rba.models._preprocessing.balanced_sample_weight`
  (scikit-learn's ``compute_sample_weight('balanced', y)``), unless the caller
  already passed an explicit ``sample_weight`` — which always wins.
- **NaN is handled natively.** XGBoost learns a default split direction for
  missing values, so — unlike the linear / kernel / neural wrappers — this model
  receives ``X`` untouched: no :class:`~rba.models._preprocessing.FfillImputer`.

:meth:`predict_proba`'s columns are the ``0..K-1`` codes, whose order matches
:attr:`classes_` (``LabelEncoder.classes_``, sklearn-sorted) exactly.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from rba.config import RANDOM_SEED
from rba.models._preprocessing import balanced_sample_weight
from rba.models.base import BaseModel


class XGBClassifierWrapper(BaseModel):
    """XGBoost gradient-boosted-trees classifier over the meeting feature matrix.

    Wraps :class:`xgboost.XGBClassifier` behind the :class:`~rba.models.base.Model`
    contract. :meth:`_fit` label-encodes ``y`` to contiguous ``0..K-1`` codes
    (XGBoost rejects strings and gappy int encodings), resolves balanced
    per-sample weights when the caller passes none, and trains the booster on the
    raw (NaN-carrying) ``X``. :meth:`predict` decodes the booster's integer codes
    back to the caller's original labels and dtype; :meth:`predict_proba` returns
    the booster's class probabilities, whose columns align with :attr:`classes_`.

    Parameters
    ----------
    n_estimators : int, default 500
        Number of boosting rounds (trees).
    max_depth : int, default 4
        Maximum tree depth per round.
    learning_rate : float, default 0.05
        Shrinkage applied to each tree's contribution (``eta``).
    subsample : float, default 0.8
        Row subsampling fraction per boosting round.
    colsample_bytree : float, default 0.8
        Column subsampling fraction per tree.
    tree_method : str, default ``"hist"``
        Tree-construction algorithm; ``"hist"`` is the fast histogram builder.
    reg_alpha : float, default 0.0
        L1 regularisation on leaf weights.
    reg_lambda : float, default 1.0
        L2 regularisation on leaf weights.
    class_weight : {'balanced', None} or dict, default 'balanced'
        How to weight classes when the caller passes no ``sample_weight``.
        ``'balanced'`` uses ``n_samples / (n_classes * class_count)`` per sample;
        a dict maps ``class → weight``; ``None`` weights every sample equally.
        XGBoost has no native ``class_weight`` arg, so this is applied via
        per-sample weights at fit time.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for XGBoost's subsampling / column sampling.
    n_jobs : int or None, default None
        Threads for tree construction; ``None`` lets XGBoost choose.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Original class labels in sklearn-sorted order
        (``LabelEncoder.classes_``); the column order of :meth:`predict_proba`.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        class_weight: Any = "balanced",
        random_state: int = RANDOM_SEED,
        n_jobs: int | None = None,
    ) -> None:
        self._n_estimators = int(n_estimators)
        self._max_depth = int(max_depth)
        self._learning_rate = float(learning_rate)
        self._subsample = float(subsample)
        self._colsample_bytree = float(colsample_bytree)
        self._tree_method = str(tree_method)
        self._reg_alpha = float(reg_alpha)
        self._reg_lambda = float(reg_lambda)
        # Pass class_weight through as-is: the string 'balanced', None, or a dict.
        self._class_weight = class_weight
        self._random_state = int(random_state)
        self._n_jobs = None if n_jobs is None else int(n_jobs)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Label-encode ``y``, resolve weights, and fit the booster on raw ``X``.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._encoder = LabelEncoder().fit(np.asarray(y))
        y_enc = self._encoder.transform(np.asarray(y))
        self.classes_ = self._encoder.classes_
        weight = balanced_sample_weight(y_enc, sample_weight, self._class_weight)
        self._model = XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            tree_method=self._tree_method,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
        )
        self._model.fit(X, y_enc, sample_weight=weight)
        logger.debug(
            "XGBClassifierWrapper fitted: {} tree(s), depth {}, classes={}.",
            self._n_estimators,
            self._max_depth,
            self.classes_.tolist(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class for each row, decoded to the original labels/dtype.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._encoder.inverse_transform(self._model.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates; columns align with :attr:`classes_`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._model.predict_proba(X)
