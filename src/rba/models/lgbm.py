"""LightGBM gradient-boosting classifier wrapper — the §7 boosted-tree model.

Wraps :class:`lightgbm.LGBMClassifier` (the scikit-learn API of Microsoft's
LightGBM) in the shared :class:`~rba.models.base.Model` contract so it is
swappable into the §7 evaluation harness exactly like every other model, and is
registered in ``src/rba/config/models.yaml`` under ``rba.models.lgbm``.

Why the wrapper is thin
-----------------------
LightGBM needs **none** of the preprocessing the linear / kernel wrappers rely
on. Two native capabilities do the work a :class:`~rba.models._preprocessing`
step would otherwise provide:

- **NaN handling.** LightGBM's histogram tree learner routes missing values to
  whichever split side lowers the loss, so a NaN feature matrix is passed
  through untouched — no :class:`~rba.models._preprocessing.FfillImputer`.
- **String labels.** LightGBM's internal label encoder ingests string targets
  directly and sets ``.classes_`` to the original labels in sorted order, so we
  need no manual :class:`~sklearn.preprocessing.LabelEncoder`; ``y`` is passed
  through verbatim and :meth:`predict` returns the original label values/dtype.

Native-alias → sklearn-param mapping
------------------------------------
The ``models.yaml`` ``default:`` block uses LightGBM's *native* hyper-parameter
aliases (``feature_fraction`` / ``bagging_fraction``), but the sklearn-API
constructor names them ``colsample_bytree`` / ``subsample``. Passing the native
names to :class:`~lightgbm.LGBMClassifier` both emits an alias warning **and**,
for bagging, silently no-ops unless the companion frequency is set. So the
wrapper maps:

- ``feature_fraction`` → ``colsample_bytree`` (per-tree feature subsampling),
- ``bagging_fraction`` → ``subsample`` **with** ``subsample_freq=1`` — the
  frequency MUST be ``> 0`` or row bagging never activates,
- ``min_data_in_leaf`` → ``min_child_samples`` (minimum samples per leaf).

``verbose=-1`` silences LightGBM's "No further splits with positive gain" /
small-data chatter that otherwise floods logs on the tiny walk-forward windows.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from loguru import logger
import numpy as np
import pandas as pd

from rba.config import RANDOM_SEED
from rba.models.base import BaseModel


class LightGBMClassifierWrapper(BaseModel):
    """LightGBM gradient-boosted-tree classifier under the shared Model contract.

    Delegates all learning to a scikit-learn-API :class:`~lightgbm.LGBMClassifier`
    built in :meth:`_fit`. NaN features and string labels are handled natively by
    LightGBM (see the module docstring), so this wrapper adds no imputation and no
    label encoding — it only maps the native-alias hyper-parameter names to their
    sklearn-API equivalents and forwards ``sample_weight``.

    Parameters
    ----------
    n_estimators : int, default 500
        Number of boosting rounds (trees).
    num_leaves : int, default 31
        Maximum leaves per tree — LightGBM's primary complexity control.
    learning_rate : float, default 0.05
        Shrinkage applied to each tree's contribution.
    feature_fraction : float, default 0.8
        Per-tree column subsampling fraction. Mapped to the sklearn-API
        ``colsample_bytree`` to avoid LightGBM's native-alias warning.
    bagging_fraction : float, default 0.8
        Per-iteration row subsampling fraction. Mapped to the sklearn-API
        ``subsample`` **and paired with** ``subsample_freq=1`` so the bagging
        actually activates (with the default ``subsample_freq=0`` it is a no-op).
    class_weight : str or dict or None, default ``"balanced"``
        Passed through as-is to :class:`~lightgbm.LGBMClassifier`: the string
        ``"balanced"`` reweights classes by inverse frequency, a dict gives
        explicit per-class weights, and ``None`` weights every class equally.
    min_data_in_leaf : int, default 20
        Minimum samples per leaf (regularisation). Mapped to the sklearn-API
        ``min_child_samples``.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for LightGBM's feature/row subsampling — reproducible fits.
    n_jobs : int or None, default None
        Thread count for LightGBM; ``None`` lets LightGBM pick.
    verbose : int, default -1
        LightGBM verbosity; ``-1`` silences its small-data chatter.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels (LightGBM's native encoder order) — the column
        order of :meth:`predict_proba`, in the original label dtype.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        class_weight: Any = "balanced",
        min_data_in_leaf: int = 20,
        random_state: int = RANDOM_SEED,
        n_jobs: int | None = None,
        verbose: int = -1,
    ) -> None:
        self._n_estimators = int(n_estimators)
        self._num_leaves = int(num_leaves)
        self._learning_rate = float(learning_rate)
        self._feature_fraction = float(feature_fraction)
        self._bagging_fraction = float(bagging_fraction)
        # Pass class_weight through untouched: 'balanced', None, or a dict.
        self._class_weight = class_weight
        self._min_data_in_leaf = int(min_data_in_leaf)
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._verbose = int(verbose)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Build and fit the sklearn-API LGBMClassifier; record ``classes_``.

        Maps the native aliases ``feature_fraction`` / ``bagging_fraction`` /
        ``min_data_in_leaf`` to ``colsample_bytree`` / ``subsample`` (with
        ``subsample_freq=1``) / ``min_child_samples`` — see the module docstring.
        ``y`` is passed through verbatim (LightGBM encodes string labels natively)
        and NaN in ``X`` is left for LightGBM to split on.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._model = LGBMClassifier(
            n_estimators=self._n_estimators,
            num_leaves=self._num_leaves,
            learning_rate=self._learning_rate,
            colsample_bytree=self._feature_fraction,  # feature_fraction alias
            subsample=self._bagging_fraction,  # bagging_fraction alias
            subsample_freq=1,  # freq MUST be >0 or bagging is a no-op
            min_child_samples=self._min_data_in_leaf,  # min_data_in_leaf alias
            class_weight=self._class_weight,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )
        self._model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self._model.classes_
        logger.debug(
            "LightGBMClassifierWrapper fitted: {} tree(s), classes={}.",
            self._n_estimators,
            self.classes_.tolist(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class label for each row of ``X`` (original label dtype).

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates, columns aligned to :attr:`classes_`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._model.predict_proba(X)
