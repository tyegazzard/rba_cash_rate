"""SMOTE resampling meta-model — the §7 imbalance experiment vs. class-weight.

`class_weight='balanced'` reweights the loss; **SMOTE** instead synthesises new
minority-class training rows (interpolating between a minority sample and its
k-nearest minority neighbours) so the learner sees a balanced set. This module
provides :class:`SMOTEModel`, a :class:`~rba.models.base.Model` that resamples the
**training fold** with SMOTE and fits a base model on the balanced data — so a
run through the §7 harness compares SMOTE (this wrapper) against the class-weight
default (the plain base model) fold-for-fold, leakage-safely (SMOTE never sees
the test fold, because the harness rebuilds per fold).

Two constraints SMOTE imposes, handled here:

- **It needs a complete numeric matrix.** SMOTE's k-NN can't span NaN, so the
  feature matrix is first densified with a per-fold
  :class:`~rba.models._preprocessing.FfillImputer` (the project's leakage-safe
  ``dense_fallback``). A consequence worth noting: a tree base that *would* have
  split on NaN natively instead receives the imputed matrix — SMOTE and
  native-NaN handling are mutually exclusive.
- **It needs enough minority samples.** ``k_neighbors`` is clamped to
  ``min_class_count - 1``; a class with a single member (which SMOTE cannot
  interpolate) disables resampling for that fold rather than raising.

For a clean SMOTE-vs-class-weight comparison the base model's ``class_weight`` is
usually set to ``None`` (via ``base_params``) so the two mechanisms aren't
stacked — the registered ``smote_xgboost`` config does this.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import RANDOM_SEED
from rba.models._preprocessing import FfillImputer
from rba.models.base import BaseModel


class SMOTEModel(BaseModel):
    """Fit a base model on SMOTE-resampled training folds (imbalance via oversampling).

    :meth:`_fit` densifies ``X`` (per-fold :class:`~rba.models._preprocessing.FfillImputer`),
    SMOTE-oversamples the minority classes up to the majority count, and fits the
    base model (named in ``models.yaml``) on the balanced matrix. :meth:`predict`
    / :meth:`predict_proba` densify with the same fitted imputer and delegate.
    ``sample_weight`` is ignored — SMOTE *is* the imbalance mechanism.

    Parameters
    ----------
    base_model_name : str, default ``"xgboost_classifier"``
        A ``models.yaml`` key naming the base estimator.
    k_neighbors : int, default 5
        SMOTE neighbourhood size; clamped down to ``min_class_count - 1`` per fold.
    base_params : dict or None, default None
        Overrides merged over the base model's ``default`` block — set
        ``{'class_weight': None}`` here for a clean SMOTE-only comparison.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for SMOTE's synthesis.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The base model's classes.
    resampled_counts_ : dict
        Post-SMOTE class counts on the most recent fit (for logging / experiments).
    """

    def __init__(
        self,
        *,
        base_model_name: str = "xgboost_classifier",
        k_neighbors: int = 5,
        base_params: dict[str, Any] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._base_model_name = str(base_model_name)
        self._k_neighbors = int(k_neighbors)
        self._base_params = dict(base_params) if base_params else {}
        self._random_state = random_state

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Densify, SMOTE-resample the training fold, and fit the base model.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: ignored.
        """
        # Lazy imports: imblearn is heavy, and build_model lives in rba.models.__init__.
        from imblearn.over_sampling import SMOTE

        from rba.config import load_model_config
        from rba.models import build_model

        self._imputer = FfillImputer().fit(X)
        x_dense = self._imputer.transform(X)
        y_arr = np.asarray(y)

        counts = pd.Series(y_arr).value_counts()
        min_count = int(counts.min())
        if len(counts) < 2 or min_count < 2:
            # SMOTE needs ≥2 classes and ≥2 samples in the smallest — a single-class
            # or singleton-class training fold (possible early in walk-forward) is
            # left unresampled rather than raising.
            logger.debug(
                "SMOTEModel: {} class(es), smallest has {} sample(s) — SMOTE skipped this fold.",
                len(counts),
                min_count,
            )
            x_res, y_res = x_dense, y_arr
        else:
            k = min(self._k_neighbors, min_count - 1)
            x_res, y_res = SMOTE(random_state=self._random_state, k_neighbors=k).fit_resample(
                x_dense, y_arr
            )
        self.resampled_counts_ = {k: int(v) for k, v in pd.Series(y_res).value_counts().items()}

        cfg = dict(load_model_config(self._base_model_name))
        if self._base_params:
            cfg["default"] = {**(cfg.get("default") or {}), **self._base_params}
        self._inner = build_model(cfg)
        # Feed the base a DataFrame so the wrappers keep feature names (no warnings).
        self._inner.fit(pd.DataFrame(x_res, columns=X.columns), pd.Series(y_res))
        self.classes_ = getattr(self._inner, "classes_", None)
        logger.debug(
            "SMOTEModel: base={}, resampled counts={}.",
            self._base_model_name,
            self.resampled_counts_,
        )

    def _dense(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute ``X`` with the fitted per-fold imputer, back to a named frame."""
        return pd.DataFrame(self._imputer.transform(X), columns=X.columns, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Densify, then delegate to the base model's :meth:`predict`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._inner.predict(self._dense(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Densify, then delegate to the base model's :meth:`predict_proba`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._inner.predict_proba(self._dense(X))
