"""Per-fold feature selection for the high-dimensional (p ≫ n) meeting problem.

The engineered feature frame carries ~681 columns against ~365 meetings, so
every model is fitting in a p ≫ n regime where untuned learners overfit noise.
This module provides supervised **filter** feature selection that is *fit per
fold* — the only leakage-safe way to select in a walk-forward setting (selecting
on the whole series, then evaluating out-of-sample, leaks the test rows into the
choice of features, CONTEXT.md Invariant #3).

Two pieces
----------
:class:`FeatureSelector`
    A scikit-learn transformer that scores every column against the target on the
    **training fold only** (mutual information / ANOVA F / absolute correlation)
    and keeps the top ``k``. :meth:`~FeatureSelector.transform` returns the
    selected columns with their **values untouched** (NaN preserved), so a
    downstream tree model still gets native-NaN splits and a downstream dense
    model still gets its own imputer. Scoring imputes NaN with the per-column
    training median for ranking purposes only.
:class:`SelectingModel`
    A :class:`~rba.models.base.Model` meta-estimator that composes a
    :class:`FeatureSelector` with any base model named in ``models.yaml``. Because
    the §7 harness builds and fits a fresh model per fold, wrapping the base model
    in :class:`SelectingModel` makes the selection **fit per fold** automatically —
    no harness change needed. Registered in ``models.yaml`` as
    ``selecting_xgboost`` (an illustrative default; the tuner varies ``k`` /
    ``method`` / the base model).

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any, overload

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from rba.config import RANDOM_SEED
from rba.models.base import BaseModel

_METHODS: tuple[str, ...] = ("mutual_info", "f_classif", "correlation")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Top-``k`` supervised filter selection, fit per fold (leakage-safe).

    On :meth:`fit` every column of ``X`` is scored against ``y`` using
    :attr:`method`, computed on the passed (training-fold) rows only, and the
    ``k`` highest-scoring columns are retained. :meth:`transform` returns those
    columns **unchanged** — original values and NaN preserved — so downstream
    NaN-native (tree) or dense (imputed) models behave exactly as they would on
    the full matrix, just narrower.

    Scoring handles NaN by imputing each column with its **training-fold median**
    for ranking only (never mutated in the output). Constant / all-NaN columns
    score ``0`` and sort last.

    Parameters
    ----------
    k : int, default 100
        Number of features to keep. Clamped to ``[1, n_features]``.
    method : {'mutual_info', 'f_classif', 'correlation'}, default 'mutual_info'
        ``'mutual_info'`` — :func:`sklearn.feature_selection.mutual_info_classif`
        (non-linear dependence; the default for the classification targets);
        ``'f_classif'`` — ANOVA F-value; ``'correlation'`` — absolute Pearson
        correlation between the feature and the label-encoded target (best suited
        to ordinal / regression targets).
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed for the mutual-information k-NN estimator (reproducibility).

    Attributes
    ----------
    n_features_in_ : int
        Number of columns seen at :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Column names seen at :meth:`fit` (when ``X`` is a DataFrame).
    scores_ : numpy.ndarray
        Per-column score, aligned to the input column order.
    support_mask_ : numpy.ndarray of bool
        Boolean mask of the selected columns, aligned to the input column order.
    selected_features_ : list
        Names (or positional indices) of the selected columns, ranked best-first.
    """

    def __init__(
        self,
        *,
        k: int = 100,
        method: str = "mutual_info",
        random_state: int = RANDOM_SEED,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}; got {method!r}.")
        self.k = int(k)
        self.method = method
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "FeatureSelector":
        """Score every column against ``y`` on this fold; keep the top ``k``.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,) -> self.
        """
        frame = self._as_frame(X)
        self.n_features_in_ = int(frame.shape[1])
        names = np.asarray(frame.columns, dtype=object)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = names

        scoring = self._median_impute(frame)
        y_codes = LabelEncoder().fit_transform(np.asarray(y))
        self.scores_ = self._score(scoring, y_codes)

        k = max(1, min(self.k, self.n_features_in_))
        # Rank by score descending; np.argsort is ascending, so negate. Stable
        # sort keeps original column order among ties (deterministic).
        order = np.argsort(-self.scores_, kind="stable")
        top = order[:k]
        self.support_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_mask_[top] = True
        self.selected_features_ = [names[i] for i in top]
        logger.debug(
            "FeatureSelector({}): kept {}/{} features.",
            self.method,
            k,
            self.n_features_in_,
        )
        return self

    @overload
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def transform(self, X: Any) -> pd.DataFrame | np.ndarray:
        """Return the selected columns, values untouched (NaN preserved).

        A ``DataFrame`` in yields a ``DataFrame`` out; an array in yields an
        array out (the two ``@overload`` signatures encode that so callers that
        pass a frame — every caller in this package — keep a statically-known
        ``DataFrame``).

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, k).
        """
        frame = self._as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"FeatureSelector fitted on {self.n_features_in_} feature(s) but "
                f"transform got {frame.shape[1]}."
            )
        selected = frame.loc[:, self.support_mask_]
        return selected if isinstance(X, pd.DataFrame) else selected.to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the selected columns (for Pipeline introspection)."""
        return np.asarray(self.selected_features_, dtype=object)

    def _score(self, scoring: np.ndarray, y_codes: np.ndarray) -> np.ndarray:
        """Per-column importance score (higher = more relevant); NaN → 0."""
        if self.method == "mutual_info":
            scores = mutual_info_classif(scoring, y_codes, random_state=self.random_state)
        elif self.method == "f_classif":
            scores, _ = f_classif(scoring, y_codes)
        else:  # correlation
            scores = self._abs_correlation(scoring, y_codes)
        return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)

    @staticmethod
    def _abs_correlation(scoring: np.ndarray, y_codes: np.ndarray) -> np.ndarray:
        """Absolute Pearson correlation of each column with the encoded target."""
        y = y_codes.astype(float)
        y_centered = y - y.mean()
        out = np.zeros(scoring.shape[1], dtype=float)
        for j in range(scoring.shape[1]):
            col = scoring[:, j]
            col_centered = col - col.mean()
            denom = np.sqrt((col_centered**2).sum() * (y_centered**2).sum())
            if denom > 0:
                out[j] = abs((col_centered * y_centered).sum() / denom)
        return out

    @staticmethod
    def _median_impute(frame: pd.DataFrame) -> np.ndarray:
        """Float matrix with NaN filled by per-column median (for scoring only)."""
        values = frame.to_numpy(dtype=float, copy=True)
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)  # all-NaN column → 0
        rows, cols = np.where(np.isnan(values))
        if rows.size:
            values[rows, cols] = np.take(medians, cols)
        return values

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X))


class SelectingModel(BaseModel):
    """A :class:`~rba.models.base.Model` that top-``k`` selects features per fold.

    Composes a :class:`FeatureSelector` with a base model named in ``models.yaml``
    (built lazily in :meth:`_fit`). Because the §7 harness rebuilds and fits the
    model afresh on every walk-forward fold, the wrapped selection is **fit per
    fold** with no harness change — the leakage-safe way to select in a p ≫ n
    time-series setting.

    :meth:`_fit` fits the selector on the training fold, reduces ``X`` to the
    selected columns, and fits the base model on the narrowed matrix.
    :meth:`predict` / :meth:`predict_proba` apply the same selection before
    delegating; :attr:`classes_` mirrors the base model's.

    Parameters
    ----------
    base_model_name : str, default ``"xgboost_classifier"``
        A ``models.yaml`` key naming the base estimator to wrap.
    k : int, default 100
        Number of features to keep (see :class:`FeatureSelector`).
    method : {'mutual_info', 'f_classif', 'correlation'}, default 'mutual_info'
        Selection scoring method.
    base_params : dict or None, default None
        Optional overrides merged over the base model's ``models.yaml`` ``default``
        block (so the tuner can vary base-model hyper-parameters through the
        wrapper).
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Seed forwarded to the selector.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The base model's classes (present for classification base models).
    selected_features_ : list
        Features chosen on the most recent :meth:`_fit`.
    """

    def __init__(
        self,
        *,
        base_model_name: str = "xgboost_classifier",
        k: int = 100,
        method: str = "mutual_info",
        base_params: dict[str, Any] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self._base_model_name = str(base_model_name)
        self._k = int(k)
        self._method = str(method)
        self._base_params = dict(base_params) if base_params else {}
        self._random_state = random_state

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit the per-fold selector, then the base model on the reduced matrix.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        # Lazy import breaks the models package import cycle (build_model lives in
        # rba.models.__init__, which does not import this module).
        from rba.config import load_model_config
        from rba.models import build_model

        self._selector = FeatureSelector(
            k=self._k, method=self._method, random_state=self._random_state
        ).fit(X, y)
        x_reduced = self._selector.transform(X)
        self.selected_features_ = self._selector.selected_features_

        cfg = dict(load_model_config(self._base_model_name))
        if self._base_params:
            cfg["default"] = {**(cfg.get("default") or {}), **self._base_params}
        self._inner = build_model(cfg)
        self._inner.fit(x_reduced, y, sample_weight=sample_weight)
        self.classes_ = getattr(self._inner, "classes_", None)
        logger.debug(
            "SelectingModel: base={}, kept {} feature(s).",
            self._base_model_name,
            len(self.selected_features_),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Select, then delegate to the base model's :meth:`predict`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._inner.predict(self._selector.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Select, then delegate to the base model's :meth:`predict_proba`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._inner.predict_proba(self._selector.transform(X))
