"""Shared preprocessing helpers for the §7 ML model wrappers.

Two small, dependency-light pieces the ML wrappers reuse so the same
leakage-safe convention is applied identically everywhere:

:class:`FfillImputer`
    The per-fold, forward-fill **dense fallback** required by the models that
    cannot ingest a NaN matrix — the linear / kernel / neural nets
    (``logistic_regression`` / ``svm_classifier`` / ``mlp_classifier`` in
    :mod:`rba.models.sklearn_wrappers`) and the ordinal threshold model
    (``ordinal_logistic`` in :mod:`rba.models.ordinal`). It realises
    ``features.yaml``'s documented ``dense_fallback: ffill`` strategy as a
    scikit-learn ``Pipeline`` transformer so it is **fit per fold on the
    training window only** (CONTEXT.md Invariant #4), never with a whole-series
    statistic. Tree / boosting models (``random_forest_classifier`` /
    ``xgboost_classifier`` / ``lightgbm_classifier``) split on NaN natively and
    do **not** use this — they receive ``X`` untouched.

:func:`balanced_sample_weight`
    The ``class_weight='balanced'`` equivalent for estimators that have **no**
    ``class_weight`` constructor argument (XGBoost). Computes the per-sample
    weight ``n_samples / (n_classes * class_count)`` — exactly scikit-learn's
    ``compute_sample_weight('balanced', y)`` — used only when the caller did not
    already pass an explicit ``sample_weight`` to :meth:`fit`.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_sample_weight


class FfillImputer(BaseEstimator, TransformerMixin):
    """Per-fold forward-fill dense fallback for models needing a complete matrix.

    Realises ``features.yaml``'s ``dense_fallback: ffill`` as a leakage-safe,
    per-fold :class:`~sklearn.pipeline.Pipeline` step (CONTEXT.md Invariant #4).
    Every series routed through the dense models is a non-stationary level /
    index / price / rate, for which carrying the last known value forward is a
    far more faithful fill than a window mean / median (the reason
    ``features.yaml`` deliberately rejects ``median`` for these columns).

    Fitting records, per column, the **last valid value in the training
    window** — the value carried forward across the train/test boundary.
    :meth:`transform` then, on any block:

    1. forward-fills within the block (``pandas`` ``ffill`` down the rows — the
       past-only carry the raw convention specifies), then
    2. fills any residual **leading** NaN (rows before the column's first valid
       reading in this block) with the recorded training carry value, then
    3. zero-fills any column that was entirely NaN in the training window (no
       information exists to carry — the paired ``<col>_is_missing`` indicator,
       added upstream in ``align.py``, preserves the missingness signal).

    No whole-series or window-median statistic is ever used; the only fitted
    state is the training-window carry vector, so the transform is safe to reuse
    on a held-out block of any length (including a single row).

    Attributes
    ----------
    n_features_in_ : int
        Number of columns seen at :meth:`fit`.
    feature_names_in_ : numpy.ndarray
        Column names seen at :meth:`fit` (``object`` dtype), when ``X`` is a
        :class:`pandas.DataFrame`.
    carry_ : numpy.ndarray
        Per-column carry value — the last valid training reading, with all-NaN
        columns set to ``0.0``. Shape ``(n_features_in_,)``.
    """

    def fit(self, X: Any, y: Any = None) -> "FfillImputer":
        """Record the per-column training-window carry vector.

        Shapes
        ------
        X: (n_samples, n_features) -> self.
        """
        frame = self._as_frame(X)
        self.n_features_in_ = int(frame.shape[1])
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        # Last row after a forward fill == last valid value per column (NaN only
        # for a column that is entirely NaN across the training window).
        filled = frame.ffill().to_numpy(dtype=float)
        carry = filled[-1].copy() if filled.shape[0] else np.full(self.n_features_in_, np.nan)
        self.carry_ = np.where(np.isnan(carry), 0.0, carry)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Forward-fill ``X``, fill residual leading NaN with the training carry.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_features) float ndarray.
        """
        frame = self._as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"FfillImputer fitted on {self.n_features_in_} feature(s) but "
                f"transform got {frame.shape[1]}."
            )
        filled = frame.ffill().to_numpy(dtype=float, copy=True)
        rows, cols = np.where(np.isnan(filled))
        if rows.size:
            filled[rows, cols] = np.take(self.carry_, cols)
        return filled

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Passthrough feature names (the transform preserves columns 1:1)."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

    @staticmethod
    def _as_frame(X: Any) -> pd.DataFrame:
        """Coerce ``X`` to a DataFrame so ``ffill`` operates down the rows."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(np.asarray(X))


def balanced_sample_weight(
    y: Any,
    sample_weight: np.ndarray | None = None,
    class_weight: Any = "balanced",
) -> np.ndarray | None:
    """Resolve the effective per-sample weight for a ``class_weight``-less estimator.

    Precedence: an explicit ``sample_weight`` always wins (returned as a float
    array); otherwise, when ``class_weight == 'balanced'``, return the balanced
    per-sample weights ``n_samples / (n_classes * class_count)`` (scikit-learn's
    :func:`~sklearn.utils.class_weight.compute_sample_weight`); a ``dict``
    ``class_weight`` is forwarded to the same function; ``None`` (or the string
    ``"none"``) yields ``None`` (uniform weighting).

    Parameters
    ----------
    y
        Target labels, one per sample.
    sample_weight
        Caller-supplied weights; when not ``None`` they are used verbatim.
    class_weight
        ``'balanced'`` (default), ``None`` / ``'none'``, or a class→weight dict.

    Returns
    -------
    numpy.ndarray or None
        The effective per-sample weight vector, or ``None`` for uniform.

    Shapes
    ------
    y: (n_samples,) -> (n_samples,) or ``None``.
    """
    if sample_weight is not None:
        return np.asarray(sample_weight, dtype=float)
    if class_weight is None or class_weight == "none":
        return None
    return compute_sample_weight(class_weight, y)
