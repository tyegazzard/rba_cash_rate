"""Ordinal threshold regression wrapper — the §7 ``ordinal`` task model.

Wraps :mod:`mord`'s cumulative-link threshold models
(:class:`mord.LogisticAT` / :class:`mord.LogisticIT`) behind the shared
:class:`~rba.models.base.Model` contract so the ordinal target — the cash-rate
*move* encoded as ordered integer buckets (e.g. ``{-50, -25, 0, +25, +50}`` bp,
or ``{0, 1, 2, 3}``) — is modelled with an estimator that *knows the classes are
ordered*, unlike a plain multinomial classifier. Registered in
``src/rba/config/models.yaml`` as ``ordinal_logistic``
(``rba.models.ordinal.OrdinalLogistic``).

The single wrapper class is :class:`OrdinalLogistic`, exposing two threshold
variants via the ``variant`` kwarg:

- ``'AT'`` — **all-threshold** (:class:`mord.LogisticAT`, the default): every
  ordinal threshold contributes to every sample's loss.
- ``'IT'`` — **immediate-threshold** (:class:`mord.LogisticIT`): only the two
  thresholds bracketing a sample's true class contribute.

Two implementation details are load-bearing:

- **Label encoding (mord requirement).** ``mord``'s threshold models require the
  target to be *exactly* the consecutive integers ``[0 .. K-1]`` that are
  present — passing the raw ordinal labels (``-25``, ``+50``, …) raises
  ``ValueError``. So :meth:`_fit` runs a :class:`~sklearn.preprocessing.LabelEncoder`,
  which maps labels to ``0..K-1`` **in sorted order** — and because the ordinal
  labels sort into their ordinal order, that encoding *is* the ordinal order.
  :meth:`predict` inverse-transforms back to the original label values.
- **Complete, scaled matrix.** These are *linear* threshold models: they cannot
  ingest NaN and are scale-sensitive. The estimator is therefore wrapped in a
  per-fold :class:`~sklearn.pipeline.Pipeline` of
  :class:`~rba.models._preprocessing.FfillImputer` (the leakage-safe
  ``dense_fallback: ffill`` — CONTEXT.md Invariants #3/#4) →
  :class:`~sklearn.preprocessing.StandardScaler` → the ``mord`` model, so every
  fit sees a dense, standardised design matrix and no whole-series statistic
  leaks across the train/test boundary.

``mord`` is scikit-learn-compatible: it supports ``sample_weight`` (forwarded to
the pipeline's final step) and ``predict_proba`` (overridden here, columns
``0..K-1`` aligned to :attr:`~OrdinalLogistic.classes_`).

No network anywhere in this module.
"""

from __future__ import annotations

from loguru import logger
import mord
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rba.models._preprocessing import FfillImputer
from rba.models.base import BaseModel

# ``variant`` -> the ``mord`` threshold estimator class it selects.
_VARIANTS = {"AT": mord.LogisticAT, "IT": mord.LogisticIT}


class OrdinalLogistic(BaseModel):
    """Ordinal (threshold) logistic regression over integer-encoded ordinal labels.

    Fits :class:`mord.LogisticAT` (all-threshold, the default) or
    :class:`mord.LogisticIT` (immediate-threshold) on the cash-rate *move* target,
    which is an **ordered** integer bucket — so the model exploits the ordering
    that a plain multinomial classifier throws away.

    The raw ordinal labels are integer-encoded to the consecutive
    ``0..K-1`` that ``mord`` requires (via
    :class:`~sklearn.preprocessing.LabelEncoder`, which sorts — and for integer
    ordinal labels the sort order *is* the ordinal order), then decoded back to
    the original label values at :meth:`predict`. Because the estimator is a
    linear threshold model, it is wrapped in a per-fold
    :class:`~sklearn.pipeline.Pipeline` —
    :class:`~rba.models._preprocessing.FfillImputer` (dense fallback) →
    :class:`~sklearn.preprocessing.StandardScaler` → the ``mord`` model — so it
    receives a complete, standardised matrix without leaking across folds. Unlike
    a pure regressor, :meth:`predict_proba` is supported.

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularisation strength passed to the ``mord`` estimator. The only
        parameter set by ``models.yaml``'s ``default:`` block.
    variant : {'AT', 'IT'}, default 'AT'
        ``'AT'`` selects :class:`mord.LogisticAT` (all-threshold); ``'IT'`` selects
        :class:`mord.LogisticIT` (immediate-threshold). Any other value raises
        :class:`ValueError`.
    max_iter : int, default 1000
        Maximum optimiser iterations for the underlying ``mord`` estimator.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The original ordinal labels in sorted (== ordinal) order; the column order
        of :meth:`predict_proba` and the label space of :meth:`predict`.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        variant: str = "AT",
        max_iter: int = 1000,
    ) -> None:
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}; got {variant!r}.")
        self._alpha = float(alpha)
        self._variant = variant
        self._max_iter = int(max_iter)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Integer-encode ``y`` to ``0..K-1`` and fit the impute→scale→mord pipeline.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._encoder = LabelEncoder().fit(np.asarray(y))
        y_enc = self._encoder.transform(np.asarray(y))
        self.classes_ = self._encoder.classes_

        base = _VARIANTS[self._variant](alpha=self._alpha, max_iter=self._max_iter)
        self._pipeline = Pipeline(
            [
                ("impute", FfillImputer()),
                ("scale", StandardScaler()),
                ("ord", base),
            ]
        )
        fit_params = {"ord__sample_weight": sample_weight} if sample_weight is not None else {}
        self._pipeline.fit(X, y_enc, **fit_params)
        logger.debug(
            "OrdinalLogistic fitted: variant={}, alpha={}, classes={}.",
            self._variant,
            self._alpha,
            self.classes_.tolist(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the ordinal move for each row, decoded to the original labels.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return self._encoder.inverse_transform(self._pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Per-class probabilities; columns ``0..K-1`` align to :attr:`classes_`.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return self._pipeline.predict_proba(X)
