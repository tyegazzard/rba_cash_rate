"""Baseline models — the §6 non-learning floors every ML model must beat.

These are the simplest possible predictors, run so that later models are judged
against a meaningful reference rather than against zero. Each implements the
:class:`~rba.models.base.Model` contract (via :class:`~rba.models.base.BaseModel`)
so it is swappable into the §7 evaluation harness exactly like an ML model, and
each is registered in ``src/rba/config/models.yaml`` under ``rba.models.baselines``.

This module will hold all four baselines named in ``models.yaml``
(``MajorityClass`` / ``Persistence`` / ``TaylorRule`` / ``MarketImplied``). Only
:class:`MajorityClass` lands here first; the other three are separate checklist
items and will be added alongside it.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.models.base import BaseModel


class MajorityClass(BaseModel):
    """Predict the most frequent class; ``predict_proba`` is the empirical prior.

    The simplest classification floor. :meth:`_fit` records the class distribution
    of ``y`` (optionally weighted by ``sample_weight``); :meth:`predict` returns the
    single most-frequent class for every row, and :meth:`predict_proba` returns that
    same empirical prior for every row — a constant, base-rate-calibrated vector so
    that log-loss / Brier are meaningful floors (a one-hot would be degenerate). It
    ignores ``X`` entirely (beyond its row count and, via
    :class:`~rba.models.base.BaseModel`, its column names).

    Ties for the modal class break to the lowest sorted class label
    (``np.argmax`` over the sorted classes). Deterministic — no random state.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels seen in ``y``; the column order of
        :meth:`predict_proba`.
    class_prior_ : numpy.ndarray
        Empirical (optionally weighted) class frequencies aligned to
        :attr:`classes_`; sums to 1.
    majority_class_ : Any
        The most frequent class label — ``classes_[argmax(class_prior_)]``.
    """

    classes_: np.ndarray
    class_prior_: np.ndarray
    majority_class_: Any

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Record the (optionally weighted) class distribution of ``y``.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        labels = np.asarray(y)
        classes, inverse = np.unique(labels, return_inverse=True)
        if sample_weight is None:
            counts = np.bincount(inverse, minlength=len(classes)).astype(float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            counts = np.bincount(inverse, weights=weights, minlength=len(classes))

        self.classes_ = classes
        self.class_prior_ = counts / counts.sum()
        self.majority_class_ = classes[int(np.argmax(counts))]
        logger.debug(
            "MajorityClass fitted: majority={!r}, prior={}.",
            self.majority_class_,
            dict(zip(classes.tolist(), self.class_prior_.round(4).tolist())),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the majority class for every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return np.full(len(X), self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the empirical class prior, broadcast to every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return np.tile(self.class_prior_, (len(X), 1))
