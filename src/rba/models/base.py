"""Shared model interface for every estimator in :mod:`rba.models`.

This module anchors the model ecosystem described in
``src/rba/config/models.yaml``: every baseline and ML model (``majority_class`` /
``persistence`` / ``taylor_rule`` / ``market_implied`` / the sklearn / XGBoost /
LightGBM / ordinal / xRFM wrappers / the ensembles) implements the same small
surface so the §7 evaluation harness can drive them interchangeably — the future
``build_model(cfg)`` factory returns *some* concrete class and the harness only
ever sees the :class:`Model` type.

Two pieces
----------
:class:`Model`
    The structural :class:`typing.Protocol` (``@runtime_checkable``) that *is* the
    contract — ``fit`` / ``predict`` / ``predict_proba`` / ``feature_names_`` —
    exactly as specified in CONTEXT.md's "Model interface contract". The harness
    and the future registry reference this type; anything with the right members
    satisfies it, no inheritance required.
:class:`BaseModel`
    A light concrete base that DRYs the boilerplate every model would otherwise
    repeat: :meth:`~BaseModel.fit` stores ``feature_names_`` from ``X.columns`` and
    returns ``self``, and a default :meth:`~BaseModel.predict_proba` raises
    :class:`NotImplementedError` — so pure regressors (``taylor_rule``) inherit the
    correct "no probabilities" behaviour for free, while classifiers override it.
    Subclasses implement only the model-specific :meth:`~BaseModel._fit` and
    :meth:`~BaseModel.predict` hooks.

Task kind (classification / regression / ordinal) is **not** carried as an in-code
marker here: ``models.yaml``'s ``task:`` field is the single source of truth for
the harness, and a regressor advertises "no probabilities" by raising
:class:`NotImplementedError` from :meth:`predict_proba` (which the harness probes).
See CONTEXT.md "Adding a new model".

No network anywhere in this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Self, runtime_checkable

from loguru import logger
import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Structural interface every model in :mod:`rba.models` satisfies.

    A model is fitted on a meeting-indexed feature matrix and target, predicts the
    target for new meetings, optionally exposes class probabilities (classification
    / ordinal only), and reports the feature names it was fitted on. Baselines and
    ML models alike implement this so they are swappable into any evaluation loop.

    Notes
    -----
    ``@runtime_checkable`` makes ``isinstance(obj, Model)`` work for tests and the
    future model registry, but note the standard caveat: it verifies only that the
    required **members are present**, never that their signatures or types match. It
    is a structural presence check, not a type check.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Fit the model and return ``self``.

        Parameters
        ----------
        X
            Feature matrix, one row per meeting.
        y
            Target aligned to ``X``'s rows.
        sample_weight
            Optional per-sample weights; ``None`` weights every sample equally.

        Returns
        -------
        Model
            ``self``, fitted — enables call chaining.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates (classification / ordinal only).

        Raises
        ------
        NotImplementedError
            For pure regressors, which have no class probabilities.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        ...

    @property
    def feature_names_(self) -> list[str]:
        """Column names seen during :meth:`fit`, in order."""
        ...


class BaseModel(ABC):
    """Concrete base DRY-ing the boilerplate shared by every :class:`Model`.

    Satisfies the :class:`Model` protocol structurally. Subclasses implement only
    the model-specific :meth:`_fit` and :meth:`predict` hooks; they inherit:

    - **feature-name storage** — :meth:`fit` records ``list(X.columns)`` before
      delegating, and :attr:`feature_names_` exposes it (raising until fitted);
    - :meth:`fit` **returning ``self``** for call chaining;
    - a default :meth:`predict_proba` that **raises** :class:`NotImplementedError`,
      the correct behaviour for a pure regressor — classification / ordinal models
      override it.
    """

    # Class-level default so ``feature_names_`` is safe even for a subclass that
    # never calls ``super().__init__`` and before ``fit`` has run.
    _feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> Self:
        """Store the feature names, run the subclass :meth:`_fit`, return ``self``.

        Parameters
        ----------
        X
            Feature matrix, one row per meeting. Its columns become
            :attr:`feature_names_`.
        y
            Target aligned to ``X``'s rows.
        sample_weight
            Optional per-sample weights; ``None`` weights every sample equally.

        Returns
        -------
        BaseModel
            ``self``, fitted.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        self._feature_names = list(X.columns)
        logger.debug(
            "Fitting {} on {} feature(s), {} sample(s).",
            type(self).__name__,
            len(self._feature_names),
            len(X),
        )
        self._fit(X, y, sample_weight)
        return self

    @abstractmethod
    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Model-specific fitting, called by :meth:`fit` after names are stored.

        Subclasses implement their learning here; feature-name storage and the
        ``return self`` contract are handled by :meth:`fit`.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class-probability estimates; not defined for pure regressors.

        The default raises :class:`NotImplementedError`, so a regressor subclass
        inherits the correct "no probabilities" behaviour. Classification / ordinal
        subclasses override this.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides it.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support predict_proba "
            "(regressors have no class probabilities)."
        )

    @property
    def feature_names_(self) -> list[str]:
        """Column names seen during :meth:`fit`, in order.

        Raises
        ------
        RuntimeError
            If accessed before :meth:`fit` has been called.
        """
        if self._feature_names is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit before "
                "accessing feature_names_."
            )
        return self._feature_names
