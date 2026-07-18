"""Ensemble meta-models — the §8 "build ensemble" experiment.

Two :class:`~rba.models.base.Model` meta-estimators that combine several §7
member models named in ``models.yaml`` and are compared, in §8, against the best
individual model:

- :class:`VotingEnsemble` — averages member class probabilities (**soft** voting,
  the default) or takes a majority vote of member predictions (**hard**). Soft
  voting exposes the averaged probabilities via :meth:`~VotingEnsemble.predict_proba`
  (so calibration still applies); hard voting has no natural probability and
  raises :class:`NotImplementedError` there.
- :class:`StackingEnsemble` — trains a **meta model** on the base models'
  out-of-fold class probabilities. The out-of-fold features are generated with an
  internal **time-series** split (:class:`sklearn.model_selection.TimeSeriesSplit`
  — CONTEXT.md Invariant #2 forbids random k-fold), so a base model never predicts
  a row it was trained on; the base models are then refit on the full training
  fold for prediction time.

Both follow the same meta-estimator shape as :mod:`rba.models.imbalance` /
:mod:`rba.models.selection`: they subclass :class:`~rba.models.base.BaseModel`,
build their members lazily inside :meth:`_fit` via
:func:`rba.models.build_model`, store ``classes_`` in sklearn-sorted order, and
are rebuilt fresh per walk-forward fold by the harness (so the whole ensemble —
members and stacking CV alike — is fit strictly out-of-sample).

Class alignment
---------------
A member trained on a small internal fold may not see every class, so its
``predict_proba`` can be narrower than the ensemble's full class set. Every member
probability matrix is therefore realigned to the ensemble's ``classes_`` (a
missing class contributes a zero column) before averaging / stacking — the same
alignment :func:`rba.validation.thresholds.collect_oof_proba` performs.

``sample_weight`` is forwarded to the members' *final* full-fold fits; it is not
threaded through the internal stacking CV (fold-relative alignment is error-prone
and the harness balances via each member's own ``class_weight``). No network
anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import RANDOM_SEED
from rba.models.base import BaseModel

_VOTING = ("soft", "hard")


def _aligned_proba(model: Any, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Return ``model.predict_proba(X)`` widened to ``classes`` (missing class → 0 col).

    Shapes
    ------
    X: (n, p), classes: (k,) -> (n, k).
    """
    proba = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(model.classes_)
    if model_classes.shape == classes.shape and np.array_equal(model_classes, classes):
        return proba
    col_of = {c: i for i, c in enumerate(classes)}
    aligned = np.zeros((len(X), len(classes)), dtype=float)
    for j, cls in enumerate(model_classes):
        aligned[:, col_of[cls]] = proba[:, j]
    return aligned


class VotingEnsemble(BaseModel):
    """Combine member models by averaging probabilities (soft) or majority vote (hard).

    :meth:`_fit` builds every member named in ``members`` from ``models.yaml`` and
    fits each on the full training fold. :meth:`predict` returns, per row,
    ``classes_[argmax(mean_member_proba)]`` for ``voting='soft'`` or the (weighted)
    majority member prediction for ``voting='hard'``. :meth:`predict_proba` returns
    the (optionally weighted) mean member probability for soft voting and raises
    :class:`NotImplementedError` for hard voting.

    Parameters
    ----------
    members : list[str]
        ``models.yaml`` keys of the member classifiers (each must expose
        ``predict_proba`` for soft voting). Duplicates are allowed (a member can be
        weighted up by listing it twice, though ``weights`` is the cleaner knob).
    voting : {'soft', 'hard'}, default 'soft'
        ``'soft'`` averages member probabilities; ``'hard'`` majority-votes member
        label predictions.
    weights : list[float] or None, default None
        Optional per-member weights (length ``len(members)``). ``None`` weights
        members equally. Applied to probabilities (soft) or vote counts (hard).
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Carried for interface symmetry; voting itself is deterministic.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique training labels — the ``predict_proba`` column order.
    members_ : list
        The fitted member models (most recent :meth:`_fit`).
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        members: list[str],
        voting: str = "soft",
        weights: list[float] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        if voting not in _VOTING:
            raise ValueError(f"voting must be one of {_VOTING}; got {voting!r}.")
        if not members:
            raise ValueError("VotingEnsemble needs at least one member model name.")
        if weights is not None and len(weights) != len(members):
            raise ValueError(
                f"weights length {len(weights)} must match members length {len(members)}."
            )
        self._member_names = list(members)
        self._voting = voting
        self._weights = None if weights is None else np.asarray(weights, dtype=float)
        self._random_state = random_state

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Build and fit every member on the training fold.

        Shapes
        ------
        X: (n, p), y: (n,), sample_weight: (n,) or ``None``.
        """
        from rba.config import load_model_config
        from rba.models import build_model

        self.classes_ = np.unique(np.asarray(y))
        self.members_ = []
        for name in self._member_names:
            model = build_model(load_model_config(name))
            model.fit(X, y, sample_weight=sample_weight)
            self.members_.append(model)
        logger.debug(
            "VotingEnsemble({}): fitted {} member(s), classes={}.",
            self._voting,
            len(self.members_),
            self.classes_.tolist(),
        )

    def _mean_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(Weighted) mean of member probabilities, aligned to ``classes_``."""
        stacked = np.stack(
            [_aligned_proba(m, X, self.classes_) for m in self.members_], axis=0
        )  # (n_members, n, k)
        if self._weights is None:
            return stacked.mean(axis=0)
        w = self._weights / self._weights.sum()
        return np.tensordot(w, stacked, axes=([0], [0]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Soft: argmax mean proba. Hard: (weighted) majority member vote.

        Shapes
        ------
        X: (n, p) -> (n,).
        """
        if self._voting == "soft":
            return self.classes_[np.argmax(self._mean_proba(X), axis=1)]
        # Hard vote: accumulate per-class weighted vote counts from member labels.
        votes = np.zeros((len(X), len(self.classes_)), dtype=float)
        col_of = {c: i for i, c in enumerate(self.classes_)}
        for m_idx, model in enumerate(self.members_):
            weight = 1.0 if self._weights is None else float(self._weights[m_idx])
            preds = np.asarray(model.predict(X))
            for row, label in enumerate(preds):
                votes[row, col_of[label]] += weight
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mean member probability (soft voting only).

        Raises
        ------
        NotImplementedError
            For ``voting='hard'`` — a majority vote has no calibrated probability.

        Shapes
        ------
        X: (n, p) -> (n, k).
        """
        if self._voting != "soft":
            raise NotImplementedError(
                "VotingEnsemble(voting='hard') has no predict_proba; use voting='soft'."
            )
        return self._mean_proba(X)


class StackingEnsemble(BaseModel):
    """Train a meta model on the base models' out-of-fold class probabilities.

    :meth:`_fit`:

    1. Generates out-of-fold (OOF) probabilities for every base model with an
       internal :class:`~sklearn.model_selection.TimeSeriesSplit` — a base model
       predicts a row only when it was trained on strictly-earlier rows, so the
       meta features carry no in-sample leakage.
    2. Fits the meta model (named in ``models.yaml``) on the concatenated OOF
       probabilities of the covered (tail) rows.
    3. Refits every base model on the **full** training fold, for prediction time.

    :meth:`predict` / :meth:`predict_proba` stack the full-fold base models'
    probabilities on ``X`` and delegate to the meta model.

    Parameters
    ----------
    base_models : list[str]
        ``models.yaml`` keys of the base classifiers (each must expose
        ``predict_proba``).
    meta_model : str, default ``"logistic_regression"``
        ``models.yaml`` key of the meta classifier trained on the stacked
        probabilities.
    n_splits : int, default 3
        Internal ``TimeSeriesSplit`` folds used to build the OOF meta features.
        Clamped to ``[2, n_samples - 1]``.
    random_state : int, default :data:`rba.config.RANDOM_SEED`
        Carried for interface symmetry (the splits are deterministic).

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique training labels — the ``predict_proba`` column order.
    base_models_ : list
        The base models refit on the full fold.
    meta_model_ : Model
        The fitted meta classifier.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        base_models: list[str],
        meta_model: str = "logistic_regression",
        n_splits: int = 3,
        random_state: int = RANDOM_SEED,
    ) -> None:
        if not base_models:
            raise ValueError("StackingEnsemble needs at least one base model name.")
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits!r}.")
        self._base_names = list(base_models)
        self._meta_name = str(meta_model)
        self._n_splits = int(n_splits)
        self._random_state = random_state

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Build OOF meta features, fit the meta model, refit bases on the full fold.

        Shapes
        ------
        X: (n, p), y: (n,), sample_weight: (n,) or ``None``.
        """
        from sklearn.model_selection import TimeSeriesSplit

        from rba.config import load_model_config
        from rba.models import build_model

        self.classes_ = np.unique(np.asarray(y))
        y_arr = np.asarray(y)
        n = len(X)
        n_splits = max(2, min(self._n_splits, n - 1))

        # 1. Internal time-series OOF probabilities → meta features for the tail rows.
        oof_rows: list[int] = []
        oof_features: list[np.ndarray] = []
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for tr_idx, te_idx in splitter.split(X):
            per_base = []
            for name in self._base_names:
                model = build_model(load_model_config(name))
                model.fit(X.iloc[tr_idx], pd.Series(y_arr[tr_idx]))
                per_base.append(_aligned_proba(model, X.iloc[te_idx], self.classes_))
            oof_features.append(np.hstack(per_base))  # (len(te), n_base * k)
            oof_rows.extend(te_idx.tolist())

        meta_x = pd.DataFrame(
            np.vstack(oof_features), columns=self._meta_feature_names(), index=oof_rows
        )
        meta_y = pd.Series(y_arr[np.asarray(oof_rows)], index=oof_rows)

        # 2. Fit the meta model on the stacked OOF probabilities.
        self.meta_model_ = build_model(load_model_config(self._meta_name))
        self.meta_model_.fit(meta_x, meta_y)

        # 3. Refit every base model on the full fold (for prediction time).
        self.base_models_ = []
        for name in self._base_names:
            model = build_model(load_model_config(name))
            model.fit(X, pd.Series(y_arr, index=X.index), sample_weight=sample_weight)
            self.base_models_.append(model)
        logger.debug(
            "StackingEnsemble: {} base(s) + meta={}, {} OOF meta-rows (n_splits={}).",
            len(self.base_models_),
            self._meta_name,
            len(oof_rows),
            n_splits,
        )

    def _meta_feature_names(self) -> list[str]:
        """Column names for the stacked probability matrix (``<base>_<class>``)."""
        return [f"{name}__{cls}" for name in self._base_names for cls in self.classes_.tolist()]

    def _meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Stack the full-fold base models' aligned probabilities on ``X``."""
        per_base = [_aligned_proba(m, X, self.classes_) for m in self.base_models_]
        return pd.DataFrame(np.hstack(per_base), columns=self._meta_feature_names(), index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Stack base probabilities, then delegate to the meta model's :meth:`predict`.

        Shapes
        ------
        X: (n, p) -> (n,).
        """
        return self.meta_model_.predict(self._meta_features(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Stack base probabilities, then delegate to the meta model's :meth:`predict_proba`.

        Shapes
        ------
        X: (n, p) -> (n, k).
        """
        return self.meta_model_.predict_proba(self._meta_features(X))
