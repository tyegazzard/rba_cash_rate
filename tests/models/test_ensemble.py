"""Tests for ``rba.models.ensemble`` — the §8 Voting / Stacking meta-models.

Covers :class:`VotingEnsemble` (soft + hard) and :class:`StackingEnsemble`: they
satisfy the Model protocol, produce valid probability distributions, combine the
members named in ``models.yaml``, align a member that missed a class on an
internal fold, build from the ``voting_ensemble`` / ``stacking_ensemble`` yaml
entries, and re-fit per fold across a walk-forward split. Members are the cheap
linear models where possible so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import load_model_config
from rba.models import Model, build_model
from rba.models.ensemble import StackingEnsemble, VotingEnsemble, _aligned_proba
from rba.validation import WalkForwardSplit


def _frame(n: int = 90, p: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """A separable-ish 3-class frame, mildly imbalanced toward 'hold'."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    # Build a label with real signal so the meta model has something to learn.
    score = X["f0"] - X["f1"]
    y = pd.Series(np.where(score > 0.6, "hike", np.where(score < -0.6, "cut", "hold")))
    return X, y


_MEMBERS = ["logistic_regression", "random_forest_classifier"]


# =============================================================================
# VotingEnsemble.
# =============================================================================
def test_voting_soft_is_a_model_and_proba_sums_to_one() -> None:
    X, y = _frame()
    model = VotingEnsemble(members=_MEMBERS, voting="soft").fit(X, y)
    assert isinstance(model, Model)
    assert model.fit(X, y) is model
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})


def test_voting_hard_predicts_but_has_no_proba() -> None:
    X, y = _frame()
    model = VotingEnsemble(members=_MEMBERS, voting="hard").fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


def test_voting_weights_shift_the_average() -> None:
    X, y = _frame()
    unweighted = VotingEnsemble(members=_MEMBERS, voting="soft").fit(X, y).predict_proba(X)
    weighted = (
        VotingEnsemble(members=_MEMBERS, voting="soft", weights=[5.0, 1.0])
        .fit(X, y)
        .predict_proba(X)
    )
    # Weighting one member heavily must change the averaged probabilities.
    assert not np.allclose(unweighted, weighted)


def test_voting_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="voting"):
        VotingEnsemble(members=_MEMBERS, voting="medium")
    with pytest.raises(ValueError, match="at least one member"):
        VotingEnsemble(members=[])
    with pytest.raises(ValueError, match="weights length"):
        VotingEnsemble(members=_MEMBERS, weights=[1.0])


# =============================================================================
# StackingEnsemble.
# =============================================================================
def test_stacking_is_a_model_and_proba_sums_to_one() -> None:
    X, y = _frame()
    model = StackingEnsemble(base_models=_MEMBERS, meta_model="logistic_regression").fit(X, y)
    assert isinstance(model, Model)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    # Meta model trained on OOF probabilities; base models refit on the full fold.
    assert len(model.base_models_) == len(_MEMBERS)


def test_stacking_meta_features_have_base_times_class_columns() -> None:
    X, y = _frame()
    model = StackingEnsemble(base_models=_MEMBERS).fit(X, y)
    feats = model._meta_features(X)
    assert feats.shape == (len(X), len(_MEMBERS) * len(model.classes_))


def test_stacking_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="at least one base"):
        StackingEnsemble(base_models=[])
    with pytest.raises(ValueError, match="n_splits"):
        StackingEnsemble(base_models=_MEMBERS, n_splits=1)


# =============================================================================
# Class alignment helper.
# =============================================================================
def test_aligned_proba_widens_a_missing_class() -> None:
    X, y = _frame()
    # Fit a member on a slice that omits 'cut' entirely → its classes_ is narrower.
    mask = y != "cut"
    member = build_model(load_model_config("logistic_regression")).fit(X[mask], y[mask])
    member_classes = np.asarray(member.classes_)  # type: ignore[attr-defined]  # concrete wrapper exposes classes_
    assert "cut" not in set(member_classes.tolist())
    full_classes = np.array(["cut", "hike", "hold"])
    aligned = _aligned_proba(member, X, full_classes)
    assert aligned.shape == (len(X), 3)
    # The missing 'cut' column is all zeros; rows still sum to 1 over seen classes.
    cut_col = list(full_classes).index("cut")
    assert np.allclose(aligned[:, cut_col], 0.0)
    np.testing.assert_allclose(aligned.sum(axis=1), np.ones(len(X)), atol=1e-6)


# =============================================================================
# models.yaml factory + walk-forward re-fit.
# =============================================================================
def test_build_model_resolves_voting_and_stacking() -> None:
    voting = build_model(load_model_config("voting_ensemble"))
    stacking = build_model(load_model_config("stacking_ensemble"))
    assert isinstance(voting, VotingEnsemble)
    assert isinstance(stacking, StackingEnsemble)


def test_stacking_refits_per_fold_in_walk_forward() -> None:
    X, y = _frame(n=90)
    splitter = WalkForwardSplit(initial_train_size=50, test_size=10, step=10)
    n_folds = 0
    for train_idx, test_idx in splitter.split(X):
        model = StackingEnsemble(base_models=_MEMBERS).fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        assert preds.shape == (len(test_idx),)
        n_folds += 1
    assert n_folds >= 2
