"""Tests for ``rba.models.imbalance`` — the SMOTE resampling meta-model.

Covers :class:`SMOTEModel`: it satisfies the Model protocol, oversamples the
minority classes up to the majority count per fold, delegates predict /
predict_proba, densifies NaN inputs, clamps ``k_neighbors`` to a small minority,
skips resampling a singleton class rather than raising, builds from ``models.yaml``
(``smote_xgboost``), and re-fits per fold across a walk-forward split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rba.config import load_model_config
from rba.models import Model, build_model
from rba.models.imbalance import SMOTEModel
from rba.validation import WalkForwardSplit


def _imbalanced_frame(
    n_hold: int = 80, n_min: int = 15, p: int = 6
) -> tuple[pd.DataFrame, pd.Series]:
    """A 3-class frame skewed to 'hold', with a NaN cell to exercise densifying."""
    rng = np.random.default_rng(12)
    n = n_hold + 2 * n_min
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    X.loc[0, "f1"] = np.nan
    labels = np.array(["hold"] * n_hold + ["hike"] * n_min + ["cut"] * n_min)
    labels = labels[rng.permutation(n)]  # interleave so walk-forward folds aren't single-class
    return X, pd.Series(labels)


def _smote(**kw: Any) -> SMOTEModel:
    return SMOTEModel(
        base_model_name="logistic_regression", base_params={"class_weight": None}, **kw
    )


def test_smote_model_is_a_model_and_balances_classes() -> None:
    X, y = _imbalanced_frame(n_hold=80, n_min=15)
    model = _smote().fit(X, y)
    assert isinstance(model, Model)
    assert model.fit(X, y) is model
    # SMOTE oversamples every minority up to the majority count → all equal.
    assert set(model.resampled_counts_.values()) == {80}


def test_smote_predict_and_proba() -> None:
    X, y = _imbalanced_frame()
    model = _smote().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    proba = model.predict_proba(X)
    assert model.classes_ is not None
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_smote_densifies_nan_inputs() -> None:
    X, y = _imbalanced_frame()
    assert X.isna().any().any()  # the fixture carries a NaN
    model = _smote().fit(X, y)  # densify → SMOTE → base fit, all NaN-free
    X_pred = X.copy()
    X_pred.loc[1, "f2"] = np.nan  # NaN at predict too → densified via the fitted imputer
    assert model.predict(X_pred).shape == (len(X_pred),)


def test_smote_clamps_k_neighbors_for_small_minority() -> None:
    # Minority of 3 < default k_neighbors=5 → clamped to 2, must not raise.
    X, y = _imbalanced_frame(n_hold=40, n_min=3)
    model = _smote(k_neighbors=5).fit(X, y)
    assert set(model.resampled_counts_.values()) == {40}


def test_smote_singleton_class_skips_resampling() -> None:
    rng = np.random.default_rng(12)
    n = 21
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(4)})
    y = pd.Series(["hold"] * 20 + ["cut"] * 1)  # 'cut' cannot be interpolated
    model = _smote().fit(X, y)
    # Resampling skipped → counts unchanged; still fits + predicts.
    assert model.resampled_counts_ == {"hold": 20, "cut": 1}
    assert model.predict(X).shape == (n,)


def test_build_model_resolves_smote_xgboost() -> None:
    model = build_model(load_model_config("smote_xgboost"))
    assert isinstance(model, SMOTEModel)
    X, y = _imbalanced_frame()
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    assert set(fitted.resampled_counts_.values()) == {80}  # balanced


def test_smote_refits_per_fold_in_walk_forward() -> None:
    X, y = _imbalanced_frame(n_hold=60, n_min=15)
    splitter = WalkForwardSplit(initial_train_size=40, test_size=10, step=10)
    n_folds = 0
    for train_idx, _test_idx in splitter.split(X):
        model = _smote().fit(X.iloc[train_idx], y.iloc[train_idx])
        assert model.classes_ is not None
        n_folds += 1
    assert n_folds >= 2
