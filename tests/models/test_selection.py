"""Tests for ``rba.models.selection`` — per-fold feature selection (p ≫ n).

Covers :class:`FeatureSelector`: it keeps the top-``k`` columns, ranks an
informative feature above noise, preserves the selected columns' values (NaN
included) on transform, clamps ``k``, supports every scoring method, is fit on
the passed (training-fold) rows ONLY — the leakage-safety property — and raises
on a bad method / feature-count mismatch. And :class:`SelectingModel`: it
satisfies the Model protocol, narrows the base model to ``k`` features, delegates
predict / predict_proba, wraps both a tree and a dense base model, builds from
``models.yaml`` (``selecting_xgboost``), and re-selects per fold across a
:class:`~rba.validation.WalkForwardSplit`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import load_model_config
from rba.models import Model, build_model
from rba.models.selection import FeatureSelector, SelectingModel
from rba.validation import WalkForwardSplit


# =============================================================================
# FeatureSelector.
# =============================================================================
def _signal_frame(n: int = 80, p: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    """``f0`` carries the class signal; ``f1..f{p-1}`` are pure noise."""
    rng = np.random.default_rng(12)
    y = pd.Series(np.tile([0, 1], n // 2))
    cols = {"f0": y.to_numpy() * 3.0 + rng.normal(0, 0.3, n)}
    for j in range(1, p):
        cols[f"f{j}"] = rng.normal(size=n)
    return pd.DataFrame(cols), y


def test_selector_keeps_k_features() -> None:
    X, y = _signal_frame()
    sel = FeatureSelector(k=3).fit(X, y)
    out = sel.transform(X)
    assert out.shape == (len(X), 3)
    assert len(sel.selected_features_) == 3


def test_selector_ranks_signal_above_noise() -> None:
    X, y = _signal_frame()
    for method in ("mutual_info", "f_classif", "correlation"):
        sel = FeatureSelector(k=1, method=method).fit(X, y)
        assert sel.selected_features_ == ["f0"], f"{method} did not pick the signal feature"


def test_selector_transform_preserves_values_and_nan() -> None:
    X = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0], "c": [5.0, 6.0, 7.0, 8.0]}
    )
    y = pd.Series([0, 1, 0, 1])
    sel = FeatureSelector(k=3).fit(X, y)  # keep all → transform is a faithful subset
    out = sel.transform(X)
    pd.testing.assert_frame_equal(out, X.loc[:, sel.support_mask_])
    assert out.isna().sum().sum() == 1  # the NaN survived (values untouched)
    assert set(out.columns) == set(sel.selected_features_)


def test_selector_clamps_k_above_n_features() -> None:
    X, y = _signal_frame(p=5)
    sel = FeatureSelector(k=100).fit(X, y)
    assert sel.transform(X).shape[1] == 5


def test_selector_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        FeatureSelector(method="chi2")


def test_selector_feature_count_mismatch_raises() -> None:
    X, y = _signal_frame(p=6)
    sel = FeatureSelector(k=2).fit(X, y)
    with pytest.raises(ValueError, match="feature"):
        sel.transform(X.iloc[:, :3])


def test_selector_is_fit_on_passed_rows_only() -> None:
    """Leakage-safety: selection depends ONLY on the rows handed to fit.

    ``A`` carries the signal in the first half, ``B`` in the second — so a
    selector fit per fold picks whichever feature is informative *in that fold*,
    proving it never peeks at held-out rows.
    """
    rng = np.random.default_rng(0)
    n = 80
    half = n // 2
    y = pd.Series(np.tile([0, 1], half))
    yv = y.to_numpy()
    idx = np.arange(n)
    a = np.where(idx < half, yv * 3.0 + rng.normal(0, 0.3, n), rng.normal(0, 1, n))
    b = np.where(idx >= half, yv * 3.0 + rng.normal(0, 0.3, n), rng.normal(0, 1, n))
    X = pd.DataFrame({"A": a, "B": b, "noise": rng.normal(size=n)})

    first = FeatureSelector(k=1, method="f_classif").fit(X.iloc[:half], y.iloc[:half])
    second = FeatureSelector(k=1, method="f_classif").fit(X.iloc[half:], y.iloc[half:])
    assert first.selected_features_ == ["A"]
    assert second.selected_features_ == ["B"]


# =============================================================================
# SelectingModel.
# =============================================================================
def _model_frame(n: int = 60, p: int = 12) -> tuple[pd.DataFrame, pd.Series]:
    """3-class target driven by ``f0``; the rest noise (with a NaN cell)."""
    rng = np.random.default_rng(12)
    cols = {f"f{j}": rng.normal(size=n) for j in range(p)}
    X = pd.DataFrame(cols)
    X.loc[0, "f1"] = np.nan  # exercise NaN pass-through to the base model
    y = pd.Series(np.where(X["f0"] > 0.4, "hike", np.where(X["f0"] < -0.4, "cut", "hold")))
    return X, y


def test_selecting_model_is_a_model_and_narrows_features() -> None:
    X, y = _model_frame()
    model = SelectingModel(base_model_name="xgboost_classifier", k=4).fit(X, y)
    assert isinstance(model, Model)
    assert model.fit(X, y) is model
    assert model.feature_names_ == list(X.columns)  # BaseModel records the FULL input
    assert len(model.selected_features_) == 4
    assert "f0" in model.selected_features_  # the informative feature survives


@pytest.mark.parametrize("base", ["xgboost_classifier", "logistic_regression"])
def test_selecting_model_predict_and_proba(base: str) -> None:
    X, y = _model_frame()
    model = SelectingModel(base_model_name=base, k=4).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_selecting_model_unfitted_predict_raises() -> None:
    model = SelectingModel(k=3)
    with pytest.raises(AttributeError):
        model.predict(_model_frame()[0])


def test_build_model_resolves_selecting_xgboost() -> None:
    model = build_model(load_model_config("selecting_xgboost"))
    assert isinstance(model, SelectingModel)
    X, y = _model_frame()
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    # yaml default k=100 clamps to the 12 available features.
    assert len(fitted.selected_features_) == 12


def test_selecting_model_base_params_override() -> None:
    X, y = _model_frame()
    model = SelectingModel(
        base_model_name="xgboost_classifier", k=5, base_params={"n_estimators": 10}
    ).fit(X, y)
    assert model._inner._n_estimators == 10  # override flowed into the base model


def test_selecting_model_reselects_per_fold_in_walk_forward() -> None:
    """Across walk-forward folds the selection is re-fit on each training window."""
    X, y = _model_frame(n=60, p=12)
    splitter = WalkForwardSplit(initial_train_size=30, test_size=10, step=10)
    selected_per_fold = []
    for train_idx, _test_idx in splitter.split(X):
        model = SelectingModel(base_model_name="xgboost_classifier", k=3).fit(
            X.iloc[train_idx], y.iloc[train_idx]
        )
        selected_per_fold.append(tuple(sorted(model.selected_features_)))
    assert len(selected_per_fold) >= 2
    assert all(len(s) == 3 for s in selected_per_fold)
    # The informative feature is picked in every fold; folds are independently fit.
    assert all("f0" in s for s in selected_per_fold)
