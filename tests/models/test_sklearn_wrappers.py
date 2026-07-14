"""Tests for ``rba.models.sklearn_wrappers`` — the §7 sklearn classifier wrappers.

No network; small inline synthetic ``(X, y)`` fixtures with three comfortably-
populated, separable-ish classes and a fixed ``random_state`` so the estimators
fit stably (MLP ``early_stopping`` needs enough per-class rows for its internal
stratified split). The shared :class:`~rba.models.base.Model` contract —
protocol conformance, ``fit`` returns ``self``, ``feature_names_`` storage,
``predict`` shape, ``predict_proba`` being a valid per-row distribution,
``classes_`` = sorted unique labels, ``sample_weight`` acceptance, label-dtype
preservation, unfitted-access errors, and NaN handling — is parametrised across
all FOUR wrappers (:class:`LogisticRegressionWrapper`,
:class:`RandomForestClassifierWrapper`, :class:`SVMClassifierWrapper`,
:class:`MLPClassifierWrapper`). NaN handling is imputed-in-Pipeline for the three
dense wrappers and native for the random forest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import RANDOM_SEED
from rba.models.base import Model
from rba.models.sklearn_wrappers import (
    LogisticRegressionWrapper,
    MLPClassifierWrapper,
    RandomForestClassifierWrapper,
    SVMClassifierWrapper,
)

# Every wrapper is exercised against the same contract.
WRAPPERS = [
    LogisticRegressionWrapper,
    RandomForestClassifierWrapper,
    SVMClassifierWrapper,
    MLPClassifierWrapper,
]
WRAPPER_IDS = [cls.__name__ for cls in WRAPPERS]

_FEATURES = ["f0", "f1", "f2", "f3"]


def _make_Xy(
    n: int = 42, *, string_labels: bool = True, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Balanced 3-class frame whose features carry a per-class signal.

    ``n`` is truncated to a multiple of three so the classes are exactly balanced
    (kind to MLP's stratified early-stopping split). Each class's feature cloud is
    shifted so the estimators can separate them; a fixed seed keeps it deterministic.
    """
    per = n // 3
    base = np.repeat(np.arange(3), per)  # [0]*per + [1]*per + [2]*per
    rng = np.random.default_rng(seed)
    signal = base[:, None] * 1.5
    X = pd.DataFrame(
        signal + rng.normal(0.0, 1.0, size=(len(base), len(_FEATURES))),
        columns=_FEATURES,
    )
    if string_labels:
        labels = np.array(["cut", "hike", "hold"])  # sorted lex already
    else:
        labels = np.array([-1, 0, 1])
    y = pd.Series(labels[base], name="target")
    return X, y


def _with_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Scatter a few NaN cells into a copy of ``X`` (never a whole column)."""
    X = X.copy()
    X.iloc[0, 0] = np.nan
    X.iloc[3, 2] = np.nan
    X.iloc[7, 1] = np.nan
    return X


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_wrapper_is_a_model_when_fitted(wrapper: type) -> None:
    X, y = _make_Xy()
    assert isinstance(wrapper().fit(X, y), Model)


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_fit_returns_self(wrapper: type) -> None:
    X, y = _make_Xy()
    model = wrapper()
    assert model.fit(X, y) is model


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_feature_names_are_stored(wrapper: type) -> None:
    X, y = _make_Xy()
    model = wrapper().fit(X, y)
    assert model.feature_names_ == _FEATURES


# -----------------------------------------------------------------------------
# predict — one label per row.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_predict_shape_and_type(wrapper: type) -> None:
    X, y = _make_Xy()
    model = wrapper().fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)
    # Every prediction is one of the fitted classes.
    assert set(np.unique(preds)).issubset(set(model.classes_.tolist()))


# -----------------------------------------------------------------------------
# predict_proba — a valid per-row distribution over classes_.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_predict_proba_is_a_valid_distribution(wrapper: type) -> None:
    X, y = _make_Xy()
    model = wrapper().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert (proba >= 0).all()
    assert (proba <= 1).all()


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_classes_are_sorted_unique_labels(wrapper: type) -> None:
    X, y = _make_Xy()
    model = wrapper().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.unique(y.to_numpy()))


# -----------------------------------------------------------------------------
# sample_weight is accepted.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_sample_weight_is_accepted(wrapper: type) -> None:
    X, y = _make_Xy()
    weights = np.linspace(0.5, 2.0, num=len(X))
    model = wrapper().fit(X, y, sample_weight=weights)
    assert model.predict(X).shape == (len(X),)


# -----------------------------------------------------------------------------
# Label-dtype preservation — strings come back as strings, ints as ints.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_string_labels_predict_returns_strings(wrapper: type) -> None:
    X, y = _make_Xy(string_labels=True)
    preds = wrapper().fit(X, y).predict(X)
    assert all(isinstance(p, str) for p in preds.tolist())


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_integer_labels_predict_dtype_preserved(wrapper: type) -> None:
    X, y = _make_Xy(string_labels=False)  # labels {-1, 0, 1}
    model = wrapper().fit(X, y)
    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


# -----------------------------------------------------------------------------
# Unfitted access raises.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_predict_before_fit_raises(wrapper: type) -> None:
    X, _ = _make_Xy()
    with pytest.raises(AttributeError):
        wrapper().predict(X)


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_predict_proba_before_fit_raises(wrapper: type) -> None:
    X, _ = _make_Xy()
    with pytest.raises(AttributeError):
        wrapper().predict_proba(X)


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_feature_names_before_fit_raises(wrapper: type) -> None:
    with pytest.raises(RuntimeError):
        _ = wrapper().feature_names_


# -----------------------------------------------------------------------------
# NaN handling — imputed in-Pipeline for dense wrappers, native for the forest.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("wrapper", WRAPPERS, ids=WRAPPER_IDS)
def test_fit_and_predict_with_nan_cells(wrapper: type) -> None:
    X, y = _make_Xy()
    X_nan = _with_nan(X)
    model = wrapper().fit(X_nan, y)
    preds = model.predict(X_nan)
    proba = model.predict_proba(X_nan)
    assert preds.shape == (len(X),)
    assert not np.isnan(proba).any()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))


# =============================================================================
# A couple of wrapper-specific structural checks.
# =============================================================================
def test_random_forest_uses_no_pipeline_and_handles_nan_natively() -> None:
    """The forest fits directly on NaN-bearing X — no imputer/scaler in front."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = _make_Xy()
    model = RandomForestClassifierWrapper().fit(_with_nan(X), y)
    assert isinstance(model._model, RandomForestClassifier)
    assert not hasattr(model, "_pipeline")


@pytest.mark.parametrize(
    "wrapper", [LogisticRegressionWrapper, SVMClassifierWrapper, MLPClassifierWrapper]
)
def test_dense_wrappers_build_the_impute_scale_clf_pipeline(wrapper: type) -> None:
    """Dense wrappers fit a per-fold ``[impute -> scale -> clf]`` Pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from rba.models._preprocessing import FfillImputer

    X, y = _make_Xy()
    model = wrapper().fit(X, y)
    assert isinstance(model._pipeline, Pipeline)
    assert [name for name, _ in model._pipeline.steps] == ["impute", "scale", "clf"]
    assert isinstance(model._pipeline.named_steps["impute"], FfillImputer)
    assert isinstance(model._pipeline.named_steps["scale"], StandardScaler)


def test_mlp_hidden_layer_sizes_coerced_from_list() -> None:
    """models.yaml passes a list [64, 32]; the wrapper stores a tuple of ints."""
    model = MLPClassifierWrapper(hidden_layer_sizes=[16, 8])
    assert model._hidden_layer_sizes == (16, 8)
    assert all(isinstance(h, int) for h in model._hidden_layer_sizes)
