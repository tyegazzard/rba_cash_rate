"""Tests for the model factory — ``rba.config.load_model_config`` + ``build_model``.

No network. Round-trips the real ``majority_class`` entry from ``models.yaml``
end to end (config → instance → fit → predict), and covers the factory's error
paths (unknown name, the ``defaults`` non-model, missing keys, a non-model class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import RANDOM_SEED, load_model_config
from rba.models import Model, build_model
from rba.models.baselines import MajorityClass, MarketImplied, Persistence, TaylorRule
from rba.models.lgbm import LightGBMClassifierWrapper
from rba.models.ordinal import OrdinalLogistic
from rba.models.sklearn_wrappers import (
    LogisticRegressionWrapper,
    MLPClassifierWrapper,
    RandomForestClassifierWrapper,
    SVMClassifierWrapper,
)
from rba.models.xgb import XGBClassifierWrapper
from rba.models.xrfm import XRFMClassifierWrapper


# -----------------------------------------------------------------------------
# load_model_config.
# -----------------------------------------------------------------------------
def test_load_model_config_returns_entry_with_name() -> None:
    cfg = load_model_config("majority_class")
    assert cfg["module"] == "rba.models.baselines"
    assert cfg["class"] == "MajorityClass"
    assert cfg["task"] == "classification"
    assert cfg["name"] == "majority_class"


def test_load_model_config_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="No model named"):
        load_model_config("no_such_model")


def test_load_model_config_defaults_section_is_not_a_model() -> None:
    with pytest.raises(KeyError, match="No model named"):
        load_model_config("defaults")


# -----------------------------------------------------------------------------
# build_model — resolution + end-to-end round-trip.
# -----------------------------------------------------------------------------
def test_build_model_resolves_majority_class() -> None:
    model = build_model(load_model_config("majority_class"))
    assert isinstance(model, MajorityClass)


def test_build_model_round_trip_fits_and_predicts() -> None:
    model = build_model(load_model_config("majority_class"))
    X = pd.DataFrame({"f0": np.arange(6, dtype=float)})
    y = pd.Series(["hold"] * 4 + ["hike"] * 2)
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    assert (fitted.predict(X) == "hold").all()
    assert fitted.predict_proba(X).shape == (6, 2)


def test_build_model_missing_keys_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="missing required key"):
        build_model({"class": "MajorityClass"})  # no 'module'


def test_build_model_non_model_class_raises_typeerror() -> None:
    """A resolvable class that isn't a Model is rejected with a clear message."""
    with pytest.raises(TypeError, match="does not implement the Model interface"):
        build_model({"module": "builtins", "class": "object"})


def test_build_model_resolves_persistence() -> None:
    model = build_model(load_model_config("persistence"))
    assert isinstance(model, Persistence)


def test_build_model_persistence_round_trip_fits_and_predicts() -> None:
    model = build_model(load_model_config("persistence"))
    X = pd.DataFrame({"f0": np.arange(6, dtype=float)})
    y = pd.Series(["hold"] * 4 + ["hike"] * 2)  # last training label: 'hike'
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    assert (fitted.predict(X) == "hike").all()
    assert fitted.predict_proba(X).shape == (6, 2)


def test_build_model_resolves_taylor_rule() -> None:
    model = build_model(load_model_config("taylor_rule"))
    assert isinstance(model, TaylorRule)


def test_build_model_taylor_rule_round_trip_uses_literature_defaults() -> None:
    """The ``default:`` yaml block flows through to Taylor's (1.5, 0.5) / (2.5, 3.0)."""
    model = build_model(load_model_config("taylor_rule"))
    X = pd.DataFrame({"cpi_headline_yoy": [2.5, 3.5], "output_gap": [0.0, 2.0]})
    y = pd.Series([0.0, 0.0])  # ignored under fit_coefficients=False
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    # 3.0 + 1.5*(π - 2.5) + 0.5*y_gap.
    np.testing.assert_allclose(fitted.predict(X), np.array([3.0, 3.0 + 1.5 + 1.0]))


def test_build_model_resolves_market_implied() -> None:
    model = build_model(load_model_config("market_implied"))
    assert isinstance(model, MarketImplied)


def test_build_model_market_implied_round_trip_fits_and_predicts() -> None:
    """Empty ``default: {}`` block → constructor defaults; classify a hike / hold / cut."""
    model = build_model(load_model_config("market_implied"))
    X = pd.DataFrame(
        {"asx_30d_implied_rate": [4.60, 4.35, 4.10], "prior_rate_pct": [4.35, 4.35, 4.35]}
    )
    y = pd.Series(["cut", "hold", "hike"])
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    np.testing.assert_array_equal(fitted.predict(X), np.array(["hike", "hold", "cut"]))
    assert fitted.predict_proba(X).shape == (3, 3)


# =============================================================================
# §7 ML models — factory resolution + end-to-end round-trip from models.yaml.
# =============================================================================
_ML_CLASSIFIER_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest_classifier",
    "svm_classifier",
    "mlp_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
)


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("logistic_regression", LogisticRegressionWrapper),
        ("random_forest_classifier", RandomForestClassifierWrapper),
        ("svm_classifier", SVMClassifierWrapper),
        ("mlp_classifier", MLPClassifierWrapper),
        ("xgboost_classifier", XGBClassifierWrapper),
        ("lightgbm_classifier", LightGBMClassifierWrapper),
        ("ordinal_logistic", OrdinalLogistic),
        ("xrfm_classifier", XRFMClassifierWrapper),
    ],
)
def test_build_model_resolves_ml_models(name: str, expected_cls: type) -> None:
    """Every §7 models.yaml entry parses and builds its named wrapper class."""
    assert isinstance(build_model(load_model_config(name)), expected_cls)


def _ml_classification_frame() -> tuple[pd.DataFrame, pd.Series]:
    """A small 3-class frame (with a NaN cell) exercising every NaN strategy."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 45
    X = pd.DataFrame(
        {"f0": rng.normal(size=n), "f1": rng.normal(size=n), "f2": rng.normal(size=n)}
    )
    X.loc[0, "f0"] = np.nan  # dense wrappers impute it; trees split on it natively
    y = pd.Series(["cut", "hold", "hike"] * (n // 3))  # balanced 15/15/15
    return X, y


@pytest.mark.parametrize("name", _ML_CLASSIFIER_NAMES)
def test_build_model_ml_classifier_round_trip(name: str) -> None:
    """The ``default:`` block flows through to a fit → predict → proba round-trip."""
    model = build_model(load_model_config(name))
    X, y = _ml_classification_frame()
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    preds = fitted.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    proba = fitted.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_build_model_ordinal_logistic_round_trip() -> None:
    """ordinal_logistic builds from yaml and round-trips integer ordinal labels."""
    model = build_model(load_model_config("ordinal_logistic"))
    rng = np.random.default_rng(RANDOM_SEED)
    n = 45
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    y = pd.Series([-25, 0, 25] * (n // 3))  # ordinal integer buckets (bp)
    fitted = model.fit(X, y)
    assert isinstance(fitted, Model)
    preds = fitted.predict(X)
    assert set(np.unique(preds)).issubset({-25, 0, 25})
    assert np.issubdtype(preds.dtype, np.integer)  # original int labels preserved
    proba = fitted.predict_proba(X)
    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(n), atol=1e-6)


def test_build_model_xrfm_is_deferred_stub() -> None:
    """xrfm_classifier resolves (factory interface check passes) but fit raises."""
    model = build_model(load_model_config("xrfm_classifier"))
    assert isinstance(model, XRFMClassifierWrapper)
    X = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    y = pd.Series(["hold", "hike", "cut"])
    with pytest.raises(NotImplementedError, match="porting course code"):
        model.fit(X, y)
