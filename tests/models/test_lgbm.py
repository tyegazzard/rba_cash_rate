"""Tests for ``rba.models.lgbm`` — the LightGBM gradient-boosting wrapper.

No network; tiny inline ``(X, y)`` fixtures with a fixed RNG seed. Covers
:class:`LightGBMClassifierWrapper`: it satisfies the ``Model`` protocol after
fit, returns ``self`` from ``fit``, records the fitted feature names, predicts an
``ndarray`` of the right shape, returns a valid ``predict_proba`` distribution
(rows sum to 1, values in ``[0, 1]``, one column per sorted class), exposes
``classes_`` as the sorted unique labels, accepts ``sample_weight`` without
error, preserves the label dtype (string labels → strings, int labels →
integers) via LightGBM's native encoder, raises on unfitted access, and fits +
predicts natively through an ``X`` that contains ``np.nan`` (no imputation). All
frames stay :class:`pandas.DataFrame` on both ``fit`` and ``predict`` to avoid
LightGBM's "no valid feature names" warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import RANDOM_SEED
from rba.models.base import Model
from rba.models.lgbm import LightGBMClassifierWrapper

try:  # LightGBM's native estimator type, for the isinstance-after-fit check.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - lightgbm is a hard dependency here.
    LGBMClassifier = None  # type: ignore[assignment,misc]


def _X(n: int, *, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """A deterministic 3-feature frame; enough rows for a stable LightGBM fit."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.uniform(-1.0, 1.0, size=n),
            "f2": rng.normal(2.0, 0.5, size=n),
        }
    )


def _y_string(n: int) -> pd.Series:
    """Three well-populated string classes cycled across ``n`` rows."""
    labels = np.array(["cut", "hold", "hike"])
    return pd.Series(labels[np.arange(n) % 3], name="target")


def _y_int(n: int) -> pd.Series:
    """Three well-populated int classes (three_class encoding) cycled over ``n``."""
    labels = np.array([-1, 0, 1])
    return pd.Series(labels[np.arange(n) % 3], name="target")


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_wraps_an_lgbm_classifier_when_fitted() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    assert isinstance(model._model, LGBMClassifier)


def test_is_a_model_when_fitted() -> None:
    X, y = _X(40), _y_string(40)
    assert isinstance(LightGBMClassifierWrapper().fit(X, y), Model)


def test_fit_returns_self() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper()
    assert model.fit(X, y) is model


def test_feature_names_are_stored() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    assert model.feature_names_ == ["f0", "f1", "f2"]


# -----------------------------------------------------------------------------
# predict / predict_proba shapes + validity.
# -----------------------------------------------------------------------------
def test_predict_returns_ndarray_of_row_count() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    preds = model.predict(_X(7, seed=1))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (7,)


def test_predict_proba_is_a_valid_distribution() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    proba = model.predict_proba(_X(7, seed=1))
    assert proba.shape == (7, 3)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(7))


def test_classes_are_sorted_unique_labels() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    # Sorted lex: 'cut' < 'hike' < 'hold'.
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))


def test_classes_are_sorted_unique_int_labels() -> None:
    X, y = _X(40), _y_int(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([-1, 0, 1]))


# -----------------------------------------------------------------------------
# sample_weight is accepted.
# -----------------------------------------------------------------------------
def test_sample_weight_is_accepted() -> None:
    X, y = _X(40), _y_string(40)
    weights = np.linspace(0.5, 1.5, num=40)
    model = LightGBMClassifierWrapper().fit(X, y, sample_weight=weights)
    assert model.predict(X).shape == (40,)


# -----------------------------------------------------------------------------
# Label-dtype preservation (LightGBM's native encoder).
# -----------------------------------------------------------------------------
def test_string_labels_predict_returns_strings() -> None:
    X, y = _X(40), _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    preds = model.predict(_X(6, seed=2))
    assert preds.dtype.kind in ("U", "O")
    assert set(preds).issubset({"cut", "hold", "hike"})


def test_integer_labels_predict_dtype_preserved() -> None:
    X, y = _X(40), _y_int(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    preds = model.predict(_X(6, seed=2))
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(preds.tolist()).issubset({-1, 0, 1})


# -----------------------------------------------------------------------------
# Unfitted access.
# -----------------------------------------------------------------------------
def test_predict_raises_before_fit() -> None:
    """The underlying estimator attribute is absent until fit."""
    with pytest.raises(AttributeError):
        LightGBMClassifierWrapper().predict(_X(3))


def test_predict_proba_raises_before_fit() -> None:
    with pytest.raises(AttributeError):
        LightGBMClassifierWrapper().predict_proba(_X(3))


def test_feature_names_raises_before_fit() -> None:
    with pytest.raises(RuntimeError):
        _ = LightGBMClassifierWrapper().feature_names_


# -----------------------------------------------------------------------------
# NaN handling — LightGBM splits on NaN natively, no imputation.
# -----------------------------------------------------------------------------
def test_fit_and_predict_with_nan_features() -> None:
    """A NaN feature matrix must fit + predict without imputation."""
    X = _X(40)
    X.iloc[::5, 0] = np.nan  # scatter NaNs through column f0
    X.iloc[1::7, 2] = np.nan
    y = _y_string(40)
    model = LightGBMClassifierWrapper().fit(X, y)
    X_pred = _X(6, seed=3)
    X_pred.iloc[0, 1] = np.nan
    preds = model.predict(X_pred)
    proba = model.predict_proba(X_pred)
    assert preds.shape == (6,)
    assert proba.shape == (6, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(6))
