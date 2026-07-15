"""Tests for ``rba.models.base`` — the shared model interface.

No network; every estimator is a trivial in-file dummy. Covers the runtime
``Model`` protocol (a conforming class is an instance, a class missing a required
member is not), the classifier / regressor ``predict_proba`` split, and the
``BaseModel`` boilerplate (feature-name storage, ``fit`` returns ``self``, the
inherited ``NotImplementedError`` default, and the not-fitted guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.models.base import BaseModel, Model


# -----------------------------------------------------------------------------
# Fixtures + trivial estimators.
# -----------------------------------------------------------------------------
def _xy(n: int = 6, n_features: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """A tiny meeting-like ``(X, y)``: ``n`` rows, ``f0..f{k-1}`` columns, 2 classes."""
    X = pd.DataFrame({f"f{j}": np.arange(n, dtype=float) + j for j in range(n_features)})
    y = pd.Series([0, 1] * (n // 2), name="target")
    return X, y


class DummyClassifier:
    """Implements the ``Model`` contract directly (no ``BaseModel`` inheritance)."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._classes: np.ndarray = np.array([0, 1])

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "DummyClassifier":
        self._names = list(X.columns)
        self._classes = np.unique(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        k = len(self._classes)
        return np.full((len(X), k), 1.0 / k)

    @property
    def feature_names_(self) -> list[str]:
        return self._names


class DummyRegressor:
    """Pure regressor implementing the contract directly — ``predict_proba`` raises."""

    def __init__(self) -> None:
        self._names: list[str] = []

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "DummyRegressor":
        self._names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("regressors have no class probabilities")

    @property
    def feature_names_(self) -> list[str]:
        return self._names


class BaseClassifier(BaseModel):
    """A ``BaseModel`` classifier — overrides the default ``predict_proba``."""

    def _fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> None:
        self._classes = np.unique(y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        k = len(self._classes)
        return np.full((len(X), k), 1.0 / k)


class BaseRegressor(BaseModel):
    """A ``BaseModel`` regressor — inherits the default ``predict_proba`` (raises)."""

    def _fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class MissingPredictProba:
    """Conforms except it lacks ``predict_proba`` — not a structural ``Model``."""

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None
    ) -> "MissingPredictProba":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    @property
    def feature_names_(self) -> list[str]:
        return []


# -----------------------------------------------------------------------------
# The runtime-checkable Model protocol.
# -----------------------------------------------------------------------------
def test_conforming_classifier_is_a_model() -> None:
    assert isinstance(DummyClassifier(), Model)


def test_conforming_regressor_is_a_model() -> None:
    assert isinstance(DummyRegressor(), Model)


def test_class_missing_required_method_is_not_a_model() -> None:
    """runtime_checkable checks member presence — no predict_proba → not a Model."""
    assert not isinstance(MissingPredictProba(), Model)


# -----------------------------------------------------------------------------
# The fit / predict / predict_proba contract on a direct implementer.
# -----------------------------------------------------------------------------
def test_fit_returns_self() -> None:
    X, y = _xy()
    model = DummyClassifier()
    assert model.fit(X, y) is model


def test_feature_names_are_the_fitted_columns() -> None:
    X, y = _xy()
    model = DummyClassifier().fit(X, y)
    assert model.feature_names_ == ["f0", "f1", "f2"]


def test_predict_returns_ndarray_of_length_n_samples() -> None:
    X, y = _xy(n=6)
    preds = DummyClassifier().fit(X, y).predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (6,)


def test_predict_proba_returns_n_samples_by_n_classes() -> None:
    X, y = _xy(n=6)  # y has 2 distinct classes
    proba = DummyClassifier().fit(X, y).predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (6, 2)


def test_regressor_predict_proba_raises_not_implemented() -> None:
    X, y = _xy()
    model = DummyRegressor().fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


# -----------------------------------------------------------------------------
# BaseModel boilerplate.
# -----------------------------------------------------------------------------
def test_basemodel_subclass_is_a_model() -> None:
    assert isinstance(BaseClassifier().fit(*_xy()), Model)


def test_basemodel_fit_returns_self() -> None:
    X, y = _xy()
    model = BaseClassifier()
    assert model.fit(X, y) is model


def test_basemodel_stores_feature_names_on_fit() -> None:
    X, y = _xy()
    model = BaseRegressor().fit(X, y)
    assert model.feature_names_ == ["f0", "f1", "f2"]


def test_basemodel_feature_names_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = BaseRegressor().feature_names_


def test_basemodel_default_predict_proba_raises() -> None:
    """A regressor subclass inherits the NotImplementedError default for free."""
    X, y = _xy()
    model = BaseRegressor().fit(X, y)
    with pytest.raises(NotImplementedError, match="predict_proba"):
        model.predict_proba(X)


def test_basemodel_classifier_overrides_predict_proba() -> None:
    X, y = _xy(n=6)
    proba = BaseClassifier().fit(X, y).predict_proba(X)
    assert proba.shape == (6, 2)


def test_basemodel_cannot_be_instantiated_directly() -> None:
    """BaseModel is abstract — _fit / predict are unimplemented hooks."""
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]
