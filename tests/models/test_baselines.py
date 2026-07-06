"""Tests for ``rba.models.baselines`` — the §6 baseline floors.

No network; tiny inline ``(X, y)`` fixtures. Covers :class:`MajorityClass`: it
satisfies the ``Model`` protocol, predicts the modal class for every row, returns
the empirical class prior from ``predict_proba``, honours ``sample_weight``, breaks
ties to the lowest label, and inherits the ``BaseModel`` feature-name storage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.models.base import Model
from rba.models.baselines import MajorityClass


def _X(n: int) -> pd.DataFrame:
    """A dummy feature frame — MajorityClass ignores its values, uses only shape."""
    return pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float)})


def _imbalanced() -> tuple[pd.DataFrame, pd.Series]:
    """7 hold / 2 hike / 1 cut → majority 'hold', prior [0.1, 0.2, 0.7] over sorted classes."""
    y = pd.Series(["hold"] * 7 + ["hike"] * 2 + ["cut"] * 1, name="target")
    return _X(len(y)), y


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_majority_class_is_a_model_when_fitted() -> None:
    X, y = _imbalanced()
    assert isinstance(MajorityClass().fit(X, y), Model)


def test_fit_returns_self() -> None:
    X, y = _imbalanced()
    model = MajorityClass()
    assert model.fit(X, y) is model


def test_feature_names_are_stored() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.feature_names_ == ["f0", "f1"]


# -----------------------------------------------------------------------------
# predict — the modal class for every row.
# -----------------------------------------------------------------------------
def test_predict_returns_majority_for_every_row() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.majority_class_ == "hold"
    preds = model.predict(_X(5))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5,)
    assert (preds == "hold").all()


def test_predict_uses_the_argument_row_count_not_the_training_count() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.predict(_X(3)).shape == (3,)


# -----------------------------------------------------------------------------
# predict_proba — the empirical class prior, constant across rows.
# -----------------------------------------------------------------------------
def test_predict_proba_is_the_empirical_prior() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    # sorted classes: ['cut', 'hike', 'hold'] -> counts 1, 2, 7
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))
    proba = model.predict_proba(_X(4))
    assert proba.shape == (4, 3)
    expected = np.array([0.1, 0.2, 0.7])
    np.testing.assert_allclose(proba, np.tile(expected, (4, 1)))
    # Every row is a valid distribution.
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_class_prior_matches_predict_proba_row() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    np.testing.assert_allclose(model.predict_proba(_X(1))[0], model.class_prior_)


# -----------------------------------------------------------------------------
# sample_weight + tie-breaking + integer labels.
# -----------------------------------------------------------------------------
def test_sample_weight_can_flip_the_majority() -> None:
    X = _X(2)
    y = pd.Series([0, 1])
    # Unweighted it's a tie -> lowest label (0); weighting class 1 makes it the majority.
    weighted = MajorityClass().fit(X, y, sample_weight=np.array([1.0, 5.0]))
    assert weighted.majority_class_ == 1
    np.testing.assert_allclose(weighted.class_prior_, np.array([1 / 6, 5 / 6]))


def test_tie_breaks_to_lowest_label() -> None:
    X = _X(2)
    y = pd.Series([1, 0])  # one each -> tie
    model = MajorityClass().fit(X, y)
    assert model.majority_class_ == 0


def test_integer_labels_predict_dtype_preserved() -> None:
    X = _X(4)
    y = pd.Series([-1, 0, 0, 1])  # three_class int encoding (cut/hold/hike)
    model = MajorityClass().fit(X, y)
    preds = model.predict(_X(3))
    assert model.majority_class_ == 0
    assert np.issubdtype(preds.dtype, np.integer)
    assert (preds == 0).all()


def test_predict_proba_raises_before_fit() -> None:
    """Unfitted access fails loudly (fitted attributes are absent)."""
    with pytest.raises(AttributeError):
        MajorityClass().predict_proba(_X(2))
