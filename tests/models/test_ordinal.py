"""Tests for ``rba.models.ordinal`` — the §7 ordinal threshold wrapper.

No network; tiny inline ``(X, y)`` fixtures with **integer ordinal labels**
(cash-rate move buckets in basis points, e.g. ``{-50, -25, 0, +25, +50}``).
Covers :class:`OrdinalLogistic`: it satisfies the ``Model`` protocol when fitted,
``fit`` returns ``self`` and stores ``feature_names_``, ``predict`` returns an
``(n,)`` ndarray of the *original* integer labels, ``predict_proba`` returns an
``(n, K)`` matrix whose rows sum to 1 with every entry in ``[0, 1]`` and columns
aligned to the sorted ``classes_``, ``sample_weight`` is accepted, NaN in ``X`` is
handled by the pipeline's :class:`FfillImputer`, unfitted access raises, and an
invalid ``variant`` raises :class:`ValueError`. Both ``'AT'`` and ``'IT'``
variants are exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import RANDOM_SEED
from rba.models.base import Model
from rba.models.ordinal import OrdinalLogistic

# The five ordinal move buckets (basis points) used across the fixtures.
_ORDINAL_LABELS = np.array([-50, -25, 0, 25, 50])


def _ordinal_data(n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """A deterministic, ordinal-signal frame with integer move labels.

    The label is a monotone (thresholded) function of a latent score built from
    the features, so the ordering is learnable; every class is well populated.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    score = 1.5 * f0 + 0.5 * f1 + rng.normal(scale=0.3, size=n)
    # Cut the latent score into the five ordered buckets by quantile so each is
    # populated, then map bucket index -> the basis-point label.
    edges = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    bucket = np.digitize(score, edges)  # 0..4
    y = _ORDINAL_LABELS[bucket]
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    return X, pd.Series(y, name="move_bps")


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_ordinal_is_a_model_when_fitted() -> None:
    X, y = _ordinal_data()
    assert isinstance(OrdinalLogistic().fit(X, y), Model)


def test_ordinal_fit_returns_self() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic()
    assert model.fit(X, y) is model


def test_ordinal_feature_names_are_stored() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic().fit(X, y)
    assert model.feature_names_ == ["f0", "f1", "f2"]


# -----------------------------------------------------------------------------
# predict — shape + original integer label dtype/values.
# -----------------------------------------------------------------------------
def test_ordinal_predict_returns_ndarray_of_shape_n() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic().fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_ordinal_predict_preserves_integer_labels() -> None:
    """Predictions come back as the original integer move buckets, not 0..K-1."""
    X, y = _ordinal_data()
    model = OrdinalLogistic().fit(X, y)
    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset(set(_ORDINAL_LABELS.tolist()))


# -----------------------------------------------------------------------------
# predict_proba — a valid distribution over the sorted classes.
# -----------------------------------------------------------------------------
def test_ordinal_predict_proba_is_a_valid_distribution() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(model.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert (proba >= 0).all() and (proba <= 1).all()


def test_ordinal_classes_are_sorted_unique_labels() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.unique(y.to_numpy()))
    np.testing.assert_array_equal(model.classes_, _ORDINAL_LABELS)


# -----------------------------------------------------------------------------
# sample_weight + NaN handling + the 'IT' variant.
# -----------------------------------------------------------------------------
def test_ordinal_sample_weight_is_accepted() -> None:
    X, y = _ordinal_data()
    weights = np.linspace(0.5, 1.5, len(X))
    model = OrdinalLogistic().fit(X, y, sample_weight=weights)
    assert model.predict(X).shape == (len(X),)


def test_ordinal_handles_nan_in_X_via_ffill_imputer() -> None:
    """A NaN matrix is completed by the pipeline's FfillImputer — fit + predict succeed."""
    X, y = _ordinal_data()
    X_nan = X.copy()
    X_nan.iloc[5, 0] = np.nan
    X_nan.iloc[10, 2] = np.nan
    X_nan.iloc[0, 1] = np.nan  # leading NaN → filled by the training carry
    model = OrdinalLogistic().fit(X_nan, y)
    preds = model.predict(X_nan)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X_nan)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))


def test_ordinal_it_variant_fits_and_predicts() -> None:
    X, y = _ordinal_data()
    model = OrdinalLogistic(variant="IT").fit(X, y)
    assert isinstance(model, Model)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert np.issubdtype(model.predict(X).dtype, np.integer)


# -----------------------------------------------------------------------------
# Error paths + unfitted access.
# -----------------------------------------------------------------------------
def test_ordinal_invalid_variant_raises_value_error() -> None:
    with pytest.raises(ValueError, match="variant"):
        OrdinalLogistic(variant="ZZ")


def test_ordinal_predict_raises_before_fit() -> None:
    """Unfitted access fails loudly (the pipeline attribute is absent)."""
    X, _ = _ordinal_data()
    with pytest.raises(AttributeError):
        OrdinalLogistic().predict(X)


def test_ordinal_predict_proba_raises_before_fit() -> None:
    X, _ = _ordinal_data()
    with pytest.raises(AttributeError):
        OrdinalLogistic().predict_proba(X)


def test_ordinal_feature_names_raises_before_fit() -> None:
    with pytest.raises(RuntimeError):
        _ = OrdinalLogistic().feature_names_
