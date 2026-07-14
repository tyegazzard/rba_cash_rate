"""Tests for ``rba.models.xgb`` — the XGBoost gradient-boosted-trees wrapper.

No network; tiny inline ``(X, y)`` fixtures with a fixed ``RANDOM_SEED``. Covers
:class:`XGBClassifierWrapper`: it satisfies the ``Model`` protocol when fitted,
``fit`` returns ``self`` and stores ``feature_names_``, ``predict`` returns an
``(n,)`` ndarray, ``predict_proba`` returns an ``(n, n_classes)`` row-stochastic
matrix in ``[0, 1]``, ``classes_`` is the sorted unique labels, ``sample_weight``
is accepted, and label dtype round-trips through the internal ``LabelEncoder`` for
BOTH string labels and the ``three_class`` ``{-1, 0, 1}`` int encoding (which
XGBoost cannot ingest natively). Also: unfitted ``predict`` / ``predict_proba``
raise, ``feature_names_`` raises before fit, and XGBoost's native NaN handling
lets ``fit`` + ``predict`` succeed on an ``X`` containing ``np.nan``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import RANDOM_SEED
from rba.models.base import Model
from rba.models.xgb import XGBClassifierWrapper


def _Xy(*, labels: str = "str", nan: bool = False, n: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """A small, class-separable synthetic frame with >=2 well-populated classes.

    ``labels='str'`` yields cut/hold/hike strings; ``labels='int'`` yields the
    ``three_class`` ``{-1, 0, 1}`` encoding. When ``nan`` is set, a scattering of
    ``np.nan`` is punched into ``X`` (XGBoost handles it natively).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    # Three balanced groups with separated feature means so the booster fits stably.
    third = n // 3
    sizes = [third, third, n - 2 * third]
    means = [-2.0, 0.0, 2.0]
    f0 = np.concatenate([rng.normal(m, 0.5, size=s) for m, s in zip(means, sizes)])
    f1 = np.concatenate([rng.normal(-m, 0.5, size=s) for m, s in zip(means, sizes)])
    f2 = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    if nan:
        X.iloc[0, 0] = np.nan
        X.iloc[5, 1] = np.nan
        X.iloc[11, 2] = np.nan
    if labels == "int":
        groups = [-1, 0, 1]
    else:
        groups = ["cut", "hold", "hike"]
    y = pd.Series(np.concatenate([np.full(s, g) for g, s in zip(groups, sizes)]))
    return X, y


def _wrapper() -> XGBClassifierWrapper:
    """A fast-to-fit wrapper — few shallow trees keep the tests snappy."""
    return XGBClassifierWrapper(n_estimators=20, max_depth=3, random_state=RANDOM_SEED)


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_is_a_model_when_fitted() -> None:
    from xgboost import XGBClassifier

    X, y = _Xy()
    model = _wrapper().fit(X, y)
    assert isinstance(model, Model)
    assert isinstance(model._model, XGBClassifier)


def test_fit_returns_self() -> None:
    X, y = _Xy()
    model = _wrapper()
    assert model.fit(X, y) is model


def test_feature_names_are_stored() -> None:
    X, y = _Xy()
    model = _wrapper().fit(X, y)
    assert model.feature_names_ == ["f0", "f1", "f2"]


# -----------------------------------------------------------------------------
# predict / predict_proba shapes + validity.
# -----------------------------------------------------------------------------
def test_predict_returns_ndarray_of_shape_n() -> None:
    X, y = _Xy()
    model = _wrapper().fit(X, y)
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_predict_proba_shape_and_is_row_stochastic() -> None:
    X, y = _Xy()
    model = _wrapper().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
    assert (proba >= 0).all() and (proba <= 1).all()


def test_classes_are_sorted_unique_labels() -> None:
    X, y = _Xy()
    model = _wrapper().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))


# -----------------------------------------------------------------------------
# sample_weight is accepted.
# -----------------------------------------------------------------------------
def test_sample_weight_is_accepted() -> None:
    X, y = _Xy()
    weights = np.linspace(0.5, 1.5, num=len(X))
    model = _wrapper().fit(X, y, sample_weight=weights)
    assert isinstance(model.predict(X), np.ndarray)


# -----------------------------------------------------------------------------
# Label-dtype preservation — strings round-trip; ints stay integers.
# -----------------------------------------------------------------------------
def test_string_labels_predict_returns_strings() -> None:
    X, y = _Xy(labels="str")
    model = _wrapper().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({"cut", "hold", "hike"})
    assert not np.issubdtype(preds.dtype, np.number)


def test_int_three_class_labels_round_trip_through_predict() -> None:
    """{-1, 0, 1} is non-contiguous — XGBoost rejects it, so the encoder must map it."""
    X, y = _Xy(labels="int")
    model = _wrapper().fit(X, y)
    np.testing.assert_array_equal(model.classes_, np.array([-1, 0, 1]))
    preds = model.predict(X)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset({-1, 0, 1})


# -----------------------------------------------------------------------------
# Unfitted access.
# -----------------------------------------------------------------------------
def test_predict_raises_before_fit() -> None:
    """The underlying estimator attribute is absent until fit."""
    X, _ = _Xy()
    with pytest.raises(AttributeError):
        XGBClassifierWrapper().predict(X)


def test_predict_proba_raises_before_fit() -> None:
    X, _ = _Xy()
    with pytest.raises(AttributeError):
        XGBClassifierWrapper().predict_proba(X)


def test_feature_names_raises_before_fit() -> None:
    with pytest.raises(RuntimeError):
        XGBClassifierWrapper().feature_names_


# -----------------------------------------------------------------------------
# NaN handling — XGBoost splits on missing values natively.
# -----------------------------------------------------------------------------
def test_fit_and_predict_with_nan_in_X_succeeds() -> None:
    X, y = _Xy(nan=True)
    assert X.isna().to_numpy().any()
    model = _wrapper().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)))
