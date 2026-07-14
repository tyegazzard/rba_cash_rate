"""Tests for ``rba.models._preprocessing`` — the shared ML-wrapper helpers.

Covers :class:`FfillImputer`: it forward-fills within a block, fills residual
leading NaN with the per-fold training carry (never a whole-series statistic),
zero-fills an entirely-NaN training column, carries the training tail across the
train/test boundary, round-trips inside a scikit-learn ``Pipeline``, and raises
on a feature-count mismatch. And :func:`balanced_sample_weight`: an explicit
``sample_weight`` wins, ``'balanced'`` reproduces
``compute_sample_weight('balanced', y)``, and ``None`` yields uniform ``None``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from rba.models._preprocessing import FfillImputer, balanced_sample_weight


# -----------------------------------------------------------------------------
# FfillImputer.
# -----------------------------------------------------------------------------
def test_ffill_carries_last_value_forward_within_block() -> None:
    X = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": [np.nan, 2.0, np.nan, np.nan]})
    out = FfillImputer().fit_transform(X)
    # 'a': 1 -> 1 -> 1 -> 4 ; 'b': leading NaN filled by carry (b's own last valid=2), then 2,2,2
    np.testing.assert_allclose(out[:, 0], [1.0, 1.0, 1.0, 4.0])
    np.testing.assert_allclose(out[:, 1], [2.0, 2.0, 2.0, 2.0])


def test_ffill_leading_nan_filled_with_training_carry() -> None:
    """A test block that STARTS with NaN gets the training tail carried across."""
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    imp = FfillImputer().fit(X_train)
    assert imp.carry_[0] == pytest.approx(3.0)
    X_test = pd.DataFrame({"a": [np.nan, np.nan, 9.0]})
    out = imp.transform(X_test)
    # First two rows have no in-block prior -> filled by the training carry (3.0).
    np.testing.assert_allclose(out[:, 0], [3.0, 3.0, 9.0])


def test_ffill_all_nan_training_column_zero_filled() -> None:
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
    imp = FfillImputer().fit(X)
    assert imp.carry_[0] == 0.0
    out = imp.transform(pd.DataFrame({"a": [np.nan, np.nan], "b": [5.0, np.nan]}))
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(out[:, 1], [5.0, 5.0])


def test_ffill_output_has_no_nan() -> None:
    X = pd.DataFrame({"a": [np.nan, 1.0, np.nan], "c": [np.nan, np.nan, np.nan]})
    out = FfillImputer().fit_transform(X)
    assert not np.isnan(out).any()


def test_ffill_single_row_transform() -> None:
    """The per-row market-fallback path in the harness transforms one row at a time."""
    imp = FfillImputer().fit(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    out = imp.transform(pd.DataFrame({"a": [np.nan], "b": [np.nan]}))
    np.testing.assert_allclose(out, [[2.0, 4.0]])


def test_ffill_round_trips_in_pipeline() -> None:
    X = pd.DataFrame({"a": [np.nan, 1.0, 2.0, 3.0], "b": [1.0, np.nan, 2.0, 3.0]})
    pipe = Pipeline([("impute", FfillImputer()), ("scale", StandardScaler())])
    out = pipe.fit_transform(X)
    assert out.shape == (4, 2)
    assert not np.isnan(out).any()


def test_ffill_feature_count_mismatch_raises() -> None:
    imp = FfillImputer().fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    with pytest.raises(ValueError, match="feature"):
        imp.transform(pd.DataFrame({"a": [1.0]}))


def test_ffill_records_feature_names() -> None:
    imp = FfillImputer().fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    assert imp.n_features_in_ == 2
    np.testing.assert_array_equal(imp.get_feature_names_out(), np.array(["a", "b"], dtype=object))


# -----------------------------------------------------------------------------
# balanced_sample_weight.
# -----------------------------------------------------------------------------
def test_balanced_weight_explicit_sample_weight_wins() -> None:
    y = np.array([0, 0, 1])
    w = np.array([3.0, 1.0, 2.0])
    np.testing.assert_allclose(balanced_sample_weight(y, sample_weight=w), w)


def test_balanced_weight_matches_sklearn() -> None:
    y = np.array(["hold"] * 7 + ["hike"] * 2 + ["cut"] * 1)
    got = balanced_sample_weight(y, class_weight="balanced")
    np.testing.assert_allclose(got, compute_sample_weight("balanced", y))


def test_balanced_weight_none_is_uniform() -> None:
    assert balanced_sample_weight(np.array([0, 1, 0]), class_weight=None) is None
    assert balanced_sample_weight(np.array([0, 1, 0]), class_weight="none") is None
