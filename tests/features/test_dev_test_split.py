"""Tests for the held-out dev/test partition (``rba.features.build.dev_test_split``).

The §7 prerequisite / CHECKLIST §1.6 guard: proves the fixed test-window boundary
(:data:`rba.config.TEST_WINDOW_START`) partitions the meeting frame cleanly into a
dev portion (everything before) and an untouchable test portion (on/after), that
both slices are leakage-free, and — on real data — that the test window actually
covers ≥1 hike and ≥1 cut so tuning can never be validated on a degenerate hold-out.

No network; a synthetic dated fixture inline. The real-data checks skip when
``features.parquet`` has not been materialised (CI, where ``data/`` is gitignored).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.config import TEST_WINDOW_START, load_target_config
from rba.features.build import (
    FEATURES_PARQUET_PATH,
    OUTCOME_COLUMNS,
    build_xy,
    dev_test_split,
    feature_columns,
)


def _dated_frame() -> pd.DataFrame:
    """A meeting frame straddling the 2022-05-01 boundary; test window has hikes + cuts."""
    dates = pd.to_datetime(
        [
            "2021-06-01",  # dev
            "2021-09-01",  # dev
            "2021-12-07",  # dev
            "2022-02-01",  # dev
            "2022-04-05",  # dev (still before 2022-05-01)
            "2022-05-03",  # test (first on/after the boundary)
            "2022-08-02",  # test — hike
            "2023-05-02",  # test — hike
            "2024-02-06",  # test — cut
            "2025-02-18",  # test — cut
        ]
    )
    change = [0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 25.0, 25.0, -25.0, -25.0]
    n = len(dates)
    return pd.DataFrame(
        {
            "meeting_date": dates,
            "statement_url": [f"/s/{i}" for i in range(n)],
            "rate_change_bps": change,
            "new_rate_pct": [3.0 + sum(change[: i + 1]) / 100.0 for i in range(n)],
            "prior_rate_pct": 3.0,
            "noise": [float(i) for i in range(n)],
        }
    )


# -----------------------------------------------------------------------------
# The boundary constant.
# -----------------------------------------------------------------------------
def test_test_window_start_is_the_agreed_boundary() -> None:
    assert TEST_WINDOW_START == date(2022, 5, 1)


# -----------------------------------------------------------------------------
# Synthetic partition mechanics.
# -----------------------------------------------------------------------------
def test_split_partitions_at_the_boundary() -> None:
    frame = _dated_frame()
    split = dev_test_split(frame, load_target_config("three_class"))
    assert len(split.y_dev) == 5
    assert len(split.y_test) == 5
    dev_dates = pd.to_datetime(frame.loc[split.x_dev.index, "meeting_date"])
    test_dates = pd.to_datetime(frame.loc[split.x_test.index, "meeting_date"])
    assert (dev_dates < pd.Timestamp(TEST_WINDOW_START)).all()
    assert (test_dates >= pd.Timestamp(TEST_WINDOW_START)).all()


def test_split_is_a_disjoint_cover_of_build_xy() -> None:
    frame = _dated_frame()
    x, y = build_xy(frame, load_target_config("three_class"))
    split = dev_test_split(frame, load_target_config("three_class"))
    # dev ∪ test == the full (X, y); dev ∩ test == ∅.
    assert set(split.x_dev.index).isdisjoint(split.x_test.index)
    assert sorted([*split.x_dev.index, *split.x_test.index]) == sorted(x.index)
    assert len(split.y_dev) + len(split.y_test) == len(y)


def test_split_slices_are_leakage_free() -> None:
    split = dev_test_split(_dated_frame(), load_target_config("three_class"))
    assert list(split.x_dev.columns) == feature_columns(_dated_frame())
    assert OUTCOME_COLUMNS.isdisjoint(split.x_dev.columns)
    assert OUTCOME_COLUMNS.isdisjoint(split.x_test.columns)


def test_split_custom_boundary_overrides_default() -> None:
    split = dev_test_split(
        _dated_frame(), load_target_config("three_class"), test_start=date(2024, 1, 1)
    )
    assert len(split.y_test) == 2  # only the two 2024/2025 meetings


# -----------------------------------------------------------------------------
# Real-data §1.6 invariant: the test window covers a hike AND a cut cycle.
# -----------------------------------------------------------------------------
@pytest.mark.skipif(
    not FEATURES_PARQUET_PATH.exists(),
    reason="features.parquet not materialised (data/ is gitignored)",
)
def test_real_test_window_covers_hike_and_cut() -> None:
    frame = pd.read_parquet(FEATURES_PARQUET_PATH)
    split = dev_test_split(frame, load_target_config("three_class"))
    x, y = build_xy(frame, load_target_config("three_class"))
    assert len(split.y_dev) + len(split.y_test) == len(y)
    assert len(split.y_dev) > len(split.y_test) > 0  # dev is the bulk; test non-empty
    assert {"hike", "cut"}.issubset(set(split.y_test.unique())), (
        "held-out test window must cover ≥1 hike and ≥1 cut cycle (CHECKLIST §1.6)"
    )
