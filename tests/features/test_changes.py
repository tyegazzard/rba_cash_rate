"""Tests for ``rba.features.changes`` — Δ, %Δ (meeting-space) and calendar %YoY.

No network — every frame is built inline. Leakage guards mirror
``tests/data/test_no_leakage.py``: a later meeting's value must never change an
earlier meeting's change feature. The YoY tests specifically assert
cadence-robustness (calendar-anchored, not a fixed meeting count).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.features.changes import (
    DEFAULT_CHANGE_HORIZONS_MEETINGS,
    DEFAULT_YOY_DAYS,
    build_changes,
    config_abs_diff_columns,
    config_change_horizons,
    config_pct_change_columns,
    config_yoy_columns,
    config_yoy_days,
)

_FUTURE_SENTINEL = 9_999_999.0


# -----------------------------------------------------------------------------
# Fixtures / builders.
# -----------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _master(
    values: list[float], name: str = "cpi", *, dates: list[str] | None = None
) -> pd.DataFrame:
    if dates is None:
        meeting_date = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    else:
        meeting_date = pd.to_datetime(dates)
    return pd.DataFrame({"meeting_date": meeting_date, name: values})


# -----------------------------------------------------------------------------
# abs_diff (Δ) — meeting-space.
# -----------------------------------------------------------------------------
def test_abs_diff_shape_and_values() -> None:
    master = _master([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    out = build_changes(
        master, abs_diff_columns=["cpi"], pct_change_columns=[], yoy_columns=[], horizons=[1, 3]
    )
    assert list(out.columns) == ["meeting_date", "cpi_diff_1", "cpi_diff_3"]
    assert out["cpi_diff_1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out.loc[3, "cpi_diff_3"] == 6.0  # 7 - 1


def test_abs_diff_nan_at_start_not_zero() -> None:
    master = _master([1.0, 2.0, 4.0, 7.0])
    out = build_changes(
        master, abs_diff_columns=["cpi"], pct_change_columns=[], yoy_columns=[], horizons=[2]
    )
    col = out["cpi_diff_2"]
    assert col.iloc[0] != 0
    assert col.iloc[:2].isna().all()
    assert col.iloc[2:].notna().all()


# -----------------------------------------------------------------------------
# pct_change (%Δ) — meeting-space.
# -----------------------------------------------------------------------------
def test_pct_change_values() -> None:
    master = _master([100.0, 110.0, 121.0, 133.1])
    out = build_changes(
        master, abs_diff_columns=[], pct_change_columns=["cpi"], yoy_columns=[], horizons=[1]
    )
    assert out.loc[1, "cpi_pct_1"] == pytest.approx(0.10)
    assert out.loc[2, "cpi_pct_1"] == pytest.approx(0.10)


def test_pct_change_zero_base_is_nan_not_inf() -> None:
    master = _master([0.0, 5.0, 10.0])
    out = build_changes(
        master, abs_diff_columns=[], pct_change_columns=["cpi"], yoy_columns=[], horizons=[1]
    )
    pct = out["cpi_pct_1"]
    assert not np.isinf(pct.to_numpy()).any()
    assert pd.isna(pct.iloc[1])  # base (row 0) is 0 → NaN


# -----------------------------------------------------------------------------
# YoY — calendar-anchored, cadence-robust.
# -----------------------------------------------------------------------------
def test_yoy_is_calendar_anchored() -> None:
    """YoY compares against the as-of value ~365d earlier, not a meeting count."""
    dates = ["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"]
    master = _master([100.0, 110.0, 120.0, 132.0], dates=dates)
    out = build_changes(master, abs_diff_columns=[], pct_change_columns=[], yoy_columns=["cpi"])

    # 2024-01-01 vs 2023-01-01 (100): (120-100)/100 = 0.20
    assert out.loc[2, "cpi_yoy_pct"] == pytest.approx(0.20)
    # 2024-06-01 vs 2023-06-01 (110): (132-110)/110 = 0.20  (2 meetings apart, not 12)
    assert out.loc[3, "cpi_yoy_pct"] == pytest.approx(0.20)
    # First two meetings have no ~1yr-old predecessor → NaN.
    assert out["cpi_yoy_pct"].iloc[:2].isna().all()


def test_yoy_robust_across_cadence_change() -> None:
    """A fixed 11-meeting shift would be wrong post-2024; calendar YoY is right.

    Dense monthly meetings through the cadence break: for a mid-2024 meeting the
    year-ago anchor is ~12 rows back pre-break but the *date* is what matters.
    """
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    values = [100.0 + i for i in range(24)]  # +1 per month
    master = pd.DataFrame({"meeting_date": dates, "cpi": values})
    out = build_changes(master, abs_diff_columns=[], pct_change_columns=[], yoy_columns=["cpi"])

    # Meeting at 2024-01-01 (row 12): target 2023-01-02 → as-of row 0 (value 100).
    assert out.loc[12, "cpi_yoy_pct"] == pytest.approx((112.0 - 100.0) / 100.0)
    # Meeting at 2023-12-01 (row 11): target 2022-12-02 → no predecessor → NaN.
    assert pd.isna(out.loc[11, "cpi_yoy_pct"])


def test_yoy_exact_365d_boundary_is_included() -> None:
    """A meeting exactly window_days earlier is a valid (past) anchor."""
    dates = ["2023-01-01", "2024-01-01"]  # exactly 365 days apart
    master = _master([200.0, 220.0], dates=dates)
    out = build_changes(
        master, abs_diff_columns=[], pct_change_columns=[], yoy_columns=["cpi"], yoy_days=365
    )
    assert out.loc[1, "cpi_yoy_pct"] == pytest.approx(0.10)


def test_yoy_zero_base_is_nan() -> None:
    dates = ["2023-01-01", "2024-01-01"]
    master = _master([0.0, 5.0], dates=dates)
    out = build_changes(master, abs_diff_columns=[], pct_change_columns=[], yoy_columns=["cpi"])
    assert pd.isna(out.loc[1, "cpi_yoy_pct"])


# -----------------------------------------------------------------------------
# Leakage — a future meeting never changes an earlier feature.
# -----------------------------------------------------------------------------
def test_future_value_never_affects_earlier_changes() -> None:
    dates = pd.date_range("2022-01-01", periods=18, freq="MS")
    base = [100.0 + i for i in range(18)]
    injected = base[:-1] + [_FUTURE_SENTINEL]

    out_base = build_changes(
        pd.DataFrame({"meeting_date": dates, "cpi": base}),
        abs_diff_columns=["cpi"],
        pct_change_columns=["cpi"],
        yoy_columns=["cpi"],
        horizons=[1, 3],
    )
    out_inj = build_changes(
        pd.DataFrame({"meeting_date": dates, "cpi": injected}),
        abs_diff_columns=["cpi"],
        pct_change_columns=["cpi"],
        yoy_columns=["cpi"],
        horizons=[1, 3],
    )

    for col in out_base.columns:
        if col == "meeting_date":
            continue
        pd.testing.assert_series_equal(
            out_base[col].iloc[:-1], out_inj[col].iloc[:-1], check_names=False
        )


def test_unsorted_input_is_sorted_before_changes() -> None:
    master = _master([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])
    shuffled = master.iloc[[5, 0, 3, 1, 4, 2]].reset_index(drop=True)
    out = build_changes(
        shuffled, abs_diff_columns=["cpi"], pct_change_columns=[], yoy_columns=[], horizons=[1]
    )
    assert out["meeting_date"].is_monotonic_increasing
    assert out["cpi_diff_1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0, 5.0]


# -----------------------------------------------------------------------------
# Ordering, skip/warn, eligibility.
# -----------------------------------------------------------------------------
def test_output_block_order_diff_then_pct_then_yoy() -> None:
    dates = pd.date_range("2022-01-01", periods=15, freq="MS")
    master = pd.DataFrame({"meeting_date": dates, "a": list(range(15)), "b": list(range(15))})
    master = master.astype({"a": float, "b": float})
    out = build_changes(
        master, abs_diff_columns=["a"], pct_change_columns=["b"], yoy_columns=["a"], horizons=[1]
    )
    assert list(out.columns) == ["meeting_date", "a_diff_1", "b_pct_1", "a_yoy_pct"]


def test_absent_column_warn_skipped(loguru_messages: list) -> None:
    master = _master([1.0, 2.0, 3.0, 4.0])
    out = build_changes(
        master,
        abs_diff_columns=["cpi", "nope"],
        pct_change_columns=[],
        yoy_columns=[],
        horizons=[1],
    )
    assert "cpi_diff_1" in out.columns
    assert not any(c.startswith("nope") for c in out.columns)
    assert any("nope" in m and "absent" in m for m in loguru_messages)


def test_companion_regime_metadata_never_transformed(loguru_messages: list) -> None:
    master = _master([1.0, 2.0, 3.0, 4.0])
    master["cpi_age_days"] = pd.array([1, 2, 3, 4], dtype="Int64")
    master["regime_covid"] = [0, 0, 1, 1]
    out = build_changes(
        master,
        abs_diff_columns=["cpi", "cpi_age_days", "regime_covid", "rate_change_bps"],
        pct_change_columns=[],
        yoy_columns=[],
        horizons=[1],
    )
    assert list(out.columns) == ["meeting_date", "cpi_diff_1"]
    for skipped in ("cpi_age_days", "regime_covid"):
        assert any(skipped in m and "not a level" in m for m in loguru_messages)


def test_no_resolved_columns_returns_meeting_key_only() -> None:
    master = _master([1.0, 2.0, 3.0])
    out = build_changes(master, abs_diff_columns=["x"], pct_change_columns=[], yoy_columns=[])
    assert list(out.columns) == ["meeting_date"]


# -----------------------------------------------------------------------------
# Config-driven defaults.
# -----------------------------------------------------------------------------
def test_defaults_from_features_yaml() -> None:
    from rba.config import load_features_config

    config = load_features_config()
    assert config_change_horizons(config) == (1, 3, 6, 12)
    assert config_yoy_days(config) == 365
    assert "unemployment_rate_sa" in config_abs_diff_columns(config)  # reconciled
    assert "xjo" in config_pct_change_columns(config)
    assert "headline_cpi_index" in config_yoy_columns(config)

    # Build with defaults on a frame carrying one series from each block.
    dates = pd.date_range("2022-01-01", periods=15, freq="MS")
    frame = pd.DataFrame(
        {
            "meeting_date": dates,
            "unemployment_rate_sa": [5.0 + 0.1 * i for i in range(15)],
            "xjo": [7000.0 + i for i in range(15)],
            "headline_cpi_index": [100.0 + i for i in range(15)],
        }
    )
    out = build_changes(frame)
    assert "unemployment_rate_sa_diff_12" in out.columns
    assert "xjo_pct_6" in out.columns
    assert "headline_cpi_index_yoy_pct" in out.columns


def test_injected_config_overrides_disk_load() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0], name="foo")
    config = {
        "changes": {
            "horizons_meetings": [1],
            "abs_diff": {"columns": ["foo"]},
            "pct_change": {"columns": []},
            "yoy": {"columns": []},
        }
    }
    out = build_changes(master, config=config)
    assert list(out.columns) == ["meeting_date", "foo_diff_1"]


def test_config_defaults_when_section_empty() -> None:
    assert config_change_horizons({"changes": {}}) == DEFAULT_CHANGE_HORIZONS_MEETINGS
    assert config_yoy_days({"changes": {}}) == DEFAULT_YOY_DAYS
    assert config_abs_diff_columns({}) == []
