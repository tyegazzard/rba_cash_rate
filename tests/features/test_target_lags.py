"""Tests for ``rba.features.target_lags`` — meeting-space lags of the target.

No network — every frame is built inline. Leakage guard mirrors
``tests/data/test_no_leakage.py`` / ``tests/features/test_changes.py``: a later
meeting's target value must never change an earlier meeting's lag feature.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.features.target_lags import (
    DEFAULT_TARGET_COLUMNS,
    DEFAULT_TARGET_LAG_HORIZONS_MEETINGS,
    build_target_lags,
    config_target_lag_horizons,
    config_target_lags_enabled,
)

_FUTURE_SENTINEL = 9_999_999.0


# -----------------------------------------------------------------------------
# Fixtures.
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


def _master(n: int = 8) -> pd.DataFrame:
    """A meeting-indexed frame with the two target columns."""
    idx = list(range(n))
    return pd.DataFrame(
        {
            "meeting_date": pd.date_range("2020-01-01", periods=n, freq="MS"),
            "rate_change_bps": [25.0 * i for i in idx],
            "new_rate_pct": [1.0 + 0.25 * i for i in idx],
        }
    )


# -----------------------------------------------------------------------------
# Shape / values.
# -----------------------------------------------------------------------------
def test_shape_and_values() -> None:
    master = _master(6)
    out = build_target_lags(master, columns=["new_rate_pct"], horizons=[1, 2])
    assert list(out.columns) == ["meeting_date", "new_rate_pct_lag_1", "new_rate_pct_lag_2"]
    # lag_1 at row 2 is row 1's value; lag_2 at row 2 is row 0's value.
    assert out.loc[2, "new_rate_pct_lag_1"] == pytest.approx(1.25)
    assert out.loc[2, "new_rate_pct_lag_2"] == pytest.approx(1.0)


def test_both_default_target_columns() -> None:
    out = build_target_lags(_master(), horizons=[1])
    assert "rate_change_bps_lag_1" in out.columns
    assert "new_rate_pct_lag_1" in out.columns


def test_nan_at_start_not_zero() -> None:
    out = build_target_lags(_master(5), columns=["new_rate_pct"], horizons=[2])
    col = out["new_rate_pct_lag_2"]
    assert col.iloc[:2].isna().all()
    assert col.iloc[2:].notna().all()
    assert col.iloc[0] != 0


def test_feature_columns_are_float() -> None:
    out = build_target_lags(_master(), horizons=[1, 2, 3])
    for col in out.columns:
        if col == "meeting_date":
            continue
        assert pd.api.types.is_float_dtype(out[col]), col


def test_output_isolation_only_new_columns() -> None:
    out = build_target_lags(_master(), horizons=[1])
    echoed = (set(out.columns) - {"meeting_date"}) & {"rate_change_bps", "new_rate_pct"}
    assert echoed == set()


def test_unsorted_input_is_sorted() -> None:
    master = _master(6)
    shuffled = master.iloc[[5, 0, 3, 1, 4, 2]].reset_index(drop=True)
    out = build_target_lags(shuffled, columns=["new_rate_pct"], horizons=[1])
    assert out["meeting_date"].is_monotonic_increasing
    assert out.loc[1, "new_rate_pct_lag_1"] == pytest.approx(1.0)  # row1 sees row0


# -----------------------------------------------------------------------------
# Leakage — a future decision never changes an earlier lag.
# -----------------------------------------------------------------------------
def test_future_value_never_affects_earlier_lags() -> None:
    base = _master(8)
    injected = base.copy()
    injected.loc[injected.index[-1], "new_rate_pct"] = _FUTURE_SENTINEL
    injected.loc[injected.index[-1], "rate_change_bps"] = _FUTURE_SENTINEL

    out_base = build_target_lags(base, horizons=[1, 2, 3])
    out_inj = build_target_lags(injected, horizons=[1, 2, 3])

    pd.testing.assert_frame_equal(out_base.iloc[:-1], out_inj.iloc[:-1])
    earlier = out_inj.iloc[:-1].drop(columns="meeting_date")
    assert not (earlier == _FUTURE_SENTINEL).to_numpy().any()


# -----------------------------------------------------------------------------
# Skip / warn.
# -----------------------------------------------------------------------------
def test_absent_column_warn_skipped(loguru_messages: list) -> None:
    master = _master()[["meeting_date", "new_rate_pct"]]  # no rate_change_bps
    out = build_target_lags(master, horizons=[1])
    assert "new_rate_pct_lag_1" in out.columns
    assert not any(c.startswith("rate_change_bps") for c in out.columns)
    assert any("rate_change_bps" in m and "absent" in m for m in loguru_messages)


def test_no_resolved_columns_returns_meeting_key_only() -> None:
    master = _master()[["meeting_date"]]
    out = build_target_lags(master, horizons=[1])
    assert list(out.columns) == ["meeting_date"]


# -----------------------------------------------------------------------------
# Config-driven defaults.
# -----------------------------------------------------------------------------
def test_defaults_from_features_yaml() -> None:
    from rba.config import load_features_config

    config = load_features_config()
    assert config_target_lag_horizons(config) == (1, 2, 3)
    assert config_target_lags_enabled(config) is True

    out = build_target_lags(_master(), config=config)
    assert "new_rate_pct_lag_3" in out.columns
    assert "rate_change_bps_lag_1" in out.columns


def test_injected_config_overrides_disk_load() -> None:
    config = {"target_lags": {"enabled": True, "horizons_meetings": [1]}}
    out = build_target_lags(_master(), config=config)
    lag_cols = [c for c in out.columns if c != "meeting_date"]
    assert lag_cols == ["rate_change_bps_lag_1", "new_rate_pct_lag_1"]


def test_config_defaults_when_section_empty() -> None:
    assert config_target_lag_horizons({"target_lags": {}}) == DEFAULT_TARGET_LAG_HORIZONS_MEETINGS
    assert config_target_lag_horizons({}) == DEFAULT_TARGET_LAG_HORIZONS_MEETINGS
    assert config_target_lags_enabled({}) is False
    assert DEFAULT_TARGET_COLUMNS == ("rate_change_bps", "new_rate_pct")
