"""Tests for ``rba.features.lags`` — meeting-space lag features.

No network — every frame is built inline. The headline guard mirrors
``tests/data/test_no_leakage.py``'s synthetic future-data injection: a sentinel
value planted at the last meeting must never surface as a lag at any meeting
(lags only ever look at earlier meetings).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.features.lags import (
    DEFAULT_LAG_HORIZONS_MEETINGS,
    build_lags,
    config_lag_columns,
    config_lag_horizons,
)

# A value no legitimate observation would ever take — if it appears as a lag at
# an earlier meeting, a future row leaked.
_FUTURE_SENTINEL = 9_999_999.0


# -----------------------------------------------------------------------------
# Fixtures / builders.
# -----------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    """Capture loguru WARNING+ messages emitted within the test into a list."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _master(n: int = 6, columns: dict[str, list[float]] | None = None) -> pd.DataFrame:
    """A tiny meeting-indexed master-like frame with ``n`` monthly meetings."""
    frame = pd.DataFrame({"meeting_date": pd.date_range("2020-01-01", periods=n, freq="MS")})
    if columns:
        for name, values in columns.items():
            frame[name] = values
    return frame


# -----------------------------------------------------------------------------
# Shape / dtype.
# -----------------------------------------------------------------------------
def test_output_shape_and_dtype() -> None:
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = build_lags(master, columns=["cpi"], horizons=[1, 3])

    # meeting_date + one column per (series, horizon).
    assert list(out.columns) == ["meeting_date", "cpi_lag_1", "cpi_lag_3"]
    assert len(out) == len(master)
    assert out["cpi_lag_1"].dtype == np.float64
    # Lag h simply shifts the source down by h meetings.
    assert out["cpi_lag_1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out.loc[3, "cpi_lag_3"] == 1.0


def test_column_order_is_series_then_horizon() -> None:
    master = _master(columns={"a": [1.0] * 6, "b": [2.0] * 6})
    out = build_lags(master, columns=["a", "b"], horizons=[1, 2])
    assert list(out.columns) == [
        "meeting_date",
        "a_lag_1",
        "a_lag_2",
        "b_lag_1",
        "b_lag_2",
    ]


def test_returns_only_meeting_key_and_lags_not_source() -> None:
    """Pure builder: the source level columns are not echoed back."""
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = build_lags(master, columns=["cpi"], horizons=[1])
    assert "cpi" not in out.columns
    assert set(out.columns) == {"meeting_date", "cpi_lag_1"}


# -----------------------------------------------------------------------------
# NaN at the series start — never zero (explicit CONTEXT pitfall).
# -----------------------------------------------------------------------------
def test_lags_produce_nan_not_zero_at_start() -> None:
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = build_lags(master, columns=["cpi"], horizons=[2])

    # First two meetings have no 2-ago predecessor → NaN, explicitly not 0.
    assert out.loc[0, "cpi_lag_2"] != 0
    assert pd.isna(out.loc[0, "cpi_lag_2"])
    assert pd.isna(out.loc[1, "cpi_lag_2"])
    assert out.loc[2, "cpi_lag_2"] == 1.0
    assert out["cpi_lag_2"].isna().sum() == 2


def test_horizon_at_least_frame_length_is_all_nan() -> None:
    master = _master(n=3, columns={"cpi": [1.0, 2.0, 3.0]})
    out = build_lags(master, columns=["cpi"], horizons=[5])
    assert out["cpi_lag_5"].isna().all()


# -----------------------------------------------------------------------------
# Leakage guard — mirrors tests/data/test_no_leakage.py.
# -----------------------------------------------------------------------------
def test_future_value_never_leaks_into_earlier_lag() -> None:
    """A sentinel at the LAST meeting never surfaces as a lag at any meeting."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, _FUTURE_SENTINEL]
    master = _master(columns={"cpi": values})
    out = build_lags(master, columns=["cpi"], horizons=[1, 2, 3])

    for h in (1, 2, 3):
        assert not (out[f"cpi_lag_{h}"] == _FUTURE_SENTINEL).any()


def test_lag_is_exactly_backward_shift() -> None:
    """Property: lag_h at row t equals the source at row t-h, else NaN (never t+h)."""
    rng = np.random.default_rng(12)
    values = rng.normal(size=8).tolist()
    master = _master(n=8, columns={"cpi": values})
    out = build_lags(master, columns=["cpi"], horizons=[1, 4])

    for h in (1, 4):
        for t in range(len(master)):
            got = out.loc[t, f"cpi_lag_{h}"]
            if t - h < 0:
                assert pd.isna(got)
            else:
                assert got == values[t - h]


def test_unsorted_input_is_sorted_before_lagging() -> None:
    """Rows given out of meeting order are sorted ascending first (no leak)."""
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    shuffled = master.iloc[[5, 0, 3, 1, 4, 2]].reset_index(drop=True)
    out = build_lags(shuffled, columns=["cpi"], horizons=[1])
    # After internal sort the sequence is monotone by date → lag_1 is the prior row.
    assert out["meeting_date"].is_monotonic_increasing
    assert out["cpi_lag_1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0, 5.0]


# -----------------------------------------------------------------------------
# Graceful skip / warn for absent or ineligible columns.
# -----------------------------------------------------------------------------
def test_absent_configured_column_skipped_with_warning(loguru_messages: list) -> None:
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = build_lags(master, columns=["cpi", "not_in_frame"], horizons=[1])

    assert "cpi_lag_1" in out.columns
    assert not any(c.startswith("not_in_frame") for c in out.columns)
    assert any("not_in_frame" in m and "absent" in m for m in loguru_messages)


def test_no_resolved_columns_returns_meeting_key_only() -> None:
    master = _master(columns={"cpi": [1.0] * 6})
    out = build_lags(master, columns=["nonexistent"], horizons=[1])
    assert list(out.columns) == ["meeting_date"]
    assert len(out) == len(master)


# -----------------------------------------------------------------------------
# Companion / regime / metadata columns are never lagged.
# -----------------------------------------------------------------------------
def test_companion_regime_and_metadata_columns_never_lagged(
    loguru_messages: list,
) -> None:
    """Explicitly requesting non-level columns warn-skips them all."""
    master = _master(columns={"cpi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    master["cpi_age_days"] = pd.array([1, 2, 3, 4, 5, 6], dtype="Int64")
    master["cpi_is_missing"] = [0, 0, 0, 0, 0, 0]
    master["regime_covid"] = [0, 0, 1, 1, 0, 0]

    requested = [
        "cpi",
        "cpi_age_days",
        "cpi_is_missing",
        "regime_covid",
        "meeting_date",
        "rate_change_bps",
    ]
    out = build_lags(master, columns=requested, horizons=[1])

    # Only the genuine level series is lagged.
    assert list(out.columns) == ["meeting_date", "cpi_lag_1"]
    for skipped in ("cpi_age_days", "cpi_is_missing", "regime_covid"):
        assert not any(c.startswith(f"{skipped}_lag_") for c in out.columns)
        assert any(skipped in m and "not a level" in m for m in loguru_messages)


def test_default_columns_and_horizons_come_from_features_yaml() -> None:
    """With no kwargs, the real features.yaml drives columns + horizons.

    Only the configured series present in the frame get lagged; the rest are
    warn-skipped, and horizons match ``lags.horizons_meetings``.
    """
    from rba.config import load_features_config

    config = load_features_config()
    cols = config_lag_columns(config)
    horizons = config_lag_horizons(config)
    assert "headline_cpi_index" in cols  # reconciled to a real align series_id
    assert horizons == (1, 2, 3, 6, 12)

    master = _master(columns={"headline_cpi_index": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    out = build_lags(master)  # defaults from features.yaml
    expected = [f"headline_cpi_index_lag_{h}" for h in horizons]
    assert [c for c in out.columns if c != "meeting_date"] == expected


def test_injected_config_overrides_disk_load() -> None:
    """A passed-in config mapping is used instead of reading features.yaml."""
    master = _master(columns={"foo": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    config = {"lags": {"monthly_columns": ["foo"], "horizons_meetings": [2]}}
    out = build_lags(master, config=config)
    assert list(out.columns) == ["meeting_date", "foo_lag_2"]


def test_config_horizons_default_when_absent() -> None:
    assert config_lag_horizons({"lags": {}}) == DEFAULT_LAG_HORIZONS_MEETINGS
    assert config_lag_columns({}) == []
