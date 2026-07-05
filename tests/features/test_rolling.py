"""Tests for ``rba.features.rolling`` — meeting-space rolling statistics.

No network — every frame is built inline. The headline guard is a **past-only /
no-future-leakage** check: changing a later meeting's value must never alter an
earlier meeting's rolling statistic (mirrors the synthetic future-injection
spirit of ``tests/data/test_no_leakage.py``, adapted to windowed stats).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.features.rolling import (
    DEFAULT_ROLLING_STATS,
    DEFAULT_ROLLING_WINDOWS_MEETINGS,
    build_rolling,
    config_rolling_columns,
    config_rolling_stats,
    config_rolling_windows,
)

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


def _master(values: list[float], name: str = "cpi") -> pd.DataFrame:
    """A meeting-indexed master-like frame: monthly meetings + one level column."""
    return pd.DataFrame(
        {
            "meeting_date": pd.date_range("2020-01-01", periods=len(values), freq="MS"),
            name: values,
        }
    )


# -----------------------------------------------------------------------------
# Shape / dtype / naming.
# -----------------------------------------------------------------------------
def test_output_shape_and_columns() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["mean", "std"])
    assert list(out.columns) == ["meeting_date", "cpi_roll_3_mean", "cpi_roll_3_std"]
    assert len(out) == len(master)
    assert out["cpi_roll_3_mean"].dtype == np.float64


def test_column_order_is_col_then_window_then_stat() -> None:
    master = _master([1.0] * 6)
    out = build_rolling(master, columns=["cpi"], windows=[3, 6], stats=["mean", "max"])
    assert list(out.columns) == [
        "meeting_date",
        "cpi_roll_3_mean",
        "cpi_roll_3_max",
        "cpi_roll_6_mean",
        "cpi_roll_6_max",
    ]


def test_returns_only_meeting_key_and_stats_not_source() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["mean"])
    assert "cpi" not in out.columns
    assert set(out.columns) == {"meeting_date", "cpi_roll_3_mean"}


# -----------------------------------------------------------------------------
# Correctness of each statistic (past-only, inclusive window).
# -----------------------------------------------------------------------------
def test_rolling_mean_is_past_only_inclusive() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["mean"])
    # Row 2 = mean(1,2,3)=2; row 5 = mean(4,5,6)=5. First two rows NaN.
    assert out.loc[2, "cpi_roll_3_mean"] == 2.0
    assert out.loc[5, "cpi_roll_3_mean"] == 5.0


def test_min_and_max_over_window() -> None:
    master = _master([5.0, 1.0, 3.0, 9.0, 2.0, 8.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["min", "max"])
    assert out.loc[3, "cpi_roll_3_min"] == 1.0  # min(1,3,9)
    assert out.loc[3, "cpi_roll_3_max"] == 9.0  # max(1,3,9)


def test_std_is_sample_ddof1() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["std"])
    # std(1,2,3) with ddof=1 == 1.0
    assert out.loc[2, "cpi_roll_3_std"] == pytest.approx(1.0)


def test_zscore_uses_rolling_mean_and_std() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["zscore"])
    # At row 2: (3 - mean(1,2,3)) / std(1,2,3) = (3-2)/1 = 1.0
    assert out.loc[2, "cpi_roll_3_zscore"] == pytest.approx(1.0)


def test_zscore_is_nan_on_flat_window_not_inf() -> None:
    """A window with zero variance → NaN z-score, never ±inf."""
    master = _master([7.0, 7.0, 7.0, 7.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["zscore"])
    z = out["cpi_roll_3_zscore"]
    assert not np.isinf(z.to_numpy()).any()
    assert z.iloc[2:].isna().all()


# -----------------------------------------------------------------------------
# NaN at the series start — never zero (explicit CONTEXT pitfall).
# -----------------------------------------------------------------------------
def test_windowed_stats_nan_at_start_not_zero() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(
        master, columns=["cpi"], windows=[3], stats=["mean", "std", "min", "max", "zscore"]
    )
    for stat in ("mean", "std", "min", "max", "zscore"):
        col = out[f"cpi_roll_3_{stat}"]
        assert col.iloc[0] != 0
        assert col.iloc[:2].isna().all()  # first window-1 rows NaN
        assert col.iloc[2:].notna().all()


def test_ewma_has_no_leading_window_nan() -> None:
    """EWMA yields a value from the first meeting (EWMA of one point)."""
    master = _master([1.0, 2.0, 3.0, 4.0])
    out = build_rolling(master, columns=["cpi"], windows=[3], stats=["ewma"])
    assert out["cpi_roll_3_ewma"].notna().all()
    assert out.loc[0, "cpi_roll_3_ewma"] == 1.0  # EWMA seeded by first value


# -----------------------------------------------------------------------------
# Leakage guard — a future meeting's value never changes an earlier stat.
# -----------------------------------------------------------------------------
def test_future_value_never_affects_earlier_rolling_stats() -> None:
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    injected = base[:-1] + [_FUTURE_SENTINEL]  # only the LAST meeting changes
    stats = ["mean", "std", "zscore", "min", "max", "ewma"]

    out_base = build_rolling(_master(base), columns=["cpi"], windows=[3], stats=stats)
    out_inj = build_rolling(_master(injected), columns=["cpi"], windows=[3], stats=stats)

    # Every row except the last must be byte-identical (no future leakage).
    for stat in stats:
        col = f"cpi_roll_3_{stat}"
        pd.testing.assert_series_equal(
            out_base[col].iloc[:-1], out_inj[col].iloc[:-1], check_names=False
        )
    # The sentinel only ever surfaces at/after its own meeting, never before.
    assert not (out_inj["cpi_roll_3_max"].iloc[:-1] == _FUTURE_SENTINEL).any()


def test_no_center_matches_manual_past_only() -> None:
    """Past-only: never center=True — reconstruct the trailing window by hand."""
    rng = np.random.default_rng(12)
    values = rng.normal(size=10).tolist()
    master = _master(values)
    out = build_rolling(master, columns=["cpi"], windows=[4], stats=["mean"])
    for t in range(len(values)):
        got = out.loc[t, "cpi_roll_4_mean"]
        if t < 3:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(np.mean(values[t - 3 : t + 1]))


def test_unsorted_input_is_sorted_before_rolling() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    shuffled = master.iloc[[5, 0, 3, 1, 4, 2]].reset_index(drop=True)
    out = build_rolling(shuffled, columns=["cpi"], windows=[3], stats=["mean"])
    assert out["meeting_date"].is_monotonic_increasing
    assert out.loc[2, "cpi_roll_3_mean"] == 2.0


def test_windows_are_trailing_not_centered() -> None:
    """Invariant #3 guard: every windowed stat is center=False (past-only).

    Directly contrasts the builder's output against ``center=True``. A trailing
    window (a) leaves the *tail* populated and the *head* NaN, and (b) at every
    row equals the pandas trailing computation and differs from the centered one
    on a monotone series — so a ``center=True`` regression fails here loudly.
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    master = _master(values)
    series = pd.Series(values)

    for stat, trailing in (
        ("mean", series.rolling(3).mean()),
        ("std", series.rolling(3).std()),
        ("min", series.rolling(3).min()),
        ("max", series.rolling(3).max()),
    ):
        out = build_rolling(master, columns=["cpi"], windows=[3], stats=[stat])
        got = out[f"cpi_roll_3_{stat}"]
        centered = series.rolling(3, center=True).agg(stat)

        # (a) Trailing window: head NaN, tail populated (centered NaNs the tail).
        assert pd.isna(got.iloc[0]) and pd.notna(got.iloc[-1])
        assert pd.isna(centered.iloc[-1])  # a centered window would drop the last row
        # (b) Matches trailing (center=False), differs from centered.
        pd.testing.assert_series_equal(got, trailing, check_names=False)
        assert not got.equals(centered)


# -----------------------------------------------------------------------------
# Graceful skip / warn + stat validation.
# -----------------------------------------------------------------------------
def test_absent_column_warn_skipped(loguru_messages: list) -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = build_rolling(master, columns=["cpi", "not_in_frame"], windows=[3], stats=["mean"])
    assert "cpi_roll_3_mean" in out.columns
    assert not any(c.startswith("not_in_frame") for c in out.columns)
    assert any("not_in_frame" in m and "absent" in m for m in loguru_messages)


def test_companion_regime_metadata_never_rolled(loguru_messages: list) -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    master["cpi_age_days"] = pd.array([1, 2, 3, 4, 5, 6], dtype="Int64")
    master["cpi_is_missing"] = [0, 0, 0, 0, 0, 0]
    master["regime_covid"] = [0, 0, 1, 1, 0, 0]

    requested = ["cpi", "cpi_age_days", "cpi_is_missing", "regime_covid", "rate_change_bps"]
    out = build_rolling(master, columns=requested, windows=[3], stats=["mean"])
    assert list(out.columns) == ["meeting_date", "cpi_roll_3_mean"]
    for skipped in ("cpi_age_days", "cpi_is_missing", "regime_covid"):
        assert any(skipped in m and "not a level" in m for m in loguru_messages)


def test_unknown_stat_raises() -> None:
    master = _master([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unknown rolling statistic"):
        build_rolling(master, columns=["cpi"], windows=[2], stats=["median"])


def test_no_resolved_columns_returns_meeting_key_only() -> None:
    master = _master([1.0, 2.0, 3.0])
    out = build_rolling(master, columns=["nonexistent"], windows=[2], stats=["mean"])
    assert list(out.columns) == ["meeting_date"]


# -----------------------------------------------------------------------------
# Config-driven defaults.
# -----------------------------------------------------------------------------
def test_defaults_from_features_yaml() -> None:
    from rba.config import load_features_config

    config = load_features_config()
    cols = config_rolling_columns(config)
    windows = config_rolling_windows(config)
    stats = config_rolling_stats(config)
    assert "headline_cpi_index" in cols  # reconciled to a real align series_id
    assert windows == (3, 6, 12)
    assert "ewma" in stats  # checklist item added EWMA

    master = _master([float(i) for i in range(15)], name="headline_cpi_index")
    out = build_rolling(master)
    expected = [
        f"headline_cpi_index_roll_{w}_{s}" for w in windows for s in stats
    ]
    assert [c for c in out.columns if c != "meeting_date"] == expected


def test_injected_config_overrides_disk_load() -> None:
    master = _master([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="foo")
    config = {"rolling": {"monthly_columns": ["foo"], "windows_meetings": [2], "stats": ["mean"]}}
    out = build_rolling(master, config=config)
    assert list(out.columns) == ["meeting_date", "foo_roll_2_mean"]


def test_config_defaults_when_section_empty() -> None:
    assert config_rolling_windows({"rolling": {}}) == DEFAULT_ROLLING_WINDOWS_MEETINGS
    assert config_rolling_stats({"rolling": {}}) == DEFAULT_ROLLING_STATS
    assert config_rolling_columns({}) == []
