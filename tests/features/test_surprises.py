"""Tests for ``rba.features.surprises`` — realised − consensus surprises.

No network — every frame is built inline. Because no consensus source exists
yet, the builder's headline behaviour is to stay **inert** (skip, don't crash)
on a consensus-free frame, and to **activate** the moment a consensus column is
present. Both are asserted here, the latter with a synthetic consensus column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.features.surprises import (
    build_surprises,
    config_surprise_pairs,
    config_surprises_enabled,
)


# -----------------------------------------------------------------------------
# Fixtures / builders.
# -----------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    """Capture loguru messages at ALL levels (surprises info-logs skips)."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _master(columns: dict[str, list[float]], n: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"meeting_date": pd.date_range("2020-01-01", periods=n, freq="MS")}
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


# -----------------------------------------------------------------------------
# Activation — where consensus IS available.
# -----------------------------------------------------------------------------
def test_surprise_is_realised_minus_consensus() -> None:
    master = _master(
        {
            "cpi": [3.0, 3.5, 4.0, 4.2],
            "cpi_consensus": [3.1, 3.2, 4.1, 4.0],
        }
    )
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    assert list(out.columns) == ["meeting_date", "cpi_surprise"]
    assert out["cpi_surprise"].tolist() == pytest.approx([-0.1, 0.3, -0.1, 0.2])


def test_returns_only_meeting_key_and_surprise() -> None:
    master = _master({"cpi": [1.0, 2.0, 3.0, 4.0], "cpi_consensus": [1.0, 1.0, 1.0, 1.0]})
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    assert set(out.columns) == {"meeting_date", "cpi_surprise"}
    assert "cpi" not in out.columns and "cpi_consensus" not in out.columns


def test_nan_when_either_side_missing_not_zero() -> None:
    master = _master(
        {
            "cpi": [3.0, float("nan"), 4.0, 4.2],
            "cpi_consensus": [3.1, 3.2, float("nan"), 4.0],
        }
    )
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    s = out["cpi_surprise"]
    assert pd.isna(s.iloc[1]) and pd.isna(s.iloc[2])  # missing side → NaN
    assert s.iloc[0] == pytest.approx(-0.1)
    assert s.iloc[1] != 0  # explicitly not zero-filled


def test_multiple_pairs_in_config_order() -> None:
    master = _master(
        {
            "cpi": [1.0, 2.0, 3.0, 4.0],
            "cpi_c": [0.5, 0.5, 0.5, 0.5],
            "unemp": [5.0, 5.1, 5.2, 5.3],
            "unemp_c": [5.0, 5.0, 5.0, 5.0],
        }
    )
    out = build_surprises(master, pairs={"cpi": "cpi_c", "unemp": "unemp_c"})
    assert list(out.columns) == ["meeting_date", "cpi_surprise", "unemp_surprise"]


# -----------------------------------------------------------------------------
# Inert — where consensus is NOT (yet) available.
# -----------------------------------------------------------------------------
def test_consensus_absent_is_skipped_not_crashed(loguru_messages: list) -> None:
    """The current-master reality: realised present, consensus never sourced."""
    master = _master({"cpi": [3.0, 3.5, 4.0, 4.2]})  # no consensus column
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    assert list(out.columns) == ["meeting_date"]  # inert, no crash
    assert any(
        "cpi_consensus" in m and "not sourced yet" in m for m in loguru_messages
    )


def test_realised_absent_is_warn_skipped(loguru_messages: list) -> None:
    master = _master({"cpi_consensus": [3.1, 3.2, 4.1, 4.0]})  # realised missing
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    assert list(out.columns) == ["meeting_date"]
    assert any("cpi" in m and "absent or not a level" in m for m in loguru_messages)


def test_non_level_sides_are_skipped(loguru_messages: list) -> None:
    """A companion / regime column may not be either side of a surprise."""
    master = _master({"cpi": [1.0, 2.0, 3.0, 4.0]})
    master["cpi_age_days"] = pd.array([1, 2, 3, 4], dtype="Int64")
    master["regime_covid"] = [0, 0, 1, 1]

    # realised is a companion → skip; consensus is a regime dummy → skip.
    out = build_surprises(
        master,
        pairs={"cpi_age_days": "cpi", "cpi": "regime_covid"},
    )
    assert list(out.columns) == ["meeting_date"]
    assert any("cpi_age_days" in m and "not a level" in m for m in loguru_messages)
    assert any("regime_covid" in m and "not a level" in m for m in loguru_messages)


def test_empty_pairs_returns_meeting_key_only() -> None:
    master = _master({"cpi": [1.0, 2.0, 3.0, 4.0]})
    out = build_surprises(master, pairs={})
    assert list(out.columns) == ["meeting_date"]
    assert len(out) == len(master)


# -----------------------------------------------------------------------------
# Point-in-time: inherited from align; row-wise independence.
# -----------------------------------------------------------------------------
def test_future_row_does_not_affect_earlier_surprises() -> None:
    """Changing the last meeting's inputs leaves every earlier surprise unchanged."""
    base = _master({"cpi": [1.0, 2.0, 3.0, 4.0], "cpi_c": [1.0, 1.0, 1.0, 1.0]})
    injected = _master({"cpi": [1.0, 2.0, 3.0, 9_999_999.0], "cpi_c": [1.0, 1.0, 1.0, 1.0]})
    out_base = build_surprises(base, pairs={"cpi": "cpi_c"})
    out_inj = build_surprises(injected, pairs={"cpi": "cpi_c"})
    pd.testing.assert_series_equal(
        out_base["cpi_surprise"].iloc[:-1],
        out_inj["cpi_surprise"].iloc[:-1],
        check_names=False,
    )


def test_unsorted_input_is_sorted() -> None:
    master = _master({"cpi": [1.0, 2.0, 3.0, 4.0], "cpi_c": [0.0, 0.0, 0.0, 0.0]})
    shuffled = master.iloc[[3, 1, 0, 2]].reset_index(drop=True)
    out = build_surprises(shuffled, pairs={"cpi": "cpi_c"})
    assert out["meeting_date"].is_monotonic_increasing
    assert out["cpi_surprise"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


# -----------------------------------------------------------------------------
# Config-driven behaviour.
# -----------------------------------------------------------------------------
def test_features_yaml_pairs_are_reconciled_and_disabled() -> None:
    from rba.config import load_features_config

    config = load_features_config()
    pairs = config_surprise_pairs(config)
    # Reconciled to real realised align series_ids; consensus sides are placeholders.
    assert "headline_cpi_index" in pairs
    assert "unemployment_rate_sa" in pairs
    assert config_surprises_enabled(config) is False  # inert until consensus sourced


def test_default_build_on_consensus_free_frame_is_inert(loguru_messages: list) -> None:
    """With config defaults and no consensus columns, nothing is produced."""
    # A frame carrying only the realised sides from features.yaml.
    master = _master(
        {
            "headline_cpi_index": [100.0, 101.0, 102.0, 103.0],
            "unemployment_rate_sa": [5.0, 5.1, 5.2, 5.3],
            "gdp_real_chain_volume_sa": [500.0, 501.0, 502.0, 503.0],
        }
    )
    out = build_surprises(master)  # defaults from features.yaml
    assert list(out.columns) == ["meeting_date"]
    assert any("not sourced yet" in m for m in loguru_messages)


def test_injected_config_overrides_disk_load() -> None:
    master = _master({"foo": [2.0, 3.0, 4.0, 5.0], "foo_c": [1.0, 1.0, 1.0, 1.0]})
    config = {"surprises": {"enabled": True, "pairs": {"foo": "foo_c"}}}
    out = build_surprises(master, config=config)
    assert list(out.columns) == ["meeting_date", "foo_surprise"]
    assert out["foo_surprise"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_config_accessors_defaults_when_absent() -> None:
    assert config_surprise_pairs({}) == {}
    assert config_surprise_pairs({"surprises": {}}) == {}
    assert config_surprises_enabled({}) is False
