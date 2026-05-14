"""Tests for ``rba.data.preprocess_target.build_decisions``."""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.preprocess_target import build_decisions


def _raw_row(
    observation_date: str,
    *,
    change_raw: str,
    new_rate_raw: str,
    statement_url: str | None = None,
    minutes_url: str | None = None,
) -> dict[str, object]:
    obs = pd.Timestamp(observation_date)
    pub = obs - pd.Timedelta(days=1) if obs >= pd.Timestamp("2008-02-06") else obs
    return {
        "observation_date": obs,
        "publication_date": pub,
        "change_raw": change_raw,
        "new_cash_rate_raw": new_rate_raw,
        "statement_url": statement_url,
        "minutes_url": minutes_url,
    }


def _raw_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_filters_pre_history_start() -> None:
    raw = _raw_frame(
        _raw_row("1990-01-23", change_raw="-0.50 to -1.00", new_rate_raw="17.00 to 17.50"),
        _raw_row("1992-12-01", change_raw="-0.50", new_rate_raw="5.75"),
        _raw_row("1993-03-23", change_raw="-0.50", new_rate_raw="5.25"),
    )
    out = build_decisions(raw)
    assert (out["observation_date"] >= pd.Timestamp("1993-01-01")).all()
    assert pd.Timestamp("1993-03-23") in out["observation_date"].values
    assert pd.Timestamp("1992-12-01") not in out["observation_date"].values


def test_drops_unparseable_range_rows() -> None:
    raw = _raw_frame(
        _raw_row("1990-01-23", change_raw="-0.50 to -1.00", new_rate_raw="17.00 to 17.50"),
        _raw_row("2025-12-10", change_raw="0.00", new_rate_raw="3.60"),
    )
    out = build_decisions(raw, history_start="1990-01-01")
    # 1990 range row dropped, modern row kept.
    assert len(out) == 1
    assert out["observation_date"].iloc[0] == pd.Timestamp("2025-12-10")


def test_computes_basis_points_correctly() -> None:
    raw = _raw_frame(
        _raw_row("2026-05-06", change_raw="+0.25", new_rate_raw="4.35"),
        _raw_row("2025-12-10", change_raw="0.00", new_rate_raw="3.60"),
        _raw_row("2025-08-13", change_raw="-0.25", new_rate_raw="3.60"),
        _raw_row("2020-03-20", change_raw="-0.25", new_rate_raw="0.25"),
    )
    out = build_decisions(raw)
    out = out.sort_values("observation_date").reset_index(drop=True)
    assert out["rate_change_bps"].tolist() == [-25, -25, 0, 25]
    assert out["rate_change_bps"].dtype == "int64"


def test_handles_50bp_and_larger_changes() -> None:
    # COVID-era 50bp emergency cut, then later larger hikes.
    raw = _raw_frame(
        _raw_row("2020-03-19", change_raw="-0.25", new_rate_raw="0.50"),
        _raw_row("2022-06-08", change_raw="+0.50", new_rate_raw="0.85"),
        _raw_row("1994-12-14", change_raw="+1.00", new_rate_raw="7.50"),
    )
    out = build_decisions(raw)
    assert set(out["rate_change_bps"]) == {-25, 50, 100}


def test_computes_prior_rate_correctly() -> None:
    raw = _raw_frame(
        _raw_row("2026-05-06", change_raw="+0.25", new_rate_raw="4.35"),
        _raw_row("2025-08-13", change_raw="-0.25", new_rate_raw="3.60"),
    )
    out = build_decisions(raw).sort_values("observation_date").reset_index(drop=True)
    assert out["prior_rate_pct"].iloc[0] == pytest.approx(3.85)  # 3.60 - (-0.25)
    assert out["prior_rate_pct"].iloc[1] == pytest.approx(4.10)  # 4.35 - 0.25


def test_holds_yield_zero_bps_and_unchanged_rate() -> None:
    raw = _raw_frame(
        _raw_row("2025-12-10", change_raw="0.00", new_rate_raw="3.60"),
    )
    out = build_decisions(raw)
    assert out["rate_change_bps"].iloc[0] == 0
    assert out["new_rate_pct"].iloc[0] == out["prior_rate_pct"].iloc[0] == 3.60


def test_output_columns_and_order() -> None:
    raw = _raw_frame(
        _raw_row("2026-05-06", change_raw="+0.25", new_rate_raw="4.35"),
    )
    out = build_decisions(raw)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "rate_change_bps",
        "new_rate_pct",
        "prior_rate_pct",
        "statement_url",
        "minutes_url",
    ]


def test_output_sorted_ascending_by_observation_date() -> None:
    raw = _raw_frame(
        _raw_row("2026-05-06", change_raw="+0.25", new_rate_raw="4.35"),
        _raw_row("1993-03-23", change_raw="-0.50", new_rate_raw="5.25"),
        _raw_row("2025-08-13", change_raw="-0.25", new_rate_raw="3.60"),
    )
    out = build_decisions(raw)
    assert out["observation_date"].is_monotonic_increasing


def test_raises_when_input_missing_required_columns() -> None:
    raw = pd.DataFrame({"observation_date": [pd.Timestamp("2026-05-06")]})
    with pytest.raises(ValueError, match="missing required columns"):
        build_decisions(raw)


def test_history_start_override() -> None:
    raw = _raw_frame(
        _raw_row("2020-12-01", change_raw="0.00", new_rate_raw="0.10"),
        _raw_row("2025-12-10", change_raw="0.00", new_rate_raw="3.60"),
    )
    out = build_decisions(raw, history_start="2022-01-01")
    assert len(out) == 1
    assert out["observation_date"].iloc[0] == pd.Timestamp("2025-12-10")
