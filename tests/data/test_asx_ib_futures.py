"""Tests for ``rba.data.sources.asx_ib_futures``.

The live ``fetch()`` path hits GitHub (Rds download) and is not
exercised here. ``_parse_upstream_dataframe``,
``_attach_publication_dates``, ``derive_meeting_implied``, and
``_to_wide`` are tested in isolation against a synthetic upstream
DataFrame mirroring the MattCowgill ``combined_data/all_data.Rds``
schema (columns ``date``, ``cash_rate``, ``scrape_date`` as
pyreadr-style object-dtype strings).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.asx_ib_futures import (
    _UPSTREAM_COVERAGE_START,
    AsxIbContract,
    _attach_publication_dates,
    _latest_trade_date_on_or_before,
    _parse_upstream_dataframe,
    _to_wide,
    derive_meeting_implied,
)


# Synthetic upstream tibble mirroring the real Rds schema: pyreadr loads
# date columns as Python object (string YYYY-MM-DD), not datetime64. The
# scrape_date values cover three trade days; each trade day has a small
# subset of the IB curve (just to keep the fixture readable).
def _synthetic_upstream() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                # 2022-04-21 trade — 3 contracts
                "2022-05-01", "2022-06-01", "2022-07-01",
                # 2024-02-05 trade (day before a notional 2024-02-06 meeting) — 4 contracts
                "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01",
                # 2025-12-08 trade — 3 contracts
                "2025-12-01", "2026-01-01", "2026-02-01",
                # A row with a NaN value to exercise drop-null
                "2026-03-01",
                # A row with a NaN trade date to exercise drop-null
                "2026-04-01",
            ],
            "cash_rate": [
                0.10, 0.25, 0.50,                  # 2022-04-21
                4.35, 4.30, 4.10, 3.95,            # 2024-02-05
                3.60, 3.55, 3.50,                  # 2025-12-08
                float("nan"),                      # null value
                3.45,                              # null trade date
            ],
            "scrape_date": [
                "2022-04-21", "2022-04-21", "2022-04-21",
                "2024-02-05", "2024-02-05", "2024-02-05", "2024-02-05",
                "2025-12-08", "2025-12-08", "2025-12-08",
                "2026-04-01",
                None,
            ],
        }
    )


# -----------------------------------------------------------------------------
# _parse_upstream_dataframe
# -----------------------------------------------------------------------------


def test_parse_extracts_canonical_long_format() -> None:
    df = _parse_upstream_dataframe(_synthetic_upstream())
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_drops_null_rows() -> None:
    """Null cash_rate and null scrape_date rows must be dropped."""
    df = _parse_upstream_dataframe(_synthetic_upstream())
    # 12 input rows (3+4+3 valid + 1 null value + 1 null trade date) → 10 surviving
    assert len(df) == 10


def test_parse_converts_cash_rate_to_settlement_price() -> None:
    """value column must be 100 - cash_rate (settlement-price convention)."""
    df = _parse_upstream_dataframe(_synthetic_upstream())
    row = df[
        (df["observation_date"] == pd.Timestamp("2024-02-05"))
        & (df["series_id"] == "ib_2024_02")
    ].iloc[0]
    # cash_rate was 4.35 → settlement = 100 - 4.35 = 95.65
    assert row["value"] == pytest.approx(95.65)


def test_parse_series_id_format() -> None:
    df = _parse_upstream_dataframe(_synthetic_upstream())
    # 2022-05-01 expiry → ib_2022_05
    assert "ib_2022_05" in set(df["series_id"])
    assert "ib_2025_12" in set(df["series_id"])
    # All series ids match the expected pattern
    assert df["series_id"].str.match(r"^ib_\d{4}_\d{2}$").all()


def test_parse_raises_on_missing_columns() -> None:
    upstream = pd.DataFrame({"date": ["2024-01-01"], "cash_rate": [1.0]})  # no scrape_date
    with pytest.raises(ValueError, match="missing required column"):
        _parse_upstream_dataframe(upstream)


def test_parse_raises_on_empty_data() -> None:
    upstream = pd.DataFrame(
        {"date": [None, None], "cash_rate": [None, None], "scrape_date": [None, None]}
    )
    with pytest.raises(ValueError, match="zero IB futures observations"):
        _parse_upstream_dataframe(upstream)


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_equals_observation_date() -> None:
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    out = _attach_publication_dates(long_df)
    assert (out["publication_date"] == out["observation_date"]).all()


def test_attach_publication_dates_drops_stale_column() -> None:
    """A pre-existing publication_date must be dropped before re-assignment."""
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    long_df["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(long_df)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    out = _attach_publication_dates(long_df)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_le_observation() -> None:
    """publication_date must never be strictly after observation_date."""
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    out = _attach_publication_dates(long_df)
    assert (out["publication_date"] <= out["observation_date"]).all()


def test_all_observations_within_upstream_coverage() -> None:
    """Every fixture trade date is on/after the documented upstream start."""
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    assert (long_df["observation_date"] >= _UPSTREAM_COVERAGE_START).all()


# -----------------------------------------------------------------------------
# derive_meeting_implied
# -----------------------------------------------------------------------------


def _curve_long(trade_date: str, contract_to_settlement: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime([trade_date] * len(contract_to_settlement)),
            "series_id": list(contract_to_settlement.keys()),
            "value": list(contract_to_settlement.values()),
        }
    )


def test_derive_meeting_implied_matches_meeting_month_contract() -> None:
    """For a meeting on Tue 2024-02-06, look up the curve as of Mon
    2024-02-05 (1 business day prior) and pick the ib_2024_02 contract.
    Settlement 95.65 → implied 100 - 95.65 = 4.35."""
    curve = _curve_long(
        "2024-02-05",
        {
            "ib_2024_02": 95.65,  # implied 4.35
            "ib_2024_03": 95.70,
            "ib_2024_04": 95.90,
        },
    )
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2024-02-06"])})
    out = derive_meeting_implied(curve, meetings)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["meeting_date"] == pd.Timestamp("2024-02-06")
    assert row["lookup_date"] == pd.Timestamp("2024-02-05")
    assert row["contract_series_id"] == "ib_2024_02"
    assert row["implied_cash_rate"] == pytest.approx(4.35)


def test_derive_meeting_implied_walks_back_on_holiday_gap() -> None:
    """If T-1 isn't a trade date (e.g. weekend / holiday), step back
    through earlier business days."""
    # Curve only has Friday 2024-02-02; meeting Mon 2024-02-05.
    # T-1 business day = Fri 2024-02-02 → resolves on first try.
    curve = _curve_long("2024-02-02", {"ib_2024_02": 95.50})
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2024-02-05"])})
    out = derive_meeting_implied(curve, meetings)
    assert out.iloc[0]["lookup_date"] == pd.Timestamp("2024-02-02")
    assert out.iloc[0]["implied_cash_rate"] == pytest.approx(4.50)


def test_derive_meeting_implied_returns_nan_when_contract_not_quoted() -> None:
    """If the meeting-month contract isn't on the curve that day, return NaN."""
    curve = _curve_long(
        "2024-02-05",
        {"ib_2024_03": 95.70, "ib_2024_04": 95.90},  # no ib_2024_02
    )
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2024-02-06"])})
    out = derive_meeting_implied(curve, meetings)
    row = out.iloc[0]
    assert row["lookup_date"] == pd.Timestamp("2024-02-05")
    assert row["contract_series_id"] == "ib_2024_02"
    assert pd.isna(row["implied_cash_rate"])


def test_derive_meeting_implied_returns_nan_when_meeting_predates_coverage() -> None:
    """Meetings before any available trade date yield NaT lookup and NaN rate."""
    curve = _curve_long("2024-02-05", {"ib_2024_02": 95.65})
    # Meeting in 2020 — well before any curve data.
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2020-03-19"])})
    out = derive_meeting_implied(curve, meetings)
    row = out.iloc[0]
    assert pd.isna(row["lookup_date"])
    assert pd.isna(row["implied_cash_rate"])


def test_derive_meeting_implied_no_future_leakage() -> None:
    """The lookup_date must be strictly before the meeting_date for
    every successfully resolved row — this is the core leakage guard."""
    # Meeting Tue 2024-02-06; curve has same-day quote and a prior-day quote.
    # Helper must pick the prior-day quote (T-1 business day) — never same-day.
    curve = pd.concat(
        [
            _curve_long("2024-02-05", {"ib_2024_02": 95.65}),
            _curve_long("2024-02-06", {"ib_2024_02": 95.50}),  # same-day; must NOT be used
        ],
        ignore_index=True,
    )
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2024-02-06"])})
    out = derive_meeting_implied(curve, meetings)
    row = out.iloc[0]
    assert row["lookup_date"] == pd.Timestamp("2024-02-05")
    assert row["lookup_date"] < row["meeting_date"]
    # Confirms we used the T-1 settlement (4.35), not the same-day one (4.50).
    assert row["implied_cash_rate"] == pytest.approx(4.35)


def test_derive_meeting_implied_raises_on_missing_meeting_column() -> None:
    curve = _curve_long("2024-02-05", {"ib_2024_02": 95.65})
    meetings = pd.DataFrame({"meeting_dt": pd.to_datetime(["2024-02-06"])})  # wrong col
    with pytest.raises(ValueError, match="observation_date"):
        derive_meeting_implied(curve, meetings)


def test_derive_meeting_implied_raises_on_malformed_long_df() -> None:
    long_df = pd.DataFrame({"observation_date": [], "series_id": []})  # no value
    meetings = pd.DataFrame({"observation_date": pd.to_datetime(["2024-02-06"])})
    with pytest.raises(ValueError, match="long_df missing"):
        derive_meeting_implied(long_df, meetings)


def test_derive_meeting_implied_multiple_meetings() -> None:
    curve = pd.concat(
        [
            _curve_long("2024-02-05", {"ib_2024_02": 95.65, "ib_2024_03": 95.70}),
            _curve_long("2024-03-04", {"ib_2024_03": 95.80, "ib_2024_04": 95.90}),
        ],
        ignore_index=True,
    )
    meetings = pd.DataFrame(
        {"observation_date": pd.to_datetime(["2024-02-06", "2024-03-05"])}
    )
    out = derive_meeting_implied(curve, meetings)
    assert len(out) == 2
    assert out.iloc[0]["implied_cash_rate"] == pytest.approx(4.35)  # 100 - 95.65
    assert out.iloc[1]["implied_cash_rate"] == pytest.approx(4.20)  # 100 - 95.80


# -----------------------------------------------------------------------------
# _latest_trade_date_on_or_before
# -----------------------------------------------------------------------------


def test_latest_trade_date_finds_exact_match() -> None:
    available = pd.Index(pd.to_datetime(["2024-02-02", "2024-02-05"]))
    out = _latest_trade_date_on_or_before(
        pd.Timestamp("2024-02-05"), available, max_lookback_bdays=7
    )
    assert out == pd.Timestamp("2024-02-05")


def test_latest_trade_date_walks_back_one_day() -> None:
    available = pd.Index(pd.to_datetime(["2024-02-02"]))  # only Fri
    out = _latest_trade_date_on_or_before(
        pd.Timestamp("2024-02-05"), available, max_lookback_bdays=7
    )
    assert out == pd.Timestamp("2024-02-02")


def test_latest_trade_date_returns_none_when_too_far_back() -> None:
    """If the only available date is way before the target, return None."""
    available = pd.Index(pd.to_datetime(["2020-01-01"]))
    out = _latest_trade_date_on_or_before(
        pd.Timestamp("2024-02-05"), available, max_lookback_bdays=7
    )
    assert out is None


def test_latest_trade_date_returns_none_when_empty() -> None:
    out = _latest_trade_date_on_or_before(
        pd.Timestamp("2024-02-05"), pd.Index([], dtype="datetime64[ns]"), max_lookback_bdays=7
    )
    assert out is None


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema() -> None:
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "n_contracts"
    # Contract columns sorted by expiry month string (alphanumeric == chronological).
    contract_cols = [c for c in wide.columns if c.startswith("ib_")]
    assert contract_cols == sorted(contract_cols)


def test_to_wide_n_contracts_counts_non_null() -> None:
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    row_22 = wide[wide["trade_date"] == pd.Timestamp("2022-04-21")].iloc[0]
    assert row_22["n_contracts"] == 3
    row_24 = wide[wide["trade_date"] == pd.Timestamp("2024-02-05")].iloc[0]
    assert row_24["n_contracts"] == 4


def test_to_wide_preserves_settlement_values() -> None:
    long_df = _parse_upstream_dataframe(_synthetic_upstream())
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    row = wide[wide["trade_date"] == pd.Timestamp("2024-02-05")].iloc[0]
    assert row["ib_2024_02"] == pytest.approx(95.65)
    assert row["ib_2024_05"] == pytest.approx(96.05)


# -----------------------------------------------------------------------------
# AsxIbContract dataclass
# -----------------------------------------------------------------------------


def test_asx_ib_contract_from_expiry_month_formats_id() -> None:
    c = AsxIbContract.from_expiry_month(pd.Timestamp("2025-09-01"))
    assert c.series_id == "ib_2025_09"
    assert c.expiry_month_start == pd.Timestamp("2025-09-01")
