"""Tests for ``rba.data.sources.commodity_prices``.

The live ``fetch()`` path hits ``fredgraph.csv?id=<SERIES_ID>`` and is
not exercised here. ``_parse_csv``, ``_attach_publication_dates``,
``_to_wide_daily``, and ``_to_wide_monthly`` are tested in isolation
against synthetic inputs mirroring the real FRED CSV layouts:

- Monthly fixture (PIORECRUSDM iron ore) — observation_date column at
  month-start, defensive trailing blank-valued row.
- Daily fixture (DCOILBRENTEU Brent) — every Mon-Fri populated except
  US-federal-holiday rows which are blank in the upstream CSV.

The structure mirrors ``test_fred_global_signals`` exactly; both
modules share parser / calendar / pivot semantics through the FRED
endpoint and the shared release-calendar dispatcher.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.commodity_prices import (
    _CALENDAR_VALIDATION_FLOOR,
    _DAILY_SERIES_IDS,
    _MONTHLY_SERIES_IDS,
    SERIES,
    CommoditySeries,
    _attach_publication_dates,
    _parse_csv,
    _to_wide_daily,
    _to_wide_monthly,
)


# Synthetic monthly FRED CSV — PIORECRUSDM layout. Three real values
# plus a defensive trailing blank row (IMF restates monthly, so the
# trailing month occasionally appears as a blank reservation cell).
_PIORECRUSDM_FIXTURE_CSV = (
    b"observation_date,PIORECRUSDM\n"
    b"1992-01-01,14.31000000000000\n"
    b"1992-02-01,14.31000000000000\n"
    b"2024-10-01,103.5\n"
    b"2024-11-01,105.2\n"
    b"2026-05-01,\n"  # trailing blank — must be dropped
)

# Synthetic daily FRED CSV — DCOILBRENTEU layout. Includes:
# - normal weekday rows,
# - a US-federal-holiday blank row (Thanksgiving 2024 = 28-Nov-2024),
# - a weekend row that FRED never actually emits (sanity check: the
#   parser drops it on the value-blank rule anyway).
_DCOILBRENTEU_FIXTURE_CSV = (
    b"observation_date,DCOILBRENTEU\n"
    b"2024-11-26,73.45\n"  # Tue
    b"2024-11-27,72.83\n"  # Wed
    b"2024-11-28,\n"       # Thu = Thanksgiving — blank in upstream
    b"2024-11-29,72.94\n"  # Fri
    b"2024-12-02,71.83\n"  # Mon
    b"2024-12-24,72.63\n"  # Tue — last full day before Christmas
)


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_four_entries() -> None:
    """Iron ore + copper + Brent + WTI = 4."""
    assert len(SERIES) == 4


def test_series_registry_fred_codes_unique() -> None:
    fred_codes = [s.fred_series_id for s in SERIES]
    assert len(fred_codes) == len(set(fred_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_fred_codes() -> None:
    by_id = {s.series_id: s.fred_series_id for s in SERIES}
    assert by_id["iron_ore_spot"] == "PIORECRUSDM"
    assert by_id["copper_spot"] == "PCOPPUSDM"
    assert by_id["brent_crude"] == "DCOILBRENTEU"
    assert by_id["wti_crude"] == "DCOILWTICO"


def test_series_registry_logical_ids_are_globally_neutral() -> None:
    """Commodity series IDs must not carry the ``us_`` prefix (these are
    global commodities, not US indicators — even though Brent and WTI
    are EIA-published and the IMF series are USD-denominated)."""
    for s in SERIES:
        assert not s.series_id.startswith("us_"), (
            f"{s.series_id!r} is a global commodity; drop the us_ prefix"
        )


def test_series_registry_frequencies_are_known() -> None:
    for s in SERIES:
        assert s.frequency in {"daily", "monthly"}


def test_daily_and_monthly_partition_covers_all_series() -> None:
    assert set(_DAILY_SERIES_IDS) | set(_MONTHLY_SERIES_IDS) == {
        s.series_id for s in SERIES
    }
    assert set(_DAILY_SERIES_IDS) & set(_MONTHLY_SERIES_IDS) == set()


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")


# -----------------------------------------------------------------------------
# _parse_csv — monthly path (PIORECRUSDM iron ore)
# -----------------------------------------------------------------------------


def test_parse_csv_monthly_extracts_rows() -> None:
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    df = _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 4 real rows; trailing blank dropped.
    assert len(df) == 4


def test_parse_csv_monthly_converts_to_month_end() -> None:
    """FRED stores monthly observations at the first day of the month;
    the parser must convert to month-end to match project-wide schema."""
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    df = _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec)
    obs_dates = set(df["observation_date"].tolist())
    assert pd.Timestamp("1992-01-31") in obs_dates
    assert pd.Timestamp("1992-02-29") in obs_dates  # 1992 was a leap year
    assert pd.Timestamp("2024-10-31") in obs_dates
    assert pd.Timestamp("2024-11-30") in obs_dates


def test_parse_csv_monthly_stamps_logical_series_id() -> None:
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    df = _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "iron_ore_spot").all()


def test_parse_csv_monthly_drops_blank_value_rows() -> None:
    """Trailing reference months for which FRED hasn't materialised a
    value yet appear as blank cells and must be dropped."""
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    df = _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("2026-05-31") not in df["observation_date"].tolist()


# -----------------------------------------------------------------------------
# _parse_csv — daily path (DCOILBRENTEU Brent)
# -----------------------------------------------------------------------------


def test_parse_csv_daily_preserves_trade_date() -> None:
    """Daily-series observation_date is the literal trade date, no
    period-end conversion."""
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    df = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec)
    dates = set(df["observation_date"].tolist())
    assert pd.Timestamp("2024-11-26") in dates
    assert pd.Timestamp("2024-11-27") in dates
    assert pd.Timestamp("2024-11-29") in dates
    assert pd.Timestamp("2024-12-02") in dates


def test_parse_csv_daily_drops_us_holiday_blank_rows() -> None:
    """Thanksgiving 2024 (Thu 28-Nov) is a blank cell in the upstream
    EIA CSV — the parser must drop it (cannot leak as a 0.0 or NaN row)."""
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    df = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("2024-11-28") not in df["observation_date"].tolist()


def test_parse_csv_daily_value_accuracy() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    df = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2024-11-26")] == pytest.approx(73.45)
    assert by_date[pd.Timestamp("2024-12-24")] == pytest.approx(72.63)


# -----------------------------------------------------------------------------
# _parse_csv — error paths
# -----------------------------------------------------------------------------


def test_parse_csv_raises_on_missing_observation_date_column() -> None:
    bad = b"DATE,PIORECRUSDM\n2024-10-01,103.5\n"
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    with pytest.raises(ValueError, match="missing expected 'observation_date'"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_missing_value_column() -> None:
    bad = b"observation_date,SOMETHING_ELSE\n2024-10-01,103.5\n"
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    with pytest.raises(ValueError, match="missing the value column"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_all_blank_values() -> None:
    """If every cell in the value column is blank, surface a ValueError
    rather than returning an empty frame silently — same defensive
    posture as the fred_global_signals parser."""
    bad = b"observation_date,PIORECRUSDM\n2024-10-01,\n2024-11-01,\n"
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    with pytest.raises(ValueError, match="Parsed zero observations"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_unknown_frequency() -> None:
    """A CommoditySeries with an unsupported frequency string must raise."""
    bad_spec = CommoditySeries(
        series_id="x", fred_series_id="PIORECRUSDM", frequency="quarterly"
    )
    with pytest.raises(ValueError, match="Unknown frequency"):
        _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=bad_spec)


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_monthly_iron_ore_plus_20_days() -> None:
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    parsed = _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    by_obs = dict(zip(out["observation_date"], out["publication_date"]))
    # 2024-10-31 → 2024-11-20 (+20 calendar days).
    assert by_obs[pd.Timestamp("2024-10-31")] == pd.Timestamp("2024-11-20")
    # 2024-11-30 → 2024-12-20.
    assert by_obs[pd.Timestamp("2024-11-30")] == pd.Timestamp("2024-12-20")


def test_attach_publication_dates_daily_brent_is_next_us_business_day() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    parsed = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    by_obs = dict(zip(out["observation_date"], out["publication_date"]))
    # Tue 26-Nov-2024 → Wed 27-Nov-2024.
    assert by_obs[pd.Timestamp("2024-11-26")] == pd.Timestamp("2024-11-27")
    # Wed 27-Nov-2024 → Fri 29-Nov-2024 (Thu 28 = Thanksgiving).
    assert by_obs[pd.Timestamp("2024-11-27")] == pd.Timestamp("2024-11-29")
    # Fri 29-Nov-2024 → Mon 02-Dec-2024 (weekend).
    assert by_obs[pd.Timestamp("2024-11-29")] == pd.Timestamp("2024-12-02")
    # Tue 24-Dec-2024 → Thu 26-Dec-2024 (Wed 25 = Christmas Day,
    # observed; Boxing Day is *not* a US federal holiday).
    assert by_obs[pd.Timestamp("2024-12-24")] == pd.Timestamp("2024-12-26")


def test_attach_publication_dates_output_column_order() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    parsed = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


def test_attach_publication_dates_empty_frame_no_raise() -> None:
    empty = pd.DataFrame(
        {
            "observation_date": pd.Series(dtype="datetime64[ns]"),
            "series_id": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    out = _attach_publication_dates(empty)
    assert out.empty
    assert "publication_date" in out.columns
    assert out["publication_date"].dtype == "datetime64[ns]"


def test_attach_publication_dates_drops_stale_column() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    parsed = _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_strictly_after_observation() -> None:
    """For every parsed commodity observation across daily and monthly
    series, publication_date must be strictly *after* observation_date —
    every commodity rule is either daily T+1 US BDay or +20 days, so
    there is no same-day rule (unlike VIX in fred_global_signals)."""
    daily_spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    monthly_spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    long_df = pd.concat(
        [
            _parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=daily_spec),
            _parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=monthly_spec),
        ],
        ignore_index=True,
    )
    out = _attach_publication_dates(long_df)
    has_pub = out["publication_date"].notna()
    assert has_pub.all()
    assert (out["publication_date"] > out["observation_date"]).all()


# -----------------------------------------------------------------------------
# _to_wide_daily / _to_wide_monthly
# -----------------------------------------------------------------------------


def test_to_wide_daily_schema_and_key() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    long_df = _attach_publication_dates(_parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec))
    wide = _to_wide_daily(long_df)
    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    assert "brent_crude" in wide.columns
    # Daily wide must not carry any monthly columns.
    for monthly_sid in _MONTHLY_SERIES_IDS:
        assert monthly_sid not in wide.columns


def test_to_wide_monthly_schema_and_key() -> None:
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    long_df = _attach_publication_dates(_parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec))
    wide = _to_wide_monthly(long_df)
    assert wide.columns[0] == "reference_month_end"
    assert wide.columns[-1] == "release_date"
    assert "iron_ore_spot" in wide.columns
    for daily_sid in _DAILY_SERIES_IDS:
        assert daily_sid not in wide.columns


def test_to_wide_daily_release_date_is_next_us_business_day() -> None:
    spec = CommoditySeries(
        series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"
    )
    long_df = _attach_publication_dates(_parse_csv(_DCOILBRENTEU_FIXTURE_CSV, spec=spec))
    wide = _to_wide_daily(long_df)
    by_obs = dict(zip(wide["trade_date"], wide["release_date"]))
    # Fri 29-Nov-2024 → Mon 02-Dec-2024.
    assert by_obs[pd.Timestamp("2024-11-29")] == pd.Timestamp("2024-12-02")


def test_to_wide_monthly_release_date_is_plus_20_days() -> None:
    spec = CommoditySeries(
        series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"
    )
    long_df = _attach_publication_dates(_parse_csv(_PIORECRUSDM_FIXTURE_CSV, spec=spec))
    wide = _to_wide_monthly(long_df)
    by_obs = dict(zip(wide["reference_month_end"], wide["release_date"]))
    assert by_obs[pd.Timestamp("2024-10-31")] == pd.Timestamp("2024-11-20")


def test_to_wide_daily_empty_input_returns_typed_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            "observation_date": pd.Series(dtype="datetime64[ns]"),
            "publication_date": pd.Series(dtype="datetime64[ns]"),
            "series_id": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    wide = _to_wide_daily(empty)
    assert wide.empty
    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"


def test_to_wide_monthly_empty_input_returns_typed_empty_frame() -> None:
    empty = pd.DataFrame(
        {
            "observation_date": pd.Series(dtype="datetime64[ns]"),
            "publication_date": pd.Series(dtype="datetime64[ns]"),
            "series_id": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
    wide = _to_wide_monthly(empty)
    assert wide.empty
    assert wide.columns[0] == "reference_month_end"
    assert wide.columns[-1] == "release_date"
