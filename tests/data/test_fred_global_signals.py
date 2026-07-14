"""Tests for ``rba.data.sources.fred_global_signals``.

The live ``fetch()`` path hits ``fredgraph.csv?id=<SERIES_ID>`` and is
not exercised here. ``_parse_csv``, ``_attach_publication_dates``,
``_to_wide_daily``, and ``_to_wide_monthly`` are tested in isolation
against synthetic inputs mirroring the real FRED CSV layouts:

- Monthly fixture (CPIAUCSL) — observation_date column at month-start,
  with the trailing month blank-valued (currently-undefined revision
  in progress).
- Daily fixture (DGS10) — every Mon-Fri populated except US-federal-
  holiday rows which are blank in the upstream CSV.

The frequency-based month-end conversion in ``_parse_csv`` is the
key novel behaviour for this source (compared to BBSW / AGB yields
which are pure daily) so the monthly fixture covers it explicitly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.fred_global_signals import (
    _CALENDAR_VALIDATION_FLOOR,
    _DAILY_SERIES_IDS,
    _MONTHLY_SERIES_IDS,
    SERIES,
    FredSeries,
    _attach_publication_dates,
    _parse_csv,
    _to_wide_daily,
    _to_wide_monthly,
)

# Synthetic monthly FRED CSV — CPIAUCSL layout. Two real values plus a
# trailing blank-valued row (CPI for the in-progress month not yet
# released by BLS — common pattern at the tail of the CSV).
_CPIAUCSL_FIXTURE_CSV = (
    b"observation_date,CPIAUCSL\n"
    b"1947-01-01,21.480\n"
    b"1947-02-01,21.620\n"
    b"2024-10-01,315.301\n"
    b"2024-11-01,316.449\n"
    b"2026-05-01,\n"  # trailing blank — must be dropped
)

# Synthetic daily FRED CSV — DGS10 layout. Includes:
# - a normal weekday row,
# - a US-federal-holiday blank row (Thanksgiving 2024 = 28-Nov-2024),
# - a weekend row that FRED never actually emits (sanity check: pandas
#   will drop it if present anyway).
_DGS10_FIXTURE_CSV = (
    b"observation_date,DGS10\n"
    b"2024-11-26,4.30\n"  # Tue
    b"2024-11-27,4.28\n"  # Wed
    b"2024-11-28,\n"  # Thu = Thanksgiving — blank in upstream
    b"2024-11-29,4.18\n"  # Fri
    b"2024-12-02,4.20\n"  # Mon
    b"2024-12-24,4.59\n"  # Tue — last full day before Christmas
)


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_six_entries() -> None:
    """Canonical 5 (CPI / FFR / 10y / DXY / VIX) plus core CPI = 6."""
    assert len(SERIES) == 6


def test_series_registry_fred_codes_unique() -> None:
    fred_codes = [s.fred_series_id for s in SERIES]
    assert len(fred_codes) == len(set(fred_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_fred_codes() -> None:
    by_id = {s.series_id: s.fred_series_id for s in SERIES}
    assert by_id["us_headline_cpi"] == "CPIAUCSL"
    assert by_id["us_core_cpi"] == "CPILFESL"
    assert by_id["us_fed_funds"] == "DFF"
    assert by_id["us_10y_treasury"] == "DGS10"
    assert by_id["us_dxy_broad"] == "DTWEXBGS"
    assert by_id["us_vix"] == "VIXCLS"


def test_series_registry_logical_ids_follow_naming_convention() -> None:
    for s in SERIES:
        assert s.series_id.startswith("us_")


def test_series_registry_frequencies_are_known() -> None:
    for s in SERIES:
        assert s.frequency in {"daily", "monthly"}


def test_daily_and_monthly_partition_covers_all_series() -> None:
    assert set(_DAILY_SERIES_IDS) | set(_MONTHLY_SERIES_IDS) == {s.series_id for s in SERIES}
    assert set(_DAILY_SERIES_IDS) & set(_MONTHLY_SERIES_IDS) == set()


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")


# -----------------------------------------------------------------------------
# _parse_csv — monthly path (CPIAUCSL)
# -----------------------------------------------------------------------------


def test_parse_csv_monthly_extracts_rows() -> None:
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    df = _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 4 real rows; trailing blank dropped.
    assert len(df) == 4


def test_parse_csv_monthly_converts_to_month_end() -> None:
    """FRED stores monthly observations at the first day of the month
    (BLS native); the parser must convert to month-end to match the
    project-wide schema convention."""
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    df = _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec)
    by_value = dict(zip(df["value"], df["observation_date"]))
    # 1947-01 ref → 1947-01-31 (31 days)
    assert by_value[21.480] == pd.Timestamp("1947-01-31")
    # 1947-02 ref → 1947-02-28 (non-leap)
    assert by_value[21.620] == pd.Timestamp("1947-02-28")
    # 2024-10 ref → 2024-10-31
    assert by_value[315.301] == pd.Timestamp("2024-10-31")
    # 2024-11 ref → 2024-11-30
    assert by_value[316.449] == pd.Timestamp("2024-11-30")


def test_parse_csv_monthly_stamps_logical_series_id() -> None:
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    df = _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "us_headline_cpi").all()


def test_parse_csv_monthly_drops_blank_value_rows() -> None:
    """Trailing reference months for which BLS hasn't released a value
    yet appear as blank cells in the FRED CSV and must be dropped."""
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    df = _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("2026-05-31") not in df["observation_date"].tolist()


# -----------------------------------------------------------------------------
# _parse_csv — daily path (DGS10)
# -----------------------------------------------------------------------------


def test_parse_csv_daily_preserves_trade_date() -> None:
    """Daily-series observation_date is the literal trade date, no
    period-end conversion."""
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    df = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec)
    dates = set(df["observation_date"].tolist())
    assert pd.Timestamp("2024-11-26") in dates
    assert pd.Timestamp("2024-11-27") in dates
    assert pd.Timestamp("2024-11-29") in dates
    assert pd.Timestamp("2024-12-02") in dates


def test_parse_csv_daily_drops_us_holiday_blank_rows() -> None:
    """Thanksgiving 2024 (Thu 28-Nov) is a blank cell in the upstream
    CSV — the parser must drop it (cannot leak as a 0.0 or NaN row)."""
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    df = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("2024-11-28") not in df["observation_date"].tolist()


def test_parse_csv_daily_value_accuracy() -> None:
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    df = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2024-11-26")] == pytest.approx(4.30)
    assert by_date[pd.Timestamp("2024-12-24")] == pytest.approx(4.59)


# -----------------------------------------------------------------------------
# _parse_csv — error paths
# -----------------------------------------------------------------------------


def test_parse_csv_raises_on_missing_observation_date_column() -> None:
    bad = b"DATE,CPIAUCSL\n2024-10-01,315.301\n"
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    with pytest.raises(ValueError, match="missing expected 'observation_date'"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_missing_value_column() -> None:
    bad = b"observation_date,SOMETHING_ELSE\n2024-10-01,315.301\n"
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    with pytest.raises(ValueError, match="missing the value column"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_all_blank_values() -> None:
    """If every cell in the value column is blank (CSV body present but
    no real data), surface a ValueError rather than returning an empty
    frame silently — same defensive posture as the AGB-yields source."""
    bad = b"observation_date,CPIAUCSL\n2024-10-01,\n2024-11-01,\n"
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    with pytest.raises(ValueError, match="Parsed zero observations"):
        _parse_csv(bad, spec=spec)


def test_parse_csv_raises_on_unknown_frequency() -> None:
    """A FredSeries with an unsupported frequency string must raise — a
    typo here would otherwise silently fall through to the daily path."""
    bad_spec = FredSeries(series_id="x", fred_series_id="CPIAUCSL", frequency="quarterly")
    with pytest.raises(ValueError, match="Unknown frequency"):
        _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=bad_spec)


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_monthly_cpi_plus_20_days() -> None:
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    parsed = _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    by_obs = dict(zip(out["observation_date"], out["publication_date"]))
    # 2024-10-31 → 2024-11-20 (+20 calendar days).
    assert by_obs[pd.Timestamp("2024-10-31")] == pd.Timestamp("2024-11-20")
    # 2024-11-30 → 2024-12-20.
    assert by_obs[pd.Timestamp("2024-11-30")] == pd.Timestamp("2024-12-20")


def test_attach_publication_dates_daily_is_next_us_business_day() -> None:
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    parsed = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec)
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
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    parsed = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec)
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
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    parsed = _parse_csv(_DGS10_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_at_or_after_observation() -> None:
    """For every parsed observation across daily and monthly series,
    publication_date must be at-or-after observation_date — never before.
    VIX is the only same-day rule; everything else is strictly after."""
    daily_spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    monthly_spec = FredSeries(
        series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly"
    )
    long_df = pd.concat(
        [
            _parse_csv(_DGS10_FIXTURE_CSV, spec=daily_spec),
            _parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=monthly_spec),
        ],
        ignore_index=True,
    )
    out = _attach_publication_dates(long_df)
    has_pub = out["publication_date"].notna()
    assert has_pub.all()
    assert (out["publication_date"] >= out["observation_date"]).all()


# -----------------------------------------------------------------------------
# _to_wide_daily / _to_wide_monthly
# -----------------------------------------------------------------------------


def test_to_wide_daily_schema_and_key() -> None:
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    long_df = _attach_publication_dates(_parse_csv(_DGS10_FIXTURE_CSV, spec=spec))
    wide = _to_wide_daily(long_df)
    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    assert "us_10y_treasury" in wide.columns
    # Daily wide must not carry any monthly columns.
    for monthly_sid in _MONTHLY_SERIES_IDS:
        assert monthly_sid not in wide.columns


def test_to_wide_monthly_schema_and_key() -> None:
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    long_df = _attach_publication_dates(_parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec))
    wide = _to_wide_monthly(long_df)
    assert wide.columns[0] == "reference_month_end"
    assert wide.columns[-1] == "release_date"
    assert "us_headline_cpi" in wide.columns
    for daily_sid in _DAILY_SERIES_IDS:
        assert daily_sid not in wide.columns


def test_to_wide_daily_release_date_is_next_us_business_day() -> None:
    spec = FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily")
    long_df = _attach_publication_dates(_parse_csv(_DGS10_FIXTURE_CSV, spec=spec))
    wide = _to_wide_daily(long_df)
    by_obs = dict(zip(wide["trade_date"], wide["release_date"]))
    # Fri 29-Nov-2024 → Mon 02-Dec-2024.
    assert by_obs[pd.Timestamp("2024-11-29")] == pd.Timestamp("2024-12-02")


def test_to_wide_monthly_release_date_is_plus_20_days() -> None:
    spec = FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly")
    long_df = _attach_publication_dates(_parse_csv(_CPIAUCSL_FIXTURE_CSV, spec=spec))
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
