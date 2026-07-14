"""Tests for ``rba.data.sources.bbsw_rates``.

The live ``fetch()`` path hits the RBA endpoints (live F1 CSV +
``f01dhist.xls`` historical archive) and is not exercised here.
``_parse_csv``, ``_extract_series`` (shared by the XLS path),
``_attach_publication_dates``, and ``_to_wide`` are tested in isolation
against synthetic inputs mirroring the real F1 layouts (Title /
Description / Frequency / Type / Units / blanks / Source / Publication
date / Series ID metadata rows followed by daily data rows).

The XLS-bytes round-trip (``_parse_xls``) is a thin wrapper around
``pd.read_excel(... engine='xlrd')`` and is not unit-tested here — the
underlying decoder is covered by pandas' own test suite, and the
post-decoding column-extraction logic is shared with ``_parse_csv`` via
``_extract_series`` and is fully exercised below (including the
datetime-input path that mirrors XLS-decoded date columns).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from rba.data.sources.bbsw_rates import (
    _BBSW_COVERAGE_START,
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    BbswSeries,
    _attach_publication_dates,
    _extract_series,
    _parse_csv,
    _snapshot_suffix,
    _to_wide,
)

# Synthetic F1 CSV mirroring the live layout. Three BBSW tenors plus a
# decoy column (Cash Rate Target) to verify Series-ID-row column lookup
# works against a sibling column. cp1252-encodable; DD-MMM-YYYY dates.
_FIXTURE_CSV = (
    b"F1 INTEREST RATES AND YIELDS \x96 MONEY MARKET\n"
    b"Title,Cash Rate Target,EOD 1-month BABs/NCDs,EOD 3-month BABs/NCDs,EOD 6-month BABs/NCDs\n"
    b"Description,Cash Rate Target on date,Bank Accepted Bills/Negotiable Certificates of Deposit-1 month,"
    b"Bank Accepted Bills/Negotiable Certificates of Deposit-3 months,"
    b"Bank Accepted Bills/Negotiable Certificates of Deposit-6 months\n"
    b"Frequency,Daily,Daily,Daily,Daily\n"
    b"Type,Original,Original,Original,Original\n"
    b"Units,Per cent,Per cent,Per cent,Per cent\n"
    b"\n"
    b"\n"
    b"Source,RBA,ASX,ASX,ASX\n"
    b"Publication date,26-May-2026,26-May-2026,26-May-2026,26-May-2026\n"
    b"Series ID,FIRMMCRTD,FIRMMBAB30D,FIRMMBAB90D,FIRMMBAB180D\n"
    # Live-CSV vintage start (2011-01-04 in real F1).
    b"04-Jan-2011,4.75,4.83,4.97,5.14\n"
    # Mid-history anchor across an Easter sequence.
    # Thu 02-Apr-2026 = pre-Good-Friday (Good Fri 03-Apr-2026).
    # Next AU business day = Tue 07-Apr-2026 (Mon 06-Apr = Easter Mon).
    b"02-Apr-2026,4.35,4.30,4.43,4.75\n"
    # Late-history anchor; verifies a regular weekday +1 BDay = Mon -> Tue.
    b"22-May-2026,4.35,4.30,4.43,4.75\n"
    b"25-May-2026,4.35,4.30,4.43,4.73\n"
    # Trailing row where the BBSW columns are empty (live CSV
    # routinely lags BBSW vs cash by 1 row at the head — only Cash
    # Rate Target / Total Return Index are populated for the latest
    # session). Must be dropped for every BBSW tenor.
    b"26-May-2026,4.35,,,\n"
)


# -----------------------------------------------------------------------------
# _parse_csv — CSV decoding path
# -----------------------------------------------------------------------------


def test_parse_csv_extracts_1m() -> None:
    spec = BbswSeries(series_id="bbsw_1m", rba_series_id="FIRMMBAB30D", tenor_months=1)
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    assert len(df) == 4  # trailing 26-May-2026 row has empty BBSW cells -> dropped


def test_parse_csv_extracts_3m() -> None:
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert df["value"].iloc[0] == pytest.approx(4.97)


def test_parse_csv_extracts_6m() -> None:
    spec = BbswSeries(series_id="bbsw_6m", rba_series_id="FIRMMBAB180D", tenor_months=6)
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert df["value"].iloc[0] == pytest.approx(5.14)


def test_parse_csv_stamps_logical_series_id() -> None:
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "bbsw_3m").all()


def test_parse_csv_drops_rows_where_bbsw_cell_empty() -> None:
    """The 26-May-2026 row has cash-rate populated but BBSW empty — the
    last live-CSV refresh routinely shows this pattern. Must be excluded
    from every BBSW tenor's output."""
    for rba_id in ["FIRMMBAB30D", "FIRMMBAB90D", "FIRMMBAB180D"]:
        spec = BbswSeries(series_id="x", rba_series_id=rba_id, tenor_months=1)
        df = _parse_csv(_FIXTURE_CSV, spec=spec)
        assert pd.Timestamp("2026-05-26") not in df["observation_date"].tolist(), (
            f"Empty-cell row leaked through for {rba_id}"
        )


def test_parse_csv_raises_on_missing_rba_series_id() -> None:
    spec = BbswSeries(series_id="bogus", rba_series_id="DOES_NOT_EXIST", tenor_months=1)
    with pytest.raises(ValueError, match="not found among"):
        _parse_csv(_FIXTURE_CSV, spec=spec)


def test_parse_csv_missing_ok_returns_empty_frame() -> None:
    """Defensive path: when the spec's column is absent and missing_ok=True
    is passed, return an empty correctly-typed frame rather than raising."""
    spec = BbswSeries(series_id="bogus", rba_series_id="DOES_NOT_EXIST", tenor_months=1)
    df = _parse_csv(_FIXTURE_CSV, spec=spec, missing_ok=True)
    assert df.empty
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_csv_correct_values_for_anchors() -> None:
    spec_3m = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    df = _parse_csv(_FIXTURE_CSV, spec=spec_3m)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2011-01-04")] == pytest.approx(4.97)
    assert by_date[pd.Timestamp("2026-04-02")] == pytest.approx(4.43)
    assert by_date[pd.Timestamp("2026-05-25")] == pytest.approx(4.43)


def test_parse_csv_handles_cp1252_byte_in_title_row() -> None:
    """The live F1 title row carries an en-dash (byte 0x96) in 'INTEREST
    RATES AND YIELDS – MONEY MARKET' that breaks UTF-8 decoding. Parsing
    must succeed via the cp1252 path baked into _parse_csv."""
    assert b"\x96" in _FIXTURE_CSV  # fixture sanity
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert not df.empty


# -----------------------------------------------------------------------------
# _extract_series — shared decoder path (covers both CSV strings and XLS datetimes)
# -----------------------------------------------------------------------------


def _build_xls_style_raw_df() -> pd.DataFrame:
    """Construct a DataFrame matching what ``pd.read_excel`` produces for
    a historical f01dhist.xls — Title column carries metadata labels and
    ``datetime`` objects (not DD-MMM-YYYY strings), other columns are
    per-tenor values. Notably the historical Source for BBSW is AFMA
    (pre-2017 administrator), not ASX as in the live CSV."""
    return pd.DataFrame(
        {
            "Title": [
                "Description",
                "Frequency",
                "Type",
                "Units",
                None,
                None,
                "Source",
                "Publication date",
                "Series ID",
                datetime(1995, 1, 3),
                datetime(1995, 1, 4),
                datetime(1995, 1, 5),
            ],
            "1-month BABs/NCDs": [
                "Bank Accepted Bills/Negotiable Certificates of Deposit-1 month",
                "Daily",
                "Original",
                "Per cent",
                None,
                None,
                "AFMA",
                datetime(2016, 5, 9),
                "FIRMMBAB30D",
                7.50,
                7.52,
                7.55,
            ],
            "3-month BABs/NCDs": [
                "Bank Accepted Bills/Negotiable Certificates of Deposit-3 months",
                "Daily",
                "Original",
                "Per cent",
                None,
                None,
                "AFMA",
                datetime(2016, 5, 9),
                "FIRMMBAB90D",
                7.75,
                7.78,
                7.80,
            ],
            # No 6-month column — verifies the missing_ok=True path used
            # historically for series with later coverage starts (e.g. EUR
            # in the F11.1 splice).
        }
    )


def test_extract_series_handles_datetime_date_column() -> None:
    """XLS files arrive with the Title column carrying ``datetime`` objects
    rather than DD-MMM-YYYY strings. ``_extract_series`` must parse these
    via the fallback ``pd.to_datetime`` (no format string) path."""
    raw_df = _build_xls_style_raw_df()
    spec = BbswSeries(series_id="bbsw_1m", rba_series_id="FIRMMBAB30D", tenor_months=1)
    df = _extract_series(raw_df, spec=spec, missing_ok=False)
    assert len(df) == 3
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["observation_date"].iloc[0] == pd.Timestamp("1995-01-03")
    assert df["value"].iloc[0] == pytest.approx(7.50)


def test_extract_series_missing_column_missing_ok() -> None:
    """When a series is absent from a historical XLS, ``missing_ok=True``
    must return an empty correctly-typed frame."""
    raw_df = _build_xls_style_raw_df()
    spec_6m = BbswSeries(series_id="bbsw_6m", rba_series_id="FIRMMBAB180D", tenor_months=6)
    df = _extract_series(raw_df, spec=spec_6m, missing_ok=True)
    assert df.empty
    assert list(df.columns) == ["observation_date", "series_id", "value"]


def test_extract_series_missing_column_strict_raises() -> None:
    raw_df = _build_xls_style_raw_df()
    spec_6m = BbswSeries(series_id="bbsw_6m", rba_series_id="FIRMMBAB180D", tenor_months=6)
    with pytest.raises(ValueError, match="not found among"):
        _extract_series(raw_df, spec=spec_6m, missing_ok=False)


def test_extract_series_raises_when_series_id_row_missing() -> None:
    raw_df = pd.DataFrame(
        {
            "Title": ["Description", datetime(1995, 1, 3)],
            "3-month BABs/NCDs": ["BBSW-3m", 7.75],
        }
    )
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _extract_series(raw_df, spec=spec, missing_ok=False)


# -----------------------------------------------------------------------------
# _attach_publication_dates — reuses agb_yields_release_calendar (+1 BDay)
# -----------------------------------------------------------------------------


def test_attach_publication_dates_is_next_business_day() -> None:
    """F1 publishes the prior session's fixings next AU business day at
    ~11:30am AEST — identical rule to F2/AGB yields (verified via 10
    Wayback samples; see source module docstring)."""
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)

    by_obs = dict(zip(out["observation_date"], out["publication_date"]))
    # Tue 04-Jan-2011 -> Wed 05-Jan-2011 (vanilla +1 BDay)
    assert by_obs[pd.Timestamp("2011-01-04")] == pd.Timestamp("2011-01-05")
    # Fri 22-May-2026 -> Mon 25-May-2026 (weekend skip)
    assert by_obs[pd.Timestamp("2026-05-22")] == pd.Timestamp("2026-05-25")
    # Mon 25-May-2026 -> Tue 26-May-2026
    assert by_obs[pd.Timestamp("2026-05-25")] == pd.Timestamp("2026-05-26")


def test_attach_publication_dates_skips_easter_holidays() -> None:
    """Thu 02-Apr-2026 is the trading day before Good Friday 03-Apr-2026.
    Mon 06-Apr-2026 = Easter Monday. Next AU business day = Tue
    07-Apr-2026 — verifies the holiday set imported via
    agb_yields_release_calendar handles Easter correctly."""
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    by_obs = dict(zip(out["observation_date"], out["publication_date"]))
    assert by_obs[pd.Timestamp("2026-04-02")] == pd.Timestamp("2026-04-07")


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before merging — otherwise stale values would leak."""
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3)
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


def test_attach_publication_dates_empty_frame_no_raise() -> None:
    """An empty input frame must round-trip without raising."""
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


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_strictly_after_observation() -> None:
    """Under the +1 AU business day rule (same as F2) every
    publication_date must sit strictly after its observation_date — by
    at least one calendar day, never before, never equal."""
    frames = []
    for rba_id in ["FIRMMBAB30D", "FIRMMBAB90D", "FIRMMBAB180D"]:
        spec = BbswSeries(series_id=f"x_{rba_id}", rba_series_id=rba_id, tenor_months=1)
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(long_df)

    has_pub = out["publication_date"].notna()
    assert has_pub.all()
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days >= 1).all()
    # Largest realistic lag = Thu/Fri before Easter where Good Fri+Mon+Anzac/etc
    # collide. Bounded loose at 7 calendar days.
    assert (lag_days <= 7).all()


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema() -> None:
    frames = []
    for rba_id, sid in [
        ("FIRMMBAB30D", "bbsw_1m"),
        ("FIRMMBAB90D", "bbsw_3m"),
        ("FIRMMBAB180D", "bbsw_6m"),
    ]:
        spec = BbswSeries(series_id=sid, rba_series_id=rba_id, tenor_months=1)
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    for sid in ["bbsw_1m", "bbsw_3m", "bbsw_6m"]:
        assert sid in wide.columns


def test_to_wide_release_date_is_next_business_day() -> None:
    """The release_date column should equal the next-AU-business-day
    publication date stamped by the calendar merge."""
    frames = []
    for rba_id, sid in [
        ("FIRMMBAB30D", "bbsw_1m"),
        ("FIRMMBAB90D", "bbsw_3m"),
    ]:
        spec = BbswSeries(series_id=sid, rba_series_id=rba_id, tenor_months=1)
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    by_obs = dict(zip(wide["trade_date"], wide["release_date"]))
    assert by_obs[pd.Timestamp("2026-05-22")] == pd.Timestamp("2026-05-25")
    assert by_obs[pd.Timestamp("2026-05-25")] == pd.Timestamp("2026-05-26")


def test_to_wide_pivot_aligns_tenors_on_trade_date() -> None:
    """All 3 tenors share the same trade dates in the fixture (apart from
    the empty trailing row) — every populated row should have non-null
    values for all 3 BBSW columns."""
    frames = []
    for rba_id, sid in [
        ("FIRMMBAB30D", "bbsw_1m"),
        ("FIRMMBAB90D", "bbsw_3m"),
        ("FIRMMBAB180D", "bbsw_6m"),
    ]:
        spec = BbswSeries(series_id=sid, rba_series_id=rba_id, tenor_months=1)
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    tenor_cols = [c for c in wide.columns if c.startswith("bbsw_")]
    for _, row in wide.iterrows():
        assert row[tenor_cols].notna().all(), (
            f"Unexpected NaN in fixture row at trade_date={row['trade_date']}"
        )


# -----------------------------------------------------------------------------
# _snapshot_suffix — filename routing for the dated raw cache
# -----------------------------------------------------------------------------


def test_snapshot_suffix_csv() -> None:
    stem, suffix = _snapshot_suffix("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv")
    assert stem == "f1"
    assert suffix == ".csv"


def test_snapshot_suffix_xls() -> None:
    stem, suffix = _snapshot_suffix(
        "https://www.rba.gov.au/statistics/tables/xls-hist/f01dhist.xls"
    )
    assert stem == "f01dhist"
    assert suffix == ".xls"


def test_snapshot_suffix_rejects_unexpected_basename() -> None:
    with pytest.raises(ValueError, match="Unexpected URL basename"):
        _snapshot_suffix("https://www.rba.gov.au/statistics/tables/something.zip")


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_three_entries() -> None:
    """F1 carries 1m / 3m / 6m BBSW only — no 2m / 4m / 5m exist."""
    assert len(SERIES) == 3


def test_series_registry_covers_expected_tenors() -> None:
    tenors = {s.tenor_months for s in SERIES}
    assert tenors == {1, 3, 6}


def test_series_registry_rba_codes_unique() -> None:
    rba_codes = [s.rba_series_id for s in SERIES]
    assert len(rba_codes) == len(set(rba_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_rba_codes() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["bbsw_1m"] == "FIRMMBAB30D"
    assert by_id["bbsw_3m"] == "FIRMMBAB90D"
    assert by_id["bbsw_6m"] == "FIRMMBAB180D"


def test_series_registry_logical_ids_follow_naming_convention() -> None:
    for s in SERIES:
        assert s.series_id.startswith("bbsw_")


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")


def test_coverage_start_predates_floor() -> None:
    """The spliced (xls-hist + live) coverage start (3m BAB 1976-04-07)
    sits before the validation floor — rows pre-1993 are emitted by
    fetch() but the calendar validator only enforces matches from
    1993-01-01 onwards."""
    assert _BBSW_COVERAGE_START < _CALENDAR_VALIDATION_FLOOR
