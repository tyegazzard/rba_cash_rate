"""Tests for ``rba.data.sources.agb_yields``.

The live ``fetch()`` path hits the RBA endpoint and is not exercised
here. ``_parse``, ``_attach_publication_dates``, and ``_to_wide`` are
tested in isolation against a synthetic CSV mirroring the real F2
layout (Title / Description / Frequency / Type / Units / blanks /
Source / Publication date / Series ID metadata rows followed by
DD-MMM-YYYY data rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.agb_yields import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    AgbYieldSeries,
    _attach_publication_dates,
    _parse,
    _to_wide,
)

# Synthetic F2 CSV mirroring the live layout, including the en-dash
# in the table title (encoded as cp1252 byte 0x96) to exercise the
# encoding path. Date format is DD-MMM-YYYY (not DD/MM/YYYY like I2).
# Series subset matches the SERIES registry: 2y, 3y, 5y, 10y.
_FIXTURE_CSV = (
    b"F2 CAPITAL MARKET YIELDS \x96 GOVERNMENT BONDS\n"
    b"Title,Australian Government 2 year bond,Australian Government 3 year bond,"
    b"Australian Government 5 year bond,Australian Government 10 year bond\n"
    b"Description,2y desc,3y desc,5y desc,10y desc\n"
    b"Frequency,Daily,Daily,Daily,Daily\n"
    b"Type,Original,Original,Original,Original\n"
    b"Units,Per cent per annum,Per cent per annum,Per cent per annum,Per cent per annum\n"
    b"\n"
    b"\n"
    b"Source,RBA,RBA,RBA,RBA\n"
    b"Publication date,22-May-2026,22-May-2026,22-May-2026,22-May-2026\n"
    b"Series ID,FCMYGBAG2D,FCMYGBAG3D,FCMYGBAG5D,FCMYGBAG10D\n"
    # First date — 10y only (mirrors real F2 where 10y starts 2013-05-20
    # but 2y/3y/5y start 2013-09-02).
    b"20-May-2013,,,,3.229\n"
    b"21-May-2013,,,,3.263\n"
    # 2y/3y/5y join 2013-09-02 (Mon — verifies BDay calendar handles weekdays correctly).
    b"02-Sep-2013,2.65,2.85,3.10,3.85\n"
    # Mid-history anchor.
    b"01-Feb-2024,3.95,3.85,3.90,4.10\n"
    # Pre-Easter row to exercise holiday-skip in pub date calendar.
    b"02-Apr-2026,3.50,3.55,3.65,3.95\n"
    # Latest row from live snapshot (2026-05-20 Wed) — pub should be Thu 21-May-2026.
    b"20-May-2026,3.40,3.50,3.65,4.00\n"
    # Trailing empty row — should be dropped for every series.
    b"21-May-2026,,,,\n"
)


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_extracts_10y() -> None:
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 6 populated 10y rows (trailing 2026-05-21 dropped).
    assert len(df) == 6


def test_parse_extracts_2y_skips_pre_2013_09() -> None:
    """2y series joins F2 on 2013-09-02 — the May-2013 rows are blank for
    2y/3y/5y in the fixture and must be dropped."""
    spec = AgbYieldSeries(series_id="agb_yield_2y", rba_series_id="FCMYGBAG2D", maturity_years=2)
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("2013-05-20") not in df["observation_date"].tolist()
    assert pd.Timestamp("2013-09-02") in df["observation_date"].tolist()
    assert len(df) == 4  # 2013-09-02, 2024-02-01, 2026-04-02, 2026-05-20


def test_parse_stamps_logical_series_id() -> None:
    spec = AgbYieldSeries(series_id="agb_yield_3y", rba_series_id="FCMYGBAG3D", maturity_years=3)
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "agb_yield_3y").all()


def test_parse_drops_trailing_empty_row() -> None:
    """The 21-May-2026 row is entirely empty across all 4 series and must
    be excluded everywhere."""
    for rba_id in ["FCMYGBAG2D", "FCMYGBAG3D", "FCMYGBAG5D", "FCMYGBAG10D"]:
        spec = AgbYieldSeries(series_id="x", rba_series_id=rba_id, maturity_years=0)
        df = _parse(_FIXTURE_CSV, spec=spec)
        assert pd.Timestamp("2026-05-21") not in df["observation_date"].tolist(), (
            f"Empty-cell row leaked through for {rba_id}"
        )


def test_parse_decodes_cp1252_title() -> None:
    """The table-title row contains an en-dash (cp1252 byte 0x96); parsing
    must not raise a UnicodeDecodeError."""
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    # Should not raise.
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert not df.empty


def test_parse_raises_on_missing_rba_series_id() -> None:
    spec = AgbYieldSeries(series_id="bogus", rba_series_id="DOES_NOT_EXIST", maturity_years=0)
    with pytest.raises(ValueError, match="not found among"):
        _parse(_FIXTURE_CSV, spec=spec)


def test_parse_raises_on_empty_series() -> None:
    csv = (
        b"F2 TEST\n"
        b"Title,Series Z\n"
        b"Description,\n"
        b"Frequency,Daily\n"
        b"Type,Original\n"
        b"Units,Per cent per annum\n"
        b"\n"
        b"\n"
        b"Source,RBA\n"
        b"Publication date,01-May-2026\n"
        b"Series ID,FCMYGBAGZ\n"
        b"01-Apr-2026,\n"
        b"02-Apr-2026,\n"
    )
    spec = AgbYieldSeries(series_id="zero", rba_series_id="FCMYGBAGZ", maturity_years=0)
    with pytest.raises(ValueError, match="zero observations"):
        _parse(csv, spec=spec)


def test_parse_raises_when_series_id_row_missing() -> None:
    csv = b"F2 TEST\nTitle,Series Q\n01-Apr-2026,1.0\n"
    spec = AgbYieldSeries(series_id="q", rba_series_id="FCMYGBAGQ", maturity_years=0)
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse(csv, spec=spec)


def test_parse_correct_values_for_anchors() -> None:
    spec_10y = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    df = _parse(_FIXTURE_CSV, spec=spec_10y)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2013-05-20")] == pytest.approx(3.229)
    assert by_date[pd.Timestamp("2024-02-01")] == pytest.approx(4.10)
    assert by_date[pd.Timestamp("2026-05-20")] == pytest.approx(4.00)


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_next_bday() -> None:
    """20-May-2026 (Wed) → 21-May-2026 (Thu)."""
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2026-05-20")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-05-21")


def test_attach_publication_dates_handles_holiday_skip() -> None:
    """02-Apr-2026 (Thu, day before Good Fri Apr 3) → 07-Apr-2026 (Tue)
    skipping Good Fri, Sat/Sun, Easter Mon."""
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2026-04-02")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-04-07")


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before merging — otherwise stale values would leak."""
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    parsed = _parse(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = AgbYieldSeries(
        series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10
    )
    parsed = _parse(_FIXTURE_CSV, spec=spec)
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
    """Every publication_date must be strictly after the corresponding
    observation_date. F2 publishes the next morning, never same-day."""
    frames = []
    for rba_id in ["FCMYGBAG2D", "FCMYGBAG3D", "FCMYGBAG5D", "FCMYGBAG10D"]:
        spec = AgbYieldSeries(series_id=f"x_{rba_id}", rba_series_id=rba_id, maturity_years=0)
        frames.append(_parse(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(long_df)

    has_pub = out["publication_date"].notna()
    assert has_pub.all(), "Every fixture observation is post-floor and should have a pub date"
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days >= 1).all(), (
        "Publication dates must be strictly after observation date; "
        f"observed minimum lag: {lag_days.min()} days"
    )
    assert (lag_days <= 7).all(), f"Unexpectedly large pub-date lag: {lag_days.max()} days"


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema() -> None:
    frames = []
    for rba_id, sid in [
        ("FCMYGBAG2D", "agb_yield_2y"),
        ("FCMYGBAG3D", "agb_yield_3y"),
        ("FCMYGBAG5D", "agb_yield_5y"),
        ("FCMYGBAG10D", "agb_yield_10y"),
    ]:
        spec = AgbYieldSeries(series_id=sid, rba_series_id=rba_id, maturity_years=0)
        frames.append(_parse(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    for sid in ["agb_yield_2y", "agb_yield_3y", "agb_yield_5y", "agb_yield_10y"]:
        assert sid in wide.columns


def test_to_wide_release_date_matches_calendar() -> None:
    """For 02-Apr-2026 the release_date column should match the
    holiday-skipped publication date (Tue 07-Apr-2026)."""
    frames = []
    for rba_id, sid in [
        ("FCMYGBAG10D", "agb_yield_10y"),
        ("FCMYGBAG2D", "agb_yield_2y"),
    ]:
        spec = AgbYieldSeries(series_id=sid, rba_series_id=rba_id, maturity_years=0)
        frames.append(_parse(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    row = wide.loc[wide["trade_date"] == pd.Timestamp("2026-04-02")].iloc[0]
    assert row["release_date"] == pd.Timestamp("2026-04-07")


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_four_entries() -> None:
    assert len(SERIES) == 4


def test_series_registry_covers_expected_maturities() -> None:
    maturities = {s.maturity_years for s in SERIES}
    assert maturities == {2, 3, 5, 10}


def test_series_registry_rba_codes_unique() -> None:
    rba_codes = [s.rba_series_id for s in SERIES]
    assert len(rba_codes) == len(set(rba_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_rba_codes() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["agb_yield_2y"] == "FCMYGBAG2D"
    assert by_id["agb_yield_3y"] == "FCMYGBAG3D"
    assert by_id["agb_yield_5y"] == "FCMYGBAG5D"
    assert by_id["agb_yield_10y"] == "FCMYGBAG10D"


def test_series_registry_logical_ids_follow_naming_convention() -> None:
    for s in SERIES:
        assert s.series_id.startswith("agb_yield_")
        assert s.series_id.endswith("y")
        # The integer in agb_yield_{N}y must match maturity_years.
        suffix = s.series_id.removeprefix("agb_yield_").removesuffix("y")
        assert int(suffix) == s.maturity_years


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")
