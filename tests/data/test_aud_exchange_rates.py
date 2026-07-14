"""Tests for ``rba.data.sources.aud_exchange_rates``.

The live ``fetch()`` path hits the RBA endpoints (live CSV + 8 historical
XLS archives) and is not exercised here. ``_parse_csv``, ``_extract_series``
(shared by the XLS path), ``_attach_publication_dates``, and ``_to_wide``
are tested in isolation against synthetic inputs mirroring the real F11.1
layouts (Title / Description / Frequency / Type / Units / blanks /
Source / Publication date / Series ID metadata rows followed by daily
data rows).

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

from rba.data.sources.aud_exchange_rates import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    AudExchangeRateSeries,
    _attach_publication_dates,
    _extract_series,
    _parse_csv,
    _snapshot_suffix,
    _to_wide,
)

# Synthetic F11.1 CSV mirroring the live layout. SoMP-core series subset
# (USD / TWI / JPY / EUR / GBP / CNY / NZD) matches the SERIES registry.
# Date format is DD-MMM-YYYY. cp1252-encodable.
_FIXTURE_CSV = (
    b"F11.1  EXCHANGE RATES\n"
    b"Title,A$1=USD,Trade-weighted Index May 1970 = 100,A$1=JPY,A$1=EUR,"
    b"A$1=GBP,A$1=CNY,A$1=NZD\n"
    b"Description,AUD/USD,AUD TWI,AUD/JPY,AUD/EUR,AUD/GBP,AUD/CNY,AUD/NZD\n"
    b"Frequency,Daily,Daily,Daily,Daily,Daily,Daily,Daily\n"
    b"Type,Indicative,Indicative,Indicative,Indicative,Indicative,Indicative,Indicative\n"
    b"Units,USD,Index,JPY,EUR,GBP,CNY,NZD\n"
    b"\n"
    b"\n"
    b"Source,WM/Reuters,RBA,RBA,RBA,RBA,RBA,RBA\n"
    b"Publication date,22-May-2026,22-May-2026,22-May-2026,22-May-2026,22-May-2026,22-May-2026,22-May-2026\n"
    b"Series ID,FXRUSD,FXRTWI,FXRJY,FXREUR,FXRUKPS,FXRCR,FXRNZD\n"
    # First populated date in the current vintage.
    b"03-Jan-2023,0.6828,61.40,88.48,0.6400,0.5656,4.6994,1.0760\n"
    # Mid-history anchor.
    b"01-Feb-2024,0.6500,62.00,98.00,0.6000,0.5100,4.6800,1.0700\n"
    # Pre-Easter (Good Fri 03-Apr-2026); same-day rule means pub == obs.
    b"02-Apr-2026,0.6470,61.90,97.50,0.6020,0.5080,4.6800,1.0760\n"
    # Latest row from a recent live snapshot.
    b"20-May-2026,0.6500,62.10,97.20,0.6040,0.5100,4.6800,1.0750\n"
    # Trailing empty row — should be dropped for every series.
    b"21-May-2026,,,,,,,\n"
)


# -----------------------------------------------------------------------------
# _parse_csv — CSV decoding path
# -----------------------------------------------------------------------------


def test_parse_csv_extracts_usd() -> None:
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    assert len(df) == 4  # 4 populated rows; trailing empty 21-May-2026 dropped


def test_parse_csv_extracts_twi() -> None:
    spec = AudExchangeRateSeries(series_id="aud_twi", rba_series_id="FXRTWI", counterparty="TWI")
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert df["value"].iloc[0] == pytest.approx(61.40)


def test_parse_csv_stamps_logical_series_id() -> None:
    spec = AudExchangeRateSeries(series_id="aud_eur", rba_series_id="FXREUR", counterparty="EUR")
    df = _parse_csv(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "aud_eur").all()


def test_parse_csv_drops_trailing_empty_row() -> None:
    """The 21-May-2026 row is entirely empty across all 7 series and must
    be excluded everywhere."""
    for rba_id in ["FXRUSD", "FXRTWI", "FXRJY", "FXREUR", "FXRUKPS", "FXRCR", "FXRNZD"]:
        spec = AudExchangeRateSeries(series_id="x", rba_series_id=rba_id, counterparty="X")
        df = _parse_csv(_FIXTURE_CSV, spec=spec)
        assert pd.Timestamp("2026-05-21") not in df["observation_date"].tolist(), (
            f"Empty-cell row leaked through for {rba_id}"
        )


def test_parse_csv_raises_on_missing_rba_series_id() -> None:
    spec = AudExchangeRateSeries(
        series_id="bogus", rba_series_id="DOES_NOT_EXIST", counterparty="X"
    )
    with pytest.raises(ValueError, match="not found among"):
        _parse_csv(_FIXTURE_CSV, spec=spec)


def test_parse_csv_missing_ok_returns_empty_frame() -> None:
    """For historical XLS files some series will be absent (CNY pre-2014,
    EUR pre-1999). ``missing_ok=True`` must return an empty frame with
    the correct schema rather than raising."""
    spec = AudExchangeRateSeries(
        series_id="bogus", rba_series_id="DOES_NOT_EXIST", counterparty="X"
    )
    df = _parse_csv(_FIXTURE_CSV, spec=spec, missing_ok=True)
    assert df.empty
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_csv_correct_values_for_anchors() -> None:
    spec_usd = AudExchangeRateSeries(
        series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD"
    )
    df = _parse_csv(_FIXTURE_CSV, spec=spec_usd)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2023-01-03")] == pytest.approx(0.6828)
    assert by_date[pd.Timestamp("2024-02-01")] == pytest.approx(0.6500)
    assert by_date[pd.Timestamp("2026-05-20")] == pytest.approx(0.6500)


def test_parse_csv_handles_cp1252_byte_in_body() -> None:
    """Verifies the cp1252 encoding path. The live CSV occasionally carries
    a 0x96 en-dash in source-note text; parsing must not raise."""
    csv = _FIXTURE_CSV.replace(
        b"AUD/USD,AUD TWI",
        b"AUD/USD \x96 4pm Sydney,AUD TWI",
    )
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    df = _parse_csv(csv, spec=spec)
    assert not df.empty


# -----------------------------------------------------------------------------
# _extract_series — shared decoder path (covers both CSV strings and XLS datetimes)
# -----------------------------------------------------------------------------


def _build_xls_style_raw_df() -> pd.DataFrame:
    """Construct a DataFrame matching what ``pd.read_excel`` produces for
    a historical F11.1 XLS — Title column carries metadata labels and
    ``datetime`` objects (not DD-MMM-YYYY strings), other columns are
    per-counterparty values."""
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
                datetime(1993, 1, 4),
                datetime(1993, 1, 5),
                datetime(1993, 1, 6),
            ],
            "A$1=USD": [
                "AUD/USD",
                "Daily",
                "Indicative",
                "USD",
                None,
                None,
                "RBA",
                datetime(1994, 12, 30),
                "FXRUSD",
                0.6850,
                0.6862,
                0.6875,
            ],
            "Trade-weighted Index May 1970 = 100": [
                "AUD TWI",
                "Daily",
                "Indicative",
                "Index",
                None,
                None,
                "RBA",
                datetime(1994, 12, 30),
                "FXRTWI",
                52.3,
                52.4,
                52.5,
            ],
            # No CNY column — historical files pre-2014 omit it.
        }
    )


def test_extract_series_handles_datetime_date_column() -> None:
    """XLS files arrive with the Title column carrying ``datetime`` objects
    rather than DD-MMM-YYYY strings. ``_extract_series`` must parse these
    via the fallback ``pd.to_datetime`` (no format string) path."""
    raw_df = _build_xls_style_raw_df()
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    df = _extract_series(raw_df, spec=spec, missing_ok=False)
    assert len(df) == 3
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["observation_date"].iloc[0] == pd.Timestamp("1993-01-04")
    assert df["value"].iloc[0] == pytest.approx(0.6850)


def test_extract_series_missing_column_missing_ok() -> None:
    """When a series is absent from a historical XLS, ``missing_ok=True``
    must return an empty correctly-typed frame."""
    raw_df = _build_xls_style_raw_df()
    spec_cny = AudExchangeRateSeries(
        series_id="aud_cny", rba_series_id="FXRCR", counterparty="CNY"
    )
    df = _extract_series(raw_df, spec=spec_cny, missing_ok=True)
    assert df.empty
    assert list(df.columns) == ["observation_date", "series_id", "value"]


def test_extract_series_missing_column_strict_raises() -> None:
    raw_df = _build_xls_style_raw_df()
    spec_cny = AudExchangeRateSeries(
        series_id="aud_cny", rba_series_id="FXRCR", counterparty="CNY"
    )
    with pytest.raises(ValueError, match="not found among"):
        _extract_series(raw_df, spec=spec_cny, missing_ok=False)


def test_extract_series_raises_when_series_id_row_missing() -> None:
    raw_df = pd.DataFrame(
        {
            "Title": ["Description", datetime(1993, 1, 4)],
            "A$1=USD": ["AUD/USD", 0.6850],
        }
    )
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _extract_series(raw_df, spec=spec, missing_ok=False)


# -----------------------------------------------------------------------------
# _attach_publication_dates — same-day rule (no calendar module)
# -----------------------------------------------------------------------------


def test_attach_publication_dates_equals_observation_date() -> None:
    """F11.1 publishes EOD 4pm AEST same-day. publication_date == observation_date."""
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] == out["observation_date"]).all()


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before stamping — otherwise stale values would leak."""
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
    parsed = _parse_csv(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD")
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


def test_no_future_leakage_publication_not_after_observation() -> None:
    """Under the same-day rule (mirroring asx_ib_futures) every
    publication_date must equal its observation_date — never strictly
    after, never before."""
    frames = []
    for rba_id in ["FXRUSD", "FXRTWI", "FXRJY", "FXREUR", "FXRUKPS", "FXRCR", "FXRNZD"]:
        spec = AudExchangeRateSeries(
            series_id=f"x_{rba_id}", rba_series_id=rba_id, counterparty="X"
        )
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(long_df)

    has_pub = out["publication_date"].notna()
    assert has_pub.all()
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days == 0).all(), (
        f"Same-day publication expected; observed lag range "
        f"[{lag_days.min()}, {lag_days.max()}] days"
    )


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema() -> None:
    frames = []
    for rba_id, sid in [
        ("FXRUSD", "aud_usd"),
        ("FXRTWI", "aud_twi"),
        ("FXRJY", "aud_jpy"),
        ("FXREUR", "aud_eur"),
        ("FXRUKPS", "aud_gbp"),
        ("FXRCR", "aud_cny"),
        ("FXRNZD", "aud_nzd"),
    ]:
        spec = AudExchangeRateSeries(series_id=sid, rba_series_id=rba_id, counterparty="X")
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    assert wide.columns[0] == "trade_date"
    assert wide.columns[-1] == "release_date"
    for sid in ["aud_usd", "aud_twi", "aud_jpy", "aud_eur", "aud_gbp", "aud_cny", "aud_nzd"]:
        assert sid in wide.columns


def test_to_wide_release_date_equals_trade_date() -> None:
    """Under the same-day rule, the release_date column should equal the
    trade_date for every row."""
    frames = []
    for rba_id, sid in [
        ("FXRUSD", "aud_usd"),
        ("FXRTWI", "aud_twi"),
    ]:
        spec = AudExchangeRateSeries(series_id=sid, rba_series_id=rba_id, counterparty="X")
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)
    assert (wide["release_date"] == wide["trade_date"]).all()


def test_to_wide_pivot_aligns_currencies_on_trade_date() -> None:
    """All 7 currencies share the same set of trade dates in the fixture
    (apart from the trailing empty row) — every populated row should
    have non-null values for all SoMP currencies."""
    frames = []
    for rba_id, sid in [
        ("FXRUSD", "aud_usd"),
        ("FXRTWI", "aud_twi"),
        ("FXRJY", "aud_jpy"),
        ("FXREUR", "aud_eur"),
        ("FXRUKPS", "aud_gbp"),
        ("FXRCR", "aud_cny"),
        ("FXRNZD", "aud_nzd"),
    ]:
        spec = AudExchangeRateSeries(series_id=sid, rba_series_id=rba_id, counterparty="X")
        frames.append(_parse_csv(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    currency_cols = [c for c in wide.columns if c.startswith("aud_")]
    for _, row in wide.iterrows():
        assert row[currency_cols].notna().all(), (
            f"Unexpected NaN in fixture row at trade_date={row['trade_date']}"
        )


# -----------------------------------------------------------------------------
# _snapshot_suffix — filename routing for the dated raw cache
# -----------------------------------------------------------------------------


def test_snapshot_suffix_csv() -> None:
    stem, suffix = _snapshot_suffix("https://www.rba.gov.au/statistics/tables/csv/f11.1-data.csv")
    assert stem == "f11.1"
    assert suffix == ".csv"


def test_snapshot_suffix_xls() -> None:
    stem, suffix = _snapshot_suffix(
        "https://www.rba.gov.au/statistics/tables/xls-hist/1991-1994.xls"
    )
    assert stem == "1991-1994"
    assert suffix == ".xls"


def test_snapshot_suffix_rejects_unexpected_basename() -> None:
    with pytest.raises(ValueError, match="Unexpected URL basename"):
        _snapshot_suffix("https://www.rba.gov.au/statistics/tables/something.zip")


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_seven_entries() -> None:
    """SoMP-core counterparties: USD, TWI, JPY, EUR, GBP, CNY, NZD."""
    assert len(SERIES) == 7


def test_series_registry_covers_expected_counterparties() -> None:
    counterparties = {s.counterparty for s in SERIES}
    assert counterparties == {"USD", "TWI", "JPY", "EUR", "GBP", "CNY", "NZD"}


def test_series_registry_rba_codes_unique() -> None:
    rba_codes = [s.rba_series_id for s in SERIES]
    assert len(rba_codes) == len(set(rba_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_matches_known_rba_codes() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["aud_usd"] == "FXRUSD"
    assert by_id["aud_twi"] == "FXRTWI"
    assert by_id["aud_jpy"] == "FXRJY"
    assert by_id["aud_eur"] == "FXREUR"
    assert by_id["aud_gbp"] == "FXRUKPS"
    assert by_id["aud_cny"] == "FXRCR"
    assert by_id["aud_nzd"] == "FXRNZD"


def test_series_registry_logical_ids_follow_naming_convention() -> None:
    for s in SERIES:
        assert s.series_id.startswith("aud_")


def test_series_registry_excludes_frozen_currencies() -> None:
    """AED (FXRUAED) and ZAR (FXRSARD) are frozen in the live CSV at
    Publication date=01-Oct-2024 and must NOT be in SERIES — see module
    docstring and CONTEXT.md 'Excluded series' on stale-feature failure
    mode."""
    excluded_rba_ids = {"FXRUAED", "FXRSARD", "FXRSDR"}
    registered = {s.rba_series_id for s in SERIES}
    assert registered.isdisjoint(excluded_rba_ids)


def test_validation_floor_is_1993() -> None:
    """Match every other source module's floor."""
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")
