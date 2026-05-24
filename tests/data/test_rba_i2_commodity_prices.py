"""Tests for ``rba.data.sources.rba_i2_commodity_prices``.

The live ``fetch()`` path hits the RBA endpoint and is not exercised
here. ``_parse``, ``_attach_publication_dates``, and ``_to_wide`` are
tested in isolation against a synthetic CSV mirroring the real I2
layout (Title / Description / Frequency / Type / Units / Source /
Publication date / Series ID metadata rows followed by DD/MM/YYYY data
rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.rba_i2_commodity_prices import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    RbaI2Series,
    _attach_publication_dates,
    _parse,
    _to_month_end,
    _to_wide,
)

# Synthetic I2 CSV mirroring the live layout. Includes:
#   - one pre-1993 row (1992-12-31) to exercise the validation floor
#   - in-window rows that resolve via the algorithmic calendar
#   - one null cell to exercise drop-null behaviour
#   - one row where only the long-history series have data and the
#     post-2009 spot series are blank (mirrors 1990s coverage)
_FIXTURE_CSV = (
    "I2 COMMODITY PRICES\n"
    "Title,Commodity prices – A$,Commodity prices – SDR,Commodity prices – US$,"
    "Rural commodity prices – A$,Base metals prices – A$,Bulk commodities prices – A$,"
    "Commodity prices (with bulk commodities spot prices) – A$,Bulk commodities spot prices – A$\n"
    "Description,Foo,Bar,Baz,Rur,BM,BC,WSp,BCSp\n"
    "Frequency,Monthly,Monthly,Monthly,Monthly,Monthly,Monthly,Monthly,Monthly\n"
    "Type,Original,Original,Original,Original,Original,Original,Original,Original\n"
    'Units,"Index, 2024/25=100","Index, 2024/25=100","Index, 2024/25=100",'
    '"Index, 2024/25=100","Index, 2024/25=100","Index, 2024/25=100",'
    '"Index, 2024/25=100","Index, 2024/25=100"\n'
    "\n"
    "\n"
    "Source,RBA,RBA,RBA,RBA,RBA,RBA,RBA,RBA\n"
    "Publication date,01-May-2026,01-May-2026,01-May-2026,01-May-2026,"
    "01-May-2026,01-May-2026,01-May-2026,01-May-2026\n"
    "Series ID,GRCPAIAD,GRCPAISDR,GRCPAIUSD,GRCPRCAD,GRCPBMAD,GRCPBCAD,"
    "GRCPAISAD,GRCPBCSAD\n"
    # Pre-1993 row — should retain NaT publication_date.
    "31/12/1992,30.0,55.0,40.0,45.0,25.0,30.0,,\n"
    # 1995 row — long-history series only.
    "31/12/1995,40.0,60.0,45.0,50.0,30.0,35.0,,\n"
    # 2010 row — both long-history and spot series populated.
    "31/01/2010,80.0,90.0,85.0,75.0,70.0,90.0,82.0,95.0\n"
    # Current vintage anchor — should pub Mar 2 2026 (Sun Mar 1).
    "28/02/2026,99.5,97.6,98.3,93.4,98.2,103.0,99.7,103.2\n"
    # Latest row from the live CSV at the time of writing (Apr 2026).
    "30/04/2026,99.9,98.7,99.5,92.1,99.7,102.0,99.6,103.0\n"
    # Trailing null row — should be dropped.
    "31/05/2026,,,,,,,,\n"
).encode("utf-8")


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_extracts_headline_aud() -> None:
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 5 non-null rows (the trailing 31/05/2026 row has empty cells -> dropped).
    assert len(df) == 5


def test_parse_stamps_logical_series_id() -> None:
    spec = RbaI2Series(series_id="commodity_prices_rural_aud", rba_series_id="GRCPRCAD")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "commodity_prices_rural_aud").all()


def test_parse_drops_null_observations() -> None:
    """The 31/05/2026 row is entirely empty across our fixture's series
    and must be excluded for every series."""
    for rba_id in ["GRCPAIAD", "GRCPAISDR", "GRCPAIUSD", "GRCPRCAD", "GRCPBMAD"]:
        spec = RbaI2Series(series_id="x", rba_series_id=rba_id)
        df = _parse(_FIXTURE_CSV, spec=spec)
        assert pd.Timestamp("2026-05-31") not in df["observation_date"].tolist(), (
            f"Empty-cell row leaked through for {rba_id}"
        )


def test_parse_drops_per_series_null_observations() -> None:
    """For the post-2009 spot series, the pre-2009 rows are blank in the
    fixture and must be excluded — but the same rows must still appear
    for the long-history series."""
    spot = RbaI2Series(series_id="x", rba_series_id="GRCPBCSAD")
    spot_df = _parse(_FIXTURE_CSV, spec=spot)
    # 2010-01-31 and onward only.
    assert spot_df["observation_date"].min() == pd.Timestamp("2010-01-31")
    assert len(spot_df) == 3  # 2010-01, 2026-02, 2026-04


def test_parse_maps_to_month_end() -> None:
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert pd.Timestamp("1992-12-31") in df["observation_date"].tolist()
    assert pd.Timestamp("2026-04-30") in df["observation_date"].tolist()


def test_parse_raises_on_missing_rba_series_id() -> None:
    spec = RbaI2Series(series_id="bogus", rba_series_id="DOES_NOT_EXIST")
    with pytest.raises(ValueError, match="not found among"):
        _parse(_FIXTURE_CSV, spec=spec)


def test_parse_raises_on_empty_series() -> None:
    csv = (
        "I2 TEST\n"
        "Title,Series Z\n"
        "Description,\n"
        "Frequency,Monthly\n"
        "Type,Original\n"
        "Units,Index\n"
        "\n"
        "\n"
        "Source,RBA\n"
        "Publication date,01-May-2026\n"
        "Series ID,GRCPZ\n"
        "31/01/2024,\n"
        "29/02/2024,\n"
    ).encode("utf-8")
    spec = RbaI2Series(series_id="zero", rba_series_id="GRCPZ")
    with pytest.raises(ValueError, match="zero observations"):
        _parse(csv, spec=spec)


def test_parse_raises_when_series_id_row_missing() -> None:
    csv = b"I2 TEST\nTitle,Series Q\n31/01/2024,1.0\n"
    spec = RbaI2Series(series_id="q", rba_series_id="GRCPQ")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse(csv, spec=spec)


# -----------------------------------------------------------------------------
# _to_month_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ts", "expected"),
    [
        (pd.Timestamp("1992-12-31"), pd.Timestamp("1992-12-31")),
        (pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30")),
        (pd.Timestamp("2024-02-29"), pd.Timestamp("2024-02-29")),
        # Mid-month input snaps forward to end-of-month.
        (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-31")),
    ],
)
def test_to_month_end(input_ts: pd.Timestamp, expected: pd.Timestamp) -> None:
    out = _to_month_end(pd.Series([input_ts]))
    assert out.iloc[0] == expected
    assert out.dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_in_window_rows() -> None:
    """2026-04-30 should resolve to Fri May 1 2026 via the algorithmic
    first-business-day rule."""
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2026-04-30")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-05-01")


def test_attach_publication_dates_handles_sunday_first_of_month() -> None:
    """2026-02-28 ref → Mar 2026 first business day. Sun Mar 1 2026 → Mon Mar 2."""
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2026-02-28")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-03-02")


def test_attach_publication_dates_allows_nat_pre_1993() -> None:
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    pre = out[out["observation_date"] < _CALENDAR_VALIDATION_FLOOR]
    assert len(pre) > 0, "fixture should contain pre-1993 rows"
    assert pre["publication_date"].isna().all()


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before merging — otherwise stale values would leak."""
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


# -----------------------------------------------------------------------------
# No-leakage invariant (Invariant #1)
# -----------------------------------------------------------------------------


def test_no_future_leakage_release_date_exceeds_month_end() -> None:
    """Every publication_date must be strictly after the corresponding
    month-end. (The RBA cannot release a month's data before that month
    has ended.)"""
    frames = []
    # Use just three series from the fixture to keep the test focused.
    for rba_id in ["GRCPAIAD", "GRCPRCAD", "GRCPBMAD"]:
        spec = RbaI2Series(series_id=f"x_{rba_id}", rba_series_id=rba_id)
        frames.append(_parse(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(long_df)

    has_pub = out["publication_date"].notna()
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days >= 1).all(), (
        "Publication dates must be strictly after month-end; "
        f"observed minimum lag: {lag_days.min()} days"
    )


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema_includes_logical_series_present_in_fixture() -> None:
    frames = []
    for rba_id, sid in [
        ("GRCPAIAD", "commodity_prices_all_aud"),
        ("GRCPRCAD", "commodity_prices_rural_aud"),
        ("GRCPBMAD", "commodity_prices_base_metals_aud"),
    ]:
        spec = RbaI2Series(series_id=sid, rba_series_id=rba_id)
        frames.append(_parse(_FIXTURE_CSV, spec=spec))
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    # The fixed-position columns must come first and last.
    assert wide.columns[0] == "reference_month_end"
    assert wide.columns[1] == "reference_label"
    assert wide.columns[-1] == "release_date"
    # Every series we materialised must appear in the wide frame.
    for sid in ["commodity_prices_all_aud", "commodity_prices_rural_aud", "commodity_prices_base_metals_aud"]:
        assert sid in wide.columns


def test_to_wide_reference_label_format() -> None:
    spec = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    long_df = _attach_publication_dates(parsed)
    wide = _to_wide(long_df)
    label_2026_04 = wide.loc[
        wide["reference_month_end"] == pd.Timestamp("2026-04-30"), "reference_label"
    ].iloc[0]
    assert label_2026_04 == "Apr 2026"


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_has_21_entries() -> None:
    assert len(SERIES) == 21


def test_series_registry_covers_all_three_currency_bases() -> None:
    suffixes = {s.series_id.rsplit("_", 1)[-1] for s in SERIES}
    assert suffixes == {"aud", "sdr", "usd"}


def test_series_registry_rba_codes_unique() -> None:
    rba_codes = [s.rba_series_id for s in SERIES]
    assert len(rba_codes) == len(set(rba_codes))


def test_series_registry_logical_ids_unique() -> None:
    ids = [s.series_id for s in SERIES]
    assert len(ids) == len(set(ids))


def test_series_registry_includes_known_headline_codes() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["commodity_prices_all_aud"] == "GRCPAIAD"
    assert by_id["commodity_prices_all_sdr"] == "GRCPAISDR"
    assert by_id["commodity_prices_all_usd"] == "GRCPAIUSD"
    assert by_id["commodity_prices_bulk_spot_aud"] == "GRCPBCSAD"


# -----------------------------------------------------------------------------
# Known-value sanity (matches the synthetic fixture; would catch a
# column-order or value-cell parser regression).
# -----------------------------------------------------------------------------


def test_parse_extracts_correct_values_for_anchor_dates() -> None:
    spec_aud = RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD")
    df = _parse(_FIXTURE_CSV, spec=spec_aud)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2026-04-30")] == pytest.approx(99.9)
    assert by_date[pd.Timestamp("2010-01-31")] == pytest.approx(80.0)

    spec_bulk_spot = RbaI2Series(
        series_id="commodity_prices_bulk_spot_aud", rba_series_id="GRCPBCSAD"
    )
    df_bs = _parse(_FIXTURE_CSV, spec=spec_bulk_spot)
    by_date = dict(zip(df_bs["observation_date"], df_bs["value"]))
    assert by_date[pd.Timestamp("2010-01-31")] == pytest.approx(95.0)
    assert by_date[pd.Timestamp("2026-04-30")] == pytest.approx(103.0)
