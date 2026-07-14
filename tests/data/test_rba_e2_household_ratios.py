"""Tests for ``rba.data.sources.rba_e2_household_ratios``.

The live ``fetch()`` path hits the RBA endpoint and is not exercised
here. ``_parse``, ``_attach_publication_dates``, and ``_to_wide`` are
tested in isolation against a synthetic CSV mirroring the real E2
layout (Title / Description / Frequency / Type / Units / Source /
Publication date / Series ID metadata rows followed by DD/MM/YYYY data
rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.rba_e2_household_ratios import (
    _CALENDAR_VALIDATION_FLOOR,
    _FLAT_OFFSET_DAYS,
    SERIES,
    RbaE2Series,
    _attach_publication_dates,
    _parse,
    _to_quarter_end,
    _to_wide,
)

# Synthetic E2 CSV. Includes:
#  - one pre-1993 row (1992-Q1) to exercise the validation floor
#  - one in-window row that matches the materialised calendar (2025-12-31)
#  - one in-window row that misses the calendar (1995-12-31) to exercise the
#    flat-offset fallback
#  - one null cell to exercise drop-null behaviour
_FIXTURE_CSV = (
    "E2 HOUSEHOLD FINANCES – SELECTED RATIOS\n"
    "Title,Household debt to income,Housing debt to income,Owner-occupier housing debt to income\n"
    "Description,Foo,Bar,Baz\n"
    "Frequency,Quarterly,Quarterly,Quarterly\n"
    "Type,Original,Original,Seasonally adjusted\n"
    "Units,Per cent,Per cent,Per cent\n"
    "\n"
    "\n"
    "Source,ABS,ABS,RBA\n"
    "Publication date,27-Mar-2026,27-Mar-2026,27-Mar-2026\n"
    "Series ID,BHFDDIT,BHFDDIH,BHFDDIO\n"
    "31/03/1992,55.0,25.0,20.0\n"
    "31/12/1995,86.1,50.8,41.2\n"
    "31/12/2014,168.4,121.3,83.8\n"
    "31/12/2025,177.0,133.7,99.6\n"
    "31/03/2026,,,\n"
).encode("utf-8")


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_extracts_bhfddit() -> None:
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 4 non-null rows (the 31/03/2026 row has empty cell -> dropped).
    assert len(df) == 4


def test_parse_stamps_logical_series_id() -> None:
    spec = RbaE2Series(series_id="housing_debt_to_income", rba_series_id="BHFDDIH")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "housing_debt_to_income").all()


def test_parse_drops_null_observations() -> None:
    spec = RbaE2Series(series_id="owner_occupier_housing_debt_to_income", rba_series_id="BHFDDIO")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # The 31/03/2026 row has a null value and should be excluded.
    assert pd.Timestamp("2026-03-31") not in df["observation_date"].tolist()


def test_parse_maps_to_quarter_end() -> None:
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # 31/03/1992 -> end of Q1 1992 = 1992-03-31
    assert pd.Timestamp("1992-03-31") in df["observation_date"].tolist()
    assert pd.Timestamp("2025-12-31") in df["observation_date"].tolist()


def test_parse_raises_on_missing_rba_series_id() -> None:
    spec = RbaE2Series(series_id="bogus", rba_series_id="DOES_NOT_EXIST")
    with pytest.raises(ValueError, match="not found among"):
        _parse(_FIXTURE_CSV, spec=spec)


def test_parse_raises_on_empty_series() -> None:
    csv = (
        "E2 TEST\n"
        "Title,Series Z\n"
        "Description,\n"
        "Frequency,Quarterly\n"
        "Type,Original\n"
        "Units,Per cent\n"
        "\n"
        "\n"
        "Source,RBA\n"
        "Publication date,27-Mar-2026\n"
        "Series ID,BHFZ\n"
        "31/03/2024,\n"
        "30/06/2024,\n"
    ).encode("utf-8")
    spec = RbaE2Series(series_id="zero", rba_series_id="BHFZ")
    with pytest.raises(ValueError, match="zero observations"):
        _parse(csv, spec=spec)


def test_parse_raises_when_series_id_row_missing() -> None:
    csv = b"E2 TEST\nTitle,Series Q\n31/01/2024,1.0\n"
    spec = RbaE2Series(series_id="q", rba_series_id="BHFQ")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse(csv, spec=spec)


# -----------------------------------------------------------------------------
# _to_quarter_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ts", "expected"),
    [
        (pd.Timestamp("1992-03-31"), pd.Timestamp("1992-03-31")),
        (pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30")),
        (pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")),
        # Mid-quarter input should snap forward to end-of-quarter.
        (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-06-30")),
    ],
)
def test_to_quarter_end(input_ts: pd.Timestamp, expected: pd.Timestamp) -> None:
    out = _to_quarter_end(pd.Series([input_ts]))
    assert out.iloc[0] == expected
    assert out.dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_scraped_quarter() -> None:
    """2025-Q4 is in the materialised calendar with publication 2026-03-27."""
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2025-12-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-03-27")


def test_attach_publication_dates_uses_flat_offset_for_wayback_gap() -> None:
    """1995-Q4 is not in the calendar; falls back to + _FLAT_OFFSET_DAYS."""
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("1995-12-31")].iloc[0]
    expected = pd.Timestamp("1995-12-31") + pd.Timedelta(days=_FLAT_OFFSET_DAYS)
    assert row["publication_date"] == expected


def test_attach_publication_dates_allows_nat_pre_1993() -> None:
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    pre = out[out["observation_date"] < _CALENDAR_VALIDATION_FLOOR]
    assert len(pre) > 0, "fixture should contain pre-1993 rows"
    assert pre["publication_date"].isna().all()


def test_attach_publication_dates_drops_stale_column() -> None:
    """If the input frame already carries a publication_date column, it
    must be dropped before merging — otherwise stale values would leak."""
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    parsed = _parse(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
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


def test_no_future_leakage_release_date_exceeds_quarter_end() -> None:
    """Every publication_date must be strictly after the corresponding
    quarter-end. (The RBA cannot release a quarter's data before that
    quarter has ended.)"""
    frames = []
    for spec in SERIES:
        df = _parse(_FIXTURE_CSV, spec=spec)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(long_df)

    has_pub = out["publication_date"].notna()
    lag_days = (
        out.loc[has_pub, "publication_date"] - out.loc[has_pub, "observation_date"]
    ).dt.days
    assert (lag_days > 7).all(), (
        "Publication dates must exceed quarter-end by more than 7 days; "
        f"observed minimum lag: {lag_days.min()} days"
    )


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema_and_dtype() -> None:
    frames = []
    for spec in SERIES:
        df = _parse(_FIXTURE_CSV, spec=spec)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    assert list(wide.columns) == [
        "reference_quarter_end",
        "reference_label",
        "household_debt_to_income",
        "housing_debt_to_income",
        "owner_occupier_housing_debt_to_income",
        "release_date",
        "source",
    ]
    assert wide["reference_quarter_end"].dtype == "datetime64[ns]"
    assert wide["release_date"].dtype == "datetime64[ns]"


def test_to_wide_source_populated() -> None:
    frames = []
    for spec in SERIES:
        df = _parse(_FIXTURE_CSV, spec=spec)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    # 2014-Q4 is in the materialised calendar via an organic archive_org
    # Wayback snapshot (stable across scraper reruns — the RBA CSV's
    # in-place-update convention doesn't affect old Wayback captures). Chosen
    # over 2025-Q4 because 2025-Q4 is now filled via ``_OVERRIDES`` (source
    # ``inferred``) — the RBA replaced its live ``Publication date`` header
    # with 2026-Q1 before Wayback captured the interim window.
    src_2014q4 = wide.loc[
        wide["reference_quarter_end"] == pd.Timestamp("2014-12-31"), "source"
    ].iloc[0]
    assert src_2014q4 in {"rba_page", "archive_org"}

    # 1995-Q4 is NOT in the calendar -> source == "inferred" (flat-offset fallback).
    src_1995q4 = wide.loc[
        wide["reference_quarter_end"] == pd.Timestamp("1995-12-31"), "source"
    ].iloc[0]
    assert src_1995q4 == "inferred"


def test_to_wide_reference_label_format() -> None:
    frames = []
    for spec in SERIES:
        df = _parse(_FIXTURE_CSV, spec=spec)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    long_df = _attach_publication_dates(long_df)
    wide = _to_wide(long_df)

    label_2025q4 = wide.loc[
        wide["reference_quarter_end"] == pd.Timestamp("2025-12-31"), "reference_label"
    ].iloc[0]
    assert label_2025q4 == "Dec 2025"


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry_covers_three_ratios() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {
        "household_debt_to_income",
        "housing_debt_to_income",
        "owner_occupier_housing_debt_to_income",
    }


def test_series_registry_rba_codes_match_known_values() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["household_debt_to_income"] == "BHFDDIT"
    assert by_id["housing_debt_to_income"] == "BHFDDIH"
    assert by_id["owner_occupier_housing_debt_to_income"] == "BHFDDIO"


# -----------------------------------------------------------------------------
# Known-value sanity (hand-verified from the live E2 CSV on 2026-05-23)
# -----------------------------------------------------------------------------


def test_known_values_match_synthetic_fixture() -> None:
    """The synthetic fixture's values for 2025-Q4, 2014-Q4, 1995-Q4 are the
    actual RBA-published numbers. If the parser ever returns different
    values for those quarters it indicates the CSV layout or column
    ordering has shifted."""
    spec_total = RbaE2Series(series_id="household_debt_to_income", rba_series_id="BHFDDIT")
    df = _parse(_FIXTURE_CSV, spec=spec_total)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2025-12-31")] == pytest.approx(177.0)
    assert by_date[pd.Timestamp("2014-12-31")] == pytest.approx(168.4)
    assert by_date[pd.Timestamp("1995-12-31")] == pytest.approx(86.1)

    spec_oo = RbaE2Series(
        series_id="owner_occupier_housing_debt_to_income", rba_series_id="BHFDDIO"
    )
    df_oo = _parse(_FIXTURE_CSV, spec=spec_oo)
    by_date = dict(zip(df_oo["observation_date"], df_oo["value"]))
    assert by_date[pd.Timestamp("2025-12-31")] == pytest.approx(99.6)
