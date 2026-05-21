"""Tests for ``rba.data.sources.rba_d`` CSV parser + publication-date attachment.

The download path (``fetch``) is not exercised here — those tests would hit
the live RBA endpoints. ``_parse`` and ``_attach_publication_dates`` are tested
in isolation against a hand-built CSV fixture that mirrors the real RBA D2
layout (Title / Description / Frequency / Type / Units / Source / Publication
date / Series ID metadata rows followed by DD/MM/YYYY observation rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.rba_d import (
    SERIES,
    RbaDSeries,
    _attach_publication_dates,
    _parse,
    _to_month_end,
)

# Synthetic CSV mirroring the real D2 layout: 8 metadata rows above the
# data rows, with the "Title" header at row index 1. Columns: 1 date
# column + 3 data columns. Includes a null-value cell, a legitimate
# numeric, and a pre-1993 row (1992-01) so we can exercise the validation
# floor in _attach_publication_dates.
_FIXTURE_CSV = (
    "D_FIXTURE TEST TABLE\n"
    "Title,Series A; SA,Series B; SA,Series C; SA\n"
    "Description,Test series A,Test series B,Test series C\n"
    "Frequency,Monthly,Monthly,Monthly\n"
    "Type,Seasonally adjusted,Seasonally adjusted,Seasonally adjusted\n"
    "Units,$ billion,$ billion,Per cent\n"
    "\n"
    "\n"
    "Source,RBA,RBA,RBA\n"
    "Publication date,30-Apr-2026,30-Apr-2026,30-Apr-2026\n"
    "Series ID,FIXA,FIXB,FIXC\n"
    "31/01/1992,100.0,50.0,2.5\n"
    "29/02/1992,101.0,50.5,2.6\n"
    "31/01/1993,110.0,55.0,3.0\n"
    "28/02/1993,111.0,,3.1\n"
    "31/03/2024,200.0,99.5,4.0\n"
).encode("utf-8")


def test_parse_extracts_series_a() -> None:
    spec = RbaDSeries(series_id="series_a", table="d_fixture", rba_series_id="FIXA")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 5 rows in the fixture, all with non-null FIXA values
    assert len(df) == 5


def test_parse_drops_null_observations() -> None:
    spec = RbaDSeries(series_id="series_b", table="d_fixture", rba_series_id="FIXB")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # Series B has a null value at 28/02/1993; should be dropped.
    assert len(df) == 4
    assert pd.Timestamp("1993-02-28") not in df["observation_date"].tolist()


def test_parse_stamps_logical_series_id() -> None:
    spec = RbaDSeries(series_id="series_c", table="d_fixture", rba_series_id="FIXC")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # The logical name (series_c), not the RBA ID (FIXC), is on every row.
    assert (df["series_id"] == "series_c").all()


def test_parse_maps_to_month_end() -> None:
    spec = RbaDSeries(series_id="series_a", table="d_fixture", rba_series_id="FIXA")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # 31/01/1992 → 1992-01-31
    assert pd.Timestamp("1992-01-31") in df["observation_date"].tolist()
    # 29/02/1992 (leap) → 1992-02-29
    assert pd.Timestamp("1992-02-29") in df["observation_date"].tolist()


def test_parse_raises_on_missing_rba_series_id() -> None:
    spec = RbaDSeries(
        series_id="bogus", table="d_fixture", rba_series_id="DOES_NOT_EXIST"
    )
    with pytest.raises(ValueError, match="not found among"):
        _parse(_FIXTURE_CSV, spec=spec)


def test_parse_raises_on_empty_series() -> None:
    # CSV where the target series has only null values.
    csv = (
        "D_FIXTURE TEST TABLE\n"
        "Title,Series Z\n"
        "Description,\n"
        "Frequency,Monthly\n"
        "Type,Seasonally adjusted\n"
        "Units,$ billion\n"
        "\n"
        "\n"
        "Source,RBA\n"
        "Publication date,30-Apr-2026\n"
        "Series ID,FIXZ\n"
        "31/01/2024,\n"
        "29/02/2024,\n"
    ).encode("utf-8")
    spec = RbaDSeries(series_id="series_z", table="d_fixture", rba_series_id="FIXZ")
    with pytest.raises(ValueError, match="zero observations"):
        _parse(csv, spec=spec)


def test_parse_raises_when_series_id_row_missing() -> None:
    # CSV without the "Series ID" metadata row.
    csv = (
        "TEST\n"
        "Title,Series Q\n"
        "31/01/2024,1.0\n"
    ).encode("utf-8")
    spec = RbaDSeries(series_id="series_q", table="d_fixture", rba_series_id="FIXQ")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse(csv, spec=spec)


# -----------------------------------------------------------------------------
# _to_month_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ts", "expected"),
    [
        (pd.Timestamp("1992-01-31"), pd.Timestamp("1992-01-31")),
        (pd.Timestamp("1992-02-29"), pd.Timestamp("1992-02-29")),  # leap
        (pd.Timestamp("2024-04-30"), pd.Timestamp("2024-04-30")),
        # Non-month-end inputs (defensive — RBA always publishes month-end
        # dates but the helper should be robust).
        (pd.Timestamp("2024-04-15"), pd.Timestamp("2024-04-30")),
    ],
)
def test_to_month_end(input_ts: pd.Timestamp, expected: pd.Timestamp) -> None:
    out = _to_month_end(pd.Series([input_ts]))
    assert out.iloc[0] == expected
    assert out.dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_recent() -> None:
    spec = RbaDSeries(series_id="series_a", table="d_fixture", rba_series_id="FIXA")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # 2024-03-31 → last weekday of Apr 2024 = Tue Apr 30 (Apr 2024 has
    # 30 days, ending on Tue) — no holiday collision because Easter 2024
    # was 31 March, and the override applies to Feb 2024 (not Mar 2024).
    row = out.loc[out["observation_date"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2024-04-30")


def test_attach_publication_dates_allows_nat_pre_1993() -> None:
    spec = RbaDSeries(series_id="series_a", table="d_fixture", rba_series_id="FIXA")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    pre_1993 = out[out["observation_date"] < pd.Timestamp("1993-01-01")]
    assert len(pre_1993) > 0, "fixture should contain pre-1993 observations"
    assert pre_1993["publication_date"].isna().all()


def test_attach_publication_dates_drops_stale_column() -> None:
    spec = RbaDSeries(series_id="series_a", table="d_fixture", rba_series_id="FIXA")
    parsed = _parse(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_raises_on_unmatched_post_1993() -> None:
    bad = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2020-05-15")],  # mid-month, no match
            "series_id": ["series_a"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing a publication date"):
        _attach_publication_dates(bad)


# -----------------------------------------------------------------------------
# SERIES registry shape
# -----------------------------------------------------------------------------


def test_series_registry_covers_expected_logical_names() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {
        "credit_total_legacy_sa",
        "credit_total_inc_fin_sa",
        "credit_business_legacy_sa",
        "credit_business_inc_fin_sa",
        "credit_owner_occupier_housing_sa",
        "credit_investor_housing_sa",
        "credit_other_personal_sa",
        "credit_total_yoy_growth_legacy_sa",
        "credit_total_inc_fin_yoy_growth_sa",
        "broad_money_yoy_growth_sa",
    }


def test_series_registry_partitions_correctly_between_d1_and_d2() -> None:
    by_id = {s.series_id: s for s in SERIES}
    # Level series live in D2; growth-rate series live in D1.
    assert by_id["credit_total_legacy_sa"].table == "d2"
    assert by_id["credit_total_inc_fin_sa"].table == "d2"
    assert by_id["credit_total_yoy_growth_legacy_sa"].table == "d1"
    assert by_id["credit_total_inc_fin_yoy_growth_sa"].table == "d1"
    assert by_id["broad_money_yoy_growth_sa"].table == "d1"


def test_series_registry_rba_ids_match_known_codes() -> None:
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["credit_total_legacy_sa"] == "DLCACS"
    assert by_id["credit_total_inc_fin_sa"] == "DLCACSFS"
    assert by_id["credit_business_legacy_sa"] == "DLCACBS"
    assert by_id["credit_business_inc_fin_sa"] == "DLCACSFBS"
    assert by_id["credit_owner_occupier_housing_sa"] == "DLCACOHS"
    assert by_id["credit_investor_housing_sa"] == "DLCACIHS"
    assert by_id["credit_other_personal_sa"] == "DLCACOPS"
    assert by_id["credit_total_yoy_growth_legacy_sa"] == "DGFAC12"
    assert by_id["credit_total_inc_fin_yoy_growth_sa"] == "DGFACNW12"
    assert by_id["broad_money_yoy_growth_sa"] == "DGFABM12"
