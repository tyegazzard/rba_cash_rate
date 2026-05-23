"""Tests for ``rba.data.sources.abs_building_approvals``.

The download path (``fetch``) is not exercised — it would hit the live
ABS Data API + the RBA H3 CSV. ``_parse_sdmx``, ``_parse_h3``, and
``_attach_publication_dates`` are tested against hand-built fixtures.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_building_approvals import (
    ABS_NSA_SERIES,
    RBA_H3_SERIES,
    RbaH3Series,
    _attach_publication_dates,
    _month_end,
    _parse_h3,
    _parse_sdmx,
    _to_month_end,
)

# Synthetic SDMX-JSON payload mirroring the BA_GCCSA shape. Seven dims
# before TIME_PERIOD (MEASURE.VALUE.SECTOR.WORK_TYPE.BUILDING_TYPE.TSEST.REGION.FREQ),
# so observation keys are eight-tuples.
_SDMX_FIXTURE = json.dumps(
    {
        "meta": {"prepared": "2026-05-22T00:00:00Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0:0:0:0:0:0": [12345, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:0:1": ["12500", 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:0:2": [None, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:0:3": [17000.5, 0, 0, None, None, 0],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "1"}]},
                            {"id": "VALUE", "values": [{"id": "1"}]},
                            {"id": "SECTOR", "values": [{"id": "9"}]},
                            {"id": "WORK_TYPE", "values": [{"id": "TOT"}]},
                            {"id": "BUILDING_TYPE", "values": [{"id": "100"}]},
                            {"id": "TSEST", "values": [{"id": "10"}]},
                            {"id": "REGION", "values": [{"id": "AUS"}]},
                            {"id": "FREQ", "values": [{"id": "M"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "1990-01"},
                                    {"id": "1990-02"},
                                    {"id": "1990-03"},
                                    {"id": "2026-03"},
                                ],
                            },
                        ]
                    },
                    "attributes": {"observation": []},
                }
            ],
        },
    }
).encode("utf-8")


# H3 fixture mirroring the RBA H3 monthly-activity-indicators CSV layout.
# Headers / metadata rows match the real D1/D2/H3 layout convention.
_H3_FIXTURE_CSV = (
    "H3 MONTHLY ACTIVITY INDICATORS\n"
    "Title,Private dwelling approvals,Private dwelling approvals trend\n"
    "Description,Private dwelling approvals,Private dwelling approvals trend\n"
    "Frequency,Monthly,Monthly\n"
    "Type,Seasonally adjusted,Trend\n"
    "Units,'000,'000\n"
    "\n"
    "\n"
    "Source,ABS,ABS\n"
    "Publication date,15-May-2026,15-May-2026\n"
    "Series ID,GISPSDA,GISDWPRITR\n"
    "31/01/1990,11.5,11.0\n"
    "28/02/1990,12.2,11.4\n"
    "31/03/1990,,11.8\n"
    "31/03/2024,15.3,14.9\n"
).encode("utf-8")


# -----------------------------------------------------------------------------
# _parse_sdmx
# -----------------------------------------------------------------------------


def test_parse_sdmx_returns_expected_columns_and_dtypes() -> None:
    df = _parse_sdmx(_SDMX_FIXTURE, series_id="building_approvals_dwellings_total_nsa")
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_sdmx_skips_null_observations() -> None:
    df = _parse_sdmx(_SDMX_FIXTURE, series_id="building_approvals_dwellings_total_nsa")
    # Fixture has 4 obs, one null -> 3 rows.
    assert len(df) == 3
    assert pd.Timestamp("1990-03-31") not in df["observation_date"].tolist()


def test_parse_sdmx_coerces_string_value_to_float() -> None:
    df = _parse_sdmx(_SDMX_FIXTURE, series_id="building_approvals_dwellings_total_nsa")
    val = df.loc[df["observation_date"] == pd.Timestamp("1990-02-28"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 12500.0


def test_parse_sdmx_stamps_series_id_on_every_row() -> None:
    df = _parse_sdmx(_SDMX_FIXTURE, series_id="any_logical_name")
    assert (df["series_id"] == "any_logical_name").all()


def test_parse_sdmx_raises_on_empty_observations() -> None:
    empty = json.dumps(
        {
            "data": {
                "dataSets": [{"observations": {}}],
                "structures": [
                    {
                        "dimensions": {
                            "observation": [
                                {"id": "MEASURE", "values": [{"id": "1"}]},
                                {"id": "TIME_PERIOD", "values": []},
                            ]
                        },
                        "attributes": {"observation": []},
                    }
                ],
            }
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="zero observations"):
        _parse_sdmx(empty, series_id="x")


# -----------------------------------------------------------------------------
# _parse_h3
# -----------------------------------------------------------------------------


def test_parse_h3_extracts_target_series() -> None:
    spec = RbaH3Series(
        series_id="building_approvals_private_dwellings_sa",
        rba_series_id="GISPSDA",
    )
    df = _parse_h3(_H3_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    # 4 fixture rows, one null at 1990-03 for GISPSDA -> 3 rows.
    assert len(df) == 3
    assert (df["series_id"] == "building_approvals_private_dwellings_sa").all()


def test_parse_h3_extracts_trend_series() -> None:
    spec = RbaH3Series(
        series_id="building_approvals_private_dwellings_trend",
        rba_series_id="GISDWPRITR",
    )
    df = _parse_h3(_H3_FIXTURE_CSV, spec=spec)
    # Trend column has no nulls -> 4 rows.
    assert len(df) == 4
    assert pd.Timestamp("1990-03-31") in df["observation_date"].tolist()


def test_parse_h3_raises_on_missing_rba_series_id() -> None:
    spec = RbaH3Series(series_id="bogus", rba_series_id="DOES_NOT_EXIST")
    with pytest.raises(ValueError, match="not found among"):
        _parse_h3(_H3_FIXTURE_CSV, spec=spec)


def test_parse_h3_raises_when_series_id_row_missing() -> None:
    csv = (
        "TEST\n"
        "Title,Series Q\n"
        "31/01/2024,1.0\n"
    ).encode("utf-8")
    spec = RbaH3Series(series_id="x", rba_series_id="GISPSDA")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse_h3(csv, spec=spec)


# -----------------------------------------------------------------------------
# month-end helpers
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1990-01", pd.Timestamp("1990-01-31")),
        ("2024-02", pd.Timestamp("2024-02-29")),
        ("2024-04", pd.Timestamp("2024-04-30")),
        ("2026-12", pd.Timestamp("2026-12-31")),
    ],
)
def test_month_end(period: str, expected: pd.Timestamp) -> None:
    assert _month_end(period) == expected


def test_to_month_end_idempotent_on_month_end() -> None:
    out = _to_month_end(pd.Series([pd.Timestamp("2024-02-29")]))
    assert out.iloc[0] == pd.Timestamp("2024-02-29")
    assert out.dtype == "datetime64[ns]"


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_post_floor_month() -> None:
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2024-02-29")],
            "series_id": ["building_approvals_dwellings_total_nsa"],
            "value": [15000.0],
        }
    )
    out = _attach_publication_dates(obs)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # Feb 2024 ref → scraped: 04 Apr 2024
    assert out["publication_date"].iloc[0] == pd.Timestamp("2024-04-04")


def test_attach_publication_dates_applies_flat_offset_pre_floor() -> None:
    """Observations before the 2019-12-31 scrape floor use observation_date + 40 days."""
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("1993-01-31")],
            "series_id": ["building_approvals_dwellings_total_nsa"],
            "value": [10000.0],
        }
    )
    out = _attach_publication_dates(obs)
    assert out["publication_date"].iloc[0] == pd.Timestamp("1993-01-31") + pd.Timedelta(
        days=40
    )


def test_attach_publication_dates_drops_stale_column() -> None:
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2024-02-29")],
            "publication_date": [pd.Timestamp("2099-01-01")],
            "series_id": ["building_approvals_dwellings_total_nsa"],
            "value": [15000.0],
        }
    )
    out = _attach_publication_dates(obs)
    assert out["publication_date"].iloc[0] != pd.Timestamp("2099-01-01")


def test_attach_publication_dates_raises_on_unmatched_post_floor_obs() -> None:
    """Post-floor observation date the scraped calendar doesn't emit."""
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2099-06-30")],
            "series_id": ["building_approvals_dwellings_total_nsa"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing a publication"):
        _attach_publication_dates(obs)


# -----------------------------------------------------------------------------
# Series registry
# -----------------------------------------------------------------------------


def test_abs_nsa_series_registry_covers_expected_ids() -> None:
    ids = {s.series_id for s in ABS_NSA_SERIES}
    assert ids == {
        "building_approvals_dwellings_total_nsa",
        "building_approvals_dwellings_houses_nsa",
        "building_approvals_dwellings_other_nsa",
        "building_approvals_value_residential_aud_thousand_nsa",
        "building_approvals_value_non_residential_aud_thousand_nsa",
    }


def test_abs_nsa_series_datakey_components() -> None:
    """Every ABS NSA series targets REGION=AUS, TSEST=10 (Original)."""
    for s in ABS_NSA_SERIES:
        parts = s.datakey.split(".")
        assert parts[5] == "10", f"{s.series_id!r} TSEST is not Original"
        assert parts[6] == "AUS", f"{s.series_id!r} REGION is not AUS"
        assert parts[7] == "M", f"{s.series_id!r} FREQ is not Monthly"
        assert parts[2] == "9", f"{s.series_id!r} SECTOR is not Total"
        assert parts[3] == "TOT", f"{s.series_id!r} WORK_TYPE is not Total Work"
        assert s.dataflow == "BA_GCCSA"


def test_rba_h3_series_registry_covers_expected_ids() -> None:
    by_id = {s.series_id: s.rba_series_id for s in RBA_H3_SERIES}
    assert by_id == {
        "building_approvals_private_dwellings_sa": "GISPSDA",
        "building_approvals_private_dwellings_trend": "GISDWPRITR",
    }
