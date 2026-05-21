"""Tests for ``rba.data.sources.abs_gdp`` SDMX-JSON parser + conventions.

The download path (``fetch``) is not exercised — it would hit the live ABS
Data API. ``_parse`` and ``_quarter_end`` are tested in isolation against a
hand-built SDMX-JSON fixture mirroring real ABS responses
(``dimensionAtObservation=AllDimensions``).

The ``_attach_publication_dates`` helper is tested against both branches of
its merge: in-window observations (>= 1993-Q1) must match the scraped
calendar; pre-window observations retain NaT.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_gdp import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    _attach_publication_dates,
    _parse,
    _quarter_end,
)

# Synthetic SDMX-JSON payload mirroring the ANA_AGG dataflow shape. Five
# series-defining dimensions before TIME_PERIOD (MEASURE, DATA_ITEM, TSEST,
# REGION, FREQ) so observation keys are colon-delimited six-tuples. We
# include:
#   - a pre-floor observation (1959-Q3 -> 9000.0) — calendar miss is OK
#   - a normal observation (1993-Q1 -> 138000.0) — first scraped quarter
#   - a numeric-string value (1993-Q2 -> "139000.0") — real ABS returns
#     strings sometimes
#   - a null/None value (1993-Q3) — must be skipped
#   - an in-window observation (2024-Q1 -> 670670.0) — exercises merge
_FIXTURE_JSON = json.dumps(
    {
        "meta": {"prepared": "2026-05-21T00:00:00Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0:0:0": [9000.0, 0, 0, None, None, 0],
                        "0:0:0:0:0:1": [138000.0, 0, 0, None, None, 0],
                        "0:0:0:0:0:2": ["139000.0", 0, 0, None, None, 0],
                        "0:0:0:0:0:3": [None, 0, 0, None, None, 0],
                        "0:0:0:0:0:4": [670670.0, 0, 0, None, None, 0],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "M1"}]},
                            {"id": "DATA_ITEM", "values": [{"id": "GPM"}]},
                            {"id": "TSEST", "values": [{"id": "20"}]},
                            {"id": "REGION", "values": [{"id": "AUS"}]},
                            {"id": "FREQ", "values": [{"id": "Q"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "1959-Q3"},
                                    {"id": "1993-Q1"},
                                    {"id": "1993-Q2"},
                                    {"id": "1993-Q3"},
                                    {"id": "2024-Q1"},
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


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_returns_expected_columns_and_dtypes() -> None:
    df = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    # _parse does not attach publication_date — that is done downstream by
    # _attach_publication_dates via the calendar merge.
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_skips_null_observations() -> None:
    df = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    # Fixture has 5 obs, one is null -> expect 4 rows.
    assert len(df) == 4
    # 1993-Q3 must be absent.
    assert pd.Timestamp("1993-09-30") not in df["observation_date"].tolist()


def test_parse_coerces_string_values_to_float() -> None:
    df = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    val = df.loc[df["observation_date"] == pd.Timestamp("1993-06-30"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 139000.0


def test_parse_maps_quarter_to_calendar_end() -> None:
    df = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    expected = {
        "1959-Q3": pd.Timestamp("1959-09-30"),
        "1993-Q1": pd.Timestamp("1993-03-31"),
        "1993-Q2": pd.Timestamp("1993-06-30"),
        "2024-Q1": pd.Timestamp("2024-03-31"),
    }
    for period, end in expected.items():
        assert end in df["observation_date"].tolist(), f"missing {period}"


def test_parse_stamps_series_id_on_every_row() -> None:
    df = _parse(_FIXTURE_JSON, series_id="gdp_nominal_current_prices_sa")
    assert (df["series_id"] == "gdp_nominal_current_prices_sa").all()


def test_parse_raises_on_empty_observations() -> None:
    empty = json.dumps(
        {
            "meta": {},
            "data": {
                "dataSets": [{"observations": {}}],
                "structures": [
                    {
                        "dimensions": {
                            "observation": [
                                {"id": "MEASURE", "values": [{"id": "M1"}]},
                                {"id": "DATA_ITEM", "values": [{"id": "GPM"}]},
                                {"id": "TSEST", "values": [{"id": "20"}]},
                                {"id": "REGION", "values": [{"id": "AUS"}]},
                                {"id": "FREQ", "values": [{"id": "Q"}]},
                                {"id": "TIME_PERIOD", "values": []},
                            ]
                        },
                        "attributes": {"observation": []},
                    }
                ],
            },
        }
    ).encode("utf-8")
    with pytest.raises(ValueError, match="zero observations"):
        _parse(empty, series_id="gdp_real_chain_volume_sa")


# -----------------------------------------------------------------------------
# _quarter_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1959-Q3", pd.Timestamp("1959-09-30")),
        ("1993-Q1", pd.Timestamp("1993-03-31")),
        ("2024-Q2", pd.Timestamp("2024-06-30")),  # leap year Q2
        ("2025-Q4", pd.Timestamp("2025-12-31")),  # year-boundary Q4
    ],
)
def test_quarter_end(period: str, expected: pd.Timestamp) -> None:
    assert _quarter_end(period) == expected


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_in_window_rows() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # 2024-Q1 reference -> scraped original release was 5 Jun 2024.
    row = out.loc[out["observation_date"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2024-06-05")
    # 1993-Q1 reference -> scraped original release was 1 Jun 1993 (Tuesday,
    # not Wednesday — pre-2003 era irregularity).
    row1993 = out.loc[out["observation_date"] == pd.Timestamp("1993-03-31")].iloc[0]
    assert row1993["publication_date"] == pd.Timestamp("1993-06-01")


def test_attach_publication_dates_leaves_pre_floor_rows_as_nat() -> None:
    """Pre-1993 GDP rows have no scraped calendar entry and retain NaT."""
    parsed = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa")
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("1959-09-30")].iloc[0]
    assert pd.isna(row["publication_date"])


def test_attach_publication_dates_drops_stale_column() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="gdp_real_chain_volume_sa").copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    # The stale 2099 value must not survive — calendar replaces it for matched
    # rows, NaT replaces it for pre-window rows.
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_raises_on_unmatched_in_window_row() -> None:
    """An in-window observation that doesn't match the scraped calendar
    indicates a data / calendar mismatch and must raise."""
    bad = pd.DataFrame(
        {
            # 2025-05-15 is not a quarter-end, so the calendar will not have it.
            # Confirmed >= the validation floor (1993-01-01).
            "observation_date": [pd.Timestamp("2025-05-15")],
            "series_id": ["gdp_real_chain_volume_sa"],
            "value": [700000.0],
        }
    )
    assert bad["observation_date"].iloc[0] >= _CALENDAR_VALIDATION_FLOOR
    with pytest.raises(ValueError, match="missing a publication"):
        _attach_publication_dates(bad)


# -----------------------------------------------------------------------------
# Series registry
# -----------------------------------------------------------------------------


def test_series_registry_covers_three_expected_series() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {
        "gdp_real_chain_volume_sa",
        "gdp_nominal_current_prices_sa",
        "gdp_per_capita_real_chain_volume_sa",
    }


def test_series_registry_datakey_positions() -> None:
    by_id = {s.series_id: s for s in SERIES}
    # All three series live in the ANA_AGG dataflow with positional datakey
    # MEASURE.DATA_ITEM.TSEST.REGION.FREQ.
    for s in SERIES:
        assert s.dataflow == "ANA_AGG"
    # Real GDP (chain volume) -> MEASURE=M1, DATA_ITEM=GPM
    assert by_id["gdp_real_chain_volume_sa"].datakey == "M1.GPM.20.AUS.Q"
    # Nominal GDP (current prices) -> MEASURE=M3, DATA_ITEM=GPM
    assert by_id["gdp_nominal_current_prices_sa"].datakey == "M3.GPM.20.AUS.Q"
    # Real GDP per capita -> MEASURE=M1, DATA_ITEM=GPM_PCA
    assert by_id["gdp_per_capita_real_chain_volume_sa"].datakey == "M1.GPM_PCA.20.AUS.Q"
