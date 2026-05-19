"""Tests for ``rba.data.sources.abs_cpi`` SDMX-JSON parser and conventions.

The download path (``fetch``) is not exercised here — it would hit the live
ABS Data API. ``_parse`` and ``_quarter_end`` are tested in isolation against
a hand-built SDMX-JSON fixture that mirrors the structure of real ABS
responses (``dimensionAtObservation=AllDimensions``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_cpi import (
    SERIES,
    _attach_publication_dates,
    _parse,
    _quarter_end,
)

# Synthetic SDMX-JSON payload mirroring the real ABS response shape. Five
# observation dimensions before TIME_PERIOD (MEASURE, INDEX, TSEST, REGION,
# FREQ) so observation keys are colon-delimited six-tuples. We include:
#   - a normal observation (1993-Q1 -> 105.2)
#   - a numeric-string value (1993-Q2 -> "106.5") — real ABS returns strings
#   - a null/None value (1993-Q3) — must be skipped
#   - a more recent quarter (2024-Q4) to exercise quarter-end mapping
_FIXTURE_JSON = json.dumps(
    {
        "meta": {"prepared": "2026-05-17T22:21:49Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0:0:0": [105.2, 0, None, None, None],
                        "0:0:0:0:0:1": ["106.5", 0, None, None, None],
                        "0:0:0:0:0:2": [None, 0, None, None, None],
                        "0:0:0:0:0:3": [142.7, 0, None, None, None],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "1"}]},
                            {"id": "INDEX", "values": [{"id": "10001"}]},
                            {"id": "TSEST", "values": [{"id": "10"}]},
                            {"id": "REGION", "values": [{"id": "50"}]},
                            {"id": "FREQ", "values": [{"id": "Q"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "1993-Q1"},
                                    {"id": "1993-Q2"},
                                    {"id": "1993-Q3"},
                                    {"id": "2024-Q4"},
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


def test_parse_returns_expected_columns_and_dtypes() -> None:
    df = _parse(_FIXTURE_JSON, series_id="headline_cpi_index")
    # _parse no longer attaches publication_date — that is done downstream by
    # _attach_publication_dates via merge against the release calendar.
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_skips_null_observations() -> None:
    df = _parse(_FIXTURE_JSON, series_id="headline_cpi_index")
    # Fixture has 4 obs, one is null -> expect 3 rows.
    assert len(df) == 3
    # 1993-Q3 must be absent.
    assert pd.Timestamp("1993-09-30") not in df["observation_date"].tolist()


def test_parse_coerces_string_values_to_float() -> None:
    df = _parse(_FIXTURE_JSON, series_id="headline_cpi_index")
    val = df.loc[df["observation_date"] == pd.Timestamp("1993-06-30"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 106.5


def test_parse_maps_quarter_to_calendar_end() -> None:
    df = _parse(_FIXTURE_JSON, series_id="headline_cpi_index")
    expected = {
        "1993-Q1": pd.Timestamp("1993-03-31"),
        "1993-Q2": pd.Timestamp("1993-06-30"),
        "2024-Q4": pd.Timestamp("2024-12-31"),
    }
    for period, end in expected.items():
        assert end in df["observation_date"].tolist(), f"missing {period}"


def test_parse_stamps_series_id_on_every_row() -> None:
    df = _parse(_FIXTURE_JSON, series_id="trimmed_mean_index")
    assert (df["series_id"] == "trimmed_mean_index").all()


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
                                {"id": "MEASURE", "values": [{"id": "1"}]},
                                {"id": "INDEX", "values": [{"id": "10001"}]},
                                {"id": "TSEST", "values": [{"id": "10"}]},
                                {"id": "REGION", "values": [{"id": "50"}]},
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
        _parse(empty, series_id="headline_cpi_index")


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1948-Q3", pd.Timestamp("1948-09-30")),
        ("1993-Q1", pd.Timestamp("1993-03-31")),
        ("2024-Q4", pd.Timestamp("2024-12-31")),
        ("2025-Q2", pd.Timestamp("2025-06-30")),
    ],
)
def test_quarter_end(period: str, expected: pd.Timestamp) -> None:
    assert _quarter_end(period) == expected


def test_series_registry_covers_three_expected_series() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {"headline_cpi_index", "trimmed_mean_index", "weighted_median_index"}


def test_attach_publication_dates_uses_calendar() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="headline_cpi_index")
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # 1993-Q1 ends 31 Mar 1993; ABS publishes on last Wed of April 1993 = Apr 28.
    row = out.loc[out["observation_date"] == pd.Timestamp("1993-03-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("1993-04-28")


def test_attach_publication_dates_drops_stale_column() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="headline_cpi_index").copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    # The stale 2099 value must not survive — it should be overwritten by the
    # algorithmic calendar value.
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_raises_on_unmatched_post_1993_row() -> None:
    # Build a frame with a quarter end the calendar will not emit (mid-month).
    bad = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2020-05-15")],
            "series_id": ["headline_cpi_index"],
            "value": [100.0],
        }
    )
    with pytest.raises(ValueError, match="missing a publication date"):
        _attach_publication_dates(bad)


def test_series_registry_datakey_positions() -> None:
    # Datakey order is MEASURE.INDEX.TSEST.REGION.FREQ.
    # Sanity-check the codes we settled on after API exploration.
    by_id = {s.series_id: s for s in SERIES}
    assert by_id["headline_cpi_index"].dataflow == "CPI"
    assert by_id["headline_cpi_index"].datakey == "1.10001.10.50.Q"
    assert by_id["trimmed_mean_index"].dataflow == "CPI_Q"
    assert by_id["trimmed_mean_index"].datakey == "1.999902.20.50.Q"
    assert by_id["weighted_median_index"].dataflow == "CPI_Q"
    assert by_id["weighted_median_index"].datakey == "1.999903.20.50.Q"
