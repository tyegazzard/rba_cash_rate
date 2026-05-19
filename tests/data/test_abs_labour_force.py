"""Tests for ``rba.data.sources.abs_labour_force`` SDMX-JSON parser + conventions.

The download path (``fetch``) is not exercised — it would hit the live ABS
Data API. ``_parse`` and ``_month_end`` are tested in isolation against a
hand-built SDMX-JSON fixture mirroring real ABS responses
(``dimensionAtObservation=AllDimensions``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_labour_force import (
    SERIES,
    _attach_publication_dates,
    _month_end,
    _parse,
)

# Synthetic SDMX-JSON payload mirroring the LF dataflow shape. Six dimensions
# before TIME_PERIOD (MEASURE, SEX, AGE, TSEST, REGION, FREQ) so observation
# keys are colon-delimited seven-tuples. We include:
#   - a normal observation (1978-02 → 6.6)
#   - a numeric-string value (1978-03 → "6.5") — real ABS returns strings sometimes
#   - a null/None value (1978-04) — must be skipped
#   - a more recent month (2024-03) to exercise month-end mapping
_FIXTURE_JSON = json.dumps(
    {
        "meta": {"prepared": "2026-05-18T22:21:49Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0:0:0:0": [6.6, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:1": ["6.5", 0, 0, None, None, 0],
                        "0:0:0:0:0:0:2": [None, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:3": [3.8, 0, 0, None, None, 0],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "M13"}]},
                            {"id": "SEX", "values": [{"id": "3"}]},
                            {"id": "AGE", "values": [{"id": "1599"}]},
                            {"id": "TSEST", "values": [{"id": "20"}]},
                            {"id": "REGION", "values": [{"id": "AUS"}]},
                            {"id": "FREQ", "values": [{"id": "M"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "1978-02"},
                                    {"id": "1978-03"},
                                    {"id": "1978-04"},
                                    {"id": "2024-03"},
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
    df = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    # _parse no longer attaches publication_date — that is done downstream by
    # _attach_publication_dates via merge against the LFS release calendar.
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_skips_null_observations() -> None:
    df = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    # Fixture has 4 obs, one is null -> expect 3 rows.
    assert len(df) == 3
    # 1978-04 must be absent.
    assert pd.Timestamp("1978-04-30") not in df["observation_date"].tolist()


def test_parse_coerces_string_values_to_float() -> None:
    df = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    val = df.loc[df["observation_date"] == pd.Timestamp("1978-03-31"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 6.5


def test_parse_maps_month_to_calendar_end() -> None:
    df = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    expected = {
        "1978-02": pd.Timestamp("1978-02-28"),
        "1978-03": pd.Timestamp("1978-03-31"),
        "2024-03": pd.Timestamp("2024-03-31"),
    }
    for period, end in expected.items():
        assert end in df["observation_date"].tolist(), f"missing {period}"


def test_parse_stamps_series_id_on_every_row() -> None:
    df = _parse(_FIXTURE_JSON, series_id="participation_rate_sa")
    assert (df["series_id"] == "participation_rate_sa").all()


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
                                {"id": "MEASURE", "values": [{"id": "M13"}]},
                                {"id": "SEX", "values": [{"id": "3"}]},
                                {"id": "AGE", "values": [{"id": "1599"}]},
                                {"id": "TSEST", "values": [{"id": "20"}]},
                                {"id": "REGION", "values": [{"id": "AUS"}]},
                                {"id": "FREQ", "values": [{"id": "M"}]},
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
        _parse(empty, series_id="unemployment_rate_sa")


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1978-02", pd.Timestamp("1978-02-28")),  # non-leap Feb
        ("2000-02", pd.Timestamp("2000-02-29")),  # leap Feb
        ("2024-04", pd.Timestamp("2024-04-30")),  # 30-day month
        ("2026-12", pd.Timestamp("2026-12-31")),  # year boundary
    ],
)
def test_month_end(period: str, expected: pd.Timestamp) -> None:
    assert _month_end(period) == expected


def test_series_registry_covers_three_expected_series() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {
        "unemployment_rate_sa",
        "underemployment_rate_sa",
        "participation_rate_sa",
    }


def test_attach_publication_dates_uses_calendar() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # March 2024 reference → 3rd Thursday of April 2024 = Apr 18 2024
    # (April 2024 Thursdays: 4, 11, 18, 25).
    row = out.loc[out["observation_date"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2024-04-18")


def test_attach_publication_dates_allows_nat_pre_1993() -> None:
    """Pre-1993 LFS rows are outside the verified era-rule window and may
    legitimately have NaT publication_date — the validation floor is 1993."""
    parsed = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa")
    out = _attach_publication_dates(parsed)
    pre_1993 = out[out["observation_date"] < pd.Timestamp("1993-01-01")]
    assert len(pre_1993) > 0, "fixture should contain pre-1993 observations"
    assert pre_1993["publication_date"].isna().all()


def test_attach_publication_dates_drops_stale_column() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="unemployment_rate_sa").copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    # The stale 2099 value must not survive — for matched rows the algorithmic
    # date replaces it; for pre-1993 rows it becomes NaT.
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_raises_on_unmatched_post_1993_row() -> None:
    # Mid-month observation date the calendar will not emit.
    bad = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2020-05-15")],
            "series_id": ["unemployment_rate_sa"],
            "value": [5.5],
        }
    )
    with pytest.raises(ValueError, match="missing a publication date"):
        _attach_publication_dates(bad)


def test_series_registry_datakey_positions() -> None:
    by_id = {s.series_id: s for s in SERIES}
    # Unemployment + participation live in LF; underemployment in LF_UNDER.
    assert by_id["unemployment_rate_sa"].dataflow == "LF"
    assert by_id["unemployment_rate_sa"].datakey == "M13.3.1599.20.AUS.M"
    assert by_id["participation_rate_sa"].dataflow == "LF"
    assert by_id["participation_rate_sa"].datakey == "M12.3.1599.20.AUS.M"
    assert by_id["underemployment_rate_sa"].dataflow == "LF_UNDER"
    assert by_id["underemployment_rate_sa"].datakey == "M23.3.1599.20.AUS.M"
