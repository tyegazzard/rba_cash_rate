"""Tests for ``rba.data.sources.abs_wpi`` SDMX-JSON parser + conventions.

The download path (``fetch``) is not exercised — it would hit the live ABS
Data API. ``_parse`` and ``_quarter_end`` are tested in isolation against a
hand-built SDMX-JSON fixture mirroring real ABS responses
(``dimensionAtObservation=AllDimensions``).

The ``_attach_publication_dates`` helper is tested against both branches of
its merge: in-window observations (>= 2019-Q3) must match the scraped
calendar; pre-window observations get the flat 60-day fallback.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_wpi import (
    _CALENDAR_VALIDATION_FLOOR,
    _PUBLICATION_OFFSET_DAYS,
    SERIES,
    _attach_publication_dates,
    _parse,
    _quarter_end,
)

# Synthetic SDMX-JSON payload mirroring the WPI dataflow shape. Seven
# series-defining dimensions before TIME_PERIOD (MEASURE, INDEX, SECTOR,
# INDUSTRY, TSEST, REGION, FREQ) so observation keys are colon-delimited
# eight-tuples. We include:
#   - a normal observation (1997-Q3 -> 66.6)
#   - a numeric-string value (1997-Q4 -> "67.0") — real ABS returns strings sometimes
#   - a null/None value (1998-Q1) — must be skipped
#   - an in-calendar-window observation (2024-Q1) to exercise the calendar merge
_FIXTURE_JSON = json.dumps(
    {
        "meta": {"prepared": "2026-05-20T00:00:00Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0:0:0:0:0": [66.6, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:1": ["67.0", 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:2": [None, 0, 0, None, None, 0],
                        "0:0:0:0:0:0:0:3": [149.2, 0, 0, None, None, 0],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "1"}]},
                            {"id": "INDEX", "values": [{"id": "THRPEB"}]},
                            {"id": "SECTOR", "values": [{"id": "7"}]},
                            {"id": "INDUSTRY", "values": [{"id": "TOT"}]},
                            {"id": "TSEST", "values": [{"id": "20"}]},
                            {"id": "REGION", "values": [{"id": "AUS"}]},
                            {"id": "FREQ", "values": [{"id": "Q"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "1997-Q3"},
                                    {"id": "1997-Q4"},
                                    {"id": "1998-Q1"},
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
    df = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    # _parse no longer attaches publication_date — that is done downstream by
    # _attach_publication_dates via the calendar merge.
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_skips_null_observations() -> None:
    df = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    # Fixture has 4 obs, one is null -> expect 3 rows.
    assert len(df) == 3
    # 1998-Q1 must be absent.
    assert pd.Timestamp("1998-03-31") not in df["observation_date"].tolist()


def test_parse_coerces_string_values_to_float() -> None:
    df = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    val = df.loc[df["observation_date"] == pd.Timestamp("1997-12-31"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 67.0


def test_parse_maps_quarter_to_calendar_end() -> None:
    df = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    expected = {
        "1997-Q3": pd.Timestamp("1997-09-30"),
        "1997-Q4": pd.Timestamp("1997-12-31"),
        "2024-Q1": pd.Timestamp("2024-03-31"),
    }
    for period, end in expected.items():
        assert end in df["observation_date"].tolist(), f"missing {period}"


def test_parse_stamps_series_id_on_every_row() -> None:
    df = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_private_sa")
    assert (df["series_id"] == "wpi_total_hourly_excl_bonuses_private_sa").all()


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
                                {"id": "INDEX", "values": [{"id": "THRPEB"}]},
                                {"id": "SECTOR", "values": [{"id": "7"}]},
                                {"id": "INDUSTRY", "values": [{"id": "TOT"}]},
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
        _parse(empty, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")


# -----------------------------------------------------------------------------
# _quarter_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("1997-Q3", pd.Timestamp("1997-09-30")),
        ("2000-Q1", pd.Timestamp("2000-03-31")),
        ("2024-Q2", pd.Timestamp("2024-06-30")),  # leap year Q2
        ("2026-Q4", pd.Timestamp("2026-12-31")),  # year-boundary Q4
    ],
)
def test_quarter_end(period: str, expected: pd.Timestamp) -> None:
    assert _quarter_end(period) == expected


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_in_window_rows() -> None:
    parsed = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # 2024-Q1 reference -> scraped original release was 15 May 2024.
    row = out.loc[out["observation_date"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2024-05-15")


def test_attach_publication_dates_applies_flat_offset_pre_window() -> None:
    """Pre-2019-Q3 WPI rows fall back to observation_date + 60 days."""
    parsed = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    out = _attach_publication_dates(parsed)
    # 1997-Q3 ends 1997-09-30; +60 days = 1997-11-29.
    row = out.loc[out["observation_date"] == pd.Timestamp("1997-09-30")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("1997-09-30") + pd.Timedelta(
        days=_PUBLICATION_OFFSET_DAYS
    )
    # 1997-Q4 ends 1997-12-31; +60 days = 1998-03-01.
    row2 = out.loc[out["observation_date"] == pd.Timestamp("1997-12-31")].iloc[0]
    assert row2["publication_date"] == pd.Timestamp("1998-03-01")


def test_attach_publication_dates_never_returns_nat() -> None:
    """Both branches must produce a publication_date — no NaT allowed."""
    parsed = _parse(_FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa")
    out = _attach_publication_dates(parsed)
    assert out["publication_date"].notna().all()


def test_attach_publication_dates_drops_stale_column() -> None:
    parsed = _parse(
        _FIXTURE_JSON, series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa"
    ).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    # The stale 2099 value must not survive — calendar replaces it for matched
    # rows and the flat offset replaces it for pre-window rows.
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_raises_on_unmatched_in_window_row() -> None:
    """An in-window observation that doesn't match the scraped calendar
    indicates a data / calendar mismatch and must raise."""
    bad = pd.DataFrame(
        {
            # 2025-05-15 is not a quarter-end, so the calendar will not have it.
            # Confirmed >= the validation floor (2019-09-30).
            "observation_date": [pd.Timestamp("2025-05-15")],
            "series_id": ["wpi_total_hourly_excl_bonuses_all_sectors_sa"],
            "value": [150.0],
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
        "wpi_total_hourly_excl_bonuses_all_sectors_sa",
        "wpi_total_hourly_excl_bonuses_private_sa",
        "wpi_total_hourly_excl_bonuses_public_sa",
    }


def test_series_registry_datakey_positions() -> None:
    by_id = {s.series_id: s for s in SERIES}
    # All three series live in the WPI dataflow with positional datakey
    # MEASURE.INDEX.SECTOR.INDUSTRY.TSEST.REGION.FREQ.
    for s in SERIES:
        assert s.dataflow == "WPI"
    # Headline (all sectors) -> SECTOR=7
    assert by_id["wpi_total_hourly_excl_bonuses_all_sectors_sa"].datakey == (
        "1.THRPEB.7.TOT.20.AUS.Q"
    )
    # Private -> SECTOR=1
    assert by_id["wpi_total_hourly_excl_bonuses_private_sa"].datakey == (
        "1.THRPEB.1.TOT.20.AUS.Q"
    )
    # Public  -> SECTOR=2
    assert by_id["wpi_total_hourly_excl_bonuses_public_sa"].datakey == (
        "1.THRPEB.2.TOT.20.AUS.Q"
    )
