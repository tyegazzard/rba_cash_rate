"""Tests for ``rba.data.sources.abs_total_value_dwellings``.

The download path (``fetch``) is not exercised — it would hit the live
ABS Data API. ``_parse``, ``_build_spliced_index``, and
``_attach_publication_dates`` are tested against hand-built fixtures.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rba.data.sources.abs_total_value_dwellings import (
    SERIES,
    SPLICED_SERIES_ID,
    _attach_publication_dates,
    _build_spliced_index,
    _parse,
    _quarter_end,
)

# Minimal SDMX-JSON payload mirroring the RES_DWELL_ST shape (3 dims
# before TIME_PERIOD: MEASURE.REGION.FREQ + TIME_PERIOD), so obs keys
# are four-tuples.
_SDMX_FIXTURE = json.dumps(
    {
        "meta": {"prepared": "2026-05-22T00:00:00Z"},
        "data": {
            "dataSets": [
                {
                    "observations": {
                        "0:0:0:0": [100.0, 0, 0, None, None, 0],
                        "0:0:0:1": ["110.5", 0, 0, None, None, 0],
                        "0:0:0:2": [None, 0, 0, None, None, 0],
                        "0:0:0:3": [200.0, 0, 0, None, None, 0],
                    }
                }
            ],
            "structures": [
                {
                    "dimensions": {
                        "observation": [
                            {"id": "MEASURE", "values": [{"id": "5"}]},
                            {"id": "REGION", "values": [{"id": "AUS"}]},
                            {"id": "FREQ", "values": [{"id": "Q"}]},
                            {
                                "id": "TIME_PERIOD",
                                "values": [
                                    {"id": "2011-Q3"},
                                    {"id": "2011-Q4"},
                                    {"id": "2012-Q1"},
                                    {"id": "2025-Q4"},
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
    df = _parse(_SDMX_FIXTURE, series_id="tvd_mean_price_aud_thousand")
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"


def test_parse_skips_null_observations() -> None:
    df = _parse(_SDMX_FIXTURE, series_id="tvd_mean_price_aud_thousand")
    assert len(df) == 3
    assert pd.Timestamp("2012-03-31") not in df["observation_date"].tolist()


def test_parse_coerces_string_value_to_float() -> None:
    df = _parse(_SDMX_FIXTURE, series_id="tvd_mean_price_aud_thousand")
    val = df.loc[df["observation_date"] == pd.Timestamp("2011-12-31"), "value"].iloc[0]
    assert isinstance(val, float)
    assert val == 110.5


def test_parse_maps_quarter_to_calendar_end() -> None:
    df = _parse(_SDMX_FIXTURE, series_id="tvd_mean_price_aud_thousand")
    assert pd.Timestamp("2011-09-30") in df["observation_date"].tolist()
    assert pd.Timestamp("2011-12-31") in df["observation_date"].tolist()
    assert pd.Timestamp("2025-12-31") in df["observation_date"].tolist()


def test_parse_raises_on_empty_observations() -> None:
    empty = json.dumps(
        {
            "data": {
                "dataSets": [{"observations": {}}],
                "structures": [
                    {
                        "dimensions": {
                            "observation": [
                                {"id": "MEASURE", "values": [{"id": "5"}]},
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
        _parse(empty, series_id="x")


# -----------------------------------------------------------------------------
# _quarter_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("2011-Q3", pd.Timestamp("2011-09-30")),
        ("2021-Q4", pd.Timestamp("2021-12-31")),
        ("2022-Q1", pd.Timestamp("2022-03-31")),
        ("2025-Q2", pd.Timestamp("2025-06-30")),
    ],
)
def test_quarter_end(period: str, expected: pd.Timestamp) -> None:
    assert _quarter_end(period) == expected


# -----------------------------------------------------------------------------
# Series registry
# -----------------------------------------------------------------------------


def test_series_registry_has_3_national_plus_32_per_city_plus_1_rppi() -> None:
    # 3 + 8 capitals × 4 measures + 1 RPPI legacy = 36 (the +1 spliced is derived)
    assert len(SERIES) == 36
    ids = [s.series_id for s in SERIES]
    assert len(set(ids)) == len(ids), "duplicate series_id"


def test_series_registry_includes_expected_national_codes() -> None:
    by_id = {s.series_id: s for s in SERIES}
    assert by_id["tvd_value_all_sectors_aud_million"].dataflow == "RES_DWELL_ST"
    assert by_id["tvd_value_all_sectors_aud_million"].datakey == "1.AUS.Q"
    assert by_id["tvd_dwelling_count_thousand"].datakey == "4.AUS.Q"
    assert by_id["tvd_mean_price_aud_thousand"].datakey == "5.AUS.Q"


def test_series_registry_includes_all_8_capitals() -> None:
    suffixes_present: set[str] = set()
    for s in SERIES:
        if s.dataflow != "RES_DWELL":
            continue
        for suffix in (
            "sydney",
            "melbourne",
            "brisbane",
            "adelaide",
            "perth",
            "hobart",
            "darwin",
            "canberra",
        ):
            if suffix in s.series_id:
                suffixes_present.add(suffix)
                break
    assert suffixes_present == {
        "sydney",
        "melbourne",
        "brisbane",
        "adelaide",
        "perth",
        "hobart",
        "darwin",
        "canberra",
    }


def test_rppi_legacy_uses_8cap_weighted_index() -> None:
    by_id = {s.series_id: s for s in SERIES}
    rppi = by_id["housing_price_index_rppi_legacy_8cap"]
    assert rppi.dataflow == "RPPI"
    assert rppi.datakey == "1.3.100.Q"


# -----------------------------------------------------------------------------
# _build_spliced_index
# -----------------------------------------------------------------------------


def _make_minimal_rppi_tvd_frame(
    *,
    rppi_2021_q3: float,
    rppi_2021_q4: float,
    tvd_2021_q4: float,
    tvd_2022_q1: float,
    tvd_2022_q2: float = 1000.0,
) -> pd.DataFrame:
    """Build the minimal RPPI + TVD-mean frame _build_spliced_index needs."""
    rows = [
        ("2021-09-30", "housing_price_index_rppi_legacy_8cap", rppi_2021_q3),
        ("2021-12-31", "housing_price_index_rppi_legacy_8cap", rppi_2021_q4),
        ("2021-12-31", "tvd_mean_price_aud_thousand", tvd_2021_q4),
        ("2022-03-31", "tvd_mean_price_aud_thousand", tvd_2022_q1),
        ("2022-06-30", "tvd_mean_price_aud_thousand", tvd_2022_q2),
    ]
    return pd.DataFrame(
        [
            {
                "observation_date": pd.Timestamp(d),
                "series_id": sid,
                "value": float(v),
            }
            for d, sid, v in rows
        ]
    )


def test_build_spliced_keeps_rppi_levels_pre_boundary() -> None:
    df = _make_minimal_rppi_tvd_frame(
        rppi_2021_q3=175.6,
        rppi_2021_q4=183.9,
        tvd_2021_q4=920.0,
        tvd_2022_q1=935.0,  # ~+1.6% QoQ
    )
    spliced = _build_spliced_index(df).set_index("observation_date")
    # Pre-boundary segments inherit the raw RPPI levels.
    assert spliced.loc[pd.Timestamp("2021-09-30"), "value"] == pytest.approx(175.6)
    assert spliced.loc[pd.Timestamp("2021-12-31"), "value"] == pytest.approx(183.9)


def test_build_spliced_extrapolates_post_boundary_from_tvd_growth() -> None:
    df = _make_minimal_rppi_tvd_frame(
        rppi_2021_q3=175.6,
        rppi_2021_q4=183.9,
        tvd_2021_q4=920.0,
        tvd_2022_q1=935.0,
        tvd_2022_q2=950.0,
    )
    spliced = _build_spliced_index(df).set_index("observation_date")
    # 2022-Q1 = 183.9 * 935 / 920
    assert spliced.loc[pd.Timestamp("2022-03-31"), "value"] == pytest.approx(183.9 * 935.0 / 920.0)
    # 2022-Q2 = 183.9 * 950 / 920
    assert spliced.loc[pd.Timestamp("2022-06-30"), "value"] == pytest.approx(183.9 * 950.0 / 920.0)


def test_build_spliced_stamps_correct_series_id() -> None:
    df = _make_minimal_rppi_tvd_frame(
        rppi_2021_q3=175.6,
        rppi_2021_q4=183.9,
        tvd_2021_q4=920.0,
        tvd_2022_q1=935.0,
    )
    spliced = _build_spliced_index(df)
    assert (spliced["series_id"] == SPLICED_SERIES_ID).all()


def test_build_spliced_raises_when_qoq_gap_exceeds_tolerance() -> None:
    """RPPI QoQ ~ +4.7%, TVD QoQ ~ +50% → gap = ~45pp > 10pp tolerance."""
    df = _make_minimal_rppi_tvd_frame(
        rppi_2021_q3=175.6,
        rppi_2021_q4=183.9,
        tvd_2021_q4=920.0,
        tvd_2022_q1=1380.0,  # ~+50% QoQ — wildly off
    )
    with pytest.raises(ValueError, match="QoQ growth gap"):
        _build_spliced_index(df)


def test_build_spliced_raises_when_boundary_quarter_missing() -> None:
    bad = pd.DataFrame(
        [
            {
                "observation_date": pd.Timestamp("2021-09-30"),
                "series_id": "housing_price_index_rppi_legacy_8cap",
                "value": 175.6,
            },
            # 2021-Q4 is missing for both RPPI and TVD.
        ]
    )
    with pytest.raises(ValueError, match="2021-Q4 missing"):
        _build_spliced_index(bad)


# -----------------------------------------------------------------------------
# _attach_publication_dates
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_calendar_for_post_floor_quarter() -> None:
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2025-12-31")],
            "series_id": ["tvd_mean_price_aud_thousand"],
            "value": [1074.7],
        }
    )
    out = _attach_publication_dates(obs)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]
    # 2025-Q4 ref → 10 Mar 2026 (scraped)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2026-03-10")


def test_attach_publication_dates_applies_flat_offset_for_rppi_era() -> None:
    """RPPI-era quarters (pre 2022-Q1) use observation_date + 76 days."""
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2021-12-31")],
            "series_id": ["housing_price_index_rppi_legacy_8cap"],
            "value": [183.9],
        }
    )
    out = _attach_publication_dates(obs)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2021-12-31") + pd.Timedelta(days=76)


def test_attach_publication_dates_drops_stale_column() -> None:
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2025-12-31")],
            "publication_date": [pd.Timestamp("2099-01-01")],
            "series_id": ["tvd_mean_price_aud_thousand"],
            "value": [1074.7],
        }
    )
    out = _attach_publication_dates(obs)
    assert out["publication_date"].iloc[0] != pd.Timestamp("2099-01-01")


def test_attach_publication_dates_raises_on_unmatched_post_floor() -> None:
    """Post-floor quarter that the scraped calendar doesn't emit."""
    obs = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2099-06-30")],
            "series_id": ["tvd_mean_price_aud_thousand"],
            "value": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing a publication"):
        _attach_publication_dates(obs)
