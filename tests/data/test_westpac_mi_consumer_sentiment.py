"""Tests for ``rba.data.sources.westpac_mi_consumer_sentiment``.

The live ``fetch()`` path hits the RBA endpoint and is not exercised
here. ``_parse``, ``_attach_publication_dates``, and ``_to_wide`` are
tested against a synthetic H3 CSV mirroring the real layout (5
columns, blank rows between Units and Source, ``Publication date`` row,
``Series ID`` row, then DD/MM/YYYY data rows).
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.sources.westpac_mi_consumer_sentiment import (
    _CALENDAR_VALIDATION_FLOOR,
    SERIES,
    RbaH3Series,
    _attach_publication_dates,
    _parse,
    _to_month_end,
    _to_wide,
)

# Synthetic H3 CSV — five columns matching the live H3 layout. Date rows
# span pre-1993, in-window, and a NaN row to exercise drop-null.
_FIXTURE_CSV = (
    "H3 MONTHLY ACTIVITY INDICATORS\n"
    "Title,Private dwelling approvals,Private dwelling approvals trend,"
    "Private non-residential building approvals,Consumer sentiment,"
    "Business conditions\n"
    "Description,Private dwelling approvals,Private dwelling approvals; Trend,"
    "Private non-residential building approvals; Current price,"
    "Westpac-Melbourne Institute consumer sentiment index,"
    "NAB business conditions index deviation from average\n"
    "Frequency,Monthly,Monthly,Monthly,Monthly,Monthly\n"
    "Type,Seasonally adjusted,Trend,Original,Seasonally adjusted,Seasonally adjusted\n"
    "Units,'000,'000,$ million,Index,Percentage points\n"
    "\n"
    "\n"
    "Source,ABS,ABS,ABS,MI,NAB\n"
    "Publication date,05-May-2026,05-May-2026,05-May-2026,20-May-2026,13-May-2026\n"
    "Series ID,GISPSDA,GISDWPRITR,GISPSNBA,GICWMICS,GICNBC\n"
    "31/01/1990,8.0,8.1,100.0,,\n"
    "31/01/2010,9.5,9.4,250.0,120.1,5.0\n"
    "30/04/2026,10.0,10.0,300.0,80.1,-0.7\n"
    "31/05/2026,,,,83.0,\n"
    "30/06/2026,,,,,\n"
).encode("utf-8")


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_extracts_gicwmics() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert list(df.columns) == ["observation_date", "series_id", "value"]
    assert df["observation_date"].dtype == "datetime64[ns]"
    assert df["value"].dtype == "float64"
    # 3 non-null rows: 2010-01, 2026-04, 2026-05
    assert len(df) == 3


def test_parse_stamps_logical_series_id() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    df = _parse(_FIXTURE_CSV, spec=spec)
    assert (df["series_id"] == "consumer_sentiment").all()


def test_parse_drops_null_observations() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    df = _parse(_FIXTURE_CSV, spec=spec)
    # 1990-01 and 2026-06 rows have null sentiment values — must be excluded.
    obs = df["observation_date"].tolist()
    assert pd.Timestamp("1990-01-31") not in obs
    assert pd.Timestamp("2026-06-30") not in obs


def test_parse_raises_on_missing_rba_series_id() -> None:
    spec = RbaH3Series(series_id="bogus", rba_series_id="DOES_NOT_EXIST")
    with pytest.raises(ValueError, match="not found among"):
        _parse(_FIXTURE_CSV, spec=spec)


def test_parse_raises_when_series_id_row_missing() -> None:
    csv = b"H3 TEST\nTitle,Series Q\n31/01/2024,1.0\n"
    spec = RbaH3Series(series_id="q", rba_series_id="BHFQ")
    with pytest.raises(ValueError, match="No 'Series ID' metadata row"):
        _parse(csv, spec=spec)


# -----------------------------------------------------------------------------
# _to_month_end
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_ts", "expected"),
    [
        (pd.Timestamp("1990-01-31"), pd.Timestamp("1990-01-31")),
        (pd.Timestamp("2024-02-29"), pd.Timestamp("2024-02-29")),
        (pd.Timestamp("2026-05-31"), pd.Timestamp("2026-05-31")),
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


def test_attach_publication_dates_uses_algorithmic_rule() -> None:
    """2026-05 → 2nd Wed of May 2026 = 2026-05-13."""
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    row = out.loc[out["observation_date"] == pd.Timestamp("2026-05-31")].iloc[0]
    assert row["publication_date"] == pd.Timestamp("2026-05-13")


def test_attach_publication_dates_precedes_month_end_for_in_window() -> None:
    """For Westpac-MI publication < reference_month_end for every in-window row."""
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    in_window = out["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    assert (out.loc[in_window, "publication_date"] < out.loc[in_window, "observation_date"]).all()


def test_attach_publication_dates_drops_stale_column() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    parsed = _parse(_FIXTURE_CSV, spec=spec).copy()
    parsed["publication_date"] = pd.Timestamp("2099-01-01")
    out = _attach_publication_dates(parsed)
    assert (out["publication_date"] != pd.Timestamp("2099-01-01")).all()


def test_attach_publication_dates_output_columns() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    parsed = _parse(_FIXTURE_CSV, spec=spec)
    out = _attach_publication_dates(parsed)
    assert list(out.columns) == [
        "observation_date",
        "publication_date",
        "series_id",
        "value",
    ]


# -----------------------------------------------------------------------------
# _to_wide
# -----------------------------------------------------------------------------


def test_to_wide_schema_and_dtype() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    long_df = _attach_publication_dates(_parse(_FIXTURE_CSV, spec=spec))
    wide = _to_wide(long_df)
    assert list(wide.columns) == [
        "reference_month_end",
        "reference_label",
        "consumer_sentiment",
        "release_date",
    ]
    assert wide["reference_month_end"].dtype == "datetime64[ns]"
    assert wide["release_date"].dtype == "datetime64[ns]"


def test_to_wide_reference_label_format() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    long_df = _attach_publication_dates(_parse(_FIXTURE_CSV, spec=spec))
    wide = _to_wide(long_df)
    label = wide.loc[
        wide["reference_month_end"] == pd.Timestamp("2026-05-31"),
        "reference_label",
    ].iloc[0]
    assert label == "May 2026"


# -----------------------------------------------------------------------------
# SERIES registry
# -----------------------------------------------------------------------------


def test_series_registry() -> None:
    ids = {s.series_id for s in SERIES}
    assert ids == {"consumer_sentiment"}
    by_id = {s.series_id: s.rba_series_id for s in SERIES}
    assert by_id["consumer_sentiment"] == "GICWMICS"


# -----------------------------------------------------------------------------
# Known-value sanity (hand-verified from the live H3 CSV on 2026-05-24)
# -----------------------------------------------------------------------------


def test_known_values_match_synthetic_fixture() -> None:
    spec = RbaH3Series(series_id="consumer_sentiment", rba_series_id="GICWMICS")
    df = _parse(_FIXTURE_CSV, spec=spec)
    by_date = dict(zip(df["observation_date"], df["value"]))
    assert by_date[pd.Timestamp("2010-01-31")] == pytest.approx(120.1)
    assert by_date[pd.Timestamp("2026-04-30")] == pytest.approx(80.1)
    assert by_date[pd.Timestamp("2026-05-31")] == pytest.approx(83.0)
