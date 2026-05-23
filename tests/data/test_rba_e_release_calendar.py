"""Tests for ``rba.data.rba_e_release_calendar``.

The harvest path (``_harvest``) is not exercised here — it hits live
Wayback + RBA endpoints. The parser ``_parse_e2_csv`` is tested against
synthetic CSV fixtures that mirror both historical E2 layouts (modern
``DD/MM/YYYY`` and legacy ``Mon-YYYY``). The CSV-loader path is verified
against the materialised ``data/external/rba_e_release_dates.csv`` checked
into the repo.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.rba_e_release_calendar import (
    _CSV_PATH,
    _OVERRIDES,
    _parse_e2_csv,
    build_rba_e_release_calendar,
    rba_e_publication_date,
)

# -----------------------------------------------------------------------------
# _parse_e2_csv — modern layout (DD/MM/YYYY)
# -----------------------------------------------------------------------------

_MODERN_FIXTURE = (
    "E2 HOUSEHOLD FINANCES – SELECTED RATIOS\n"
    "Title,Household debt to income,Housing debt to income\n"
    "Description,Foo,Bar\n"
    "Frequency,Quarterly,Quarterly\n"
    "Type,Original,Original\n"
    "Units,Per cent,Per cent\n"
    "\n"
    "\n"
    "Source,ABS,ABS\n"
    "Publication date,27-Mar-2026,27-Mar-2026\n"
    "Series ID,BHFDDIT,BHFDDIH\n"
    "30/09/2025,176.2,132.7\n"
    "31/12/2025,177.0,133.7\n"
    "31/03/2026,,\n"
).encode("utf-8")


def test_parse_e2_modern_returns_pub_and_last_data_quarter() -> None:
    pub, qe = _parse_e2_csv(_MODERN_FIXTURE)
    assert pub == pd.Timestamp("2026-03-27")
    # Last row with any non-null value across series columns is 2025-12-31.
    assert qe == pd.Timestamp("2025-12-31")


# -----------------------------------------------------------------------------
# _parse_e2_csv — legacy layout (Mon-YYYY)
# -----------------------------------------------------------------------------

_LEGACY_FIXTURE = (
    "E2 HOUSEHOLD FINANCES – SELECTED RATIOS\n"
    "Title,Household debt to income,Housing debt to income\n"
    "Description,Foo,Bar\n"
    "Frequency,Quarterly,Quarterly\n"
    "Type,Original,Original\n"
    "Units,Per cent,Per cent\n"
    "\n"
    "\n"
    "Source,ABS,ABS\n"
    "Publication date,27-Mar-2015,27-Mar-2015\n"
    "Series ID,BHFDDIT,BHFDDIH\n"
    "Sep-2014,151.9,138.1\n"
    "Dec-2014,153.8,140.3\n"
    "Mar-2015,,\n"
).encode("utf-8")


def test_parse_e2_legacy_format() -> None:
    """Wayback snapshots from 2015-2018 use ``Mon-YYYY``; must still parse."""
    pub, qe = _parse_e2_csv(_LEGACY_FIXTURE)
    assert pub == pd.Timestamp("2015-03-27")
    # Last row with data is Dec 2014 -> end of month is 2014-12-31.
    assert qe == pd.Timestamp("2014-12-31")


def test_parse_e2_handles_latin1_encoded_bytes() -> None:
    """Some Wayback snapshots ship with non-utf8 bytes (e.g. en-dash); the
    parser falls back to latin-1 so they still parse."""
    fixture = _MODERN_FIXTURE.replace(b"\xe2\x80\x93", b"\x96")  # en-dash -> latin1
    pub, qe = _parse_e2_csv(fixture)
    assert pub == pd.Timestamp("2026-03-27")
    assert qe == pd.Timestamp("2025-12-31")


def test_parse_e2_returns_nones_when_no_data() -> None:
    """A CSV with only metadata rows yields (pub, None)."""
    no_data = ("E2 HOUSEHOLD FINANCES – SELECTED RATIOS\nTitle,A\nSeries ID,X\n").encode("utf-8")
    pub, qe = _parse_e2_csv(no_data)
    assert pub is None
    assert qe is None


# -----------------------------------------------------------------------------
# Materialised CSV loader
# -----------------------------------------------------------------------------


def test_csv_is_materialised_in_repo() -> None:
    """The CSV must be checked in; the rba_e2 source depends on it for
    scraped publication dates."""
    assert _CSV_PATH.exists(), (
        f"Expected materialised CSV at {_CSV_PATH}. Run "
        "`python -m rba.data.rba_e_release_calendar` to regenerate."
    )


def test_build_rba_e_release_calendar_shape_and_dtypes() -> None:
    df = build_rba_e_release_calendar()
    assert list(df.columns) == [
        "reference_quarter_end",
        "publication_date",
        "source",
    ]
    assert df["reference_quarter_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"
    assert df["reference_quarter_end"].is_monotonic_increasing
    # Sources must come from the documented enum.
    assert set(df["source"].unique()).issubset({"rba_page", "archive_org", "inferred"})


def test_build_rba_e_release_calendar_unique_reference_quarter() -> None:
    """No duplicate (reference_quarter_end) entries — dedupe should keep
    one row per quarter, preferring rba_page over archive_org."""
    df = build_rba_e_release_calendar()
    assert df["reference_quarter_end"].is_unique


@pytest.mark.parametrize(
    ("ref_qe", "expected_pub"),
    [
        # Earliest scraped quarter.
        (date(2014, 12, 31), date(2015, 3, 27)),
        # Christmas-shifted release (would have been Dec 25 algorithmically).
        (date(2015, 9, 30), date(2015, 12, 18)),
        # Easter-shifted release (Mar 25 2016 was Good Friday).
        (date(2015, 12, 31), date(2016, 4, 1)),
        # Most recent quarter (rba_page source from live CSV).
        (date(2025, 12, 31), date(2026, 3, 27)),
    ],
)
def test_rba_e_publication_date_known_quarters(ref_qe: date, expected_pub: date) -> None:
    assert rba_e_publication_date(ref_qe) == expected_pub


def test_rba_e_publication_date_returns_none_for_pre_floor_quarter() -> None:
    """Pre-2014-Q4 quarters are outside the scraped window."""
    assert rba_e_publication_date(date(2010, 3, 31)) is None


def test_rba_e_publication_date_returns_none_for_wayback_gap_quarter() -> None:
    """Some post-floor quarters fall in Wayback gaps — they have no calendar
    entry and the source module fills via flat offset."""
    # 2018-Q4 is in a Wayback gap (the harvest jumps from 2018-Q3 to 2019-Q4).
    assert rba_e_publication_date(date(2018, 12, 31)) is None


def test_rba_page_takes_precedence_over_archive_org() -> None:
    """The live CSV header is authoritative for the latest release; rba_page
    rows should not be displaced by older archive_org snapshots of the
    same quarter."""
    df = build_rba_e_release_calendar()
    latest = df.iloc[-1]
    # In the materialised CSV the last row should be the rba_page entry.
    assert latest["source"] == "rba_page"


# -----------------------------------------------------------------------------
# Overrides
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_csv() -> None:
    override_key = date(2025, 12, 31)
    forced = date(2099, 1, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert rba_e_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_extend_calendar_for_missing_quarter() -> None:
    """An override for a quarter not in the CSV materialises a row."""
    qe = date(1995, 6, 30)
    forced = date(1995, 9, 29)
    _OVERRIDES[qe] = forced
    try:
        df = build_rba_e_release_calendar()
        hit = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "publication_date"]
        assert not hit.empty
        assert hit.iloc[0] == pd.Timestamp(forced)
        src = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "source"].iloc[0]
        assert src == "inferred"
    finally:
        del _OVERRIDES[qe]
