"""Tests for ``rba.data.wpi_release_calendar``.

The scrape path is not exercised over the live network. Instead, the regex
``_RELEASED_RE`` is tested against a hand-built HTML fragment mirroring the
real ABS page, and the CSV-loader path is verified against the materialised
``data/external/wpi_release_dates.csv`` checked into the repo.

Spot-check coverage targets known historical ABS WPI release dates verified
by hand from the ABS release pages.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.wpi_release_calendar import (
    _CSV_PATH,
    _MONTH_SLUGS,
    _OVERRIDES,
    _RELEASED_RE,
    _build_url,
    build_wpi_release_calendar,
    quarter_end,
    wpi_publication_date,
)

# -----------------------------------------------------------------------------
# Regex against a representative HTML fragment
# -----------------------------------------------------------------------------

# Mirrors the visible "Released" block on https://www.abs.gov.au/.../mar-2024 —
# also includes the dcterms.issued meta tag (revision date) and the
# "Release date and time" field (also a revision date) so we verify the
# regex selects the original publication, not either revision.
_HTML_FRAGMENT = """
<head>
<meta name="dcterms.issued" content="Mon, 02/11/2020 - 11:30" />
<meta name="dcterms.modified" content="Mon, 02/11/2020 - 11:30" />
</head>
<body>
<div id="release-date-section">
<div class="field field--name-dynamic-twig-fieldnode-release-or-orig-publish field--type-ds field--label-inline">
<div class="field__label">Released</div><div class="field__item"> 12/08/2020</div>
</div>
</div>
<div class="field field--name-field-abs-release-date">
<div class="field__label">Release date and time</div><div class="field__item">02/11/2020 11:30am AEDT</div>
</div>
</body>
"""


def test_released_regex_matches_original_not_revision() -> None:
    match = _RELEASED_RE.search(_HTML_FRAGMENT)
    assert match is not None, "regex failed to match the Released field"
    # Must be the ORIGINAL 12 Aug 2020, not the 02 Nov 2020 revision dates.
    assert match.group(1) == "12/08/2020"


def test_released_regex_does_not_match_outside_field_label() -> None:
    """A stray DD/MM/YYYY elsewhere on the page must not match."""
    decoy = "<p>Some text mentioning 01/01/1999 unrelated to release.</p>"
    assert _RELEASED_RE.search(decoy) is None


# -----------------------------------------------------------------------------
# URL builder + helpers
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "qm", "expected_slug"),
    [
        (2024, 3, "mar-2024"),
        (2020, 6, "jun-2020"),
        (2019, 9, "sep-2019"),
        (2023, 12, "dec-2023"),
    ],
)
def test_build_url_slug(year: int, qm: int, expected_slug: str) -> None:
    url = _build_url(year, qm)
    assert url.endswith(f"/{expected_slug}")
    assert "wage-price-index-australia" in url


def test_month_slugs_cover_all_quarter_end_months() -> None:
    assert set(_MONTH_SLUGS) == {3, 6, 9, 12}


@pytest.mark.parametrize(
    ("year", "qm", "expected"),
    [
        (2024, 3, date(2024, 3, 31)),
        (2020, 6, date(2020, 6, 30)),
        (2019, 9, date(2019, 9, 30)),
        (2023, 12, date(2023, 12, 31)),
    ],
)
def test_quarter_end(year: int, qm: int, expected: date) -> None:
    assert quarter_end(year, qm) == expected


def test_quarter_end_rejects_non_quarter_month() -> None:
    with pytest.raises(ValueError, match="quarter_month"):
        quarter_end(2024, 4)


# -----------------------------------------------------------------------------
# Materialised CSV loader
# -----------------------------------------------------------------------------


def test_csv_is_materialised_in_repo() -> None:
    """Sanity check that the CSV is checked in; the abs_wpi source depends
    on it for in-window observations."""
    assert _CSV_PATH.exists(), (
        f"Expected materialised CSV at {_CSV_PATH}. Run "
        "`python -m rba.data.wpi_release_calendar` to regenerate."
    )


def test_build_wpi_release_calendar_shape_and_dtypes() -> None:
    df = build_wpi_release_calendar()
    assert list(df.columns) == ["reference_quarter_end", "publication_date"]
    assert df["reference_quarter_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"
    # Sorted ascending by reference quarter end.
    assert df["reference_quarter_end"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("ref_qe", "expected_pub"),
    [
        # Original Aug 2020 release (NOT the Nov 2 revision shown by dcterms.issued)
        (date(2020, 6, 30), date(2020, 8, 12)),
        # 3rd Wednesday of May 2024 — most-recent stable quarter pre-test
        (date(2024, 3, 31), date(2024, 5, 15)),
        # Tuesday release — the calendar must preserve non-Wednesday dates
        (date(2024, 6, 30), date(2024, 8, 13)),
        # Sep 2019 — earliest scraped quarter
        (date(2019, 9, 30), date(2019, 11, 13)),
    ],
)
def test_wpi_publication_date_known_quarters(
    ref_qe: date, expected_pub: date
) -> None:
    assert wpi_publication_date(ref_qe) == expected_pub


def test_wpi_publication_date_returns_none_for_pre_floor_quarter() -> None:
    """Pre-2019-Q3 quarters are outside the scraped window."""
    assert wpi_publication_date(date(2015, 3, 31)) is None


def test_wpi_publication_date_returns_none_for_unscraped_2019_q1() -> None:
    """2019-Q1 page returns 404 from the post-redesign ABS URL pattern."""
    assert wpi_publication_date(date(2019, 3, 31)) is None


# -----------------------------------------------------------------------------
# Overrides
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_csv() -> None:
    override_key = date(2024, 3, 31)
    forced = date(2099, 1, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert wpi_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_extend_calendar_for_missing_quarter() -> None:
    """An override for a pre-floor quarter materialises a row in the
    calendar DataFrame."""
    qe = date(2018, 12, 31)
    forced = date(2019, 2, 20)
    _OVERRIDES[qe] = forced
    try:
        df = build_wpi_release_calendar()
        hit = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "publication_date"]
        assert not hit.empty
        assert hit.iloc[0] == pd.Timestamp(forced)
    finally:
        del _OVERRIDES[qe]
