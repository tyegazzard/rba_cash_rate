"""Tests for ``rba.data.gdp_release_calendar``.

The scrape path is not exercised over the live network. Instead, both regex
extractors are tested against hand-built HTML fragments mirroring the real
ABS pages (legacy ``/ausstats/`` and modern ``/statistics/`` namespaces),
and the CSV-loader path is verified against the materialised
``data/external/gdp_release_dates.csv`` checked into the repo.

Spot-check coverage targets known historical ABS GDP release dates verified
by hand from the ABS release pages — including the pre-2003 anomalies that
disqualified the algorithmic approach (non-Wednesday releases, 2-month
lags, non-first-Wednesday slots).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.gdp_release_calendar import (
    _CSV_PATH,
    _LEGACY_MONTH_SLUGS,
    _LEGACY_RELEASED_RE,
    _MODERN_MONTH_SLUGS,
    _MODERN_RELEASED_RE,
    _OVERRIDES,
    _build_url,
    _uses_modern_url,
    build_gdp_release_calendar,
    gdp_publication_date,
    quarter_end,
)

# -----------------------------------------------------------------------------
# Regex against representative HTML fragments
# -----------------------------------------------------------------------------

# Legacy ABS catalogue page (Cat. 5206.0, pre-2019). DC.Date.issued meta tag
# is the canonical original-release marker; the visible "Released at" field
# agrees on every sampled page.
_LEGACY_HTML_FRAGMENT = """
<head>
<META NAME="DC.Date.issued" SCHEME="ISO8601" CONTENT="2018-12-05">
<META NAME="DC.Date.modified" SCHEME="ISO8601" CONTENT="2019-02-01">
</head>
<body>
<p>Released at 11:30 AM (CANBERRA TIME) 05/12/2018&nbsp;&nbsp;</p>
</body>
"""

# Modern ABS statistics page (post-2019 redesign). dcterms.issued may reflect
# a later revision; the visible "Released" field is the original release.
_MODERN_HTML_FRAGMENT = """
<head>
<meta name="dcterms.issued" content="Wed, 04/09/2024 - 11:30" />
<meta name="dcterms.modified" content="Mon, 02/12/2024 - 11:30" />
</head>
<body>
<div id="release-date-section">
<div class="field field--name-dynamic-twig-fieldnode-release-or-orig-publish field--type-ds field--label-inline">
<div class="field__label">Released</div><div class="field__item"> 04/09/2024</div>
</div>
</div>
<div class="field field--name-field-abs-release-date">
<div class="field__label">Release date and time</div><div class="field__item">02/12/2024 11:30am AEDT</div>
</div>
</body>
"""


def test_legacy_released_regex_matches_dc_date_issued() -> None:
    match = _LEGACY_RELEASED_RE.search(_LEGACY_HTML_FRAGMENT)
    assert match is not None, "legacy regex failed to match DC.Date.issued"
    assert match.group(1) == "2018-12-05"


def test_legacy_released_regex_does_not_match_dc_date_modified() -> None:
    """The modified meta tag must not be mistaken for the issued date."""
    only_modified = '<META NAME="DC.Date.modified" SCHEME="ISO8601" CONTENT="2019-02-01">'
    assert _LEGACY_RELEASED_RE.search(only_modified) is None


def test_modern_released_regex_matches_original_not_revision() -> None:
    match = _MODERN_RELEASED_RE.search(_MODERN_HTML_FRAGMENT)
    assert match is not None, "modern regex failed to match the Released field"
    # Must be the ORIGINAL 04/09/2024, not the 02/12/2024 revision.
    assert match.group(1) == "04/09/2024"


def test_modern_released_regex_does_not_match_outside_field_label() -> None:
    """A stray DD/MM/YYYY elsewhere on the page must not match."""
    decoy = "<p>Some text mentioning 01/01/1999 unrelated to release.</p>"
    assert _MODERN_RELEASED_RE.search(decoy) is None


# -----------------------------------------------------------------------------
# URL builder + helpers
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "qm", "expect_modern"),
    [
        (1993, 3, False),
        (2010, 12, False),
        (2019, 3, False),  # boundary: 2019-Q1 uses legacy
        (2019, 6, True),  # boundary: 2019-Q2 uses modern
        (2020, 9, True),
        (2026, 3, True),
    ],
)
def test_uses_modern_url_boundary(year: int, qm: int, expect_modern: bool) -> None:
    assert _uses_modern_url(year, qm) is expect_modern


@pytest.mark.parametrize(
    ("year", "qm", "url_contains"),
    [
        (2024, 3, "/statistics/economy/national-accounts/"),
        (2024, 3, "mar-2024"),
        (2019, 6, "jun-2019"),
        (1995, 9, "/ausstats/abs@.nsf/PreviousProducts/"),
        (1995, 9, "Sep%201995"),
        (2018, 12, "Dec%202018"),
    ],
)
def test_build_url_contains_expected(year: int, qm: int, url_contains: str) -> None:
    url = _build_url(year, qm)
    assert url_contains in url, f"url={url!r} missing {url_contains!r}"


def test_legacy_month_slugs_cover_all_quarter_end_months() -> None:
    assert set(_LEGACY_MONTH_SLUGS) == {3, 6, 9, 12}


def test_modern_month_slugs_cover_all_quarter_end_months() -> None:
    assert set(_MODERN_MONTH_SLUGS) == {3, 6, 9, 12}


def test_month_slug_casing_matches_url_conventions() -> None:
    """Legacy slugs are title-case (`Mar`) because the page is keyed by
    'Main Features1Mar 2024'; modern slugs are lowercase (`mar`) because
    the URL path is all-lowercase."""
    for s in _LEGACY_MONTH_SLUGS.values():
        assert s[0].isupper() and s[1:].islower()
    for s in _MODERN_MONTH_SLUGS.values():
        assert s.islower()


@pytest.mark.parametrize(
    ("year", "qm", "expected"),
    [
        (1993, 3, date(1993, 3, 31)),
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
    """Sanity check that the CSV is checked in; the abs_gdp source depends
    on it for in-window observations."""
    assert _CSV_PATH.exists(), (
        f"Expected materialised CSV at {_CSV_PATH}. Run "
        "`python -m rba.data.gdp_release_calendar` to regenerate."
    )


def test_build_gdp_release_calendar_shape_and_dtypes() -> None:
    df = build_gdp_release_calendar()
    assert list(df.columns) == ["reference_quarter_end", "publication_date"]
    assert df["reference_quarter_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"
    # Sorted ascending by reference quarter end.
    assert df["reference_quarter_end"].is_monotonic_increasing


def test_build_gdp_release_calendar_covers_inflation_targeting_era() -> None:
    """Calendar must extend back to 1993-Q1 (scrape floor) and forward to
    at least the most recently released quarter."""
    df = build_gdp_release_calendar()
    assert df["reference_quarter_end"].min() == pd.Timestamp("1993-03-31")
    # Conservative bound: the last released quarter is at least 2024-Q4 by
    # the time these tests run.
    assert df["reference_quarter_end"].max() >= pd.Timestamp("2024-12-31")


@pytest.mark.parametrize(
    ("ref_qe", "expected_pub"),
    [
        # Pre-2003 anomalies — these are exactly why the algorithmic
        # "first Wednesday of M+3" rule was rejected:
        # 1993-Q1 -> 1 Jun 1993 (Tuesday, not Wednesday)
        (date(1993, 3, 31), date(1993, 6, 1)),
        # 1993-Q4 -> 16 Mar 1994 (3rd Wednesday, not 1st)
        (date(1993, 12, 31), date(1994, 3, 16)),
        # 1995-Q3 -> 29 Nov 1995 (2-month lag, not 3)
        (date(1995, 9, 30), date(1995, 11, 29)),
        # 2000-Q2 -> 13 Sep 2000 (2nd Wednesday, not 1st)
        (date(2000, 6, 30), date(2000, 9, 13)),
        # Post-2003 cleanly follows the rule:
        (date(2018, 12, 31), date(2019, 3, 6)),
        (date(2019, 9, 30), date(2019, 12, 4)),
        (date(2024, 3, 31), date(2024, 6, 5)),
        (date(2025, 12, 31), date(2026, 3, 4)),
    ],
)
def test_gdp_publication_date_known_quarters(ref_qe: date, expected_pub: date) -> None:
    assert gdp_publication_date(ref_qe) == expected_pub


def test_gdp_publication_date_returns_none_for_pre_floor_quarter() -> None:
    """Pre-1993-Q1 quarters are outside the scraped window."""
    assert gdp_publication_date(date(1985, 6, 30)) is None


def test_gdp_publication_date_returns_none_for_unreleased_future_quarter() -> None:
    """Future quarters that the scraper 404'd on are absent from the CSV."""
    assert gdp_publication_date(date(2099, 3, 31)) is None


# -----------------------------------------------------------------------------
# Overrides
# -----------------------------------------------------------------------------


def test_overrides_take_precedence_over_csv() -> None:
    override_key = date(2024, 3, 31)
    forced = date(2099, 1, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert gdp_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_extend_calendar_for_missing_quarter() -> None:
    """An override for a pre-floor quarter materialises a row in the
    calendar DataFrame."""
    qe = date(1985, 6, 30)
    forced = date(1985, 9, 4)
    _OVERRIDES[qe] = forced
    try:
        df = build_gdp_release_calendar()
        hit = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "publication_date"]
        assert not hit.empty
        assert hit.iloc[0] == pd.Timestamp(forced)
    finally:
        del _OVERRIDES[qe]
