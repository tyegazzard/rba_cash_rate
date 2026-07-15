"""Tests for ``rba.data.abs_ba_release_calendar``.

The scrape path is not exercised over the live network. Instead, the
regex ``_RELEASED_RE`` is tested against a hand-built HTML fragment
mirroring the real ABS page, and the CSV-loader path is verified against
the materialised ``data/external/abs_ba_release_dates.csv`` checked into
the repo.

Spot-check coverage targets known historical ABS BA release dates
verified by hand from the ABS release pages during the design probe.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.abs_ba_release_calendar import (
    _CSV_PATH,
    _MONTH_SLUGS,
    _OVERRIDES,
    _RELEASED_RE,
    _build_url,
    ba_publication_date,
    build_ba_release_calendar,
    month_end,
)

# Mirrors the visible "Released" block on a modern ABS BA page. Also
# includes the dcterms.issued meta tag (the revision date) so we verify
# the regex selects the original publication, not the revision.
_HTML_FRAGMENT = """
<head>
<meta name="dcterms.issued" content="Mon, 02/11/2025 - 11:30" />
<meta name="dcterms.modified" content="Mon, 02/11/2025 - 11:30" />
</head>
<body>
<div id="release-date-section">
<div class="field field--name-dynamic-twig-fieldnode-release-or-orig-publish field--type-ds field--label-inline">
<div class="field__label">Released</div><div class="field__item"> 04/04/2024</div>
</div>
</div>
<div class="field field--name-field-abs-release-date">
<div class="field__label">Release date and time</div><div class="field__item">02/11/2025 11:30am AEDT</div>
</div>
</body>
"""


def test_released_regex_matches_original_not_revision() -> None:
    match = _RELEASED_RE.search(_HTML_FRAGMENT)
    assert match is not None, "regex failed to match the Released field"
    assert match.group(1) == "04/04/2024"


def test_released_regex_does_not_match_outside_field_label() -> None:
    decoy = "<p>Some text mentioning 01/01/1999 unrelated to release.</p>"
    assert _RELEASED_RE.search(decoy) is None


@pytest.mark.parametrize(
    ("year", "month", "expected_slug"),
    [
        (2024, 3, "mar-2024"),
        (2020, 6, "jun-2020"),
        (2019, 12, "dec-2019"),
        (2026, 1, "jan-2026"),
    ],
)
def test_build_url_slug(year: int, month: int, expected_slug: str) -> None:
    url = _build_url(year, month)
    assert url.endswith(f"/{expected_slug}")
    assert "building-approvals-australia" in url


def test_month_slugs_cover_all_months() -> None:
    assert set(_MONTH_SLUGS) == set(range(1, 13))


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2024, 2, date(2024, 2, 29)),  # leap Feb
        (2025, 2, date(2025, 2, 28)),  # non-leap Feb
        (2026, 1, date(2026, 1, 31)),
        (2024, 4, date(2024, 4, 30)),
    ],
)
def test_month_end(year: int, month: int, expected: date) -> None:
    assert month_end(year, month) == expected


def test_csv_is_materialised_in_repo() -> None:
    """abs_building_approvals depends on the CSV for in-window dates."""
    assert _CSV_PATH.exists(), (
        f"Expected materialised CSV at {_CSV_PATH}. Run "
        "`python -m rba.data.abs_ba_release_calendar` to regenerate."
    )


def test_build_ba_release_calendar_shape_and_dtypes() -> None:
    df = build_ba_release_calendar()
    assert list(df.columns) == ["reference_month_end", "publication_date", "source"]
    assert df["reference_month_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"
    assert df["source"].dtype == object
    assert df["reference_month_end"].is_monotonic_increasing


def test_build_ba_release_calendar_source_values_are_in_enum() -> None:
    df = build_ba_release_calendar()
    valid = {"abs_page", "archive_org", "inferred"}
    extras = set(df["source"].unique()) - valid
    assert not extras, f"unexpected source values: {extras}"


@pytest.mark.parametrize(
    ("ref_me", "expected_pub"),
    [
        (date(2026, 3, 31), date(2026, 5, 4)),  # Mon — Mar 2026 ref
        (date(2026, 2, 28), date(2026, 4, 1)),  # Wed — Feb 2026 ref
        (date(2025, 12, 31), date(2026, 2, 3)),  # Tue — Dec 2025 ref
        (date(2024, 2, 29), date(2024, 4, 4)),  # Thu — Feb 2024 ref (Easter year)
        (date(2023, 3, 31), date(2023, 5, 8)),  # Mon — 8th of M+2 (2nd Mon of May)
        (date(2019, 12, 31), date(2020, 2, 3)),  # earliest scraped month
    ],
)
def test_ba_publication_date_known_months(ref_me: date, expected_pub: date) -> None:
    assert ba_publication_date(ref_me) == expected_pub


def test_ba_publication_date_returns_none_for_pre_floor_month() -> None:
    """Pre-Dec-2019 reference months are outside the scraped window."""
    assert ba_publication_date(date(2015, 6, 30)) is None


def test_overrides_take_precedence_over_csv() -> None:
    override_key = date(2024, 2, 29)
    forced = date(2099, 1, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert ba_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_extend_calendar_for_missing_month() -> None:
    me = date(2010, 6, 30)
    forced = date(2010, 8, 4)
    _OVERRIDES[me] = forced
    try:
        df = build_ba_release_calendar()
        hit = df.loc[df["reference_month_end"] == pd.Timestamp(me), "publication_date"]
        assert not hit.empty
        assert hit.iloc[0] == pd.Timestamp(forced)
        src = df.loc[df["reference_month_end"] == pd.Timestamp(me), "source"].iloc[0]
        assert src == "inferred"
    finally:
        del _OVERRIDES[me]
