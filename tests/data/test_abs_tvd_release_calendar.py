"""Tests for ``rba.data.abs_tvd_release_calendar``.

The scrape path is not exercised over the live network. The CSV-loader
path is verified against the materialised
``data/external/abs_tvd_release_dates.csv``; spot-checks target the
known sampled releases used in the design probe.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rba.data.abs_tvd_release_calendar import (
    _CSV_PATH,
    _MONTH_SLUGS,
    _OVERRIDES,
    _RELEASED_RE,
    _build_url,
    build_tvd_release_calendar,
    quarter_end,
    tvd_publication_date,
)

_HTML_FRAGMENT = """
<head>
<meta name="dcterms.issued" content="Mon, 02/11/2025 - 11:30" />
</head>
<body>
<div class="field field--name-dynamic-twig-fieldnode-release-or-orig-publish">
<div class="field__label">Released</div><div class="field__item"> 10/03/2026</div>
</div>
</body>
"""


def test_released_regex_matches_visible_field() -> None:
    match = _RELEASED_RE.search(_HTML_FRAGMENT)
    assert match is not None
    assert match.group(1) == "10/03/2026"


def test_released_regex_does_not_match_arbitrary_date() -> None:
    decoy = "<p>The 02/11/2025 revision rewrote the dataset.</p>"
    assert _RELEASED_RE.search(decoy) is None


@pytest.mark.parametrize(
    ("year", "qm", "expected_slug"),
    [
        (2025, 12, "dec-quarter-2025"),
        (2022, 3, "mar-quarter-2022"),
        (2024, 6, "jun-quarter-2024"),
        (2023, 9, "sep-quarter-2023"),
    ],
)
def test_build_url_slug(year: int, qm: int, expected_slug: str) -> None:
    url = _build_url(year, qm)
    assert url.endswith(f"/{expected_slug}")
    assert "total-value-dwellings" in url


def test_month_slugs_cover_only_quarter_end_months() -> None:
    assert set(_MONTH_SLUGS) == {3, 6, 9, 12}


@pytest.mark.parametrize(
    ("year", "qm", "expected"),
    [
        (2025, 12, date(2025, 12, 31)),
        (2022, 3, date(2022, 3, 31)),
        (2024, 6, date(2024, 6, 30)),
        (2023, 9, date(2023, 9, 30)),
    ],
)
def test_quarter_end(year: int, qm: int, expected: date) -> None:
    assert quarter_end(year, qm) == expected


def test_quarter_end_rejects_non_quarter_month() -> None:
    with pytest.raises(ValueError, match="quarter_month"):
        quarter_end(2024, 5)


def test_csv_is_materialised_in_repo() -> None:
    assert _CSV_PATH.exists(), (
        f"Expected materialised CSV at {_CSV_PATH}. Run "
        "`python -m rba.data.abs_tvd_release_calendar` to regenerate."
    )


def test_build_tvd_release_calendar_shape_and_dtypes() -> None:
    df = build_tvd_release_calendar()
    assert list(df.columns) == ["reference_quarter_end", "publication_date", "source"]
    assert df["reference_quarter_end"].dtype == "datetime64[ns]"
    assert df["publication_date"].dtype == "datetime64[ns]"
    assert df["source"].dtype == object
    assert df["reference_quarter_end"].is_monotonic_increasing


def test_source_values_are_in_enum() -> None:
    df = build_tvd_release_calendar()
    valid = {"abs_page", "archive_org", "inferred"}
    extras = set(df["source"].unique()) - valid
    assert not extras, f"unexpected source values: {extras}"


@pytest.mark.parametrize(
    ("ref_qe", "expected_pub"),
    [
        (date(2025, 12, 31), date(2026, 3, 10)),  # 2nd Tuesday of Mar
        (date(2025, 9, 30), date(2025, 12, 2)),   # 1st Tuesday of Dec
        (date(2024, 12, 31), date(2025, 3, 11)),  # 2nd Tuesday of Mar
        (date(2023, 9, 30), date(2023, 12, 5)),   # 1st Tuesday of Dec
        (date(2022, 3, 31), date(2022, 6, 14)),   # 2nd Tuesday of Jun — earliest scraped
    ],
)
def test_tvd_publication_date_known_quarters(
    ref_qe: date, expected_pub: date
) -> None:
    assert tvd_publication_date(ref_qe) == expected_pub


def test_tvd_publication_date_returns_none_for_pre_floor_quarter() -> None:
    """RPPI-era quarters (≤ 2021-Q4) are outside the TVD scrape window."""
    assert tvd_publication_date(date(2021, 12, 31)) is None
    assert tvd_publication_date(date(2015, 3, 31)) is None


def test_overrides_take_precedence_over_csv() -> None:
    override_key = date(2025, 12, 31)
    forced = date(2099, 1, 1)
    _OVERRIDES[override_key] = forced
    try:
        assert tvd_publication_date(override_key) == forced
    finally:
        del _OVERRIDES[override_key]


def test_overrides_extend_calendar_for_pre_floor_quarter() -> None:
    qe = date(2018, 12, 31)
    forced = date(2019, 3, 12)
    _OVERRIDES[qe] = forced
    try:
        df = build_tvd_release_calendar()
        hit = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "publication_date"]
        assert not hit.empty
        assert hit.iloc[0] == pd.Timestamp(forced)
        src = df.loc[df["reference_quarter_end"] == pd.Timestamp(qe), "source"].iloc[0]
        assert src == "inferred"
    finally:
        del _OVERRIDES[qe]
