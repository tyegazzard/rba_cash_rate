"""Tests for ``rba.data.sources.rba_media_releases``.

The live ``fetch()`` path hits the RBA over HTTPS and is not exercised
here. ``_parse``, ``_extract_governor``, ``_extract_paragraphs``,
``_select_targets``, ``_validate_cross_check``, and ``_normalise_whitespace``
are tested in isolation against synthetic HTML fixtures that mirror the three
title eras (1990s-2010 Fraser/Macfarlane/early-Stevens, 2010-Jan-2024
Stevens/Lowe, Feb-2024+ Reserve Bank Board) and the two body-container
variants (modern ``<div class="rss-mr-content">`` vs older
``<p>`` direct children).
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag
import pandas as pd
import pytest

from rba.data.sources.rba_media_releases import (
    _CALENDAR_VALIDATION_FLOOR,
    _extract_governor,
    _extract_paragraphs,
    _normalise_whitespace,
    _parse,
    _select_targets,
    _validate_cross_check,
)


def _content_div(soup: BeautifulSoup) -> Tag:
    """The ``<div id='content'>`` node, narrowed from ``Tag | None``.

    The fixtures always contain it, so a miss is a broken fixture rather than a
    real data case — the assert documents that invariant for the type checker.
    """
    div = soup.find("div", id="content")
    assert div is not None, "fixture missing <div id='content'>"
    return div


# -----------------------------------------------------------------------------
# Synthetic HTML fixtures — one per era. Mirror the real RBA layout closely
# enough that the parser exercises the same code path it would in production:
# the <div id="content"> wrapper, the itemprop=headline title span, the
# itemprop=datePublished date span, the box-article-info / box-enquiries side-
# blocks (must be excluded from body), and the era-specific body container
# (rss-mr-content vs <section>).
# -----------------------------------------------------------------------------


def _fixture_modern_lowe() -> bytes:
    """2020 Lowe-era release: rss-mr-content body wrapper, contact box."""
    return b"""
<html><body>
<div id="content" tabindex="-1" role="main" class="column-content content-style">
  <div itemscope itemtype="https://schema.org/PublicationIssue">
    <h1 class="page-title">
      <span class="page-title-prefix">Media Release</span>
      <span class="rss-mr-title" itemprop="headline">Statement by Philip Lowe, Governor: Monetary Policy Decision</span>
    </h1>
    <div class="box-article-info article-data">
      <div class="item">
        <span class="key">Number</span>
        <span class="value" itemprop="issueNumber">2020-08</span>
      </div>
      <div class="item">
        <span class="key">Date</span>
        <span class="value" itemprop="datePublished">19 March 2020</span>
      </div>
    </div>
    <div class="rss-mr-content">
      <p>The coronavirus is first and foremost a public health issue. It is also having a very major impact on the economy.</p>
      <p>At an out-of-cycle meeting today, the Board decided to lower the cash rate by 25 basis points to 0.25 per cent.</p>
      <p>There will be a press conference with further details at 4.00pm AEDT today.</p>
    </div>
    <aside>
      <div class="nav-page-contents aside-component monetary-policy-related-links">
        <p>Related: Statement on Monetary Policy</p>
      </div>
    </aside>
    <section>
      <div class="box-enquiries">
        <div class="item">
          <p>Media and Communications</p>
          <p>+61 2 9551 8111</p>
        </div>
      </div>
    </section>
  </div>
</div>
</body></html>
"""


def _fixture_modern_board() -> bytes:
    """Feb-2024+ Reserve-Bank-Board era release (post-governance-reform)."""
    return b"""
<html><body>
<div id="content" tabindex="-1" role="main" class="column-content content-style">
  <div itemscope itemtype="https://schema.org/PublicationIssue">
    <h1 class="page-title">
      <span class="page-title-prefix">Media Release</span>
      <span class="rss-mr-title" itemprop="headline">Statement by the Reserve Bank Board: Monetary Policy Decision</span>
    </h1>
    <div class="box-article-info article-data">
      <div class="item"><span class="key">Number</span><span class="value">2024-01</span></div>
      <div class="item"><span class="key">Date</span><span class="value" itemprop="datePublished">6 February 2024</span></div>
    </div>
    <div class="rss-mr-content">
      <p>At its meeting today, the Board decided to leave the cash rate target unchanged at 4.35&nbsp;per&nbsp;cent.</p>
      <p>While recent data indicate that inflation is easing, it remains high.</p>
    </div>
    <section>
      <div class="box-enquiries"><p>Phone: +61 2 9551 8111</p></div>
    </section>
  </div>
</div>
</body></html>
"""


def _fixture_legacy_fraser() -> bytes:
    """1990s Fraser-era release: no rss-mr-content wrapper; <p> directly under
    a sibling div inside #content. Contact box lives under <section>."""
    return b"""
<html><body>
<div id="content" tabindex="-1" role="main" class="column-content content-style">
  <div itemscope itemtype="https://schema.org/PublicationIssue">
    <h1 class="page-title">
      <span class="page-title-prefix">Media Release</span>
      <span class="rss-mr-title" itemprop="headline">Statement by the Governor, Mr Bernie Fraser: Reduction in Interest Rates - July 1996</span>
    </h1>
    <div class="box-article-info article-data">
      <div class="item"><span class="key">Number</span><span class="value">96-09</span></div>
      <div class="item"><span class="key">Date</span><span class="value" itemprop="datePublished">31 July 1996</span></div>
    </div>
    <div>
      <p>Following on from the improvement in the inflation outlook, the Board has decided to reduce interest rates.</p>
      <p>From today, the cash rate target for the Bank's domestic market operations will be reduced by 50 basis points.</p>
    </div>
    <section>
      <div class="box-enquiries">
        <div class="item"><p>Dr S.A. Grenville</p><p>Assistant Governor (Economic)</p></div>
      </div>
    </section>
  </div>
</div>
</body></html>
"""


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_modern_lowe_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_modern_lowe(), decision_date=pd.Timestamp("2020-03-19"))
    assert parsed["title"] == "Statement by Philip Lowe, Governor: Monetary Policy Decision"
    assert parsed["publication_date"] == pd.Timestamp("2020-03-19")
    assert parsed["governor"] == "Philip Lowe"
    assert len(parsed["paragraphs"]) == 3
    assert parsed["paragraphs"][0].startswith("The coronavirus is first and foremost")
    assert "press conference" in parsed["paragraphs"][-1]


def test_parse_modern_board_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    assert parsed["title"] == (
        "Statement by the Reserve Bank Board: Monetary Policy Decision"
    )
    assert parsed["publication_date"] == pd.Timestamp("2024-02-06")
    assert parsed["governor"] == "Reserve Bank Board"
    assert len(parsed["paragraphs"]) == 2


def test_parse_legacy_fraser_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_legacy_fraser(), decision_date=pd.Timestamp("1996-07-31"))
    assert "Mr Bernie Fraser" in parsed["title"]
    assert parsed["publication_date"] == pd.Timestamp("1996-07-31")
    assert parsed["governor"] == "Bernie Fraser"
    assert len(parsed["paragraphs"]) == 2


def test_parse_body_text_joins_paragraphs_with_blank_line() -> None:
    parsed = _parse(_fixture_modern_lowe(), decision_date=pd.Timestamp("2020-03-19"))
    assert parsed["body_text"].count("\n\n") == 2  # 3 paragraphs → 2 separators
    assert parsed["body_text"].split("\n\n") == list(parsed["paragraphs"])


def test_parse_excludes_contact_box_and_aside() -> None:
    """Phone numbers and related-links must not leak into paragraphs."""
    parsed = _parse(_fixture_modern_lowe(), decision_date=pd.Timestamp("2020-03-19"))
    joined = parsed["body_text"]
    assert "Media and Communications" not in joined
    assert "9551 8111" not in joined
    assert "Related: Statement on Monetary Policy" not in joined


def test_parse_normalises_nbsp() -> None:
    """The 2024 fixture uses &nbsp; in "4.35 per cent" — must collapse to spaces."""
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    assert "\xa0" not in parsed["body_text"]
    assert "4.35 per cent" in parsed["body_text"]


def test_parse_raises_on_missing_content_div() -> None:
    html = b"<html><body><div>no content div</div></body></html>"
    with pytest.raises(ValueError, match="No <div id='content'>"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_missing_headline() -> None:
    html = b'<html><body><div id="content"><span itemprop="datePublished">6 February 2024</span><p>body</p></div></body></html>'
    with pytest.raises(ValueError, match="itemprop='headline'"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_missing_date() -> None:
    html = b'<html><body><div id="content"><span itemprop="headline">Statement by the Reserve Bank Board</span><p>body</p></div></body></html>'
    with pytest.raises(ValueError, match="itemprop='datePublished'"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_unparseable_date() -> None:
    html = (
        b'<html><body><div id="content">'
        b'<span itemprop="headline">Statement by the Reserve Bank Board</span>'
        b'<span itemprop="datePublished">Feb 6, 2024</span>'  # wrong format
        b'<div class="rss-mr-content"><p>body</p></div>'
        b'</div></body></html>'
    )
    with pytest.raises(ValueError, match="Could not parse datePublished"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_empty_body() -> None:
    html = (
        b'<html><body><div id="content">'
        b'<span itemprop="headline">Statement by the Reserve Bank Board</span>'
        b'<span itemprop="datePublished">6 February 2024</span>'
        b'<div class="rss-mr-content"></div>'  # no paragraphs
        b'</div></body></html>'
    )
    with pytest.raises(ValueError, match="zero body paragraphs"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


# -----------------------------------------------------------------------------
# _extract_governor — title-pattern coverage per era
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        # Feb-2024+ board era
        (
            "Statement by the Reserve Bank Board: Monetary Policy Decision - February 2024",
            "Reserve Bank Board",
        ),
        # 2010-Jan-2024: "Statement by <Name>, Governor"
        (
            "Statement by Philip Lowe, Governor: Monetary Policy Decision - June 2023",
            "Philip Lowe",
        ),
        (
            "Statement by Glenn Stevens, Governor: Monetary Policy Decision - May 2010",
            "Glenn Stevens",
        ),
        (
            "Statement by Michele Bullock, Governor: Monetary Policy Decision - November 2023",
            "Michele Bullock",
        ),
        # Pre-2010: "Statement by the Governor, Mr <Name>"
        (
            "Statement by the Governor, Mr Bernie Fraser: Reduction in Interest Rates - July 1996",
            "Bernie Fraser",
        ),
        (
            "Statement by the Governor, Mr Ian Macfarlane: Monetary Policy",
            "Ian Macfarlane",
        ),
        (
            "Statement by the Governor, Mr Glenn Stevens: Monetary Policy",
            "Glenn Stevens",
        ),
    ],
)
def test_extract_governor_per_era(title: str, expected: str) -> None:
    assert _extract_governor(title) == expected


def test_extract_governor_returns_none_on_unrecognised_title() -> None:
    assert _extract_governor("Report on Clearing and Settlement Facilities") is None


# -----------------------------------------------------------------------------
# _extract_paragraphs — modern container vs fallback
# -----------------------------------------------------------------------------


def test_extract_paragraphs_prefers_modern_container() -> None:
    soup = BeautifulSoup(_fixture_modern_lowe(), "html.parser")
    paras = _extract_paragraphs(_content_div(soup))
    assert len(paras) == 3
    assert all("Media and Communications" not in p for p in paras)


def test_extract_paragraphs_falls_back_for_legacy_layout() -> None:
    soup = BeautifulSoup(_fixture_legacy_fraser(), "html.parser")
    paras = _extract_paragraphs(_content_div(soup))
    assert len(paras) == 2
    assert all("Grenville" not in p for p in paras)


# -----------------------------------------------------------------------------
# _normalise_whitespace
# -----------------------------------------------------------------------------


def test_normalise_whitespace_collapses_nbsp_and_tabs() -> None:
    assert _normalise_whitespace("a\xa0b\t\tc\n d") == "a b c d"


def test_normalise_whitespace_strips() -> None:
    assert _normalise_whitespace("  \n leading-trailing  \t") == "leading-trailing"


# -----------------------------------------------------------------------------
# _select_targets — F11 filter behaviour
# -----------------------------------------------------------------------------


def _f11_stub() -> pd.DataFrame:
    """Synthetic F11 frame covering: pre-floor row, in-window row with URL,
    in-window row without URL (hold without statement, pre-2010), in-window
    row with empty-string URL (should be filtered), and a duplicate URL."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["1990-01-23", "1996-07-31", "2008-07-02", "2020-03-19", "2024-02-07"]
            ),
            "publication_date": pd.to_datetime(
                ["1990-01-23", "1996-07-31", "2008-07-01", "2020-03-19", "2024-02-06"]
            ),
            "statement_url": [
                "/media-releases/1990/mr-90-01.html",  # pre-floor — excluded
                "/media-releases/1996/mr-96-09.html",
                None,  # hold pre-2010, no statement
                "/media-releases/2020/mr-20-08.html",
                "/media-releases/2024/mr-24-01.html",
            ],
            "change_raw": ["-1.00", "-0.50", "0.00", "-0.25", "0.00"],
            "new_cash_rate_raw": ["17.50", "7.00", "7.25", "0.25", "4.35"],
            "minutes_url": [None, None, None, None, None],
        }
    )


def test_select_targets_excludes_pre_floor_rows() -> None:
    out = _select_targets(_f11_stub())
    assert (out["decision_date"] >= _CALENDAR_VALIDATION_FLOOR).all()


def test_select_targets_excludes_rows_without_statement_url() -> None:
    out = _select_targets(_f11_stub())
    assert len(out) == 3
    assert set(out["decision_date"]) == {
        pd.Timestamp("1996-07-31"),
        pd.Timestamp("2020-03-19"),
        pd.Timestamp("2024-02-06"),
    }


def test_select_targets_excludes_empty_string_url() -> None:
    f11 = _f11_stub()
    f11.loc[f11["publication_date"] == pd.Timestamp("2020-03-19"), "statement_url"] = ""
    out = _select_targets(f11)
    assert pd.Timestamp("2020-03-19") not in set(out["decision_date"])


def test_select_targets_decision_date_equals_publication_date() -> None:
    """The 'decision date' for media-release scope is F11.publication_date —
    the announcement day, not the effective day."""
    out = _select_targets(_f11_stub())
    feb_2024 = out[out["decision_date"] == pd.Timestamp("2024-02-06")]
    assert len(feb_2024) == 1  # not 2024-02-07 (the effective date)


def test_select_targets_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"observation_date": [pd.Timestamp("2024-02-07")]})
    with pytest.raises(ValueError, match="missing required column"):
        _select_targets(bad)


# -----------------------------------------------------------------------------
# _validate_cross_check
# -----------------------------------------------------------------------------


def _media_row(decision: str, pub: str | None = None, url_slug: str = "mr-24-01") -> dict:
    pub = pub or decision
    return {
        "decision_date": pd.Timestamp(decision),
        "publication_date": pd.Timestamp(pub),
        "governor": "Reserve Bank Board",
        "title": "Statement by the Reserve Bank Board: Monetary Policy Decision",
        "url": f"https://www.rba.gov.au/media-releases/2024/{url_slug}.html",
        "sha256": "deadbeef",
        "raw_html_filename": f"{decision}__{url_slug}.html",
        "body_text": "body",
        "paragraphs": ["body"],
    }


def test_validate_cross_check_passes_on_full_match() -> None:
    f11 = _f11_stub()
    media = pd.DataFrame(
        [
            _media_row("1996-07-31", url_slug="mr-96-09"),
            _media_row("2020-03-19", url_slug="mr-20-08"),
            _media_row("2024-02-06", url_slug="mr-24-01"),
        ]
    )
    _validate_cross_check(media, f11)  # no raise


def test_validate_cross_check_raises_on_missing_decision() -> None:
    f11 = _f11_stub()
    media = pd.DataFrame(
        [
            _media_row("1996-07-31", url_slug="mr-96-09"),
            _media_row("2024-02-06", url_slug="mr-24-01"),
            # 2020-03-19 missing
        ]
    )
    with pytest.raises(ValueError, match="have no matching media-release"):
        _validate_cross_check(media, f11)


def test_validate_cross_check_raises_on_extra_media_row() -> None:
    f11 = _f11_stub()
    media = pd.DataFrame(
        [
            _media_row("1996-07-31", url_slug="mr-96-09"),
            _media_row("2020-03-19", url_slug="mr-20-08"),
            _media_row("2024-02-06", url_slug="mr-24-01"),
            _media_row("2050-01-01", url_slug="mr-50-01"),  # not in F11
        ]
    )
    with pytest.raises(ValueError, match="without a matching F11 decision"):
        _validate_cross_check(media, f11)


def test_validate_cross_check_raises_when_publication_ne_decision() -> None:
    """The publication-date == decision-date invariant must be enforced."""
    f11 = _f11_stub()
    media = pd.DataFrame(
        [
            _media_row("1996-07-31", url_slug="mr-96-09"),
            _media_row("2020-03-19", url_slug="mr-20-08"),
            _media_row("2024-02-06", pub="2024-02-07", url_slug="mr-24-01"),  # off by one
        ]
    )
    with pytest.raises(ValueError, match="publication_date != decision_date"):
        _validate_cross_check(media, f11)


def test_validate_cross_check_empty_media_with_no_targets_passes() -> None:
    """Empty F11 (or all pre-floor) ↔ empty media frame is a valid state."""
    f11 = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["1980-01-01"]),
            "publication_date": pd.to_datetime(["1980-01-01"]),
            "statement_url": [None],
            "change_raw": ["0.00"],
            "new_cash_rate_raw": ["10.00"],
            "minutes_url": [None],
        }
    )
    media = pd.DataFrame(
        columns=[
            "decision_date",
            "publication_date",
            "governor",
            "title",
            "url",
            "sha256",
            "raw_html_filename",
            "body_text",
            "paragraphs",
        ]
    )
    _validate_cross_check(media, f11)  # no raise


# -----------------------------------------------------------------------------
# No-leakage invariant — Invariant #1 contract
# -----------------------------------------------------------------------------


def test_no_future_leakage_publication_eq_decision_in_parse() -> None:
    """Every fixture's parsed publication_date equals the decision_date the
    test caller passed in — guaranteeing the same-day rule end-to-end."""
    for fixture, decision in [
        (_fixture_modern_lowe(), pd.Timestamp("2020-03-19")),
        (_fixture_modern_board(), pd.Timestamp("2024-02-06")),
        (_fixture_legacy_fraser(), pd.Timestamp("1996-07-31")),
    ]:
        parsed = _parse(fixture, decision_date=decision)
        assert parsed["publication_date"] == decision
