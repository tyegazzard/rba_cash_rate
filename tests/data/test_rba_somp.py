"""Tests for ``rba.data.sources.rba_somp``.

The live ``fetch()`` path hits the RBA over HTTPS and is not exercised here.
``_enumerate_issues``, ``_select_targets``, ``_parse``, ``_extract_dc_date``,
``_extract_blocks``, ``_normalise_whitespace``, ``_validate_cross_check``,
``_softcheck_publication_gap`` and ``_issue_year_mon`` are tested in isolation
against synthetic HTML fixtures that mirror the SoMP archive index and the
Overview-page layout across eras (the modern ``<h2>`` key-message Overview, the
2006–2014 flat ``<p>`` "Introduction", the ``<div id="content">`` wrapper, the
``<meta name="dc.date">`` marker, and the in-page-contents / related-links
side-blocks that must be excluded).

Because the SoMP is quarterly and not F11-driven, the tests also cover the
issue-to-meeting **month mapping** and the **mixed publication-vs-decision sign**
(the SoMP may be released before, on, or after its accompanying decision by
era), which is the no-leakage finding specific to this source.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag
import pandas as pd
import pytest

from rba.data.sources.rba_somp import (
    _CALENDAR_VALIDATION_FLOOR,
    _enumerate_issues,
    _extract_blocks,
    _extract_dc_date,
    _issue_year_mon,
    _normalise_whitespace,
    _parse,
    _select_targets,
    _softcheck_publication_gap,
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
# Synthetic HTML fixtures — archive index + Overview pages per era.
# -----------------------------------------------------------------------------


def _fixture_index() -> bytes:
    """SoMP archive index: pre-floor PDF-era issues, in-window issues, noise."""
    return b"""
<html><head></head><body>
<div id="content">
  <ul>
    <li><a href="/publications/smp/2005/boxes.html">2005 Boxes</a></li>
    <li><a href="/publications/smp/2005/nov/">November 2005</a></li>
    <li><a href="/publications/smp/2006/feb/">February 2006</a></li>
    <li><a href="/publications/smp/2006/may/">May 2006</a></li>
    <li><a href="/publications/smp/2018/may/">May 2018</a></li>
    <li><a href="/publications/smp/2024/may/">May 2024</a></li>
    <li><a href="/publications/smp/2024/boxes.html">2024 Boxes</a></li>
    <li><a href="/publications/smp/2024/">2024 index</a></li>
    <li><a href="/publications/smp/2024/may/overview.html">Overview chapter link</a></li>
    <li><a href="/publications/smp/forecasts-archive.html">Forecasts archive</a></li>
  </ul>
</div>
</body></html>
"""


def _fixture_modern_overview() -> bytes:
    """2024 redesign Overview: <h2> key messages, in-page contents rail."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-05-07">
</head><body>
<div id="content" role="main" class="column-content content-style">
  <h1 class="page-title">Statement on Monetary Policy &ndash; May&nbsp;2024 Overview</h1>
  <div class="nav-publication-contents">
    <h2>Contents</h2>
    <p>Overview</p><p>International economic conditions</p>
  </div>
  <p>Inflation is still high and is falling more slowly than expected.</p>
  <h2>Inflation has remained higher than expected.</h2>
  <p>Underlying inflation was 4.0&nbsp;per&nbsp;cent over the year to the March quarter.</p>
  <h2>The labour market has eased.</h2>
  <p>Conditions in the labour market continued to ease over recent months.</p>
  <div class="publication-related-links aside-component">
    <h2>Related Information</h2>
    <p>Media Release</p>
  </div>
  <section>
    <div class="box-enquiries"><p>Media and Communications: +61 2 9551 8111</p></div>
  </section>
</div>
</body></html>
"""


def _fixture_legacy_intro() -> bytes:
    """2006-era Overview titled "Introduction": flat <p>, no <h2> key messages."""
    return b"""
<html><head>
<meta name="dc.date" content="2006-02-07">
</head><body>
<div id="content" role="main" class="column-content content-style">
  <h1 class="page-title">Statement on Monetary Policy &ndash; February&nbsp;2006 Introduction</h1>
  <p>Growth in the world economy remained robust in 2005.</p>
  <p>The Australian economy continued to expand at a solid pace.</p>
  <p>The Board left the cash rate unchanged at 5.5 per cent.</p>
  <section>
    <div class="box-enquiries"><p>Dr Tony Richards: +61 2 9551 8111</p></div>
  </section>
</div>
</body></html>
"""


# -----------------------------------------------------------------------------
# Synthetic F11 frame
# -----------------------------------------------------------------------------


def _f11_stub() -> pd.DataFrame:
    """F11 frame with Feb/May decisions for the in-window fixture issues."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["2006-02-09", "2006-05-04", "2018-05-02", "2024-05-08"]
            ),
            "publication_date": pd.to_datetime(
                # 2006-02: SoMP (Feb 7) released BEFORE the decision (Feb 8).
                # 2018-05: SoMP (May 4) released AFTER the decision (May 1).
                # 2024-05: SoMP released SAME DAY as the decision (May 7).
                ["2006-02-08", "2006-05-03", "2018-05-01", "2024-05-07"]
            ),
            "change_raw": ["0.00", "0.00", "0.00", "-0.25"],
            "new_cash_rate_raw": ["5.50", "5.50", "1.50", "4.10"],
        }
    )


# -----------------------------------------------------------------------------
# _enumerate_issues
# -----------------------------------------------------------------------------


def test_enumerate_issues_excludes_pre_floor_and_noise() -> None:
    issues = _enumerate_issues(_fixture_index())
    # Pre-floor 2005/nov excluded; boxes.html / year-root / chapter / archive
    # links ignored; in-window issue roots kept.
    assert issues == [
        (2006, "feb", "/publications/smp/2006/feb/"),
        (2006, "may", "/publications/smp/2006/may/"),
        (2018, "may", "/publications/smp/2018/may/"),
        (2024, "may", "/publications/smp/2024/may/"),
    ]


def test_enumerate_issues_sorted_by_year_then_month() -> None:
    issues = _enumerate_issues(_fixture_index())
    keys = [(y, m) for y, m, _ in issues]
    assert keys == sorted(keys, key=lambda k: (k[0], {"feb": 2, "may": 5}[k[1]]))


def test_enumerate_issues_floor_is_2006_feb() -> None:
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("2006-02-01")


# -----------------------------------------------------------------------------
# _select_targets — issue → meeting month mapping
# -----------------------------------------------------------------------------


def test_select_targets_maps_each_issue_to_same_month_decision() -> None:
    out = _select_targets(_fixture_index(), _f11_stub())
    assert list(out["decision_date"]) == [
        pd.Timestamp("2006-02-08"),
        pd.Timestamp("2006-05-03"),
        pd.Timestamp("2018-05-01"),
        pd.Timestamp("2024-05-07"),
    ]
    assert out["decision_date"].is_monotonic_increasing
    assert list(out["mon"]) == ["feb", "may", "may", "may"]


def test_select_targets_raises_when_month_has_no_decision() -> None:
    f11 = _f11_stub()
    # Drop the 2018-05 decision: the 2018/may SoMP can no longer be anchored.
    f11 = f11[f11["publication_date"] != pd.Timestamp("2018-05-01")]
    with pytest.raises(ValueError, match="no F11 decision"):
        _select_targets(_fixture_index(), f11)


def test_select_targets_raises_on_ambiguous_double_decision() -> None:
    f11 = _f11_stub()
    extra = f11[f11["publication_date"] == pd.Timestamp("2024-05-07")].copy()
    extra["publication_date"] = pd.Timestamp("2024-05-21")
    f11 = pd.concat([f11, extra], ignore_index=True)
    with pytest.raises(ValueError, match="ambiguous"):
        _select_targets(_fixture_index(), f11)


def test_select_targets_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"observation_date": [pd.Timestamp("2024-05-08")]})
    with pytest.raises(ValueError, match="missing required column"):
        _select_targets(_fixture_index(), bad)


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_modern_overview_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_modern_overview(), decision_date=pd.Timestamp("2024-05-07"))
    assert parsed["title"] == "Statement on Monetary Policy – May 2024 Overview"
    assert parsed["publication_date"] == pd.Timestamp("2024-05-07")
    assert parsed["paragraphs"][0].startswith("Inflation is still high")
    assert "Inflation has remained higher than expected." in parsed["paragraphs"]


def test_parse_legacy_intro_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_legacy_intro(), decision_date=pd.Timestamp("2006-02-08"))
    assert parsed["title"].endswith("February 2006 Introduction")
    assert parsed["publication_date"] == pd.Timestamp("2006-02-07")
    assert parsed["paragraphs"][-1].startswith("The Board left the cash rate")


def test_parse_captures_h2_as_standalone_entries() -> None:
    parsed = _parse(_fixture_modern_overview(), decision_date=pd.Timestamp("2024-05-07"))
    paras = parsed["paragraphs"]
    i_head = paras.index("The labour market has eased.")
    assert paras[i_head + 1].startswith("Conditions in the labour market")


def test_parse_body_text_joins_blocks_with_blank_line() -> None:
    parsed = _parse(_fixture_legacy_intro(), decision_date=pd.Timestamp("2006-02-08"))
    assert parsed["body_text"].split("\n\n") == list(parsed["paragraphs"])


def test_parse_excludes_contents_rail_related_links_and_contact() -> None:
    parsed = _parse(_fixture_modern_overview(), decision_date=pd.Timestamp("2024-05-07"))
    joined = parsed["body_text"]
    assert "Contents" not in joined
    assert "International economic conditions" not in joined
    assert "Related Information" not in joined
    assert "Media Release" not in joined
    assert "9551 8111" not in joined


def test_parse_normalises_nbsp() -> None:
    parsed = _parse(_fixture_modern_overview(), decision_date=pd.Timestamp("2024-05-07"))
    assert "\xa0" not in parsed["body_text"]
    assert "4.0 per cent" in parsed["body_text"]


def test_parse_raises_on_missing_content_div() -> None:
    html = b"<html><body><div>no content div</div></body></html>"
    with pytest.raises(ValueError, match="No <div id='content'>"):
        _parse(html, decision_date=pd.Timestamp("2024-05-07"))


def test_parse_raises_on_missing_title() -> None:
    html = (
        b'<html><head><meta name="dc.date" content="2024-05-07"></head>'
        b'<body><div id="content"><p>body</p></div></body></html>'
    )
    with pytest.raises(ValueError, match="No <h1> title"):
        _parse(html, decision_date=pd.Timestamp("2024-05-07"))


def test_parse_raises_on_missing_dc_date() -> None:
    html = (
        b'<html><body><div id="content"><h1>SoMP</h1>'
        b"<p>body</p></div></body></html>"
    )
    with pytest.raises(ValueError, match="dc.date"):
        _parse(html, decision_date=pd.Timestamp("2024-05-07"))


def test_parse_raises_on_empty_body() -> None:
    html = (
        b'<html><head><meta name="dc.date" content="2024-05-07"></head>'
        b'<body><div id="content"><h1>SoMP</h1></div></body></html>'
    )
    with pytest.raises(ValueError, match="zero body blocks"):
        _parse(html, decision_date=pd.Timestamp("2024-05-07"))


# -----------------------------------------------------------------------------
# _extract_dc_date / _extract_blocks / _normalise_whitespace / _issue_year_mon
# -----------------------------------------------------------------------------


def test_extract_dc_date_parses_iso() -> None:
    soup = BeautifulSoup(_fixture_modern_overview(), "html.parser")
    assert _extract_dc_date(soup) == pd.Timestamp("2024-05-07")


def test_extract_dc_date_none_on_missing_meta() -> None:
    soup = BeautifulSoup(b"<html><head></head><body></body></html>", "html.parser")
    assert _extract_dc_date(soup) is None


def test_extract_dc_date_none_on_unparseable() -> None:
    soup = BeautifulSoup(
        b'<html><head><meta name="dc.date" content="not-a-date"></head></html>',
        "html.parser",
    )
    assert _extract_dc_date(soup) is None


def test_extract_blocks_drops_nav_headings_and_rails() -> None:
    soup = BeautifulSoup(_fixture_modern_overview(), "html.parser")
    blocks = _extract_blocks(_content_div(soup))
    assert "Related Information" not in blocks
    assert "Contents" not in blocks


def test_extract_blocks_preserves_document_order() -> None:
    soup = BeautifulSoup(_fixture_modern_overview(), "html.parser")
    blocks = _extract_blocks(_content_div(soup))
    assert blocks.index("Inflation has remained higher than expected.") < blocks.index(
        "The labour market has eased."
    )


def test_normalise_whitespace_collapses_nbsp_and_tabs() -> None:
    assert _normalise_whitespace("a\xa0b\t\tc\n d") == "a b c d"


def test_issue_year_mon_parses_path() -> None:
    assert _issue_year_mon("/publications/smp/2024/may/") == (2024, "may")


def test_issue_year_mon_raises_on_bad_path() -> None:
    with pytest.raises(ValueError, match="Unrecognised SoMP issue path"):
        _issue_year_mon("/publications/smp/2024/")


# -----------------------------------------------------------------------------
# _validate_cross_check
# -----------------------------------------------------------------------------


def _somp_row(decision: str, pub: str, year: int, mon: str) -> dict:
    page = "intro" if year <= 2014 else "overview"
    return {
        "decision_date": pd.Timestamp(decision),
        "publication_date": pd.Timestamp(pub),
        "title": f"Statement on Monetary Policy – {mon} {year} Overview",
        "url": f"https://www.rba.gov.au/publications/smp/{year}/{mon}/{page}.html",
        "sha256": "deadbeef",
        "raw_html_filename": f"{decision}__{year}-{mon}-{page}.html",
        "body_text": "body",
        "paragraphs": ["body"],
    }


def _full_somp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _somp_row("2006-02-08", "2006-02-07", 2006, "feb"),
            _somp_row("2006-05-03", "2006-05-05", 2006, "may"),
            _somp_row("2018-05-01", "2018-05-04", 2018, "may"),
            _somp_row("2024-05-07", "2024-05-07", 2024, "may"),
        ]
    )


def test_validate_cross_check_passes_on_full_match() -> None:
    targets = _select_targets(_fixture_index(), _f11_stub())
    _validate_cross_check(_full_somp(), targets, _f11_stub())  # no raise


def test_validate_cross_check_raises_on_missing_issue() -> None:
    targets = _select_targets(_fixture_index(), _f11_stub())
    somp = _full_somp().iloc[:3]  # drop the 2024 issue
    with pytest.raises(ValueError, match="have no matching output row"):
        _validate_cross_check(somp, targets, _f11_stub())


def test_validate_cross_check_raises_on_extra_row() -> None:
    targets = _select_targets(_fixture_index(), _f11_stub())
    somp = pd.concat(
        [_full_somp(), pd.DataFrame([_somp_row("2050-02-01", "2050-02-01", 2050, "feb")])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not derived from the archive index"):
        _validate_cross_check(somp, targets, _f11_stub())


def test_validate_cross_check_raises_on_phantom_decision() -> None:
    # A target whose decision_date is not present in F11 (phantom mapping).
    targets = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2099-02-01")],
            "year": [2099],
            "mon": ["feb"],
            "issue_path": ["/publications/smp/2099/feb/"],
        }
    )
    somp = pd.DataFrame([_somp_row("2099-02-01", "2099-02-01", 2099, "feb")])
    with pytest.raises(ValueError, match="absent from F11"):
        _validate_cross_check(somp, targets, _f11_stub())


def test_validate_cross_check_raises_on_duplicate_decision() -> None:
    dup = _somp_row("2024-05-07", "2024-05-07", 2024, "may")
    targets = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-05-07")],
            "year": [2024],
            "mon": ["may"],
            "issue_path": ["/publications/smp/2024/may/"],
        }
    )
    somp = pd.DataFrame([dup, dup])
    with pytest.raises(ValueError, match="Mapping collision"):
        _validate_cross_check(somp, targets, _f11_stub())


def test_validate_cross_check_empty_passes() -> None:
    targets = pd.DataFrame(columns=["decision_date", "year", "mon", "issue_path"])
    somp = pd.DataFrame(
        columns=[
            "decision_date",
            "publication_date",
            "title",
            "url",
            "sha256",
            "raw_html_filename",
            "body_text",
            "paragraphs",
        ]
    )
    _validate_cross_check(somp, targets, _f11_stub())  # no raise


# -----------------------------------------------------------------------------
# Publication-vs-decision sign — mixed by era (the SoMP no-leakage finding)
# -----------------------------------------------------------------------------


def test_publication_sign_is_mixed_across_eras() -> None:
    somp = _full_somp()
    by_decision = {r["decision_date"]: r["publication_date"] for _, r in somp.iterrows()}
    # 2006: SoMP released BEFORE the decision.
    assert by_decision[pd.Timestamp("2006-02-08")] < pd.Timestamp("2006-02-08")
    # 2018: SoMP released AFTER the decision.
    assert by_decision[pd.Timestamp("2018-05-01")] > pd.Timestamp("2018-05-01")
    # 2024: SoMP released SAME DAY as the decision.
    assert by_decision[pd.Timestamp("2024-05-07")] == pd.Timestamp("2024-05-07")


@pytest.fixture
def loguru_messages():
    """Capture loguru WARNING+ messages emitted within the test into a list."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def test_softcheck_silent_within_window(loguru_messages) -> None:
    _softcheck_publication_gap(
        publication_date=pd.Timestamp("2018-05-04"),
        decision_date=pd.Timestamp("2018-05-01"),  # +3 days, within window
        url="u",
    )
    assert not any("days from decision_date" in m for m in loguru_messages)


def test_softcheck_silent_when_released_before_decision(loguru_messages) -> None:
    _softcheck_publication_gap(
        publication_date=pd.Timestamp("2006-02-07"),
        decision_date=pd.Timestamp("2006-02-08"),  # -1 day, within window
        url="u",
    )
    assert not any("days from decision_date" in m for m in loguru_messages)


def test_softcheck_silent_for_november_three_week_lag(loguru_messages) -> None:
    # November issues historically released ~3 weeks after the decision; this is
    # a genuine same-policy-round gap and must not warn under the 31-day window.
    _softcheck_publication_gap(
        publication_date=pd.Timestamp("2015-11-27"),
        decision_date=pd.Timestamp("2015-11-03"),  # +24 days, genuine
        url="u",
    )
    assert not any("days from decision_date" in m for m in loguru_messages)


def test_softcheck_warns_beyond_window(loguru_messages) -> None:
    _softcheck_publication_gap(
        publication_date=pd.Timestamp("2024-06-30"),
        decision_date=pd.Timestamp("2024-05-07"),  # ~54 days — implausible
        url="u",
    )
    assert any("days from decision_date" in m for m in loguru_messages)
