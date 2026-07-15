"""Tests for ``rba.data.sources.rba_speeches``.

The live ``fetch()`` / ``_build_targets()`` paths hit the RBA over HTTPS and are
not exercised here. The pure helpers — ``_enumerate_years``,
``_enumerate_speeches``, ``_parse``, ``_classify_speech_type``,
``_split_speaker_role``, ``_extract_publication_date``, ``_extract_dc_date``,
``_extract_author``, ``_extract_blocks``, ``_normalise_whitespace``,
``_softcheck_dc_date``, ``_validate_cross_check`` and ``_report_parse_coverage``
— are tested in isolation against synthetic HTML fixtures that mirror the
speeches archive index, the per-year listings, and the per-speech page layout
across eras (the modern speech, a 1990s speech with a ``DDMMYY`` slug, an
Assistant-Governor author with a parenthetical role and footnote marker, an MPB
member with a post-nominal honorific, a media-conference transcript with
interleaved speaker turns, and a Q&A-transcript companion).

Speeches are the one text source that does NOT inherit the decision-keyed wide
schema (see CONTEXT.md): the tests therefore exercise the **event-keyed**
``speech_id`` key, the multi-speaker / multi-type metadata parse, and the
no-F11 archive-completeness cross-check.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag
import pandas as pd
import pytest

from rba.data.sources.rba_speeches import (
    _CALENDAR_VALIDATION_FLOOR,
    _classify_speech_type,
    _enumerate_speeches,
    _enumerate_years,
    _extract_author,
    _extract_blocks,
    _extract_dc_date,
    _extract_publication_date,
    _normalise_whitespace,
    _parse,
    _report_parse_coverage,
    _softcheck_dc_date,
    _split_speaker_role,
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
# Synthetic HTML fixtures — archive index + year page + per-speech pages.
# -----------------------------------------------------------------------------


def _fixture_index() -> bytes:
    """Archive index: year-page links spanning pre-floor + in-window + noise."""
    return b"""
<html><head></head><body>
<div id="content">
  <ul>
    <li><a href="/speeches/1990/">1990</a></li>
    <li><a href="/speeches/1992/">1992</a></li>
    <li><a href="/speeches/1993/">1993</a></li>
    <li><a href="/speeches/2024/">2024</a></li>
    <li><a href="/speeches/2026/">2026</a></li>
    <li><a href="/publications/smp/">Not a speech year</a></li>
    <li><a href="/speeches/">Speeches home</a></li>
  </ul>
</div>
</body></html>
"""


def _fixture_year_page() -> bytes:
    """A 2024 year page: speech links, self-link, non-speech, cross-year link."""
    return b"""
<html><head></head><body>
<div id="content">
  <ul>
    <li><a href="/speeches/2024/sp-gov-2024-02-06.html">A Governor speech</a></li>
    <li><a href="/speeches/2024/mc-gov-2024-02-06.html">Media conference</a></li>
    <li><a href="/speeches/2024/sp-ag-2024-04-02.html">An AG speech</a></li>
    <li><a href="/speeches/2024/sp-ag-2024-04-02-q-and-a-transcript.html">Q&amp;A</a></li>
    <li><a href="/speeches/2024/background-paper-2024.html">Background paper</a></li>
    <li><a href="/speeches/2024/index.html">2024 index</a></li>
    <li><a href="/speeches/2023/sp-gov-2023-12-01.html">Prev-year sidebar link</a></li>
  </ul>
</div>
</body></html>
"""


def _fixture_modern_speech() -> bytes:
    """2024 Governor speech: itemprop author/datePublished, dc.date, h1, body."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-02-06">
</head><body>
<div id="content" role="main" class="column-content content-style">
  <h1 class="page-title">Speech The Economic Outlook and Monetary Policy</h1>
  <div class="box-article-info">
    <p itemprop="author">Michele Bullock Governor</p>
    <p>Published <span itemprop="datePublished">6&nbsp;February 2024</span></p>
  </div>
  <p>Thank you for the opportunity to speak today.</p>
  <h2>The international economy</h2>
  <p>Global growth has been resilient over the past year.</p>
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


def _fixture_assistant_governor_speech() -> bytes:
    """AG speech: parenthetical role + footnote marker in the author byline."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-04-02">
</head><body>
<div id="content">
  <h1>Speech Inflation Dynamics in a Changing Economy</h1>
  <div class="box-article-info">
    <p itemprop="author">Sarah Hunter [ * ] Assistant Governor (Economic)</p>
    <p><span itemprop="datePublished">2 April 2024</span></p>
  </div>
  <p>Inflation has eased over the past year but remains above target.</p>
</div>
</body></html>
"""


def _fixture_media_conference() -> bytes:
    """Media conference: interleaved official + journalist turns as plain <p>."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-02-06">
</head><body>
<div id="content">
  <h1>Media conference Monetary Policy Decision</h1>
  <div class="box-article-info">
    <p itemprop="author">Michele Bullock Governor</p>
    <p><span itemprop="datePublished">6 February 2024</span></p>
  </div>
  <p>Good afternoon. Today the Board decided to hold the cash rate.</p>
  <p>David Chau, ABC News. What is your message to households?</p>
  <p>We understand many households are under pressure.</p>
</div>
</body></html>
"""


def _fixture_qanda() -> bytes:
    """Q&A transcript companion (own slug, own type)."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-04-02">
</head><body>
<div id="content">
  <h1>Transcript of Question &amp; Answer Session Inflation Dynamics</h1>
  <div class="box-article-info">
    <p itemprop="author">Sarah Hunter Assistant Governor (Economic)</p>
    <p><span itemprop="datePublished">2 April 2024</span></p>
  </div>
  <p>John Kehoe, AFR. Do you expect another rate rise?</p>
  <p>I am not going to give forward guidance.</p>
</div>
</body></html>
"""


def _fixture_legacy_speech() -> bytes:
    """1997 speech (DDMMYY slug era): same itemprop structure, flat <p> body."""
    return b"""
<html><head>
<meta name="dc.date" content="1997-12-04">
</head><body>
<div id="content">
  <h1>Speech The Changing Nature of Economic Crises</h1>
  <div class="box-article-info">
    <p itemprop="author">I.J. Macfarlane Governor</p>
    <p><span itemprop="datePublished">4 December 1997</span></p>
  </div>
  <p>The world economy has experienced several crises this decade.</p>
</div>
</body></html>
"""


def _fixture_mpb_speech() -> bytes:
    """Monetary Policy Board member with a post-nominal honorific."""
    return b"""
<html><head>
<meta name="dc.date" content="2026-06-02">
</head><body>
<div id="content">
  <h1>Speech Economic Conditions and the Outlook</h1>
  <div class="box-article-info">
    <p itemprop="author">Ian Harper AO Monetary Policy Board member</p>
    <p><span itemprop="datePublished">2 June 2026</span></p>
  </div>
  <p>The economy is in a period of transition.</p>
</div>
</body></html>
"""


# -----------------------------------------------------------------------------
# _enumerate_years
# -----------------------------------------------------------------------------


def test_enumerate_years_excludes_pre_floor_and_non_speech_links() -> None:
    years = _enumerate_years(_fixture_index())
    # 1990 / 1992 are pre-floor (floor year 1993); /publications/smp/ ignored.
    assert years == [1993, 2024, 2026]


def test_enumerate_years_floor_is_1993() -> None:
    assert _CALENDAR_VALIDATION_FLOOR == pd.Timestamp("1993-01-01")


# -----------------------------------------------------------------------------
# _enumerate_speeches
# -----------------------------------------------------------------------------


def test_enumerate_speeches_keeps_speech_family_only() -> None:
    out = _enumerate_speeches(_fixture_year_page(), 2024)
    ids = [sid for sid, _ in out]
    assert ids == [
        "mc-gov-2024-02-06",
        "sp-ag-2024-04-02",
        "sp-ag-2024-04-02-q-and-a-transcript",
        "sp-gov-2024-02-06",
    ]
    # background-paper / index excluded; cross-year sidebar link excluded.
    assert "background-paper-2024" not in ids
    assert "index" not in ids
    assert "sp-gov-2023-12-01" not in ids


def test_enumerate_speeches_builds_year_scoped_url_path() -> None:
    out = dict(_enumerate_speeches(_fixture_year_page(), 2024))
    assert out["sp-gov-2024-02-06"] == "/speeches/2024/sp-gov-2024-02-06.html"


def test_enumerate_speeches_ignores_cross_year_links() -> None:
    # The page lists a 2023 sidebar link; enumerating it as 2024 drops it.
    out = _enumerate_speeches(_fixture_year_page(), 2024)
    assert all(path.startswith("/speeches/2024/") for _, path in out)


# -----------------------------------------------------------------------------
# _classify_speech_type
# -----------------------------------------------------------------------------


def test_classify_speech_type_speech() -> None:
    assert _classify_speech_type("Speech The Economic Outlook") == (
        "Speech",
        "The Economic Outlook",
    )


def test_classify_speech_type_media_conference() -> None:
    assert _classify_speech_type("Media conference Monetary Policy Decision") == (
        "Media conference",
        "Monetary Policy Decision",
    )


def test_classify_speech_type_qanda_longest_match_wins() -> None:
    label, title = _classify_speech_type(
        "Transcript of Question & Answer Session Inflation Dynamics"
    )
    assert label == "Transcript of Question & Answer Session"
    assert title == "Inflation Dynamics"


def test_classify_speech_type_unknown_returns_none_and_full_h1() -> None:
    label, title = _classify_speech_type("Keynote Remarks on Financial Stability")
    assert label is None
    assert title == "Keynote Remarks on Financial Stability"


# -----------------------------------------------------------------------------
# _split_speaker_role
# -----------------------------------------------------------------------------


def test_split_speaker_role_governor() -> None:
    assert _split_speaker_role("Michele Bullock Governor") == (
        "Michele Bullock",
        "Governor",
    )


def test_split_speaker_role_deputy_governor_not_greedy() -> None:
    assert _split_speaker_role("Andrew Hauser Deputy Governor") == (
        "Andrew Hauser",
        "Deputy Governor",
    )


def test_split_speaker_role_assistant_governor_with_paren_and_footnote() -> None:
    assert _split_speaker_role("Sarah Hunter [ * ] Assistant Governor (Economic)") == (
        "Sarah Hunter",
        "Assistant Governor (Economic)",
    )


def test_split_speaker_role_strips_post_nominal_honorific() -> None:
    assert _split_speaker_role("Ian Harper AO Monetary Policy Board member") == (
        "Ian Harper",
        "Monetary Policy Board member",
    )


def test_split_speaker_role_department_head() -> None:
    assert _split_speaker_role("Ellis Connolly Head of Payments Policy Department") == (
        "Ellis Connolly",
        "Head of Payments Policy Department",
    )


def test_split_speaker_role_initials_name() -> None:
    assert _split_speaker_role("I.J. Macfarlane Governor") == (
        "I.J. Macfarlane",
        "Governor",
    )


def test_split_speaker_role_none_on_empty() -> None:
    assert _split_speaker_role(None) == (None, None)
    assert _split_speaker_role("") == (None, None)


def test_split_speaker_role_unrecognised_role_keeps_name_only() -> None:
    speaker, role = _split_speaker_role("Jane Doe Visiting Fellow")
    assert speaker == "Jane Doe Visiting Fellow"
    assert role is None


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_modern_speech_extracts_all_fields() -> None:
    parsed = _parse(_fixture_modern_speech(), speech_id="sp-gov-2024-02-06")
    assert parsed["speech_type"] == "Speech"
    assert parsed["title"] == "The Economic Outlook and Monetary Policy"
    assert parsed["publication_date"] == pd.Timestamp("2024-02-06")
    assert parsed["speaker"] == "Michele Bullock"
    assert parsed["speaker_role"] == "Governor"
    assert parsed["paragraphs"][0].startswith("Thank you for the opportunity")
    assert "The international economy" in parsed["paragraphs"]


def test_parse_excludes_related_links_and_contact() -> None:
    parsed = _parse(_fixture_modern_speech(), speech_id="sp-gov-2024-02-06")
    joined = parsed["body_text"]
    assert "Related Information" not in joined
    assert "Media Release" not in joined
    assert "9551 8111" not in joined


def test_parse_normalises_nbsp_in_date() -> None:
    parsed = _parse(_fixture_modern_speech(), speech_id="sp-gov-2024-02-06")
    assert parsed["publication_date"] == pd.Timestamp("2024-02-06")


def test_parse_assistant_governor_speech() -> None:
    parsed = _parse(_fixture_assistant_governor_speech(), speech_id="sp-ag-2024-04-02")
    assert parsed["speaker"] == "Sarah Hunter"
    assert parsed["speaker_role"] == "Assistant Governor (Economic)"
    assert parsed["speech_type"] == "Speech"


def test_parse_media_conference_keeps_interleaved_turns_verbatim() -> None:
    parsed = _parse(_fixture_media_conference(), speech_id="mc-gov-2024-02-06")
    assert parsed["speech_type"] == "Media conference"
    # Journalist question retained verbatim alongside the official's answers.
    assert any("David Chau" in p for p in parsed["paragraphs"])
    assert any("households are under pressure" in p for p in parsed["paragraphs"])


def test_parse_qanda_companion_classifies_as_qanda() -> None:
    parsed = _parse(_fixture_qanda(), speech_id="sp-ag-2024-04-02-q-and-a-transcript")
    assert parsed["speech_type"] == "Transcript of Question & Answer Session"
    assert parsed["title"] == "Inflation Dynamics"


def test_parse_legacy_speech() -> None:
    parsed = _parse(_fixture_legacy_speech(), speech_id="sp-gov-041297")
    assert parsed["publication_date"] == pd.Timestamp("1997-12-04")
    assert parsed["speaker"] == "I.J. Macfarlane"
    assert parsed["speaker_role"] == "Governor"


def test_parse_mpb_speech() -> None:
    parsed = _parse(_fixture_mpb_speech(), speech_id="sp-mpb-2026-06-02")
    assert parsed["speaker"] == "Ian Harper"
    assert parsed["speaker_role"] == "Monetary Policy Board member"


def test_parse_body_text_joins_blocks_with_blank_line() -> None:
    parsed = _parse(_fixture_legacy_speech(), speech_id="sp-gov-041297")
    assert parsed["body_text"].split("\n\n") == list(parsed["paragraphs"])


def test_parse_raises_on_missing_content_div() -> None:
    html = b"<html><body><div>no content div</div></body></html>"
    with pytest.raises(ValueError, match="No <div id='content'>"):
        _parse(html, speech_id="sp-gov-2024-02-06")


def test_parse_raises_on_missing_title() -> None:
    html = (
        b'<html><body><div id="content">'
        b'<p itemprop="author">X Governor</p>'
        b'<span itemprop="datePublished">6 February 2024</span>'
        b"</div></body></html>"
    )
    with pytest.raises(ValueError, match="No <h1> title"):
        _parse(html, speech_id="sp-gov-2024-02-06")


def test_parse_raises_on_missing_date() -> None:
    html = (
        b'<html><body><div id="content"><h1>Speech X</h1>'
        b'<p itemprop="author">X Governor</p><p>body</p></div></body></html>'
    )
    with pytest.raises(ValueError, match="datePublished"):
        _parse(html, speech_id="sp-gov-2024-02-06")


def test_parse_raises_on_empty_body() -> None:
    # h1 + a valid date, but the only <p> blocks live in the excluded
    # box-article-info byline — so the body proper has zero blocks.
    html = (
        b'<html><body><div id="content"><h1>Speech X</h1>'
        b'<div class="box-article-info">'
        b'<p itemprop="author">X Governor</p>'
        b'<p><span itemprop="datePublished">6 February 2024</span></p>'
        b"</div></div></body></html>"
    )
    with pytest.raises(ValueError, match="zero body blocks"):
        _parse(html, speech_id="sp-gov-2024-02-06")


# -----------------------------------------------------------------------------
# _extract_* helpers
# -----------------------------------------------------------------------------


def test_extract_publication_date_parses_d_month_yyyy() -> None:
    soup = BeautifulSoup(_fixture_modern_speech(), "html.parser")
    assert _extract_publication_date(soup) == pd.Timestamp("2024-02-06")


def test_extract_publication_date_none_on_missing() -> None:
    soup = BeautifulSoup(b"<html><body></body></html>", "html.parser")
    assert _extract_publication_date(soup) is None


def test_extract_dc_date_parses_iso() -> None:
    soup = BeautifulSoup(_fixture_modern_speech(), "html.parser")
    assert _extract_dc_date(soup) == pd.Timestamp("2024-02-06")


def test_extract_author_normalises() -> None:
    soup = BeautifulSoup(_fixture_assistant_governor_speech(), "html.parser")
    assert _extract_author(soup) == "Sarah Hunter [ * ] Assistant Governor (Economic)"


def test_extract_blocks_drops_nav_headings_and_rails() -> None:
    soup = BeautifulSoup(_fixture_modern_speech(), "html.parser")
    blocks = _extract_blocks(_content_div(soup))
    assert "Related Information" not in blocks
    assert "The international economy" in blocks


def test_normalise_whitespace_collapses_nbsp_and_tabs() -> None:
    assert _normalise_whitespace("a\xa0b\t\tc\n d") == "a b c d"


# -----------------------------------------------------------------------------
# _softcheck_dc_date
# -----------------------------------------------------------------------------


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


def test_softcheck_dc_date_silent_when_agrees(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=pd.Timestamp("2024-02-06"),
        publication_date=pd.Timestamp("2024-02-06"),
        speech_id="sp-gov-2024-02-06",
    )
    assert not any("disagrees with datePublished" in m for m in loguru_messages)


def test_softcheck_dc_date_warns_on_disagreement(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=pd.Timestamp("2024-02-07"),
        publication_date=pd.Timestamp("2024-02-06"),
        speech_id="sp-gov-2024-02-06",
    )
    assert any("disagrees with datePublished" in m for m in loguru_messages)


def test_softcheck_dc_date_silent_on_missing(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=None,
        publication_date=pd.Timestamp("2024-02-06"),
        speech_id="sp-gov-2024-02-06",
    )
    assert not any("disagrees with datePublished" in m for m in loguru_messages)


# -----------------------------------------------------------------------------
# _validate_cross_check
# -----------------------------------------------------------------------------


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2024, 2024],
            "speech_id": ["sp-gov-2024-02-06", "sp-ag-2024-04-02"],
            "url_path": [
                "/speeches/2024/sp-gov-2024-02-06.html",
                "/speeches/2024/sp-ag-2024-04-02.html",
            ],
        }
    )


def _speech_row(speech_id: str, pub: str, **over) -> dict:
    row = {
        "speech_id": speech_id,
        "publication_date": pd.Timestamp(pub),
        "speaker": "Michele Bullock",
        "speaker_role": "Governor",
        "speech_type": "Speech",
        "title": "A title",
        "url": f"https://www.rba.gov.au/speeches/2024/{speech_id}.html",
        "sha256": "deadbeef",
        "raw_html_filename": f"{speech_id}.html",
        "body_text": "body",
        "paragraphs": ["body"],
    }
    row.update(over)
    return row


def _full_speeches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _speech_row("sp-gov-2024-02-06", "2024-02-06"),
            _speech_row("sp-ag-2024-04-02", "2024-04-02"),
        ]
    )


def test_validate_cross_check_passes_on_full_match() -> None:
    _validate_cross_check(_full_speeches(), _targets())  # no raise


def test_validate_cross_check_raises_on_missing_speech() -> None:
    speeches = _full_speeches().iloc[:1]
    with pytest.raises(ValueError, match="have no matching output row"):
        _validate_cross_check(speeches, _targets())


def test_validate_cross_check_raises_on_extra_row() -> None:
    speeches = pd.concat(
        [_full_speeches(), pd.DataFrame([_speech_row("sp-gov-2099-01-01", "2099-01-01")])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="not derived from the archive index"):
        _validate_cross_check(speeches, _targets())


def test_validate_cross_check_raises_on_duplicate_speech_id() -> None:
    dup = _speech_row("sp-gov-2024-02-06", "2024-02-06")
    targets = pd.DataFrame(
        {
            "year": [2024],
            "speech_id": ["sp-gov-2024-02-06"],
            "url_path": ["/speeches/2024/sp-gov-2024-02-06.html"],
        }
    )
    speeches = pd.DataFrame([dup, dup])
    with pytest.raises(ValueError, match="appear in more than one row"):
        _validate_cross_check(speeches, targets)


def test_validate_cross_check_raises_on_pre_floor_date() -> None:
    targets = pd.DataFrame(
        {
            "year": [1992],
            "speech_id": ["sp-gov-1992-01-01"],
            "url_path": ["/speeches/1992/sp-gov-1992-01-01.html"],
        }
    )
    speeches = pd.DataFrame([_speech_row("sp-gov-1992-01-01", "1992-01-01")])
    with pytest.raises(ValueError, match="pre-floor publication_date"):
        _validate_cross_check(speeches, targets)


def test_validate_cross_check_empty_passes() -> None:
    targets = pd.DataFrame(columns=["year", "speech_id", "url_path"])
    speeches = pd.DataFrame(
        columns=[
            "speech_id",
            "publication_date",
            "speaker",
            "speaker_role",
            "speech_type",
            "title",
            "url",
            "sha256",
            "raw_html_filename",
            "body_text",
            "paragraphs",
        ]
    )
    _validate_cross_check(speeches, targets)  # no raise


def test_validate_cross_check_warns_on_archive_year_mismatch(loguru_messages) -> None:
    targets = pd.DataFrame(
        {
            "year": [2024],
            "speech_id": ["sp-gov-2024-02-06"],
            "url_path": ["/speeches/2024/sp-gov-2024-02-06.html"],
        }
    )
    # Listed under 2024 but datePublished says 2025 — soft warn, not a raise.
    speeches = pd.DataFrame([_speech_row("sp-gov-2024-02-06", "2025-02-06")])
    _validate_cross_check(speeches, targets)
    assert any("listed under archive year" in m for m in loguru_messages)


def test_report_parse_coverage_warns_on_missing_fields(loguru_messages) -> None:
    speeches = pd.DataFrame(
        [
            _speech_row("sp-gov-2024-02-06", "2024-02-06", speaker=None),
            _speech_row("sp-ag-2024-04-02", "2024-04-02", speech_type=None),
        ]
    )
    _report_parse_coverage(speeches)
    joined = "".join(loguru_messages)
    assert "no parsed speaker" in joined
    assert "no parsed speech_type" in joined
