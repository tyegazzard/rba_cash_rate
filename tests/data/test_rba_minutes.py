"""Tests for ``rba.data.sources.rba_minutes`` and its release calendar.

The live ``fetch()`` path hits the RBA over HTTPS and is not exercised here.
``_parse``, ``_extract_blocks``, ``_extract_dc_date``, ``_select_targets``,
``_attach_publication_dates``, ``_softcheck_dc_date``, ``_validate_cross_check``,
``_normalise_whitespace`` and the algorithmic release-calendar functions are
tested in isolation against synthetic HTML fixtures that mirror the minutes
layout (``<div id="content">`` with a flat ``<h2>``/``<p>`` sequence, the
uniform ``<h1>`` title, ``<meta name="dc.date">``, and the related-links /
contact side-blocks that must be excluded).
"""

from __future__ import annotations

from datetime import date

from bs4 import BeautifulSoup
import pandas as pd
import pytest

from rba.data.rba_minutes_release_calendar import (
    build_rba_minutes_release_calendar,
    rba_minutes_publication_date,
    second_tuesday_after,
)
from rba.data.sources.rba_minutes import (
    _CALENDAR_VALIDATION_FLOOR,
    _attach_publication_dates,
    _extract_blocks,
    _extract_dc_date,
    _normalise_whitespace,
    _parse,
    _select_targets,
    _softcheck_dc_date,
    _validate_cross_check,
)

# -----------------------------------------------------------------------------
# Synthetic HTML fixtures — mirror the real RBA minutes layout closely enough
# that the parser exercises the same code path it would in production: the
# <div id="content"> wrapper, the <h1> page title, the <meta name="dc.date">
# marker, the flat <h2>/<p> body sequence, and the related-links / contact
# side-blocks (must be excluded from the body).
# -----------------------------------------------------------------------------


def _fixture_modern_board() -> bytes:
    """2024 Reserve-Bank-Board era minutes: dc.date == algorithmic, tile rail."""
    return b"""
<html><head>
<meta name="dc.date" content="2024-02-20">
</head><body>
<div id="content" tabindex="-1" role="main" class="column-content content-style">
  <h1 class="page-title">Minutes of the Monetary Policy Meeting of the Reserve Bank Board</h1>
  <p>Sydney &ndash; 5&nbsp;and 6&nbsp;February 2024</p>
  <h2>Members present</h2>
  <p>Michele Bullock (Governor and Chair), Andrew Hauser (Deputy Governor)</p>
  <h2>International economic developments</h2>
  <p>Members commenced their discussion of the global economy by noting that conditions remained volatile.</p>
  <p>Inflation had eased across most advanced economies.</p>
  <h2>The decision</h2>
  <p>The Board decided to leave the cash rate target unchanged at 4.35&nbsp;per&nbsp;cent.</p>
  <div class="nav-page-contents aside-component monetary-policy-related-links minutes">
    <h2>Related Information</h2>
    <p>Statement on Monetary Policy</p>
  </div>
  <div class="tile-container">
    <div class="clickable"><p>Governor Michele Bullock addresses the media</p></div>
  </div>
  <section>
    <div class="box-enquiries"><p>Media and Communications: +61 2 9551 8111</p></div>
  </section>
</div>
</body></html>
"""


def _fixture_legacy_2008() -> bytes:
    """2008-era minutes: no related-links rail; contact box under <section>."""
    return b"""
<html><head>
<meta name="dc.date" content="2008-02-19">
</head><body>
<div id="content" tabindex="-1" role="main" class="column-content content-style">
  <h1 class="page-title">Minutes of the Monetary Policy Meeting of the Reserve Bank Board</h1>
  <p>Sydney &ndash; 5 February 2008</p>
  <h2>Members Present</h2>
  <p>Glenn Stevens (Chairman and Governor), Ric Battellino (Deputy Governor)</p>
  <h2>Financial Markets</h2>
  <p>Members were briefed on conditions in financial markets.</p>
  <h2>The Decision</h2>
  <p>The Board decided to raise the cash rate by 25 basis points to 7.0 per cent.</p>
  <section>
    <div class="box-enquiries"><p>Dr Tony Richards: +61 2 9551 8111</p></div>
  </section>
</div>
</body></html>
"""


def _fixture_artifact_dc_date() -> bytes:
    """Minutes page whose dc.date is a CMS artifact (equals the meeting date)."""
    return b"""
<html><head>
<meta name="dc.date" content="2016-12-06">
</head><body>
<div id="content">
  <h1>Minutes of the Monetary Policy Meeting of the Reserve Bank Board</h1>
  <p>Sydney &ndash; 6 December 2016</p>
  <h2>The Decision</h2>
  <p>The Board decided to leave the cash rate unchanged at 1.5 per cent.</p>
</div>
</body></html>
"""


# -----------------------------------------------------------------------------
# Release-calendar rule
# -----------------------------------------------------------------------------


def test_second_tuesday_after_tuesday_meeting_is_plus_14() -> None:
    # 2024-02-06 is a Tuesday.
    assert second_tuesday_after(date(2024, 2, 6)) == date(2024, 2, 20)


def test_second_tuesday_after_thursday_covid_meeting() -> None:
    # 2020-03-19 was a Thursday (out-of-cycle COVID meeting); minutes 2020-03-31.
    assert second_tuesday_after(date(2020, 3, 19)) == date(2020, 3, 31)
    assert date(2020, 3, 31).weekday() == 1  # Tuesday


def test_second_tuesday_after_always_lands_on_tuesday() -> None:
    for d in (date(2015, 2, 3), date(2008, 2, 5), date(2026, 3, 17), date(2019, 4, 2)):
        assert second_tuesday_after(d).weekday() == 1


def test_rba_minutes_publication_date_uses_rule_by_default() -> None:
    assert rba_minutes_publication_date(date(2015, 2, 3)) == date(2015, 2, 17)


def test_build_release_calendar_shape_and_sort() -> None:
    cal = build_rba_minutes_release_calendar(
        [date(2024, 2, 6), date(2008, 2, 5), date(2020, 3, 19)]
    )
    assert list(cal.columns) == ["meeting_date", "publication_date"]
    assert len(cal) == 3
    assert cal["meeting_date"].is_monotonic_increasing
    assert (cal["publication_date"] > cal["meeting_date"]).all()


def test_build_release_calendar_empty() -> None:
    cal = build_rba_minutes_release_calendar([])
    assert list(cal.columns) == ["meeting_date", "publication_date"]
    assert cal.empty


# -----------------------------------------------------------------------------
# _parse
# -----------------------------------------------------------------------------


def test_parse_modern_board_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    assert parsed["title"] == (
        "Minutes of the Monetary Policy Meeting of the Reserve Bank Board"
    )
    assert parsed["dc_date"] == pd.Timestamp("2024-02-20")
    # Sydney line + 3 headings + 4 paragraphs (related-links / tile excluded).
    assert parsed["paragraphs"][0].startswith("Sydney")
    assert "Members present" in parsed["paragraphs"]
    assert "The decision" in parsed["paragraphs"]


def test_parse_legacy_2008_extracts_canonical_fields() -> None:
    parsed = _parse(_fixture_legacy_2008(), decision_date=pd.Timestamp("2008-02-05"))
    assert parsed["dc_date"] == pd.Timestamp("2008-02-19")
    assert "Financial Markets" in parsed["paragraphs"]
    assert parsed["paragraphs"][-1].startswith("The Board decided to raise")


def test_parse_captures_h2_as_standalone_entries() -> None:
    """<h2> headings appear as their own interleaved entries (decision #3)."""
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    paras = parsed["paragraphs"]
    i_head = paras.index("International economic developments")
    # The heading immediately precedes its first body paragraph.
    assert paras[i_head + 1].startswith("Members commenced their discussion")


def test_parse_body_text_joins_blocks_with_blank_line() -> None:
    parsed = _parse(_fixture_legacy_2008(), decision_date=pd.Timestamp("2008-02-05"))
    assert parsed["body_text"].split("\n\n") == list(parsed["paragraphs"])


def test_parse_excludes_related_links_tiles_and_contact() -> None:
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    joined = parsed["body_text"]
    assert "Related Information" not in joined
    assert "addresses the media" not in joined
    assert "9551 8111" not in joined
    assert "Statement on Monetary Policy" not in joined


def test_parse_normalises_nbsp() -> None:
    parsed = _parse(_fixture_modern_board(), decision_date=pd.Timestamp("2024-02-06"))
    assert "\xa0" not in parsed["body_text"]
    assert "4.35 per cent" in parsed["body_text"]


def test_parse_raises_on_missing_content_div() -> None:
    html = b"<html><body><div>no content div</div></body></html>"
    with pytest.raises(ValueError, match="No <div id='content'>"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_missing_title() -> None:
    html = b'<html><body><div id="content"><p>body</p></div></body></html>'
    with pytest.raises(ValueError, match="No <h1> title"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_raises_on_empty_body() -> None:
    html = b'<html><body><div id="content"><h1>Minutes</h1></div></body></html>'
    with pytest.raises(ValueError, match="zero body blocks"):
        _parse(html, decision_date=pd.Timestamp("2024-02-06"))


def test_parse_dc_date_none_when_meta_absent() -> None:
    html = (
        b'<html><body><div id="content"><h1>Minutes</h1>'
        b"<p>Sydney</p><p>body</p></div></body></html>"
    )
    parsed = _parse(html, decision_date=pd.Timestamp("2024-02-06"))
    assert parsed["dc_date"] is None


# -----------------------------------------------------------------------------
# _extract_dc_date
# -----------------------------------------------------------------------------


def test_extract_dc_date_parses_iso() -> None:
    soup = BeautifulSoup(_fixture_modern_board(), "html.parser")
    assert _extract_dc_date(soup) == pd.Timestamp("2024-02-20")


def test_extract_dc_date_none_on_missing_meta() -> None:
    soup = BeautifulSoup(b"<html><head></head><body></body></html>", "html.parser")
    assert _extract_dc_date(soup) is None


def test_extract_dc_date_none_on_unparseable() -> None:
    soup = BeautifulSoup(
        b'<html><head><meta name="dc.date" content="not-a-date"></head></html>',
        "html.parser",
    )
    assert _extract_dc_date(soup) is None


# -----------------------------------------------------------------------------
# _extract_blocks
# -----------------------------------------------------------------------------


def test_extract_blocks_drops_nav_headings() -> None:
    soup = BeautifulSoup(_fixture_modern_board(), "html.parser")
    blocks = _extract_blocks(soup.find("div", id="content"))
    assert "Related Information" not in blocks
    assert all("addresses the media" not in b for b in blocks)


def test_extract_blocks_preserves_document_order() -> None:
    soup = BeautifulSoup(_fixture_legacy_2008(), "html.parser")
    blocks = _extract_blocks(soup.find("div", id="content"))
    assert blocks.index("Members Present") < blocks.index("Financial Markets")
    assert blocks.index("Financial Markets") < blocks.index("The Decision")


# -----------------------------------------------------------------------------
# _normalise_whitespace
# -----------------------------------------------------------------------------


def test_normalise_whitespace_collapses_nbsp_and_tabs() -> None:
    assert _normalise_whitespace("a\xa0b\t\tc\n d") == "a b c d"


# -----------------------------------------------------------------------------
# _select_targets
# -----------------------------------------------------------------------------


def _f11_stub() -> pd.DataFrame:
    """Synthetic F11 frame: pre-floor row with minutes, in-window rows with and
    without minutes (most-recent NaN gap), and an empty-string URL row."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["2006-10-04", "2008-02-06", "2015-02-04", "2020-03-19", "2026-05-06"]
            ),
            "publication_date": pd.to_datetime(
                ["2006-10-04", "2008-02-05", "2015-02-03", "2020-03-19", "2026-05-05"]
            ),
            "minutes_url": [
                "/monetary-policy/rba-board-minutes/2006/03102006.html",  # pre-floor
                "/monetary-policy/rba-board-minutes/2008/05022008.html",
                "/monetary-policy/rba-board-minutes/2015/2015-02-03.html",
                "/monetary-policy/rba-board-minutes/2020/2020-03-19.html",
                None,  # most recent meeting — minutes not yet published
            ],
            "change_raw": ["0.25", "0.25", "-0.25", "-0.25", "0.00"],
            "new_cash_rate_raw": ["6.00", "7.00", "2.25", "0.25", "3.85"],
            "statement_url": [None, None, None, None, None],
        }
    )


def test_select_targets_excludes_pre_floor_rows() -> None:
    out = _select_targets(_f11_stub())
    assert (out["decision_date"] >= _CALENDAR_VALIDATION_FLOOR).all()
    assert pd.Timestamp("2006-10-04") not in set(out["decision_date"])


def test_select_targets_excludes_nan_minutes_url() -> None:
    """The most-recent-meeting NaN gap is excluded, never treated as a miss."""
    out = _select_targets(_f11_stub())
    assert pd.Timestamp("2026-05-05") not in set(out["decision_date"])
    assert set(out["decision_date"]) == {
        pd.Timestamp("2008-02-05"),
        pd.Timestamp("2015-02-03"),
        pd.Timestamp("2020-03-19"),
    }


def test_select_targets_excludes_empty_string_url() -> None:
    f11 = _f11_stub()
    f11.loc[f11["publication_date"] == pd.Timestamp("2015-02-03"), "minutes_url"] = ""
    out = _select_targets(f11)
    assert pd.Timestamp("2015-02-03") not in set(out["decision_date"])


def test_select_targets_raises_on_missing_columns() -> None:
    bad = pd.DataFrame({"observation_date": [pd.Timestamp("2024-02-07")]})
    with pytest.raises(ValueError, match="missing required column"):
        _select_targets(bad)


# -----------------------------------------------------------------------------
# _attach_publication_dates / _softcheck_dc_date
# -----------------------------------------------------------------------------


def test_attach_publication_dates_uses_algorithmic_rule() -> None:
    df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-02-06", "2020-03-19"]),
            "dc_date": [pd.Timestamp("2024-02-20"), pd.Timestamp("2020-03-31")],
            "url": ["u1", "u2"],
        }
    )
    out = _attach_publication_dates(df)
    assert list(out["publication_date"]) == [
        pd.Timestamp("2024-02-20"),
        pd.Timestamp("2020-03-31"),
    ]
    assert (out["publication_date"] > out["decision_date"]).all()


def test_attach_publication_dates_ignores_artifact_dc_date() -> None:
    """An implausible dc.date (<= decision_date) is ignored; rule still wins."""
    df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2016-12-06"]),
            "dc_date": [pd.Timestamp("2016-11-15")],  # artifact, before the meeting
            "url": ["u"],
        }
    )
    out = _attach_publication_dates(df)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2016-12-20")


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


def test_softcheck_dc_date_warns_on_plausible_divergence(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=pd.Timestamp("2024-02-27"),  # plausible (after) but != rule
        publication_date=pd.Timestamp("2024-02-20"),
        decision_date=pd.Timestamp("2024-02-06"),
        url="u",
    )
    assert any("disagrees" in m for m in loguru_messages)


def test_softcheck_dc_date_silent_on_artifact(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=pd.Timestamp("2016-11-15"),  # before the announcement
        publication_date=pd.Timestamp("2016-12-20"),
        decision_date=pd.Timestamp("2016-12-06"),
        url="u",
    )
    assert not any("disagrees" in m for m in loguru_messages)


def test_softcheck_dc_date_silent_on_none(loguru_messages) -> None:
    _softcheck_dc_date(
        dc_date=None,
        publication_date=pd.Timestamp("2024-02-20"),
        decision_date=pd.Timestamp("2024-02-06"),
        url="u",
    )
    assert not any("disagrees" in m for m in loguru_messages)


# -----------------------------------------------------------------------------
# _validate_cross_check
# -----------------------------------------------------------------------------


def _minutes_row(decision: str, pub: str | None = None, slug: str = "2015-02-03") -> dict:
    pub = pub or rba_minutes_publication_date(pd.Timestamp(decision).date()).isoformat()
    return {
        "decision_date": pd.Timestamp(decision),
        "publication_date": pd.Timestamp(pub),
        "title": "Minutes of the Monetary Policy Meeting of the Reserve Bank Board",
        "url": f"https://www.rba.gov.au/monetary-policy/rba-board-minutes/2015/{slug}.html",
        "sha256": "deadbeef",
        "raw_html_filename": f"{decision}__{slug}.html",
        "body_text": "body",
        "paragraphs": ["body"],
    }


def test_validate_cross_check_passes_on_full_match() -> None:
    f11 = _f11_stub()
    minutes = pd.DataFrame(
        [
            _minutes_row("2008-02-05"),
            _minutes_row("2015-02-03"),
            _minutes_row("2020-03-19"),
        ]
    )
    _validate_cross_check(minutes, f11)  # no raise


def test_validate_cross_check_raises_on_missing_decision() -> None:
    f11 = _f11_stub()
    minutes = pd.DataFrame([_minutes_row("2008-02-05"), _minutes_row("2015-02-03")])
    with pytest.raises(ValueError, match="have no matching minutes"):
        _validate_cross_check(minutes, f11)


def test_validate_cross_check_raises_on_extra_row() -> None:
    f11 = _f11_stub()
    minutes = pd.DataFrame(
        [
            _minutes_row("2008-02-05"),
            _minutes_row("2015-02-03"),
            _minutes_row("2020-03-19"),
            _minutes_row("2050-01-01"),  # not in F11
        ]
    )
    with pytest.raises(ValueError, match="without a matching F11 decision"):
        _validate_cross_check(minutes, f11)


def test_validate_cross_check_raises_when_publication_not_after_decision() -> None:
    f11 = _f11_stub()
    minutes = pd.DataFrame(
        [
            _minutes_row("2008-02-05"),
            _minutes_row("2015-02-03"),
            _minutes_row("2020-03-19", pub="2020-03-19"),  # same day — leakage
        ]
    )
    with pytest.raises(ValueError, match="publication_date <= "):
        _validate_cross_check(minutes, f11)


def test_validate_cross_check_empty_with_no_targets_passes() -> None:
    f11 = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["1980-01-01"]),
            "publication_date": pd.to_datetime(["1980-01-01"]),
            "minutes_url": [None],
            "change_raw": ["0.00"],
            "new_cash_rate_raw": ["10.00"],
            "statement_url": [None],
        }
    )
    minutes = pd.DataFrame(
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
    _validate_cross_check(minutes, f11)  # no raise


# -----------------------------------------------------------------------------
# No-leakage invariant — minutes are released AFTER the decision
# -----------------------------------------------------------------------------


def test_no_leakage_publication_strictly_after_decision() -> None:
    for decision in ("2008-02-05", "2015-02-03", "2024-02-06", "2020-03-19"):
        pub = rba_minutes_publication_date(pd.Timestamp(decision).date())
        assert pd.Timestamp(pub) > pd.Timestamp(decision)
