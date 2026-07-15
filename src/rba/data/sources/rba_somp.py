"""RBA Statement on Monetary Policy (SoMP) — quarterly text payload.

Four times a year (historically February, May, August and November) the RBA
publishes its **Statement on Monetary Policy** — a long, multi-chapter document
setting out the Bank's assessment of the economy and the forecasts underpinning
the policy decision. The SoMP is the third text source for the project (after
:mod:`rba.data.sources.rba_media_releases` and
:mod:`rba.data.sources.rba_minutes`) and feeds the hawkish / dovish sentiment,
tone-shift and embedding features (CHECKLIST §5 ``Text features``).

This module inherits the **decision-keyed wide schema** established by
:mod:`rba.data.sources.rba_media_releases` — see CONTEXT.md "Text-data schema
convention" — with the same two differences the minutes module made: no
``governor`` column (the SoMP is a Bank document with no individual
attribution), and ``publication_date`` is *not* tied to ``decision_date`` by a
fixed sign (see below).

Why the SoMP diverges from the first two text sources
-----------------------------------------------------
The media-release and minutes scrapers were both *F11-driven* (keyed on
``f11.statement_url`` / ``f11.minutes_url``) and *1:1 with every meeting*. The
SoMP breaks both assumptions:

1. **Quarterly, not per-meeting.** The SoMP accompanies only the ~4 policy
   rounds per year that fall in February / May / August / November, so it joins
   to the meeting frame **sparsely** — most meetings have no SoMP. The frame is
   still decision-keyed (one row per accompanying Board meeting); it is simply
   absent for the other meetings.
2. **No F11 driver column.** There is no ``f11.somp_url``. The crawl is driven
   from the SoMP **archive index** (``/publications/smp/``), and each scraped
   issue is mapped to the Board meeting it accompanied by **calendar month**
   (the SoMP for month ``M`` of year ``Y`` is anchored to the F11 decision in
   that same month). The cross-check validates that mapping rather than an exact
   F11 URL set.
3. **Long multi-chapter document.** The modern SoMP spans an Overview page plus
   several chapter pages and "Box" inserts. v1 captures the **Overview page
   only** — the highest-signal, cross-era-comparable summary — fetched from the
   issue's Overview page, whose filename is ``overview.html`` (most issues),
   ``00-overview.html`` (a handful of mid-era issues) or ``intro.html`` (the
   2006–2014 era, which titled the summary "Introduction"); see
   :data:`_OVERVIEW_PAGES`.

Coverage floor (HTML-only) and the PDF era
-------------------------------------------
:data:`_CALENDAR_VALIDATION_FLOOR` is **2006-02-01**, the first quarterly SoMP
published as browsable HTML chapters. The quarterly SoMP itself began in 1997,
but the 1997–2005 issues are **PDF-only** (the per-month HTML folders return
HTTP 403 / 404; the year landing pages link only a ``boxes.html`` HTML page and
otherwise point at PDFs). Those issues are excluded from v1 by the floor and
documented as a coverage limitation rather than silently dropped; a future
iteration can add a PDF-extraction path to reach back to 1997. This floor is
SoMP-specific, **not** the project-wide 1993-01-01.

URL pattern and HTML structure observed
---------------------------------------
- Issue root: ``/publications/smp/<YYYY>/<mon>/`` where ``<mon>`` is one of
  ``feb`` / ``may`` / ``aug`` / ``nov``. The archive index lists exactly these
  issue roots (plus per-year ``boxes.html`` and cross-year navigation, which the
  enumerator ignores).
- Overview page: ``<issue>/overview.html`` (2015 → present) or
  ``<issue>/intro.html`` (2006 → 2014). Both carry the body in
  ``<div id="content">`` and the same ``<meta name="dc.date">`` marker as the
  issue root.
- Body container: ``<div id="content">`` is stable across every era probed
  (2006 → 2026). The pre-2015 summary is a flat ``<p>`` sequence; the 2015+ /
  2024-redesign Overview interleaves ``<h2>`` key-message headings with ``<p>``
  paragraphs (captured as standalone entries, like the minutes module).
- Title: the ``<h1>`` is ``"Statement on Monetary Policy – <Month YYYY>
  Overview"`` (or ``… Introduction`` pre-2015), whitespace-normalised.
- Publication date: SoMP pages carry **no** ``itemprop="datePublished"``. The
  only per-page date marker is ``<meta name="dc.date">`` — but unlike the
  minutes pages it is **reliable** on every era probed (it matches the genuine
  release date), so it is scraped as the **canonical** ``publication_date``.

Publication-date rule and the no-leakage sign
----------------------------------------------
``publication_date`` is the scraped ``dc.date``. There is deliberately **no**
release-calendar module: the relationship to the accompanying decision is
era-dependent and does not fit a clean algorithm —

- 2024 → present (8-meeting cadence): SoMP released the **same day** as the
  decision (``publication_date == decision_date``);
- ~2008 → 2023: SoMP released a few days **after** the decision (the Friday
  after the first-Tuesday meeting — ``+3`` days typical, ``+10`` observed);
- 2006 → 2007: SoMP occasionally released **before** the rate announcement
  (e.g. Feb 2006 SoMP on 2006-02-07, decision 2006-02-08).

Because the sign inverts across eras, the cross-check does **not** assert
``publication_date`` is before/after/equal to ``decision_date`` (unlike media
releases and minutes). Instead :func:`_validate_cross_check` enforces mapping
integrity and warns (never raises) when ``|publication_date - decision_date|``
exceeds :data:`_SOFT_PUBLICATION_GAP_DAYS` — a plausibility signal for a stray
``dc.date``, mirroring the minutes module's warn-only ``dc.date`` soft-check.

Network notes (RBA WAF)
-----------------------
Identical to the first two text modules: the RBA fronting WAF rejects a custom
``User-Agent`` header (HTTP 403) but accepts the default ``Python-urllib`` UA.
We call ``urlopen`` with the URL string directly and never wrap it in a
``Request`` carrying headers.

Cross-check :func:`_validate_cross_check` enforces:

- every in-window SoMP issue maps to exactly one F11 decision and yields exactly
  one row (misses are allow-listed in :data:`_KNOWN_MISSING_DECISIONS`, empty by
  default — treat misses as bugs, not data conditions);
- no SoMP row has a ``decision_date`` absent from F11 (a phantom mapping);
- no two SoMP rows share a ``decision_date`` (a mapping collision);
- ``|publication_date - decision_date|`` within :data:`_SOFT_PUBLICATION_GAP_DAYS`
  (warn-only — the sign is era-dependent).

Output schema
-------------
``fetch()`` returns a wide-format ``pandas.DataFrame`` with columns:

- ``decision_date`` (datetime64[ns]) — accompanying Board announcement day
  (== F11 publication_date) for the SoMP's calendar month.
- ``publication_date`` (datetime64[ns]) — scraped ``dc.date`` (the SoMP release
  day; may be ``<``, ``==`` or ``>`` ``decision_date`` by era).
- ``title`` (object) — SoMP Overview headline, whitespace-normalised.
- ``url`` (object) — fully-qualified Overview-page URL.
- ``sha256`` (object) — SHA-256 of the raw HTML bytes.
- ``raw_html_filename`` (object) — filename of the cached snapshot under
  ``data/raw/rba_somp/`` (``<YYYY-MM-DD>__<YYYY>-<mon>-<page>.html``; the date
  prefix is the decision date, ``<page>`` is ``overview`` or ``intro``).
- ``body_text`` (object) — Overview plain-text body, blocks joined by
  ``\\n\\n``.
- ``paragraphs`` (object) — ``list[str]`` of body blocks (``<h2>`` headings
  interleaved with ``<p>`` paragraphs; parallel to
  ``body_text.split("\\n\\n")``).

Running this module as ``__main__`` also materialises the wide frame at
``data/external/rba_somp.parquet``.

Side effects
------------
``fetch()`` writes under ``data/raw/rba_somp/``:

- ``<YYYY-MM-DD>__<YYYY>-<mon>-<page>.html`` — verbatim HTML bytes per issue
  (the date prefix is the decision date).
- ``_metadata.json`` — provenance manifest: one entry per SoMP document with
  URL, SHA-256, download timestamp (UTC), byte count, and decision_date.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import TypedDict
import urllib.error
import urllib.request

from bs4 import BeautifulSoup, Tag
from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR
from rba.data.sources import rba_f11

SOURCE_NAME = "rba_somp"
SOURCE_BASE_URL = "https://www.rba.gov.au"
SOURCE_INDEX_URL = "https://www.rba.gov.au/publications/smp/"

# First quarterly SoMP published as browsable HTML chapters. The quarterly SoMP
# began in 1997, but the 1997–2005 issues are PDF-only (per-month HTML folders
# 403/404) and excluded from v1 by this floor. SoMP-specific — NOT the
# project-wide 1993-01-01 floor.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("2006-02-01")

# Decisions intentionally allowed to have no matching SoMP row. Empty by
# default — missing SoMP issues are bugs, not data conditions.
_KNOWN_MISSING_DECISIONS: frozenset[pd.Timestamp] = frozenset()

# A scraped dc.date further than this from the accompanying decision triggers a
# warning (never a raise). Across the full 2006 → 2026 corpus the genuine gap
# spans -14 days (pre-2008 February issues released ~2 weeks *before* the
# decision) to +24 days (November issues historically released ~3 weeks
# *after*), all within the same policy round. A threshold of 31 days (≈ one
# month) therefore never warns on legitimate data while still catching a dc.date
# that points to a wholly different month — i.e. a CMS artifact or a mis-mapped
# issue. The sign is era-dependent and is deliberately NOT asserted.
_SOFT_PUBLICATION_GAP_DAYS = 31

# SoMP issue months and their calendar month numbers.
_ISSUE_MONTHS: dict[str, int] = {"feb": 2, "may": 5, "aug": 8, "nov": 11}

# Overview-page candidates, in preference order. Three filename schemes are
# observed across the HTML era (probed over every in-window issue 2006 → 2026):
# ``overview.html`` (most issues), ``00-overview.html`` (a handful of mid-era
# issues that numbered their chapter files), and ``intro.html`` (2006–2014
# issues, which titled the summary "Introduction"). The first that resolves
# (HTTP 200) is used.
_OVERVIEW_PAGES: tuple[str, ...] = ("overview", "00-overview", "intro")

# Issue-root link pattern in the archive index (trailing slash, month segment).
_ISSUE_LINK_RE = re.compile(r"^/publications/smp/(\d{4})/(feb|may|aug|nov)/$")

# Ancestor classes/tags whose <h2>/<p> descendants are excluded from body
# extraction. Reuses the minutes / media-release CMS set plus the SoMP-specific
# in-page contents rail (``nav-publication-contents``).
_EXCLUDE_ANCESTOR_CLASSES: frozenset[str] = frozenset(
    {
        "box-enquiries",
        "box-article-info",
        "rss-mr-contact",
        "nav-page-contents",
        "nav-publication-contents",
        "aside-component",
        "monetary-policy-related-links",
        "publication-related-links",
        "tile-container",
        "clickable",
        "complementary",
    }
)
_EXCLUDE_ANCESTOR_TAGS: frozenset[str] = frozenset({"aside", "footer", "nav"})

# Trailing navigation headings (case-insensitive) dropped from the body.
_NAV_HEADINGS: frozenset[str] = frozenset({"related information", "related content"})


def fetch(
    *,
    f11_meetings: pd.DataFrame | None = None,
    force_download: bool = True,
) -> pd.DataFrame:
    """Pull every in-window SoMP Overview document from the archive index.

    Parameters
    ----------
    f11_meetings
        Pre-loaded F11 meetings frame (output of
        :func:`rba.data.sources.rba_f11.fetch`). If ``None``, ``rba_f11.fetch``
        is called with ``force_download=False`` to reuse the most recent cached
        snapshot — the SoMP crawl does not need a fresh F11 download.
    force_download
        If True (default), refresh every SoMP Overview HTML from the live RBA
        endpoint. If False, reuse the dated snapshots already under
        ``data/raw/rba_somp/`` and only download missing files. The archive
        index is always fetched fresh (it is small and drives enumeration).

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``decision_date``.

    Shapes
    ------
    Returns: (n_issues, 8) where ``n_issues`` is the count of SoMP issues in the
    archive index on/after :data:`_CALENDAR_VALIDATION_FLOOR`.
    """
    if f11_meetings is None:
        f11_meetings = rba_f11.fetch(force_download=False)

    index_html = _download_index()
    targets = _select_targets(index_html, f11_meetings)
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        decision_date = pd.Timestamp(target["decision_date"])
        issue_path = str(target["issue_path"])
        snapshot_path, entry, raw_bytes, full_url = _resolve_snapshot(
            dest_dir,
            decision_date=decision_date,
            issue_path=issue_path,
            force_download=force_download,
        )
        parsed = _parse(raw_bytes, decision_date=decision_date)
        rows.append(
            {
                "decision_date": decision_date,
                "publication_date": parsed["publication_date"],
                "title": parsed["title"],
                "url": full_url,
                "sha256": entry["sha256"],
                "raw_html_filename": snapshot_path.name,
                "body_text": parsed["body_text"],
                "paragraphs": parsed["paragraphs"],
            }
        )
        manifest.append(entry)

    _write_manifest(dest_dir, manifest)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["decision_date"] = pd.to_datetime(df["decision_date"]).astype("datetime64[ns]")
        df["publication_date"] = pd.to_datetime(df["publication_date"]).astype(
            "datetime64[ns]"
        )

    _validate_cross_check(df, targets, f11_meetings)

    return df.sort_values("decision_date").reset_index(drop=True)


def _download_index() -> bytes:
    """Download the SoMP archive index HTML (the enumeration source)."""
    logger.info("Downloading SoMP archive index {}", SOURCE_INDEX_URL)
    try:
        with urllib.request.urlopen(SOURCE_INDEX_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download SoMP index {SOURCE_INDEX_URL}: HTTP {exc.code}. "
            "Note: the RBA WAF rejects custom User-Agent headers — the default "
            "Python-urllib UA is required."
        ) from exc


def _enumerate_issues(index_html: bytes) -> list[tuple[int, str, str]]:
    """Enumerate in-window SoMP issues from the archive index HTML.

    Parameters
    ----------
    index_html
        Raw bytes of ``/publications/smp/``.

    Returns
    -------
    list[tuple[int, str, str]]
        One ``(year, mon, issue_path)`` per issue root link matching
        :data:`_ISSUE_LINK_RE` and on/after
        :data:`_CALENDAR_VALIDATION_FLOOR`, where ``mon`` is one of
        ``feb``/``may``/``aug``/``nov`` and ``issue_path`` is the relative issue
        root (with trailing slash). De-duplicated and sorted ascending by
        ``(year, month-number)``.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    seen: dict[tuple[int, str], str] = {}
    for anchor in soup.find_all("a", href=True):
        match = _ISSUE_LINK_RE.match(str(anchor["href"]))
        if match is None:
            continue
        year = int(match.group(1))
        mon = match.group(2)
        issue_start = pd.Timestamp(year=year, month=_ISSUE_MONTHS[mon], day=1)
        if issue_start < _CALENDAR_VALIDATION_FLOOR:
            continue
        seen[(year, mon)] = match.group(0)
    return [
        (year, mon, path)
        for (year, mon), path in sorted(
            seen.items(), key=lambda kv: (kv[0][0], _ISSUE_MONTHS[kv[0][1]])
        )
    ]


def _select_targets(index_html: bytes, f11_meetings: pd.DataFrame) -> pd.DataFrame:
    """Map every in-window SoMP issue to its accompanying F11 decision.

    Each SoMP issue for calendar month ``M`` of year ``Y`` is anchored to the
    single F11 decision in that same month. A SoMP month with **no** F11
    decision, or with **more than one**, is an error (the SoMP-to-meeting mapping
    is otherwise ambiguous).

    Parameters
    ----------
    index_html
        Raw bytes of the archive index (input to :func:`_enumerate_issues`).
    f11_meetings
        Output of :func:`rba.data.sources.rba_f11.fetch`. Must contain at least
        ``observation_date`` and ``publication_date``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``decision_date`` (datetime64[ns], == F11 publication_date),
        ``year`` (int), ``mon`` (object), ``issue_path`` (object, relative issue
        root). Rows sorted ascending by ``decision_date``.

    Shapes
    ------
    Returns: (n_issues, 4).
    """
    needed = {"observation_date", "publication_date"}
    missing = needed - set(f11_meetings.columns)
    if missing:
        raise ValueError(
            f"f11_meetings missing required column(s): {sorted(missing)}. "
            f"Got columns: {list(f11_meetings.columns)}"
        )

    pubs = pd.to_datetime(f11_meetings["publication_date"])
    rows: list[dict[str, object]] = []
    for year, mon, issue_path in _enumerate_issues(index_html):
        month_num = _ISSUE_MONTHS[mon]
        same_month = pubs[(pubs.dt.year == year) & (pubs.dt.month == month_num)]
        if len(same_month) == 0:
            raise ValueError(
                f"SoMP issue {year}-{mon} has no F11 decision in {year}-"
                f"{month_num:02d}; cannot anchor it to a Board meeting. "
                "Investigate the F11 frame or the SoMP archive index."
            )
        if len(same_month) > 1:
            raise ValueError(
                f"SoMP issue {year}-{mon} maps to {len(same_month)} F11 decisions "
                f"in {year}-{month_num:02d}: "
                f"{[d.date() for d in sorted(same_month)]}. Mapping is ambiguous."
            )
        rows.append(
            {
                "decision_date": same_month.iloc[0],
                "year": year,
                "mon": mon,
                "issue_path": issue_path,
            }
        )

    df = pd.DataFrame(rows, columns=["decision_date", "year", "mon", "issue_path"])
    if not df.empty:
        df["decision_date"] = pd.to_datetime(df["decision_date"]).astype("datetime64[ns]")
    return df.sort_values("decision_date").reset_index(drop=True)


def _resolve_snapshot(
    dest_dir: Path,
    *,
    decision_date: pd.Timestamp,
    issue_path: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes, str]:
    """Return (snapshot path, manifest entry, raw bytes, full URL) for one issue.

    Tries the Overview-page candidates in :data:`_OVERVIEW_PAGES` order
    (``overview.html`` then ``intro.html``). On reuse (``force_download=False``)
    the first matching cached snapshot wins; on download the first candidate that
    resolves (HTTP 200) wins.
    """
    year, mon = _issue_year_mon(issue_path)
    date_str = decision_date.strftime("%Y-%m-%d")

    if not force_download:
        for page in _OVERVIEW_PAGES:
            snapshot_path = dest_dir / f"{date_str}__{year}-{mon}-{page}.html"
            if snapshot_path.exists():
                raw_bytes = snapshot_path.read_bytes()
                full_url = SOURCE_BASE_URL + issue_path + f"{page}.html"
                entry: dict[str, object] = {
                    "decision_date": date_str,
                    "url": full_url,
                    "snapshot_filename": snapshot_path.name,
                    "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "bytes": len(raw_bytes),
                    "downloaded_at_utc": None,
                    "reused": True,
                }
                logger.info("Reusing cached SoMP at {}", snapshot_path)
                return snapshot_path, entry, raw_bytes, full_url

    return _download(dest_dir, issue_path=issue_path, date_str=date_str)


def _download(
    dest_dir: Path, *, issue_path: str, date_str: str
) -> tuple[Path, dict[str, object], bytes, str]:
    """Download an issue's Overview HTML (overview.html → intro.html fallback).

    The RBA WAF rejects custom ``User-Agent`` headers (403) but accepts the
    default ``Python-urllib`` UA. We pass the URL string directly to
    :func:`urllib.request.urlopen` and never wrap it in a ``Request`` carrying
    headers.
    """
    year, mon = _issue_year_mon(issue_path)
    last_error: urllib.error.HTTPError | None = None
    for page in _OVERVIEW_PAGES:
        full_url = SOURCE_BASE_URL + issue_path + f"{page}.html"
        logger.info("Downloading {} ", full_url)
        try:
            with urllib.request.urlopen(full_url, timeout=60) as response:  # noqa: S310 — public RBA URL
                raw_bytes = bytes(response.read())
        except urllib.error.HTTPError as exc:
            last_error = exc
            continue
        snapshot_path = dest_dir / f"{date_str}__{year}-{mon}-{page}.html"
        snapshot_path.write_bytes(raw_bytes)
        entry: dict[str, object] = {
            "decision_date": date_str,
            "url": full_url,
            "snapshot_filename": snapshot_path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "reused": False,
        }
        return snapshot_path, entry, raw_bytes, full_url

    code = last_error.code if last_error is not None else "?"
    raise RuntimeError(
        f"No Overview page resolved for SoMP issue {year}-{mon} "
        f"(tried {_OVERVIEW_PAGES}); last HTTP {code}. "
        "Note: the RBA WAF rejects custom User-Agent headers — the default "
        "Python-urllib UA is required."
    )


def _issue_year_mon(issue_path: str) -> tuple[int, str]:
    """Return ``(year, mon)`` parsed from a SoMP issue root path."""
    match = re.search(r"/publications/smp/(\d{4})/(feb|may|aug|nov)/", issue_path)
    if match is None:
        raise ValueError(f"Unrecognised SoMP issue path: {issue_path!r}")
    return int(match.group(1)), match.group(2)


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest ({} entries) to {}", len(manifest), metadata_path)


class _ParsedSoMP(TypedDict):
    """Typed shape of :func:`_parse`'s output — one SoMP Overview document."""

    title: str
    publication_date: pd.Timestamp
    paragraphs: list[str]
    body_text: str


def _parse(html: bytes, *, decision_date: pd.Timestamp) -> _ParsedSoMP:
    """Parse one SoMP Overview HTML into title / publication_date / body.

    Parameters
    ----------
    html
        Raw HTML bytes from the Overview page.
    decision_date
        Decision date this document corresponds to, used only for error context.

    Returns
    -------
    dict
        Keys: ``title`` (str), ``publication_date`` (pd.Timestamp — scraped from
        ``<meta name="dc.date">``), ``paragraphs`` (list[str]), ``body_text``
        (str — blocks joined by ``"\\n\\n"``).
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="content")
    if content is None:
        raise ValueError(
            f"No <div id='content'> in SoMP HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )

    heading = content.find("h1")
    if heading is None:
        raise ValueError(
            f"No <h1> title in SoMP HTML for decision {decision_date.date()}. "
            "Page layout may have changed."
        )
    title = _normalise_whitespace(heading.get_text(" ", strip=True))

    publication_date = _extract_dc_date(soup)
    if publication_date is None:
        raise ValueError(
            f"No parseable <meta name='dc.date'> in SoMP HTML for decision "
            f"{decision_date.date()}. dc.date is the canonical publication date "
            "for this source; page layout may have changed."
        )

    paragraphs = _extract_blocks(content)
    if not paragraphs:
        raise ValueError(
            f"Extracted zero body blocks from SoMP HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )
    body_text = "\n\n".join(paragraphs)

    return {
        "title": title,
        "publication_date": publication_date,
        "paragraphs": paragraphs,
        "body_text": body_text,
    }


def _extract_dc_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """Return the ``<meta name='dc.date'>`` value as a Timestamp, or None.

    Unlike the minutes pages (where ``dc.date`` is ~10% CMS artifacts and used
    only as a soft check), the SoMP ``dc.date`` is reliable and used as the
    canonical ``publication_date``. A missing or unparseable value yields
    ``None`` (the caller raises).
    """
    meta = soup.find("meta", attrs={"name": "dc.date"})
    if meta is None:
        return None
    content = meta.get("content")
    if not content:
        return None
    parsed = pd.to_datetime(str(content), format="%Y-%m-%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _extract_blocks(content_div: Tag) -> list[str]:
    """Extract body blocks (``<h2>`` headings + ``<p>`` paragraphs) in order.

    Walks every ``<h2>`` and ``<p>`` under ``content_div`` in document order,
    excluding any whose ancestor chain crosses an element in
    :data:`_EXCLUDE_ANCESTOR_CLASSES` or :data:`_EXCLUDE_ANCESTOR_TAGS`
    (in-page contents rail, related-links rails, contact box, navigation asides,
    tiles). ``<h2>`` key-message headings are kept as standalone entries
    interleaved with their following ``<p>`` content; trailing navigation
    headings in :data:`_NAV_HEADINGS` are dropped.
    """
    out: list[str] = []
    for el in content_div.find_all(["h2", "p"]):
        if _has_excluded_ancestor(el, content_div):
            continue
        text = _normalise_whitespace(el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name == "h2" and text.lower() in _NAV_HEADINGS:
            continue
        out.append(text)
    return out


def _has_excluded_ancestor(node: Tag, container: Tag) -> bool:
    """Return True if any ancestor of ``node`` (up to ``container``) is excluded."""
    for ancestor in node.parents:
        if ancestor is container:
            return False
        if ancestor.name in _EXCLUDE_ANCESTOR_TAGS:
            return True
        classes = ancestor.get_attribute_list("class")
        if set(classes) & _EXCLUDE_ANCESTOR_CLASSES:
            return True
    return False


def _normalise_whitespace(text: str) -> str:
    """Collapse all internal whitespace (including NBSP) to single spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _validate_cross_check(
    somp: pd.DataFrame, targets: pd.DataFrame, f11_meetings: pd.DataFrame
) -> None:
    """Enforce the SoMP ↔ F11 mapping invariants.

    Raises ``ValueError`` if:

    - any in-window SoMP issue (a ``targets`` row) has no matching SoMP output
      row (unless allow-listed in :data:`_KNOWN_MISSING_DECISIONS`);
    - any SoMP row has no matching target (an extra not derived from the index);
    - any SoMP row's ``decision_date`` is absent from F11 (a phantom mapping);
    - two SoMP rows share a ``decision_date`` (a mapping collision).

    Warns (never raises) when ``|publication_date - decision_date|`` exceeds
    :data:`_SOFT_PUBLICATION_GAP_DAYS`: the sign is era-dependent (the SoMP can
    be released before, on, or after the decision), so only an implausibly large
    gap is flagged.
    """
    expected = set(pd.to_datetime(targets["decision_date"])) if not targets.empty else set()
    actual = set(pd.to_datetime(somp["decision_date"])) if not somp.empty else set()

    allow = {pd.Timestamp(d) for d in _KNOWN_MISSING_DECISIONS}
    missing = (expected - actual) - allow
    extra = actual - expected

    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} in-window SoMP issue(s) have no matching output "
            f"row. Sample decision dates: {[d.date() for d in sample]}. "
            "Investigate the Overview download/parse or add to "
            "_KNOWN_MISSING_DECISIONS if a known data gap."
        )
    if extra:
        sample = sorted(extra)[:5]
        raise ValueError(
            f"{len(extra)} SoMP row(s) not derived from the archive index: "
            f"{[d.date() for d in sample]}. Stale cache, or a manual ingest of a "
            "document the index doesn't list?"
        )

    if somp.empty:
        return

    f11_decisions = set(pd.to_datetime(f11_meetings["publication_date"]))
    phantom = actual - f11_decisions
    if phantom:
        sample = sorted(phantom)[:5]
        raise ValueError(
            f"{len(phantom)} SoMP row(s) whose decision_date is absent from F11: "
            f"{[d.date() for d in sample]}. The month-mapping produced a phantom "
            "decision."
        )

    dupes = somp["decision_date"][somp["decision_date"].duplicated()]
    if not dupes.empty:
        sample = sorted(set(dupes))[:5]
        raise ValueError(
            f"{len(set(dupes))} decision_date(s) mapped by more than one SoMP "
            f"issue: {[pd.Timestamp(d).date() for d in sample]}. Mapping collision."
        )

    for _, row in somp.iterrows():
        _softcheck_publication_gap(
            publication_date=row["publication_date"],
            decision_date=row["decision_date"],
            url=str(row.get("url", "")),
        )


def _softcheck_publication_gap(
    *,
    publication_date: pd.Timestamp,
    decision_date: pd.Timestamp,
    url: str,
) -> None:
    """Warn (never raise) if the SoMP release is implausibly far from the decision.

    The SoMP can be released before, on, or after its accompanying decision
    depending on the era, so no sign is asserted; only an absolute gap larger
    than :data:`_SOFT_PUBLICATION_GAP_DAYS` is flagged as a plausibility signal
    for a stray ``dc.date`` or a mis-mapped issue.
    """
    if publication_date is None or pd.isna(publication_date):
        return
    gap_days = abs((pd.Timestamp(publication_date) - pd.Timestamp(decision_date)).days)
    if gap_days > _SOFT_PUBLICATION_GAP_DAYS:
        logger.warning(
            "SoMP publication_date {} is {} days from decision_date {} ({}); "
            "expected within {} days. Soft check only; scraped dc.date retained. "
            "Verify the issue-to-meeting mapping and the page's dc.date.",
            pd.Timestamp(publication_date).date(),
            gap_days,
            pd.Timestamp(decision_date).date(),
            url,
            _SOFT_PUBLICATION_GAP_DAYS,
        )


if __name__ == "__main__":
    df = fetch()
    dest = EXTERNAL_DATA_DIR / "rba_somp.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(
        "Wrote {} SoMP documents (decision range {} → {}) to {}",
        len(df),
        df["decision_date"].min().date() if not df.empty else "n/a",
        df["decision_date"].max().date() if not df.empty else "n/a",
        dest,
    )
