"""RBA Board minutes — text payload per monetary-policy meeting.

Roughly two weeks after every monetary-policy Board meeting the RBA publishes
the **minutes** of that meeting — a ~30-50 paragraph document covering the
international and domestic economic discussion, financial-market conditions,
and the considerations leading to the decision. The minutes are the second
text source for the project (after :mod:`rba.data.sources.rba_media_releases`)
and feed the hawkish / dovish sentiment, tone-shift, and embedding features
(CHECKLIST §5 ``Text features``).

This module inherits the **decision-keyed wide schema** established by
:mod:`rba.data.sources.rba_media_releases` — see CONTEXT.md "Text-data schema
convention". The only schema differences are:

- no ``governor`` column — the minutes are a Board document with no individual
  attribution; and
- ``publication_date > decision_date`` (minutes are released ~14 days *after*
  the announcement) rather than equal to it.

Why F11-driven crawl, not an archive index?
--------------------------------------------
:mod:`rba.data.sources.rba_f11` already populates ``minutes_url`` on every
meeting row that has published minutes (NaN otherwise — the most recent
meeting always has ``minutes_url=NaN`` for ~2 weeks until its minutes are
released). Driving the crawl directly from ``F11.minutes_url`` gives an exact
one-to-one expected set and an exact cross-check, with no archive-index
scraping or per-era title regex. This mirrors the media-release module exactly.

The structural "most recent meeting has no minutes yet" gap is *not* a data
condition to allow-list: ``_select_targets`` filters on ``minutes_url.notna()``
so a NaN row is simply not a target, never a miss.

URL pattern and HTML structure observed
---------------------------------------
- URL slug: ``/monetary-policy/rba-board-minutes/<YYYY>/<slug>.html`` where the
  modern slug is the meeting date ``<YYYY-MM-DD>`` and the pre-2009 slug is
  ``<DDMMYYYY>``. F11 carries the exact URL, so the slug form is irrelevant to
  this module.
- Body container: ``<div id="content">`` is stable across every era probed
  (2006 → 2026). Inside, the body is a flat sequence of ``<h2>`` section
  headings and ``<p>`` paragraphs in document order; the first ``<p>`` is
  always a ``Sydney - <D Month YYYY>`` location/date line.
- Title: the ``<h1>`` page title is uniform every era —
  ``"Minutes of the Monetary Policy Meeting of the Reserve Bank Board"``.
- Publication date: minutes pages carry **no** ``itemprop="datePublished"``.
  The only per-page date marker is ``<meta name="dc.date">``, which is
  unreliable (~10% CMS artifacts), so the publication date is computed
  algorithmically (see below) and ``dc.date`` is used only as a warn-only soft
  cross-check.

Embedded section headings
--------------------------
Unlike media releases, minutes contain ``<h2>`` section headings (e.g.
``Members present``, ``Financial Markets``, ``International economic
developments``, ``The Decision``). These are captured as **standalone entries**
in the ``paragraphs`` list, interleaved with their following ``<p>`` content,
so downstream features can split / filter on heading prefixes. Trailing
navigation headings (``Related Information`` / ``Related Content``) are
dropped, as are related-links / contact / aside blocks via the same ancestor
exclusion the media-release module uses.

Publication-date rule
----------------------
``publication_date`` is the **second Tuesday strictly after the meeting** —
``decision_date + 14 days`` for the regular Tuesday cadence, and the second
Tuesday after a non-Tuesday meeting (the COVID out-of-cycle Thursday meeting
2020-03-19 → 2020-03-31). The rule (and its ``_OVERRIDES`` dict) lives in
:mod:`rba.data.rba_minutes_release_calendar`, mirroring
:mod:`rba.data.rba_i2_release_calendar`. The scraped ``dc.date`` is compared
against the algorithmic value per row and a divergence is *warned* (not raised)
when the ``dc.date`` is plausible (strictly after the announcement); implausible
``dc.date`` values (lag <= 0 — the known CMS artifacts) are ignored silently.

Network notes (RBA WAF)
-----------------------
Identical to the media-release module: the RBA fronting WAF rejects a custom
``User-Agent`` header (HTTP 403) but accepts the default ``Python-urllib`` UA.
We call ``urlopen`` with the URL string directly and never wrap it in a
``Request`` carrying headers.

Decision-date convention
------------------------
``decision_date`` is the Board *announcement* day, identical to
:attr:`rba_f11.publication_date` (one calendar day before the *effective* date
from Feb 2008 onward) — the same anchor the media-release module uses. The
``minutes_url`` slug date (the meeting day) can differ from ``decision_date``
by a day in the pre-2008 era; F11's ``publication_date`` is the canonical
anchor.

Cross-check :func:`_validate_cross_check` enforces:

- every F11 row on/after :data:`_CALENDAR_VALIDATION_FLOOR` with a
  ``minutes_url`` yields exactly one minutes row;
- no minutes row lacks a matching F11 decision;
- every minutes row has ``publication_date > decision_date`` (minutes are
  released later — the no-leakage sign flips vs media releases);
- known-missing decisions live in :data:`_KNOWN_MISSING_DECISIONS` (empty by
  default — treat misses as bugs, not data conditions).

Output schema
-------------
``fetch()`` returns a wide-format ``pandas.DataFrame`` with columns:

- ``decision_date`` (datetime64[ns]) — Board announcement day (== F11
  publication_date).
- ``publication_date`` (datetime64[ns]) — algorithmic minutes release date
  (``> decision_date``).
- ``title`` (object) — minutes headline, whitespace-normalised.
- ``url`` (object) — fully-qualified minutes URL.
- ``sha256`` (object) — SHA-256 of the raw HTML bytes.
- ``raw_html_filename`` (object) — filename of the cached snapshot under
  ``data/raw/rba_minutes/`` (``<YYYY-MM-DD>__<slug>.html``; date prefix is the
  decision date).
- ``body_text`` (object) — full plain-text body, blocks joined by ``\\n\\n``.
- ``paragraphs`` (object) — ``list[str]`` of body blocks (``<h2>`` headings
  interleaved with ``<p>`` paragraphs; parallel to
  ``body_text.split("\\n\\n")``).

Running this module as ``__main__`` also materialises the wide frame at
``data/external/rba_minutes.parquet``.

Side effects
------------
``fetch()`` writes under ``data/raw/rba_minutes/``:

- ``<YYYY-MM-DD>__<slug>.html`` — verbatim HTML bytes per meeting (the date
  prefix is the decision date).
- ``_metadata.json`` — provenance manifest: one entry per minutes document with
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
from rba.data.rba_minutes_release_calendar import rba_minutes_publication_date
from rba.data.sources import rba_f11

SOURCE_NAME = "rba_minutes"
SOURCE_BASE_URL = "https://www.rba.gov.au"

# Start of the regular minutes-publication regime (second Tuesday after each
# meeting). The 2006-10 → 2007-11 minutes exist in F11 but were backfilled and
# only published ~2007-12-05, so the algorithmic rule does not describe them;
# they are excluded by this floor. NOT the project-wide 1993-01-01 floor.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("2008-02-05")

# Decisions intentionally allowed to have no matching minutes row. Empty by
# default — missing minutes are bugs, not data conditions. The structural
# "most recent meeting has no minutes for ~2 weeks" gap is handled by the
# minutes_url.notna() filter in _select_targets, NOT by this allowlist.
_KNOWN_MISSING_DECISIONS: frozenset[pd.Timestamp] = frozenset()

# Ancestor classes/tags whose <h2>/<p> descendants are excluded from body
# extraction (related-links rails, contact box, navigation asides, tiles).
# Same RBA CMS as the media-release module.
_EXCLUDE_ANCESTOR_CLASSES: frozenset[str] = frozenset(
    {
        "box-enquiries",
        "box-article-info",
        "rss-mr-contact",
        "nav-page-contents",
        "aside-component",
        "monetary-policy-related-links",
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
    """Pull every Board-minutes document referenced by F11.

    Parameters
    ----------
    f11_meetings
        Pre-loaded F11 meetings frame (output of
        :func:`rba.data.sources.rba_f11.fetch`). If ``None``, ``rba_f11.fetch``
        is called with ``force_download=False`` to reuse the most recent cached
        snapshot — the minutes crawl does not need a fresh F11 download.
    force_download
        If True (default), refresh every minutes HTML from the live RBA
        endpoint. If False, reuse the dated snapshots already under
        ``data/raw/rba_minutes/`` and only download missing files.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``decision_date``.

    Shapes
    ------
    Returns: (n_decisions, 8) where ``n_decisions`` is the count of F11 rows
    on/after :data:`_CALENDAR_VALIDATION_FLOOR` with a ``minutes_url``.
    """
    if f11_meetings is None:
        f11_meetings = rba_f11.fetch(force_download=False)

    targets = _select_targets(f11_meetings)
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        decision_date = pd.Timestamp(target["decision_date"])
        url_path = str(target["minutes_url"])
        snapshot_path, entry, raw_bytes = _resolve_snapshot(
            dest_dir,
            decision_date=decision_date,
            url_path=url_path,
            force_download=force_download,
        )
        parsed = _parse(raw_bytes, decision_date=decision_date)
        rows.append(
            {
                "decision_date": decision_date,
                "dc_date": parsed["dc_date"],
                "title": parsed["title"],
                "url": SOURCE_BASE_URL + url_path,
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
        df = _attach_publication_dates(df)
        df = df.drop(columns=["dc_date"])
        df = df[
            [
                "decision_date",
                "publication_date",
                "title",
                "url",
                "sha256",
                "raw_html_filename",
                "body_text",
                "paragraphs",
            ]
        ]

    _validate_cross_check(df, f11_meetings)

    return df.sort_values("decision_date").reset_index(drop=True)


def _select_targets(f11_meetings: pd.DataFrame) -> pd.DataFrame:
    """Filter F11 to in-window decisions that have a ``minutes_url``.

    Parameters
    ----------
    f11_meetings
        Output of :func:`rba.data.sources.rba_f11.fetch`. Must contain at least
        ``observation_date``, ``publication_date``, and ``minutes_url``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``decision_date`` (datetime64[ns], == F11 publication_date),
        ``minutes_url`` (object, relative path from the RBA root). Rows sorted
        ascending by ``decision_date``. Rows with ``minutes_url=NaN`` (e.g. the
        most recent meeting, whose minutes are not yet published) are excluded.

    Shapes
    ------
    Returns: (n_targets, 2).
    """
    needed = {"observation_date", "publication_date", "minutes_url"}
    missing = needed - set(f11_meetings.columns)
    if missing:
        raise ValueError(
            f"f11_meetings missing required column(s): {sorted(missing)}. "
            f"Got columns: {list(f11_meetings.columns)}"
        )

    df = f11_meetings.copy()
    df = df[df["publication_date"] >= _CALENDAR_VALIDATION_FLOOR]
    has_url = df["minutes_url"].notna() & (df["minutes_url"].astype(str).str.len() > 0)
    df = df[has_url]
    df = df.assign(decision_date=pd.to_datetime(df["publication_date"]))
    return (
        df[["decision_date", "minutes_url"]]
        .sort_values("decision_date")
        .reset_index(drop=True)
    )


def _resolve_snapshot(
    dest_dir: Path,
    *,
    decision_date: pd.Timestamp,
    url_path: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one minutes doc."""
    slug = url_path.rsplit("/", 1)[-1].removesuffix(".html")
    date_str = decision_date.strftime("%Y-%m-%d")
    snapshot_path = dest_dir / f"{date_str}__{slug}.html"
    full_url = SOURCE_BASE_URL + url_path

    if snapshot_path.exists() and not force_download:
        raw_bytes = snapshot_path.read_bytes()
        entry: dict[str, object] = {
            "decision_date": date_str,
            "url": full_url,
            "snapshot_filename": snapshot_path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        logger.info("Reusing cached minutes at {}", snapshot_path)
        return snapshot_path, entry, raw_bytes

    return _download(snapshot_path, full_url=full_url, date_str=date_str)


def _download(
    snapshot_path: Path, *, full_url: str, date_str: str
) -> tuple[Path, dict[str, object], bytes]:
    """Download one minutes HTML, save dated snapshot, return path + entry + bytes.

    The RBA WAF rejects custom ``User-Agent`` headers (403) but accepts the
    default ``Python-urllib`` UA. We therefore pass the URL string directly to
    :func:`urllib.request.urlopen` and never wrap it in a ``Request`` carrying
    headers.
    """
    logger.info("Downloading {} -> {}", full_url, snapshot_path)
    try:
        with urllib.request.urlopen(full_url, timeout=60) as response:  # noqa: S310 — public RBA URL
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {full_url}: HTTP {exc.code}. "
            "Note: the RBA WAF rejects custom User-Agent headers — the default "
            "Python-urllib UA is required."
        ) from exc
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
    return snapshot_path, entry, raw_bytes


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest ({} entries) to {}", len(manifest), metadata_path)


class _ParsedMinutes(TypedDict):
    """Typed shape of :func:`_parse`'s output — one minutes document."""

    title: str
    dc_date: pd.Timestamp | None
    paragraphs: list[str]
    body_text: str


def _parse(html: bytes, *, decision_date: pd.Timestamp) -> _ParsedMinutes:
    """Parse one minutes HTML into title / dc_date / body.

    Parameters
    ----------
    html
        Raw HTML bytes from the minutes page.
    decision_date
        Decision date this document corresponds to, used only for error context.

    Returns
    -------
    dict
        Keys: ``title`` (str), ``dc_date`` (pd.Timestamp | None — parsed from
        ``<meta name="dc.date">``, used only for the soft cross-check),
        ``paragraphs`` (list[str]), ``body_text`` (str — blocks joined by
        ``"\\n\\n"``).
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="content")
    if content is None:
        raise ValueError(
            f"No <div id='content'> in minutes HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )

    heading = content.find("h1")
    if heading is None:
        raise ValueError(
            f"No <h1> title in minutes HTML for decision {decision_date.date()}. "
            "Page layout may have changed."
        )
    title = _normalise_whitespace(heading.get_text(" ", strip=True))

    dc_date = _extract_dc_date(soup)

    paragraphs = _extract_blocks(content)
    if not paragraphs:
        raise ValueError(
            f"Extracted zero body blocks from minutes HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )
    body_text = "\n\n".join(paragraphs)

    return {
        "title": title,
        "dc_date": dc_date,
        "paragraphs": paragraphs,
        "body_text": body_text,
    }


def _extract_dc_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """Return the ``<meta name='dc.date'>`` value as a Timestamp, or None.

    Minutes pages carry no ``itemprop='datePublished'``; ``dc.date`` is the only
    per-page date marker and is unreliable (~10% CMS artifacts), so it is used
    only as a warn-only soft cross-check in :func:`_attach_publication_dates`.
    A missing or unparseable value yields ``None`` (no raise).
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
    (related-links rails, contact box, navigation asides, tiles). ``<h2>``
    section headings are kept as standalone entries interleaved with their
    following ``<p>`` content; trailing navigation headings in
    :data:`_NAV_HEADINGS` are dropped.
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


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add the algorithmic ``publication_date`` column and soft-check ``dc.date``.

    ``publication_date`` is computed per row from ``decision_date`` via
    :func:`rba.data.rba_minutes_release_calendar.rba_minutes_publication_date`
    (second Tuesday strictly after the meeting). The scraped ``dc_date`` is
    compared against it: a plausible divergence (``dc_date`` strictly after the
    announcement but unequal to the algorithmic value) is *warned*; an
    implausible ``dc_date`` (``<= decision_date`` — the known CMS artifacts) is
    ignored silently.

    Parameters
    ----------
    df
        Frame with ``decision_date`` (datetime64[ns]) and ``dc_date``
        (Timestamp | None) columns.

    Returns
    -------
    pandas.DataFrame
        ``df`` with a ``publication_date`` (datetime64[ns]) column added.

    Shapes
    ------
    Returns: (len(df), df.shape[1] + 1).
    """
    out = df.copy()
    pub = [
        pd.Timestamp(rba_minutes_publication_date(pd.Timestamp(d).date()))
        for d in out["decision_date"]
    ]
    out["publication_date"] = pd.Series(pub, index=out.index).astype("datetime64[ns]")

    for _, row in out.iterrows():
        _softcheck_dc_date(
            dc_date=row["dc_date"],
            publication_date=row["publication_date"],
            decision_date=row["decision_date"],
            url=str(row.get("url", "")),
        )
    return out


def _softcheck_dc_date(
    *,
    dc_date: pd.Timestamp | None,
    publication_date: pd.Timestamp,
    decision_date: pd.Timestamp,
    url: str,
) -> None:
    """Warn (never raise) if a plausible ``dc.date`` disagrees with the rule."""
    if dc_date is None or pd.isna(dc_date):
        return
    dc = pd.Timestamp(dc_date)
    if dc <= decision_date:
        return
    if dc != publication_date:
        logger.warning(
            "dc.date {} disagrees with algorithmic publication_date {} for "
            "minutes decision {} ({}). Soft check only; algorithmic value "
            "retained — add to rba_minutes_release_calendar._OVERRIDES if this "
            "is a genuine schedule deviation.",
            dc.date(),
            publication_date.date(),
            decision_date.date(),
            url,
        )


def _validate_cross_check(minutes: pd.DataFrame, f11_meetings: pd.DataFrame) -> None:
    """Enforce the F11 ↔ minutes invariants.

    Raises ``ValueError`` if:

    - any in-window F11 decision with a ``minutes_url`` has no matching minutes
      row (unless allow-listed in :data:`_KNOWN_MISSING_DECISIONS`);
    - any minutes row has no matching F11 decision (extras indicate a stale
      cache or a manual ingest of a document F11 doesn't reference);
    - any minutes row has ``publication_date <= decision_date`` (minutes are
      released ~14 days *after* the announcement — the no-leakage sign flips vs
      media releases).
    """
    expected_targets = _select_targets(f11_meetings)
    expected = set(pd.to_datetime(expected_targets["decision_date"]))
    actual = (
        set(pd.to_datetime(minutes["decision_date"])) if not minutes.empty else set()
    )

    allow = {pd.Timestamp(d) for d in _KNOWN_MISSING_DECISIONS}
    missing = (expected - actual) - allow
    extra = actual - expected

    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} F11 decision(s) with a minutes_url have no matching "
            f"minutes row. Sample: {[d.date() for d in sample]}. Investigate the "
            "upstream minutes URL or add to _KNOWN_MISSING_DECISIONS if a known "
            "data gap."
        )
    if extra:
        sample = sorted(extra)[:5]
        raise ValueError(
            f"{len(extra)} minutes row(s) without a matching F11 decision: "
            f"{[d.date() for d in sample]}. Stale cache from a removed decision, "
            "or an upstream F11 change?"
        )

    if not minutes.empty:
        not_later = minutes[minutes["publication_date"] <= minutes["decision_date"]]
        if not not_later.empty:
            sample_df = not_later[["decision_date", "publication_date", "url"]].head(5)
            raise ValueError(
                f"{len(not_later)} minutes row(s) where publication_date <= "
                "decision_date (minutes must be released after the announcement). "
                f"Sample:\n{sample_df.to_string(index=False)}"
            )


if __name__ == "__main__":
    df = fetch()
    dest = EXTERNAL_DATA_DIR / "rba_minutes.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(
        "Wrote {} minutes documents (decision range {} → {}) to {}",
        len(df),
        df["decision_date"].min().date() if not df.empty else "n/a",
        df["decision_date"].max().date() if not df.empty else "n/a",
        dest,
    )
