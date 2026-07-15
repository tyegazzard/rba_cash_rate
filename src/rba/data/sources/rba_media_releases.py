"""RBA post-decision media releases — text payload per Board meeting.

After every monetary-policy Board meeting the RBA publishes a media-release
``Statement`` summarising the decision and the supporting reasoning. The full
text of these statements is the primary text input for the project's hawkish /
dovish sentiment, tone-shift, and embedding features (CHECKLIST §5 ``Text
features``).

Why F11-driven crawl, not yearly-archive title regex?
-----------------------------------------------------
:mod:`rba.data.sources.rba_f11` already populates ``statement_url`` on every
meeting row that issued a post-decision release (NaN otherwise — pre-Apr 2010
the Board only issued statements at rate-change meetings; from Apr 2010 a
statement accompanies every meeting). Driving the crawl directly from
``F11.statement_url`` therefore:

- gives an *exact* one-to-one set of expected releases (no false positives from
  regulatory / payments / appointments releases that happen to share the
  ``Statement by ...`` title prefix);
- gives an *exact* cross-check (every F11 row with a ``statement_url`` must
  yield one media-release row, and vice versa); and
- avoids any per-era title-regex maintenance cost.

The yearly archive (``/media-releases/<YYYY>/``) is therefore *not* scraped —
F11 is the index.

URL pattern and HTML structure observed
---------------------------------------
- URL slug: ``/media-releases/<YYYY>/(j)?mr-<YY|YYYY>-<NN>.html``. ``jmr-``
  marks joint releases with Treasury (e.g. RBA + Treasurer statements). Year
  2000 uniquely uses a 4-digit year segment; every other year uses ``YY``.
  All variants resolve in the same way and carry the same body structure.
- Body container: ``<div id="content" role="main" class="column-content
  content-style">`` is stable across every era probed (1993 → 2024). Inside
  this container, the modern era (2018+) wraps the body in
  ``<div class="rss-mr-content">``; older eras have body paragraphs as direct
  ``<p>`` children of a ``<section>``. Excluded ancestors (contact details,
  related-links rails, navigation): see ``_EXCLUDE_ANCESTOR_CLASSES`` /
  ``_EXCLUDE_ANCESTOR_TAGS``.
- Publication date: every release exposes ``<span itemprop="datePublished">D
  Month YYYY</span>`` — used as the canonical date stamp. The label rendered
  inside ``box-article-info`` varies across eras; the itemprop does not.
- Title patterns observed (post-decision releases only):

  ===================================  =====================================================
  Era                                  Title pattern
  -----------------------------------  -----------------------------------------------------
  1993 → ~Apr 2010 (Fraser/Macfarlane/ ``Statement by the Governor, Mr <Surname>: <topic>``
    early-Stevens; rate-change only)
  ~Apr 2010 → Jan 2024 (post-template- ``Statement by <FirstName Surname>, Governor:
    standardisation Stevens/Lowe/        Monetary Policy Decision - <Month YYYY>``
    early-Bullock)
  Feb 2024 → present (post-governance-  ``Statement by the Reserve Bank Board: Monetary
    reform: separate Monetary             Policy Decision - <Month YYYY>``
    Policy Board)
  ===================================  =====================================================

  ``governor`` is extracted from the title best-effort via
  :data:`_GOVERNOR_PATTERNS`. A regex miss yields ``governor=None`` plus a
  warning log (does not raise) — text features that need an attribution can
  fall back on ``decision_date`` against a hand-coded governor tenure table.

Network notes (RBA WAF)
-----------------------
The RBA fronting WAF rejects requests with a custom ``User-Agent`` header
(``HTTP 403``) but accepts the default ``Python-urllib`` UA. This is the
reverse of the GitHub raw endpoint used by :mod:`rba.data.sources.asx_ib_futures`
(which *requires* a non-empty UA). We therefore call ``urlopen`` with the URL
string directly and never wrap it in a ``Request`` carrying headers.

Schema convention — text-data sources
-------------------------------------
Numerical sources use the long-format
``[observation_date, publication_date, series_id, value]`` shape because every
observation is one numeric reading anchored to a single series. Text releases
don't fit: each statement is a unique document, there is no recurring
``series_id``, and ``value`` would have to become an ``object`` text column.

This module therefore emits a **decision-keyed wide schema** — one row per
decision — that the three remaining text scrapers (minutes, SoMP, Governor
speeches) will inherit. The schema joins 1:1 on ``decision_date`` to the F11
meeting frame (and indirectly to every numeric source via the meeting frame).

Decision-date convention
------------------------
For RBA media releases ``decision_date == publication_date``: the Board
announces at 14:30 AEST and the statement is on the website at the same
moment. ``decision_date`` is the *announcement* day, identical to
:attr:`rba_f11.publication_date` (which is itself one calendar day before the
*effective* date from Feb 2008 onward). Cross-check
:func:`_validate_cross_check` enforces:

- every F11 row on/after :data:`_CALENDAR_VALIDATION_FLOOR` with a
  ``statement_url`` yields exactly one media-release row;
- every media-release row's parsed ``publication_date`` equals its
  ``decision_date``;
- known-missing decisions live in :data:`_KNOWN_MISSING_DECISIONS` (empty by
  default — treat misses as bugs, not data conditions).

Output schema
-------------
``fetch()`` returns a wide-format ``pandas.DataFrame`` with columns:

- ``decision_date`` (datetime64[ns]) — Board announcement day (== F11
  publication_date).
- ``publication_date`` (datetime64[ns]) — parsed from the page's
  ``itemprop="datePublished"`` field. Always equals ``decision_date`` (post-
  cross-check); kept distinct for schema-uniformity with numerical sources.
- ``governor`` (object | None) — best-effort attribution from the title (see
  patterns above). ``None`` on regex miss with a warning logged.
- ``title`` (object) — release headline, whitespace-normalised.
- ``url`` (object) — fully-qualified release URL.
- ``sha256`` (object) — SHA-256 of the raw HTML bytes (for vintage tracking).
- ``raw_html_filename`` (object) — filename of the cached snapshot under
  ``data/raw/rba_media_releases/`` (i.e. ``<YYYY-MM-DD>__<slug>.html``).
- ``body_text`` (object) — full plain-text body, paragraphs joined by
  ``\\n\\n``. Whitespace-normalised; contact / navigation / related-links
  blocks excluded.
- ``paragraphs`` (object) — ``list[str]`` of body paragraphs (parallel to
  ``body_text.split("\\n\\n")``).

Running this module as ``__main__`` also materialises the wide frame at
``data/external/rba_media_releases.parquet``.

Side effects
------------
``fetch()`` writes under ``data/raw/rba_media_releases/``:

- ``<YYYY-MM-DD>__<slug>.html`` — verbatim HTML bytes per release (one file
  per decision; the date prefix is the decision date).
- ``_metadata.json`` — provenance manifest: one entry per release with URL,
  SHA-256, download timestamp (UTC), byte count, and decision_date.
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

SOURCE_NAME = "rba_media_releases"
SOURCE_BASE_URL = "https://www.rba.gov.au"

# Inflation-targeting era start — matches every other source module.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Decisions intentionally allowed to have no matching media-release row.
# Empty by default — missing statements are bugs, not data conditions.
_KNOWN_MISSING_DECISIONS: frozenset[pd.Timestamp] = frozenset()

# Body extraction: modern era wraps the body in this class; older eras have
# paragraphs as direct children of <section> inside <div id="content">.
_MODERN_BODY_CLASS = "rss-mr-content"

# Ancestor classes/tags whose <p> children must be excluded from body
# extraction (contact box, navigation, related-links rail, etc.).
_EXCLUDE_ANCESTOR_CLASSES: frozenset[str] = frozenset(
    {
        "box-enquiries",
        "box-article-info",
        "rss-mr-contact",
        "nav-page-contents",
        "aside-component",
        "monetary-policy-related-links",
        "tile-container",
        "complementary",
    }
)
_EXCLUDE_ANCESTOR_TAGS: frozenset[str] = frozenset({"aside", "footer", "nav"})

# Governor attribution patterns. Ordered most-specific-first; each entry is
# (regex, fixed_string_or_None). When fixed_string is None, group(1) is used.
_GOVERNOR_PATTERNS: tuple[tuple[re.Pattern[str], str | None], ...] = (
    (
        re.compile(r"Statement\s+by\s+the\s+Reserve\s+Bank\s+Board", re.IGNORECASE),
        "Reserve Bank Board",
    ),
    (
        re.compile(
            r"Statement\s+by\s+([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){1,2})\s*,\s*Governor",
            re.IGNORECASE,
        ),
        None,
    ),
    (
        re.compile(
            r"Statement\s+by\s+the\s+Governor\s*,\s+(?:Mr|Mrs|Ms|Dr)\s+"
            r"([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){1,2})",
            re.IGNORECASE,
        ),
        None,
    ),
)


def fetch(
    *,
    f11_meetings: pd.DataFrame | None = None,
    force_download: bool = True,
) -> pd.DataFrame:
    """Pull every post-decision media release referenced by F11.

    Parameters
    ----------
    f11_meetings
        Pre-loaded F11 meetings frame (output of
        :func:`rba.data.sources.rba_f11.fetch`). If ``None``, ``rba_f11.fetch``
        is called with ``force_download=False`` to reuse the most recent
        cached snapshot — the media-release crawl does not need a fresh F11
        download to be valid.
    force_download
        If True (default), refresh every release HTML from the live RBA
        endpoint. If False, reuse the dated snapshots already under
        ``data/raw/rba_media_releases/`` (matched by decision-date filename
        prefix) and only download missing files.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``decision_date``.

    Shapes
    ------
    Returns: (n_decisions, 9) where ``n_decisions`` is the count of F11 rows
    on/after :data:`_CALENDAR_VALIDATION_FLOOR` with a ``statement_url``.
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
        url_path = str(target["statement_url"])
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
                "publication_date": parsed["publication_date"],
                "governor": parsed["governor"],
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
        df["publication_date"] = pd.to_datetime(df["publication_date"]).astype("datetime64[ns]")

    _validate_cross_check(df, f11_meetings)

    return df.sort_values("decision_date").reset_index(drop=True)


def _select_targets(f11_meetings: pd.DataFrame) -> pd.DataFrame:
    """Filter F11 to in-window decisions that have a ``statement_url``.

    Parameters
    ----------
    f11_meetings
        Output of :func:`rba.data.sources.rba_f11.fetch`. Must contain at
        least ``observation_date``, ``publication_date``, and
        ``statement_url``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``decision_date`` (datetime64[ns], == F11 publication_date),
        ``statement_url`` (object, relative path from the RBA root). Rows
        sorted ascending by ``decision_date``.

    Shapes
    ------
    Returns: (n_targets, 2).
    """
    needed = {"observation_date", "publication_date", "statement_url"}
    missing = needed - set(f11_meetings.columns)
    if missing:
        raise ValueError(
            f"f11_meetings missing required column(s): {sorted(missing)}. "
            f"Got columns: {list(f11_meetings.columns)}"
        )

    df = f11_meetings.copy()
    df = df[df["publication_date"] >= _CALENDAR_VALIDATION_FLOOR]
    has_url = df["statement_url"].notna() & (df["statement_url"].astype(str).str.len() > 0)
    df = df[has_url]
    df = df.assign(decision_date=pd.to_datetime(df["publication_date"]))
    return (
        df[["decision_date", "statement_url"]].sort_values("decision_date").reset_index(drop=True)
    )


def _resolve_snapshot(
    dest_dir: Path,
    *,
    decision_date: pd.Timestamp,
    url_path: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one release."""
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
        logger.info("Reusing cached media release at {}", snapshot_path)
        return snapshot_path, entry, raw_bytes

    return _download(snapshot_path, full_url=full_url, date_str=date_str)


def _download(
    snapshot_path: Path, *, full_url: str, date_str: str
) -> tuple[Path, dict[str, object], bytes]:
    """Download one release HTML, save dated snapshot, return path + entry + bytes.

    The RBA WAF rejects custom ``User-Agent`` headers (403) but accepts the
    default ``Python-urllib`` UA. We therefore pass the URL string directly
    to :func:`urllib.request.urlopen` and never wrap it in a ``Request``
    carrying headers.
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


class _ParsedMediaRelease(TypedDict):
    """Typed shape of :func:`_parse`'s output — one media-release document."""

    title: str
    publication_date: pd.Timestamp
    governor: str | None
    paragraphs: list[str]
    body_text: str


def _parse(html: bytes, *, decision_date: pd.Timestamp) -> _ParsedMediaRelease:
    """Parse one release HTML into title / publication_date / governor / body.

    Parameters
    ----------
    html
        Raw HTML bytes from the release page.
    decision_date
        Decision date this release corresponds to, used only for error context.

    Returns
    -------
    dict
        Keys: ``title`` (str), ``publication_date`` (pd.Timestamp),
        ``governor`` (str | None), ``paragraphs`` (list[str]),
        ``body_text`` (str — paragraphs joined by ``"\\n\\n"``).
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="content")
    if content is None:
        raise ValueError(
            f"No <div id='content'> in release HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )

    headline = content.find(attrs={"itemprop": "headline"})
    if headline is None:
        raise ValueError(
            f"No element with itemprop='headline' in release HTML for "
            f"decision {decision_date.date()}. Page layout may have changed."
        )
    title = _normalise_whitespace(headline.get_text(" ", strip=True))

    date_el = soup.find(attrs={"itemprop": "datePublished"})
    if date_el is None:
        raise ValueError(
            f"No element with itemprop='datePublished' in release HTML for "
            f"decision {decision_date.date()}. Page layout may have changed."
        )
    date_text = _normalise_whitespace(date_el.get_text(" ", strip=True))
    try:
        publication_date = pd.to_datetime(date_text, format="%d %B %Y", errors="raise")
    except ValueError as exc:
        raise ValueError(
            f"Could not parse datePublished {date_text!r} as 'D Month YYYY' for "
            f"decision {decision_date.date()}."
        ) from exc

    governor = _extract_governor(title)
    if governor is None:
        logger.warning(
            "Could not extract governor from title {!r} (decision {}). "
            "Title may have used an unrecognised pattern.",
            title,
            decision_date.date(),
        )

    paragraphs = _extract_paragraphs(content)
    if not paragraphs:
        raise ValueError(
            f"Extracted zero body paragraphs from release HTML for decision "
            f"{decision_date.date()}. Page layout may have changed."
        )
    body_text = "\n\n".join(paragraphs)

    return {
        "title": title,
        "publication_date": publication_date,
        "governor": governor,
        "paragraphs": paragraphs,
        "body_text": body_text,
    }


def _extract_governor(title: str) -> str | None:
    """Return governor attribution from a release title, or None on miss."""
    for pattern, fixed in _GOVERNOR_PATTERNS:
        match = pattern.search(title)
        if match:
            return fixed if fixed is not None else _normalise_whitespace(match.group(1))
    return None


def _extract_paragraphs(content_div: Tag) -> list[str]:
    """Extract body paragraphs from a ``<div id='content'>`` element.

    Prefers the modern ``<div class='rss-mr-content'>`` body wrapper when
    present; otherwise walks every ``<p>`` under ``content_div`` and excludes
    those whose ancestor chain crosses an element in
    :data:`_EXCLUDE_ANCESTOR_CLASSES` or :data:`_EXCLUDE_ANCESTOR_TAGS`
    (contact box, related-links rails, navigation asides, etc.).
    """
    container = content_div.find("div", class_=_MODERN_BODY_CLASS) or content_div
    out: list[str] = []
    for p in container.find_all("p"):
        if _has_excluded_ancestor(p, container):
            continue
        text = _normalise_whitespace(p.get_text(" ", strip=True))
        if text:
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


def _validate_cross_check(media: pd.DataFrame, f11_meetings: pd.DataFrame) -> None:
    """Enforce the F11 ↔ media-releases invariants.

    Raises ``ValueError`` if:

    - any in-window F11 decision with a ``statement_url`` has no matching
      media-release row (unless allow-listed in
      :data:`_KNOWN_MISSING_DECISIONS`);
    - any media-release row has no matching F11 decision (extras indicate a
      stale cache or a manual ingest of a release F11 doesn't reference);
    - any media-release row has ``publication_date != decision_date``
      (publication-date convention for post-decision statements is same-day).
    """
    expected_targets = _select_targets(f11_meetings)
    expected = set(pd.to_datetime(expected_targets["decision_date"]))
    actual = set(pd.to_datetime(media["decision_date"])) if not media.empty else set()

    allow = {pd.Timestamp(d) for d in _KNOWN_MISSING_DECISIONS}
    missing = (expected - actual) - allow
    extra = actual - expected

    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} F11 decision(s) with a statement_url have no "
            f"matching media-release row. Sample: {[d.date() for d in sample]}. "
            "Investigate the upstream release URL or add to "
            "_KNOWN_MISSING_DECISIONS if a known data gap."
        )
    if extra:
        sample = sorted(extra)[:5]
        raise ValueError(
            f"{len(extra)} media-release row(s) without a matching F11 "
            f"decision: {[d.date() for d in sample]}. Stale cache from a "
            "removed decision, or an upstream F11 change?"
        )

    if not media.empty:
        mismatched = media[media["publication_date"] != media["decision_date"]]
        if not mismatched.empty:
            sample_df = mismatched[["decision_date", "publication_date", "url"]].head(5)
            raise ValueError(
                f"{len(mismatched)} media-release row(s) where "
                "publication_date != decision_date (post-decision statements "
                f"must be same-day). Sample:\n{sample_df.to_string(index=False)}"
            )


if __name__ == "__main__":
    df = fetch()
    dest = EXTERNAL_DATA_DIR / "rba_media_releases.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(
        "Wrote {} media releases (decision range {} → {}) to {}",
        len(df),
        df["decision_date"].min().date() if not df.empty else "n/a",
        df["decision_date"].max().date() if not df.empty else "n/a",
        dest,
    )
