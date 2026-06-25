"""RBA speeches — the event-keyed text corpus of official remarks.

The RBA publishes the full text of every speech, testimony, media conference,
panel appearance, interview, podcast and Q&A transcript given by its officials
— the Governor, Deputy Governor, Assistant Governors, senior department heads
and (from 2025) Monetary Policy Board members. Roughly 60 speeches are
published a year (1,300+ across the archive). They are the fourth and final
text source for the project (after :mod:`rba.data.sources.rba_media_releases`,
:mod:`rba.data.sources.rba_minutes` and :mod:`rba.data.sources.rba_somp`) and
feed the hawkish / dovish sentiment, tone-shift and embedding features
(CHECKLIST §5 ``Text features``).

Why speeches do NOT inherit the decision-keyed wide schema
----------------------------------------------------------
The first three text sources were all **decision-keyed** — one row per Board
meeting, joining 1:1 (media releases, minutes) or sparsely (SoMP) to the F11
meeting frame. Speeches break that fundamentally, so this module is the
**documented exception** to the "Text-data schema convention" in CONTEXT.md:

1. **Not decision-keyed — a continuous event stream.** There is no natural
   meeting anchor: dozens of speeches a year by multiple officials, none of
   which map to a decision. The honest schema is one row **per speech**, keyed
   on the speech itself (the URL slug, :data:`speech_id`). Speeches join to the
   meeting frame later, in the feature layer, by **point-in-time aggregation**
   (``publication_date <= meeting_date``, many-to-one), NOT a 1:1 key.
2. **No F11 driver.** There is no ``f11.speech_url``. The crawl is driven from
   the speeches **archive index** (``/speeches/``), which links a per-year page
   for every year, each listing that year's speech pages. The cross-check
   therefore validates archive-enumeration completeness and per-row field
   integrity, not an exact F11 URL set.
3. **Multiple speakers / roles / types.** Unlike the minutes / SoMP (Bank
   documents, no attribution) and media releases (single governor regex),
   speeches broaden the ``governor`` column into ``speaker`` + ``speaker_role``
   + ``speech_type``.

Scope (agreed with the project owner)
-------------------------------------
- **Speakers:** all RBA officials (Governor, Deputy Governor, Assistant
  Governors, senior officials, Monetary Policy Board members). The
  ``speaker_role`` column lets the feature layer narrow later.
- **Types:** all (formal speeches, testimony, media conferences, panels,
  interviews, podcasts, Q&A transcripts). The ``speech_type`` column lets the
  feature layer narrow later.
- **Q&A / media-conference companions:** kept as **separate rows** (each is its
  own URL / slug / type). Their bodies interleave the official's turns with
  journalist / interviewer turns; v1 captures the body **verbatim** (every
  ``<p>`` in document order, questions included). Downstream features can
  attribute turns from the cached raw HTML via ``class="speaker"``.

Coverage floor
--------------
:data:`_CALENDAR_VALIDATION_FLOOR` is the **project-wide 1993-01-01**
(inflation-targeting era), unlike the minutes / SoMP floors which were dictated
by HTML availability. The speeches archive resolves as browsable HTML all the
way back to **1990** with identical page structure (``itemprop="author"`` and
``itemprop="datePublished"`` are present even on the 1990 pages — there is **no
PDF-only early era**), but the 1990–1992 speeches (8 in total) predate the
project's economic window and are excluded by the floor, consistent with every
other source. Enumeration filters on the **archive year** (``year >= 1993``);
the per-page parsed ``publication_date`` is asserted ``>=`` the floor in the
cross-check.

URL pattern and HTML structure observed
---------------------------------------
- Archive index: ``/speeches/`` links a per-year page ``/speeches/<YYYY>/`` for
  every year 1990 → present.
- Per-speech URL: ``/speeches/<YYYY>/<slug>.html``. The slug encodes
  ``<type>-<role>-<date>[-extra][-q-and-a-transcript]`` where ``type`` is ``sp``
  (speech family) or ``mc`` (media conference), and ``date`` is ``DDMMYY``
  (pre-2017) or ``YYYY-MM-DD`` (2017+). The role code (``gov``/``dg``/``ag``/
  ``so``/``mpb``) and infix tokens are **noisy** (``qanda``, ``y2k``, ``ec``,
  ``fm``, ``fs``, ``dm`` all observed), so the slug is used only as the stable
  unique :data:`speech_id` / cache key — **not** parsed for speaker / role /
  type. The reliable source for those is the on-page metadata (below).
- Body container: ``<div id="content">`` is stable across every era probed
  (1990 → 2026). Reuses the SoMP / minutes ancestor-exclusion set.
- Title + type: the ``<h1>`` is ``"<TypeLabel> <Title>"`` where ``TypeLabel`` is
  one of :data:`_SPEECH_TYPE_LABELS` (``Speech`` / ``Testimony`` / ``Media
  conference`` / ``Interview`` / ``Panel participation`` / ``Fireside chat`` /
  ``Podcast`` / ``Transcript of Question & Answer Session``). ``speech_type`` is
  the matched leading label; ``title`` is the remainder.
- Speaker: ``itemprop="author"`` = ``"<Name> [ * ] <RoleTitle>"`` (e.g.
  ``"Michele Bullock Governor"``, ``"Les Austin Assistant Governor (Financial
  Institutions)"``, ``"Ian Harper AO Monetary Policy Board member"``). Split into
  ``speaker`` (name, footnote / honorific stripped) and ``speaker_role`` (the
  on-page role title) by :func:`_split_speaker_role`.
- Publication date: both ``itemprop="datePublished"`` (``"D Month YYYY"``, the
  media-releases format) and ``<meta name="dc.date">`` (``"YYYY-MM-DD"``) are
  present and agree on every era probed. ``datePublished`` is the **canonical**
  ``publication_date``; ``dc.date`` is a warn-only soft cross-check. Speeches are
  published the day they are delivered, so ``publication_date`` == the speech
  date.

Best-effort parses warn, never raise
-------------------------------------
``speaker`` / ``speaker_role`` / ``speech_type`` are best-effort parses of the
on-page metadata: a miss logs a warning and yields ``None`` (mirroring the
media-release governor regex), so an unrecognised author or h1 layout never
breaks the crawl. A missing ``<div id="content">``, ``<h1>`` or
``datePublished`` *does* raise (a structural change worth a hard failure).

Network notes (RBA WAF)
-----------------------
Identical to the first three text modules: the RBA fronting WAF rejects a custom
``User-Agent`` header (HTTP 403) but accepts the default ``Python-urllib`` UA.
We call ``urlopen`` with the URL string directly and never wrap it in a
``Request`` carrying headers. The crawl is large (~1,300 pages), so
``force_download=False`` reuses cached snapshots and only downloads missing
files.

Cross-check :func:`_validate_cross_check` enforces:

- every enumerated in-window archive speech yields exactly one row (misses are
  allow-listed in :data:`_KNOWN_MISSING_SPEECHES`, empty by default — treat
  misses as bugs, not data conditions);
- no extras (every row derives from an enumerated archive speech);
- :data:`speech_id` is unique (no collisions);
- every row has a parseable ``publication_date`` on/after the floor;
- warns (never raises) when a row's ``publication_date`` year disagrees with the
  archive year it was listed under, and reports ``speaker`` / ``speaker_role`` /
  ``speech_type`` parse-coverage gaps.

Output schema
-------------
``fetch()`` returns an **event-keyed** wide-format ``pandas.DataFrame`` with
columns:

- ``speech_id`` (object) — the URL slug; unique per speech (the event key).
- ``publication_date`` (datetime64[ns]) — scraped ``datePublished`` (== the
  speech date).
- ``speaker`` (object | None) — best-effort name from ``itemprop="author"``.
- ``speaker_role`` (object | None) — best-effort on-page role title.
- ``speech_type`` (object | None) — best-effort leading ``<h1>`` type label.
- ``title`` (object) — speech headline (``<h1>`` with the type label removed;
  full ``<h1>`` if the label is unrecognised).
- ``url`` (object) — fully-qualified speech URL.
- ``sha256`` (object) — SHA-256 of the raw HTML bytes.
- ``raw_html_filename`` (object) — filename of the cached snapshot under
  ``data/raw/rba_speeches/`` (``<slug>.html``).
- ``body_text`` (object) — full plain-text body, blocks joined by ``\\n\\n``.
- ``paragraphs`` (object) — ``list[str]`` of body blocks (``<h2>`` headings
  interleaved with ``<p>`` paragraphs; parallel to
  ``body_text.split("\\n\\n")``).

Rows are sorted ascending by ``(publication_date, speech_id)``.

Running this module as ``__main__`` also materialises the wide frame at
``data/external/rba_speeches.parquet``.

Side effects
------------
``fetch()`` writes under ``data/raw/rba_speeches/``:

- ``<slug>.html`` — verbatim HTML bytes per speech.
- ``_metadata.json`` — provenance manifest: one entry per speech with URL,
  SHA-256, download timestamp (UTC), byte count, and speech_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from bs4 import BeautifulSoup, Tag
from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

SOURCE_NAME = "rba_speeches"
SOURCE_BASE_URL = "https://www.rba.gov.au"
SOURCE_INDEX_URL = "https://www.rba.gov.au/speeches/"

# Project-wide inflation-targeting floor. The speeches archive is browsable HTML
# back to 1990 (no PDF-only era — itemprop author/datePublished are present even
# then), but 1990–1992 (8 speeches) predate the project's economic window and
# are excluded by this floor, consistent with every other source. Enumeration
# filters on the archive year; the parsed publication_date is asserted >= this
# floor in the cross-check.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Speeches intentionally allowed to have no matching output row, keyed by
# speech_id (the slug). Empty by default — missing speeches are bugs, not data
# conditions.
_KNOWN_MISSING_SPEECHES: frozenset[str] = frozenset()

# Speech-page slug prefixes that denote a speech-family document. Year pages may
# also list non-speech artefacts (e.g. ``background-paper-...``) and the literal
# ``index`` self-link; both are excluded by this prefix filter.
_SPEECH_SLUG_PREFIXES: tuple[str, ...] = ("sp", "mc")

# Per-speech link in a year page (its own year only — sidebar cross-year links
# are ignored). Group 1 = year, group 2 = slug.
_SPEECH_LINK_RE = re.compile(r"/speeches/(\d{4})/((?:sp|mc)-[a-z0-9\-]+)\.html")

# Year-page link in the archive index. Group 1 = year.
_YEAR_LINK_RE = re.compile(r"/speeches/(\d{4})/")

# Leading <h1> type labels, longest-first so multi-word labels win. The matched
# label becomes ``speech_type``; the remainder of the <h1> becomes ``title``.
# Validated to cover 100% of a cross-era sample.
_SPEECH_TYPE_LABELS: tuple[str, ...] = (
    "Transcript of Question & Answer Session",
    "Question & Answer Session",
    "Monetary Policy Decision Media Conference",
    "Media conference",
    "Panel participation",
    "Panel discussion",
    "Fireside chat",
    "Testimony",
    "Interview",
    "Podcast",
    "Address",
    "Speech",
)

# Role-title lead keywords (longest / most-specific first) used to split an
# ``itemprop="author"`` string into name + role. Everything from the first lead
# keyword to the end of the string is the role title.
_ROLE_LEADS: tuple[str, ...] = (
    "Deputy Governor",
    "Assistant Governor",
    "Monetary Policy Board member",
    "Governor",
    "Head",
    "Chief",
    "Secretary",
    "Senior",
    "Special",
    "Acting",
    "Director",
    "Counsellor",
    "Economist",
    "Adviser",
    "Manager",
)
_AUTHOR_SPLIT_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<role>(?:"
    + "|".join(re.escape(lead) for lead in _ROLE_LEADS)
    + r")\b.*)$"
)

# Footnote markers ("[ * ]", "[*]", "*") embedded in author strings.
_FOOTNOTE_RE = re.compile(r"\[\s*\*\s*\]|\*")

# Trailing post-nominal honorifics stripped from a parsed speaker name.
_HONORIFIC_RE = re.compile(
    r"(?:\s+(?:AO|AC|AM|OAM|QC|KC|CBE|OBE|MBE|CB|CSM|PSM|FAA|FACE|AField))+$"
)

# Ancestor classes/tags whose <h2>/<p> descendants are excluded from body
# extraction. Reuses the SoMP / minutes CMS set.
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
_NAV_HEADINGS: frozenset[str] = frozenset(
    {"related information", "related content", "endnotes", "references"}
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull every in-window speech enumerated from the archive index.

    Parameters
    ----------
    force_download
        If True (default), refresh every speech HTML from the live RBA endpoint.
        If False, reuse the snapshots already under ``data/raw/rba_speeches/``
        (matched by ``<slug>.html`` filename) and only download missing files.
        The archive index and per-year pages are always fetched fresh (they are
        small and drive enumeration).

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(publication_date,
        speech_id)``.

    Shapes
    ------
    Returns: (n_speeches, 11) where ``n_speeches`` is the count of speech pages
    in the archive listed under a year on/after
    :data:`_CALENDAR_VALIDATION_FLOOR`.
    """
    targets = _build_targets()
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for _, target in targets.iterrows():
        speech_id = str(target["speech_id"])
        url_path = str(target["url_path"])
        snapshot_path, entry, raw_bytes, full_url = _resolve_snapshot(
            dest_dir,
            speech_id=speech_id,
            url_path=url_path,
            force_download=force_download,
        )
        parsed = _parse(raw_bytes, speech_id=speech_id)
        rows.append(
            {
                "speech_id": speech_id,
                "publication_date": parsed["publication_date"],
                "speaker": parsed["speaker"],
                "speaker_role": parsed["speaker_role"],
                "speech_type": parsed["speech_type"],
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
        df["publication_date"] = pd.to_datetime(df["publication_date"]).astype(
            "datetime64[ns]"
        )

    _validate_cross_check(df, targets)

    if df.empty:
        return df
    return df.sort_values(["publication_date", "speech_id"]).reset_index(drop=True)


def _build_targets(*, index_html: bytes | None = None) -> pd.DataFrame:
    """Enumerate every in-window speech from the archive index + year pages.

    Downloads the archive index, enumerates its year-page links on/after
    :data:`_CALENDAR_VALIDATION_FLOOR`, then downloads each year page and
    enumerates that year's speech links. De-duplicated by ``speech_id``.

    Parameters
    ----------
    index_html
        Optional pre-fetched archive-index bytes (used by tests). When ``None``
        the index is downloaded.

    Returns
    -------
    pandas.DataFrame
        Columns: ``year`` (int, the archive year the speech is listed under),
        ``speech_id`` (object, the slug), ``url_path`` (object, relative path
        ``/speeches/<YYYY>/<slug>.html``). Sorted ascending by
        ``(year, speech_id)``.

    Shapes
    ------
    Returns: (n_speeches, 3).
    """
    if index_html is None:
        index_html = _download_index()
    years = _enumerate_years(index_html)

    seen: dict[str, tuple[int, str]] = {}
    for year in years:
        year_html = _download_year_page(year)
        for speech_id, url_path in _enumerate_speeches(year_html, year):
            seen.setdefault(speech_id, (year, url_path))

    rows = [
        {"year": year, "speech_id": speech_id, "url_path": url_path}
        for speech_id, (year, url_path) in seen.items()
    ]
    df = pd.DataFrame(rows, columns=["year", "speech_id", "url_path"])
    return df.sort_values(["year", "speech_id"]).reset_index(drop=True)


def _download_index() -> bytes:
    """Download the speeches archive index HTML (the year-page enumeration)."""
    logger.info("Downloading speeches archive index {}", SOURCE_INDEX_URL)
    return _get(SOURCE_INDEX_URL)


def _download_year_page(year: int) -> bytes:
    """Download one ``/speeches/<YYYY>/`` page (the per-year speech listing)."""
    url = f"{SOURCE_INDEX_URL}{year}/"
    logger.info("Downloading speeches year page {}", url)
    return _get(url)


def _get(url: str) -> bytes:
    """GET a public RBA URL with the default UA (custom UA → WAF 403)."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — public RBA URL
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Failed to download {url}: HTTP {exc.code}. "
            "Note: the RBA WAF rejects custom User-Agent headers — the default "
            "Python-urllib UA is required."
        ) from exc


def _enumerate_years(index_html: bytes) -> list[int]:
    """Enumerate in-window archive years from the index HTML.

    Parameters
    ----------
    index_html
        Raw bytes of ``/speeches/``.

    Returns
    -------
    list[int]
        Sorted unique years on/after ``_CALENDAR_VALIDATION_FLOOR.year`` for
        which the index links a ``/speeches/<YYYY>/`` page.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    floor_year = _CALENDAR_VALIDATION_FLOOR.year
    years: set[int] = set()
    for anchor in soup.find_all("a", href=True):
        match = _YEAR_LINK_RE.search(str(anchor["href"]))
        if match is None:
            continue
        year = int(match.group(1))
        if year >= floor_year:
            years.add(year)
    return sorted(years)


def _enumerate_speeches(year_html: bytes, year: int) -> list[tuple[str, str]]:
    """Enumerate ``(speech_id, url_path)`` for one year page.

    Only links whose href year equals ``year`` are kept (so sidebar cross-year
    links are ignored), and only speech-family slugs (``sp-`` / ``mc-`` prefix);
    the ``index`` self-link and non-speech artefacts (e.g. ``background-paper``)
    are excluded by the prefix filter.

    Parameters
    ----------
    year_html
        Raw bytes of ``/speeches/<year>/``.
    year
        The archive year of this page (used to drop cross-year sidebar links).

    Returns
    -------
    list[tuple[str, str]]
        ``(speech_id, url_path)`` per in-page speech link, de-duplicated and
        sorted by ``speech_id``. ``url_path`` is ``/speeches/<year>/<slug>.html``.
    """
    soup = BeautifulSoup(year_html, "html.parser")
    seen: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        match = _SPEECH_LINK_RE.search(str(anchor["href"]))
        if match is None:
            continue
        link_year = int(match.group(1))
        if link_year != year:
            continue
        slug = match.group(2)
        seen[slug] = f"/speeches/{year}/{slug}.html"
    return sorted(seen.items())


def _resolve_snapshot(
    dest_dir: Path,
    *,
    speech_id: str,
    url_path: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes, str]:
    """Return (snapshot path, manifest entry, raw bytes, full URL) for a speech."""
    snapshot_path = dest_dir / f"{speech_id}.html"
    full_url = SOURCE_BASE_URL + url_path

    if snapshot_path.exists() and not force_download:
        raw_bytes = snapshot_path.read_bytes()
        entry: dict[str, object] = {
            "speech_id": speech_id,
            "url": full_url,
            "snapshot_filename": snapshot_path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        logger.info("Reusing cached speech at {}", snapshot_path)
        return snapshot_path, entry, raw_bytes, full_url

    return _download(snapshot_path, speech_id=speech_id, full_url=full_url)


def _download(
    snapshot_path: Path, *, speech_id: str, full_url: str
) -> tuple[Path, dict[str, object], bytes, str]:
    """Download one speech HTML, save the snapshot, return path + entry + bytes.

    The RBA WAF rejects custom ``User-Agent`` headers (403) but accepts the
    default ``Python-urllib`` UA. We pass the URL string directly to
    :func:`urllib.request.urlopen` and never wrap it in a ``Request`` carrying
    headers.
    """
    logger.info("Downloading {} -> {}", full_url, snapshot_path)
    raw_bytes = _get(full_url)
    snapshot_path.write_bytes(raw_bytes)
    entry: dict[str, object] = {
        "speech_id": speech_id,
        "url": full_url,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry, raw_bytes, full_url


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Wrote provenance manifest ({} entries) to {}", len(manifest), metadata_path
    )


def _parse(html: bytes, *, speech_id: str) -> dict[str, object]:
    """Parse one speech HTML into its event-keyed fields.

    Parameters
    ----------
    html
        Raw HTML bytes from the speech page.
    speech_id
        The slug, used only for error / warning context.

    Returns
    -------
    dict
        Keys: ``title`` (str), ``speech_type`` (str | None), ``publication_date``
        (pd.Timestamp — scraped ``itemprop="datePublished"``), ``speaker``
        (str | None), ``speaker_role`` (str | None), ``paragraphs`` (list[str]),
        ``body_text`` (str — blocks joined by ``"\\n\\n"``).

    Raises
    ------
    ValueError
        If ``<div id="content">``, the ``<h1>`` title or
        ``itemprop="datePublished"`` is missing (a structural change). Best-effort
        speaker / role / type misses warn and yield ``None`` instead.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", id="content")
    if content is None:
        raise ValueError(
            f"No <div id='content'> in speech HTML for {speech_id}. "
            "Page layout may have changed."
        )

    heading = content.find("h1")
    if heading is None:
        raise ValueError(
            f"No <h1> title in speech HTML for {speech_id}. "
            "Page layout may have changed."
        )
    h1_text = _normalise_whitespace(heading.get_text(" ", strip=True))
    speech_type, title = _classify_speech_type(h1_text)
    if speech_type is None:
        logger.warning(
            "Could not classify speech_type from <h1> {!r} ({}). h1 may use an "
            "unrecognised type label; storing full <h1> as title.",
            h1_text,
            speech_id,
        )

    publication_date = _extract_publication_date(soup)
    if publication_date is None:
        raise ValueError(
            f"No parseable itemprop='datePublished' in speech HTML for "
            f"{speech_id}. datePublished is the canonical publication date for "
            "this source; page layout may have changed."
        )
    _softcheck_dc_date(
        dc_date=_extract_dc_date(soup),
        publication_date=publication_date,
        speech_id=speech_id,
    )

    author = _extract_author(soup)
    speaker, speaker_role = _split_speaker_role(author)
    if speaker is None or speaker_role is None:
        logger.warning(
            "Could not fully parse speaker/role from author {!r} ({}). "
            "Storing best-effort speaker={!r} speaker_role={!r}.",
            author,
            speech_id,
            speaker,
            speaker_role,
        )

    paragraphs = _extract_blocks(content)
    if not paragraphs:
        raise ValueError(
            f"Extracted zero body blocks from speech HTML for {speech_id}. "
            "Page layout may have changed."
        )
    body_text = "\n\n".join(paragraphs)

    return {
        "title": title,
        "speech_type": speech_type,
        "publication_date": publication_date,
        "speaker": speaker,
        "speaker_role": speaker_role,
        "paragraphs": paragraphs,
        "body_text": body_text,
    }


def _classify_speech_type(h1_text: str) -> tuple[str | None, str]:
    """Split an ``<h1>`` into ``(speech_type, title)``.

    Matches the leading text against :data:`_SPEECH_TYPE_LABELS` (longest-first,
    case-insensitive). On a match, ``speech_type`` is the matched label and
    ``title`` is the remainder; on no match, ``speech_type`` is ``None`` and
    ``title`` is the full ``<h1>``.
    """
    for label in _SPEECH_TYPE_LABELS:
        if h1_text.lower().startswith(label.lower()):
            remainder = h1_text[len(label) :].strip(" :-–—")
            return label, remainder or h1_text
    return None, h1_text


def _extract_publication_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """Return ``itemprop='datePublished'`` as a Timestamp, or ``None``.

    The text is ``"D Month YYYY"`` (the media-releases format), possibly with an
    NBSP between day and month; whitespace is normalised before parsing.
    """
    el = soup.find(attrs={"itemprop": "datePublished"})
    if el is None:
        return None
    date_text = _normalise_whitespace(el.get_text(" ", strip=True))
    parsed = pd.to_datetime(date_text, format="%d %B %Y", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _extract_dc_date(soup: BeautifulSoup) -> pd.Timestamp | None:
    """Return the ``<meta name='dc.date'>`` value as a Timestamp, or ``None``.

    Used only as a warn-only soft cross-check against the canonical
    ``datePublished`` (the two agree on every era probed).
    """
    meta = soup.find("meta", attrs={"name": "dc.date"})
    if meta is None:
        return None
    content = meta.get("content")
    if not content:
        return None
    parsed = pd.to_datetime(str(content), format="%Y-%m-%d", errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _extract_author(soup: BeautifulSoup) -> str | None:
    """Return the whitespace-normalised ``itemprop='author'`` text, or ``None``."""
    el = soup.find(attrs={"itemprop": "author"})
    if el is None:
        return None
    text = _normalise_whitespace(el.get_text(" ", strip=True))
    return text or None


def _split_speaker_role(author: str | None) -> tuple[str | None, str | None]:
    """Split an author string into ``(speaker_name, speaker_role)``.

    The author is ``"<Name> [ * ] <RoleTitle>"``; the role title begins at the
    first :data:`_ROLE_LEADS` keyword. Footnote markers (``[ * ]`` / ``*``) and
    trailing post-nominal honorifics (``AO`` / ``AC`` / …) are stripped from the
    name. On no recognised role lead, the (footnote-stripped) author is returned
    as a best-effort ``speaker`` with ``speaker_role=None``.

    Parameters
    ----------
    author
        Raw ``itemprop="author"`` text, or ``None``.

    Returns
    -------
    tuple[str | None, str | None]
        ``(speaker_name, speaker_role)``. Either element may be ``None`` on a
        parse miss (the caller warns).
    """
    if not author:
        return None, None
    cleaned = _normalise_whitespace(_FOOTNOTE_RE.sub(" ", author))
    if not cleaned:
        return None, None
    match = _AUTHOR_SPLIT_RE.match(cleaned)
    if match is None:
        return cleaned, None
    name = _HONORIFIC_RE.sub("", match.group("name").strip()).strip()
    role = _normalise_whitespace(match.group("role"))
    return (name or None), (role or None)


def _extract_blocks(content_div: Tag) -> list[str]:
    """Extract body blocks (``<h2>`` headings + ``<p>`` paragraphs) in order.

    Walks every ``<h2>`` and ``<p>`` under ``content_div`` in document order,
    excluding any whose ancestor chain crosses an element in
    :data:`_EXCLUDE_ANCESTOR_CLASSES` or :data:`_EXCLUDE_ANCESTOR_TAGS`
    (related-links rails, contact box, navigation asides, tiles). For media
    conferences and Q&A transcripts the interleaved speaker turns are plain
    ``<p>`` elements and are therefore captured **verbatim** in document order.
    ``<h2>`` section headings are kept as standalone entries; trailing navigation
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


def _softcheck_dc_date(
    *,
    dc_date: pd.Timestamp | None,
    publication_date: pd.Timestamp,
    speech_id: str,
) -> None:
    """Warn (never raise) if ``dc.date`` disagrees with the canonical date."""
    if dc_date is None or pd.isna(dc_date):
        return
    if pd.Timestamp(dc_date) != pd.Timestamp(publication_date):
        logger.warning(
            "dc.date {} disagrees with datePublished {} for speech {}. Soft "
            "check only; datePublished retained as canonical.",
            pd.Timestamp(dc_date).date(),
            pd.Timestamp(publication_date).date(),
            speech_id,
        )


def _validate_cross_check(speeches: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Enforce the archive ↔ speeches invariants.

    Raises ``ValueError`` if:

    - any enumerated in-window archive speech has no matching output row (unless
      allow-listed in :data:`_KNOWN_MISSING_SPEECHES`);
    - any output row has no matching enumerated speech (an extra);
    - any ``speech_id`` appears more than once (a collision);
    - any row has a missing (``NaT``) ``publication_date`` or one before
      :data:`_CALENDAR_VALIDATION_FLOOR`.

    Warns (never raises) when a row's ``publication_date`` year disagrees with
    the archive year it was listed under, and reports ``speaker`` /
    ``speaker_role`` / ``speech_type`` parse-coverage gaps.
    """
    expected = set(targets["speech_id"]) if not targets.empty else set()
    actual = set(speeches["speech_id"]) if not speeches.empty else set()

    allow = set(_KNOWN_MISSING_SPEECHES)
    missing = (expected - actual) - allow
    extra = actual - expected

    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} enumerated archive speech(es) have no matching "
            f"output row. Sample speech_ids: {sample}. Investigate the "
            "download/parse or add to _KNOWN_MISSING_SPEECHES if a known gap."
        )
    if extra:
        sample = sorted(extra)[:5]
        raise ValueError(
            f"{len(extra)} output row(s) not derived from the archive index: "
            f"{sample}. Stale cache, or a manual ingest of a page the index "
            "doesn't list?"
        )

    if speeches.empty:
        return

    dupes = speeches["speech_id"][speeches["speech_id"].duplicated()]
    if not dupes.empty:
        sample = sorted(set(dupes))[:5]
        raise ValueError(
            f"{len(set(dupes))} speech_id(s) appear in more than one row: "
            f"{sample}. Enumeration or cache collision."
        )

    pubs = pd.to_datetime(speeches["publication_date"], errors="coerce")
    bad_date = speeches[pubs.isna() | (pubs < _CALENDAR_VALIDATION_FLOOR)]
    if not bad_date.empty:
        sample = bad_date[["speech_id", "publication_date"]].head(5)
        raise ValueError(
            f"{len(bad_date)} speech row(s) with a missing or pre-floor "
            f"publication_date (floor {_CALENDAR_VALIDATION_FLOOR.date()}). "
            f"Sample:\n{sample.to_string(index=False)}"
        )

    _softcheck_archive_year(speeches, targets)
    _report_parse_coverage(speeches)


def _softcheck_archive_year(speeches: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Warn (never raise) when a row's publication year ≠ its archive year."""
    year_by_id = dict(zip(targets["speech_id"], targets["year"], strict=False))
    pubs = pd.to_datetime(speeches["publication_date"])
    for speech_id, pub in zip(speeches["speech_id"], pubs, strict=False):
        archive_year = year_by_id.get(speech_id)
        if archive_year is None or pd.isna(pub):
            continue
        if pd.Timestamp(pub).year != int(archive_year):
            logger.warning(
                "Speech {} listed under archive year {} but datePublished is {}. "
                "Soft check only; scraped date retained.",
                speech_id,
                int(archive_year),
                pd.Timestamp(pub).date(),
            )


def _report_parse_coverage(speeches: pd.DataFrame) -> None:
    """Warn with counts when best-effort metadata parses missed."""
    total = len(speeches)
    for column in ("speaker", "speaker_role", "speech_type"):
        misses = int(speeches[column].isna().sum())
        if misses:
            logger.warning(
                "{}/{} speech row(s) have no parsed {} (best-effort field; "
                "None retained).",
                misses,
                total,
                column,
            )


if __name__ == "__main__":
    df = fetch()
    dest = EXTERNAL_DATA_DIR / "rba_speeches.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info(
        "Wrote {} speeches (publication range {} → {}) to {}",
        len(df),
        df["publication_date"].min().date() if not df.empty else "n/a",
        df["publication_date"].max().date() if not df.empty else "n/a",
        dest,
    )
