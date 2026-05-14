"""RBA cash rate target — full meeting-indexed decision history.

Despite the module name (kept for CHECKLIST alignment with the "F1.1" item),
the authoritative source for the meeting-indexed cash rate target history is
the HTML table on ``https://www.rba.gov.au/statistics/cash-rate/``.

Why this URL and not statistical table A2 (a02hist.xlsx)?
---------------------------------------------------------
A2 only contains *change events* (~98 rows since 1990). The HTML table on
``/statistics/cash-rate/`` contains every Board meeting since 23 Jan 1990
(~399 rows), with hold meetings encoded as ``Change=0.00``. The HTML table is
a strict superset and is what we need to predict every meeting outcome (cut /
hold / hike).

Effective-date convention
-------------------------
Per RBA documentation (A2 ``Notes`` sheet, verbatim):

    "Commencing February 2008, the announcement is made on the day of the
    Reserve Bank Board meeting, effective the following day. Prior to this
    time, the announcement and effective dates were the same."

The first row affected is ``2008-02-06`` (Wed effective ≡ Tue 2008-02-05
announcement). All post-Feb-2008 effective dates are exactly one calendar day
after the announcement, including the 2020-03-20 inter-meeting COVID cut.

Output schema
-------------
``fetch()`` returns a DataFrame with the columns required by the
``CONTEXT.md`` source contract:

- ``observation_date`` (datetime64[ns]) — effective date as published.
- ``publication_date`` (datetime64[ns]) — announcement date, derived per the
  convention above (same as ``observation_date`` pre-Feb-2008;
  ``observation_date - 1 calendar day`` from Feb 2008 onward).
- ``change_raw`` (object) — change column verbatim from the HTML (e.g.
  ``"+0.25"``, ``"0.00"``, ``"-1.00"``, ``"-1.00 to -1.50"``). Range strings
  appear only in the pre-1990s rows and are passed through; downstream
  preprocessing parses to numeric and drops rows outside the project's history
  window.
- ``new_cash_rate_raw`` (object) — new cash rate target verbatim (e.g.
  ``"4.35"``, ``"15.00 to 15.50"``).
- ``statement_url`` (object | None) — relative URL to the media release, if
  one was issued.
- ``minutes_url`` (object | None) — relative URL to the meeting minutes, if
  one is published.

Side effects
------------
``fetch()`` writes:

- ``data/raw/rba_f11/<YYYY-MM-DD>.html`` — verbatim HTML bytes (immutable raw).
- ``data/raw/rba_f11/_metadata.json`` — provenance manifest with URL, SHA-256,
  download timestamp, row count, and bytes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd

from rba.config import RAW_DATA_DIR

SOURCE_NAME = "rba_f11"
SOURCE_URL = "https://www.rba.gov.au/statistics/cash-rate/"

# First "new convention" effective date. From this date onward, the listed
# effective date is one calendar day after the Board announcement.
CUTOVER_EFFECTIVE_DATE = pd.Timestamp("2008-02-06")

_TABLE_ID = "datatable"


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull RBA cash rate meeting history, persist raw, return canonical frame.

    Parameters
    ----------
    force_download
        If True (default), download a fresh copy from the RBA. If False, reuse
        the most recent existing snapshot in ``data/raw/rba_f11/`` and only
        download when no snapshot is present.

    Returns
    -------
    pandas.DataFrame
        See module docstring for schema. Sorted ascending by
        ``observation_date``.

    Shapes
    ------
    Returns: (n_meetings, 6) where ``n_meetings`` ≈ 400 as of 2026-05.
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = _resolve_snapshot(dest_dir, force_download=force_download)
    df = _parse(snapshot_path.read_bytes())
    df = _apply_date_convention(df)
    return df.sort_values("observation_date").reset_index(drop=True)


def _resolve_snapshot(dest_dir: Path, *, force_download: bool) -> Path:
    """Return path to a usable HTML snapshot, downloading if needed."""
    existing = sorted(dest_dir.glob("[0-9]" * 4 + "-*.html"))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing cash-rate snapshot at {}", path)
        return path
    return _download(dest_dir)


def _download(dest_dir: Path) -> Path:
    """Download the cash-rate HTML page, save dated snapshot, write metadata."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}.html"
    logger.info("Downloading {} -> {}", SOURCE_URL, snapshot_path)
    with urllib.request.urlopen(SOURCE_URL) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    metadata = {
        "url": SOURCE_URL,
        "snapshot_filename": snapshot_path.name,
        "sha256": sha256,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": len(raw_bytes),
    }
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)
    return snapshot_path


def _parse(html: bytes) -> pd.DataFrame:
    """Parse the cash-rate HTML table into a canonical event-level frame.

    Parameters
    ----------
    html
        Raw HTML bytes containing a ``<table id="datatable">`` with rows of
        ``<th scope="row">effective date</th><td>change</td><td>rate</td>
        <td>links</td>``. The links cell may be missing entirely on some early
        rows.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns]), ``change_raw`` (object),
        ``new_cash_rate_raw`` (object), ``statement_url`` (object | None),
        ``minutes_url`` (object | None). Rows ordered as in source (descending
        by date).

    Shapes
    ------
    Returns: (n_meetings, 5).
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id=_TABLE_ID)
    if table is None:
        raise ValueError(f"No <table id={_TABLE_ID!r}> in HTML; page layout changed.")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        th = tr.find("th", attrs={"scope": "row"})
        if th is None:
            continue
        date_str = th.get_text(strip=True)
        tds = tr.find_all("td")
        if len(tds) < 2:
            logger.warning("Skipping malformed row at date {!r}: <2 <td> cells", date_str)
            continue
        change_raw = tds[0].get_text(strip=True)
        new_rate_raw = tds[1].get_text(strip=True)

        statement_url: str | None = None
        minutes_url: str | None = None
        if len(tds) >= 3:
            for a in tds[2].find_all("a", href=True):
                href = a["href"]
                label = (a.get("aria-label") or a.get_text(strip=True)).lower()
                if "minutes" in href or "minutes" in label:
                    minutes_url = href
                elif "media-release" in href or "statement" in label:
                    statement_url = href

        rows.append(
            {
                "observation_date": pd.to_datetime(date_str, dayfirst=False, format="mixed"),
                "change_raw": change_raw,
                "new_cash_rate_raw": new_rate_raw,
                "statement_url": statement_url,
                "minutes_url": minutes_url,
            }
        )

    if not rows:
        raise ValueError("Parsed zero rows from cash-rate table; page layout changed.")

    return pd.DataFrame(rows)


def _apply_date_convention(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``publication_date`` per the Feb-2008 announcement-day convention.

    Pre-Feb-2008 rows: ``publication_date == observation_date``.
    From ``CUTOVER_EFFECTIVE_DATE`` (2008-02-06) onward:
    ``publication_date = observation_date - 1 calendar day``.

    Returns
    -------
    pandas.DataFrame
        Same as input plus a ``publication_date`` (datetime64[ns]) column.
        Column order is reset to put both date columns first.
    """
    out = df.copy()
    obs = out["observation_date"]
    one_day = pd.Timedelta(days=1)
    out["publication_date"] = obs.where(obs < CUTOVER_EFFECTIVE_DATE, obs - one_day)

    cols = [c for c in out.columns if c != "publication_date"]
    obs_idx = cols.index("observation_date")
    cols.insert(obs_idx + 1, "publication_date")
    return out[cols]
