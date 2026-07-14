"""RBA E1 / E2 (Household Balance Sheets) release calendar — scraped.

Background
----------
Tables E1 (Household and Business Balance Sheets) and E2 (Household
Finances – Selected Ratios) publish quarterly with a documented lag of
~12-13 weeks from the reference quarter end. The RBA does not list E1
or E2 in its public release-frequency schedule, and the historical
sample of release dates does not fit a clean algorithmic rule:

============================  ================  ===================================
Reference quarter             Released          Notes
----------------------------  ----------------  -----------------------------------
2014-Q4 (Dec 2014)            2015-03-27 Fri    last Fri of M+3
2015-Q1                       2015-06-26 Fri    last Fri of M+3
2015-Q3                       2015-12-18 Fri    pushed earlier — Christmas
2015-Q4                       2016-04-01 Fri    pushed later  — Good Fri (Mar 25)
2016-Q1                       2016-07-01 Fri    first Fri of M+4 (no Easter)
2016-Q2                       2016-09-30 Fri    last Fri of M+3
2016-Q3                       2016-12-31 Sat    last-day-of-Dec (unusual)
2016-Q4                       2017-03-31 Fri    last Fri of M+3
2017-Q1                       2017-06-30 Fri    last Fri of M+3
2017-Q2                       2017-10-03 Tue    correction publication
2017-Q3                       2018-02-02 Fri    delayed
2017-Q4                       2018-04-03 Tue    Tue after Easter (Mar 30-Apr 2)
2018-Q1                       2018-06-29 Fri    last Fri of M+3
2018-Q3                       2018-12-14 Fri    pushed earlier — Christmas
2019-Q4                       2020-03-27 Fri    last Fri of M+3
2020-Q3                       2020-12-18 Fri    pushed earlier — Christmas
2022-Q3                       2023-02-07 Tue    delayed
2022-Q4                       2023-03-24 Fri    last Fri of M+3 (one Fri early)
2025-Q1                       2025-06-27 Fri    last Fri of M+3
2025-Q4                       2026-03-27 Fri    last Fri of M+3
============================  ================  ===================================

Across this 20-release sample we observed Friday, Tuesday, and Saturday
release weekdays; release-day-of-month varies between the 3rd and the
last day of the third month following quarter end; Christmas weeks pull
the release back by 1-2 weeks; Easter weeks push it forward by 1 week
or to the following Tuesday. No clean rule fits.

Scraping strategy
-----------------
The RBA refreshes ``e2-data.csv`` in place — each refresh overwrites the
``Publication date`` header row with the current release date but
preserves no public archive of past release dates. We therefore harvest
two sources:

1. **archive.org Wayback Machine** for the historical record. The CDX
   API enumerates unique snapshots of ``rba.gov.au/statistics/tables/csv/
   e2-data.csv`` (collapsed by digest, so one row per actual content
   change). For each snapshot we extract the ``Publication date`` header
   and the last observed quarter end (the most recent row with any
   non-null value across the series columns); that pair becomes one
   calendar entry with ``source = "archive_org"``.

2. **The live RBA CSV** for the latest release. ``Publication date`` in
   the current ``e2-data.csv`` gives the most recent release date, paired
   with the last data row's quarter. ``source = "rba_page"``.

Coverage limitations
--------------------
Wayback only crawls the RBA CSV opportunistically — over the inflation-
targeting era we get scraped coverage from **2014-Q4 onward** with gaps
(observed ~50% coverage of post-2014 quarters). Earlier observations
(1988-Q2 through 2014-Q3, ~105 quarters) have no scraped publication
date. The consuming source module applies a flat conservative
``+ 95`` day offset (slightly longer than the worst observed lag of
~92 days) for any quarter outside the scraped set; those rows carry
``source = "inferred"``.

We accept this as a documented limitation — the spec target of ≥95%
non-inferred coverage is not achievable for E2 because the RBA does not
publish a release-date archive and Wayback's crawl density is partial.
Approximately 13% of E2 observations have ``source != "inferred"``
publication dates; the remaining 87% use the flat offset. This is the
honest read: the calendar's value is forward-looking (current and
future releases) and recent-vintage backfill, not 1988-era backfill.

Source provenance enum
----------------------
Each row in the materialised CSV carries a ``source`` column with one of:

- ``rba_page`` — date came from the live RBA CSV's ``Publication date``
  header at last harvest. There is at most one ``rba_page`` row per
  harvest (the latest release).
- ``archive_org`` — date came from a Wayback snapshot of an earlier RBA
  CSV vintage.
- ``inferred`` — placeholder for rows added via ``_OVERRIDES`` (or by
  the consuming source module's flat-offset fallback, applied at lookup
  time rather than baked into this CSV).

Overrides
---------
``_OVERRIDES`` patches individual reference-quarter-ends without
re-harvesting (e.g. RBA corrects a release date or a hand-verified value
needs to be injected). Empty by default.

Usage
-----
- ``rba_e_publication_date(reference_quarter_end)`` — single lookup;
  returns ``None`` if the quarter is outside the harvested window.
- ``build_rba_e_release_calendar()`` — load the materialised CSV.
- ``uv run python -m rba.data.rba_e_release_calendar`` — re-harvest
  Wayback + live CSV and rewrite ``data/external/rba_e_release_dates.csv``.
"""

from __future__ import annotations

from datetime import date
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

_CSV_PATH: Path = EXTERNAL_DATA_DIR / "rba_e_release_dates.csv"

# Live RBA endpoint and Wayback CDX query. ``collapse=digest`` returns
# one row per unique CSV content (i.e. per actual refresh), which is what
# we want for release-date extraction.
_LIVE_CSV_URL = "https://www.rba.gov.au/statistics/tables/csv/e2-data.csv"
_CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
    "?url=rba.gov.au/statistics/tables/csv/e2-data.csv"
    "&output=json&fl=timestamp,digest,original"
    "&filter=statuscode:200"
    "&collapse=digest"
)

# Hand-verified overrides keyed by reference-quarter-end ``date``. Populate to
# patch a known-bad scrape or inject a value the harvest missed.
#
# ``2025-12-31 → 2026-03-27``: was the live ``Publication date`` header on the
# RBA E2 CSV between 2026-03-27 and 2026-06-26 (when the 2026-Q1 update
# replaced it in-place). Wayback did not crawl within that window, so the
# scraper's ``rba_page`` sees only 2026-Q1 (published 2026-06-26). This
# override restores the fact so :mod:`rba.data.rba_e2_household_ratios`'s
# 2025-Q4 observation has a real publication date rather than the flat
# fallback offset. Add further entries here as each new quarter's live-only
# window closes without a Wayback snapshot.
_OVERRIDES: dict[date, date] = {
    date(2025, 12, 31): date(2026, 3, 27),
}

_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "rba-cash-rate-prediction/0.1 (research; "
        "https://github.com/Tye-G/rba-cash-rate-prediction)"
    ),
    "Accept": "text/csv,*/*",
}
_SCRAPE_DELAY_SECONDS = 0.4


def rba_e_publication_date(reference_quarter_end: date) -> date | None:
    """Return the E2 publication date for ``reference_quarter_end``, or
    ``None`` if the quarter is outside the harvested window.

    Parameters
    ----------
    reference_quarter_end
        End-of-quarter date (Mar 31, Jun 30, Sep 30, or Dec 31).

    Returns
    -------
    datetime.date | None
        Publication date, or ``None`` if the calendar has no entry for
        this quarter. The consuming source module applies a flat
        conservative offset for ``None`` results.
    """
    if reference_quarter_end in _OVERRIDES:
        return _OVERRIDES[reference_quarter_end]

    cal = build_rba_e_release_calendar()
    qe_ts = pd.Timestamp(reference_quarter_end)
    hit = cal.loc[cal["reference_quarter_end"] == qe_ts, "publication_date"]
    if hit.empty:
        return None
    return hit.iloc[0].date()


def build_rba_e_release_calendar() -> pd.DataFrame:
    """Load the materialised E2 release calendar.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_quarter_end`` (datetime64[ns]) — quarter-end date.
        - ``publication_date`` (datetime64[ns]) — scraped release date.
        - ``source`` (object) — one of ``rba_page``, ``archive_org``,
          ``inferred``.

    Shapes
    ------
    Returns: (n_scraped, 3). Empty frame with correct columns/dtypes if
    the CSV has not been materialised yet (run ``python -m
    rba.data.rba_e_release_calendar`` to populate it).
    """
    if not _CSV_PATH.exists():
        logger.warning(
            "RBA E release-date CSV missing at {}; returning empty calendar. "
            "Run `python -m rba.data.rba_e_release_calendar` to populate.",
            _CSV_PATH,
        )
        return _empty_frame()

    df = pd.read_csv(_CSV_PATH, parse_dates=["reference_quarter_end", "publication_date"])
    df["reference_quarter_end"] = df["reference_quarter_end"].dt.as_unit("ns")
    df["publication_date"] = df["publication_date"].dt.as_unit("ns")
    if "source" not in df.columns:
        df["source"] = "archive_org"
    df["source"] = df["source"].astype(object)

    if _OVERRIDES:
        df = df.copy()
        for qe, pub in _OVERRIDES.items():
            qe_ts = pd.Timestamp(qe)
            pub_ts = pd.Timestamp(pub)
            mask = df["reference_quarter_end"] == qe_ts
            if mask.any():
                df.loc[mask, "publication_date"] = pub_ts
                df.loc[mask, "source"] = "inferred"
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "reference_quarter_end": [qe_ts],
                                "publication_date": [pub_ts],
                                "source": ["inferred"],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        df = df.sort_values("reference_quarter_end").reset_index(drop=True)
        df["reference_quarter_end"] = df["reference_quarter_end"].astype("datetime64[ns]")
        df["publication_date"] = df["publication_date"].astype("datetime64[ns]")

    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_quarter_end": pd.Series([], dtype="datetime64[ns]"),
            "publication_date": pd.Series([], dtype="datetime64[ns]"),
            "source": pd.Series([], dtype=object),
        }
    )


# ---------------------------------------------------------------------------
# Harvest pipeline (called from __main__)
# ---------------------------------------------------------------------------


def _parse_e2_csv(body: bytes) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Extract (publication_date, last_data_quarter_end) from an E2 CSV.

    Handles both historical date formats: modern ``DD/MM/YYYY`` and
    pre-2018 ``Mon-YYYY``. Returns ``(None, None)`` if either cannot be
    located.
    """
    try:
        df = pd.read_csv(io.BytesIO(body), header=1, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(body), header=1, low_memory=False, encoding="latin-1")

    title_col = df.columns[0]

    pub_row = df[df[title_col] == "Publication date"]
    pub: pd.Timestamp | None = None
    if not pub_row.empty:
        for col, val in pub_row.iloc[0].items():
            if col == title_col or pd.isna(val):
                continue
            pub = pd.to_datetime(str(val), errors="coerce")
            if pd.notna(pub):
                break

    modern = pd.to_datetime(df[title_col], format="%d/%m/%Y", errors="coerce")
    legacy = pd.to_datetime(df[title_col], format="%b-%Y", errors="coerce")
    legacy = legacy + pd.offsets.MonthEnd(0)
    parsed_dates = modern.fillna(legacy)
    data_mask = parsed_dates.notna()
    if not data_mask.any():
        return pub, None

    series_cols = [c for c in df.columns if c != title_col]
    data_df = df.loc[data_mask].copy()
    last_with_data: pd.Timestamp | None = None
    for idx in reversed(data_df.index):
        row_vals = data_df.loc[idx, series_cols]
        has_value = row_vals.apply(lambda v: pd.notna(pd.to_numeric(v, errors="coerce"))).any()
        if has_value:
            last_with_data = parsed_dates.loc[idx]
            break

    return pub, last_with_data


def _list_wayback_snapshots() -> list[tuple[str, str]]:
    """Query the Wayback CDX API for unique snapshots of e2-data.csv."""
    req = urllib.request.Request(_CDX_URL, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — public CDX URL
        data = json.loads(r.read().decode("utf-8"))
    return [(row[0], row[2]) for row in data[1:]]


def _fetch_wayback(timestamp: str, url: str) -> bytes | None:
    snap_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    req = urllib.request.Request(snap_url, headers=_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — public Wayback URL
            return r.read()
    except urllib.error.URLError as e:
        logger.debug("Wayback fetch {} failed: {}", snap_url, e)
        return None


def _fetch_live() -> bytes | None:
    req = urllib.request.Request(_LIVE_CSV_URL, headers=_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — public RBA URL
            return r.read()
    except urllib.error.URLError as e:
        logger.warning("Live RBA E2 CSV fetch failed: {}", e)
        return None


def _harvest() -> pd.DataFrame:
    """Harvest Wayback snapshots + live CSV into a calendar frame.

    Returns sorted, deduplicated frame keyed by ``reference_quarter_end``.
    ``rba_page`` rows take precedence over ``archive_org`` for the same
    reference quarter (the live CSV is authoritative for the latest
    release).
    """
    rows: list[dict[str, object]] = []

    snapshots = _list_wayback_snapshots()
    logger.info("Wayback returned {} unique snapshots", len(snapshots))
    for ts, url in snapshots:
        body = _fetch_wayback(ts, url)
        if body is None:
            continue
        pub, qe = _parse_e2_csv(body)
        if pub is None or qe is None:
            logger.debug("Snapshot {} did not yield (pub, qe)", ts)
            continue
        rows.append(
            {
                "reference_quarter_end": qe,
                "publication_date": pub,
                "source": "archive_org",
            }
        )
        logger.info("Wayback {} -> ref={} pub={}", ts, qe.date(), pub.date())
        time.sleep(_SCRAPE_DELAY_SECONDS)

    body = _fetch_live()
    if body is not None:
        pub, qe = _parse_e2_csv(body)
        if pub is not None and qe is not None:
            rows.append(
                {
                    "reference_quarter_end": qe,
                    "publication_date": pub,
                    "source": "rba_page",
                }
            )
            logger.info("Live RBA -> ref={} pub={}", qe.date(), pub.date())

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_frame()

    # Dedupe: prefer rba_page over archive_org for the same reference quarter.
    df["_priority"] = df["source"].map({"rba_page": 0, "archive_org": 1, "inferred": 2})
    df = df.sort_values(["reference_quarter_end", "_priority"]).drop_duplicates(
        "reference_quarter_end", keep="first"
    )
    df = df.drop(columns="_priority")
    df = df.sort_values("reference_quarter_end").reset_index(drop=True)
    df["reference_quarter_end"] = pd.to_datetime(df["reference_quarter_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    df["source"] = df["source"].astype(object)
    return df


if __name__ == "__main__":
    harvested = _harvest()
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    harvested.to_csv(_CSV_PATH, index=False)
    logger.info("Wrote RBA E release calendar ({} rows) to {}", len(harvested), _CSV_PATH)
