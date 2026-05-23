"""ABS Total Value of Dwellings (Cat. 6432.0) release calendar — scraped.

TVD is the live successor to the discontinued Residential Property Price
Indexes (Cat. 6416.0). It publishes quarterly with no clean algorithmic
release rule: across an 8-quarter sample spanning 2022-Q1 → 2025-Q4 the
release was always a Tuesday but the *week* alternated between the 1st and
2nd Tuesday of the third month following the reference quarter end:

============================  ================  =====================
Reference quarter             Released          Week of M+3 Tuesday
----------------------------  ----------------  ---------------------
2022-Q1 (mar-quarter-2022)    2022-06-14        2nd Tuesday of Jun
2022-Q3                       2022-12-06        1st Tuesday of Dec
2023-Q3                       2023-12-05        1st Tuesday of Dec
2024-Q4                       2025-03-11        2nd Tuesday of Mar
2025-Q1                       2025-06-10        2nd Tuesday of Jun
2025-Q2                       2025-09-09        2nd Tuesday of Sep
2025-Q3                       2025-12-02        1st Tuesday of Dec
2025-Q4                       2026-03-10        2nd Tuesday of Mar
============================  ================  =====================

The 1st-vs-2nd-Tuesday choice is unstable enough that we **scrape**
rather than apply a rule. The scraped URL pattern is::

    https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/
      total-value-dwellings/<mon>-quarter-<yyyy>

where ``<mon>`` is the three-letter lower-case month of the reference
quarter end (``mar``, ``jun``, ``sep``, ``dec``) and ``<yyyy>`` is the
four-digit year. The visible "Released" field is preferred over any meta
tag because the meta tag tracks the most recent page revision rather than
the original publication date — same defensive choice as wpi / gdp / ba.

URL coverage
------------
The modern ABS TVD pages exist back to **2022-Q1**. The 2021-Q4 page
404s, as does the legacy ``residential-property-price-indexes-...``
namespace. The scrape floor is therefore ``date(2022, 3, 31)``;
pre-floor quarters (the RPPI-era 2011-Q3 → 2021-Q4) are not scrapable.

Pre-floor fallback
------------------
The consuming source module ``abs_total_value_dwellings`` applies a flat
conservative ``+ 76`` day offset (slightly longer than the worst observed
TVD lag of 73 days) for any observation whose reference quarter pre-dates
the scrape floor. This covers the RPPI legacy series (2003-Q3 → 2021-Q4)
that the source module pulls as a splice baseline.

Source provenance
-----------------
Each row carries a ``source`` column with one of:

- ``abs_page`` — date came from the ABS release-page scrape.
- ``archive_org`` — reserved fallback if a live ABS page 404s but a
  Wayback Machine snapshot exists. Not actively populated in v1.
- ``inferred`` — date came from an override or a flat-offset fallback
  applied at lookup time.

Overrides
---------
``_OVERRIDES`` lets you patch individual reference-quarter-ends without
re-scraping (e.g. ABS corrects a release date). Empty by default.

Usage
-----
- ``tvd_publication_date(quarter_end)`` — single lookup; returns ``None``
  if the quarter is outside the scraped window (the source module
  applies its flat offset instead).
- ``build_tvd_release_calendar()`` — DataFrame keyed by
  ``reference_quarter_end``.
- ``uv run python -m rba.data.abs_tvd_release_calendar`` — re-scrapes
  the ABS pages and rewrites ``data/external/abs_tvd_release_dates.csv``.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

_CSV_PATH: Path = EXTERNAL_DATA_DIR / "abs_tvd_release_dates.csv"

# Earliest scrape-able reference quarter on the modern ABS TVD URL space.
_SCRAPE_FLOOR_YEAR = 2022
_SCRAPE_FLOOR_QUARTER_MONTH = 3  # 2022-Q1 ends 31 Mar

# Quarter-end month -> URL slug component.
_MONTH_SLUGS: dict[int, str] = {3: "mar", 6: "jun", 9: "sep", 12: "dec"}

# Hand-verified overrides keyed by reference-quarter-end. Empty by default.
_OVERRIDES: dict[date, date] = {}

# Modern ABS release-page "Released" field — same as wpi/gdp/ba.
_RELEASED_RE = re.compile(
    r'<div class="field__label">Released</div>\s*'
    r'<div class="field__item">\s*(\d{1,2}/\d{1,2}/\d{4})'
)

_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "rba-cash-rate-prediction/0.1 (research; "
        "https://github.com/Tye-G/rba-cash-rate-prediction)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}
_SCRAPE_DELAY_SECONDS = 0.5


def quarter_end(year: int, quarter_month: int) -> date:
    """Return the calendar end-of-quarter date for the quarter ending in
    ``quarter_month`` (3, 6, 9, or 12).
    """
    if quarter_month == 3:
        return date(year, 3, 31)
    if quarter_month == 6:
        return date(year, 6, 30)
    if quarter_month == 9:
        return date(year, 9, 30)
    if quarter_month == 12:
        return date(year, 12, 31)
    raise ValueError(
        f"quarter_month must be one of 3, 6, 9, 12 (got {quarter_month})"
    )


def _build_url(year: int, quarter_month: int) -> str:
    slug = _MONTH_SLUGS[quarter_month]
    return (
        "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
        f"total-value-dwellings/{slug}-quarter-{year}"
    )


def scrape_release_date(year: int, quarter_month: int) -> date | None:
    """Scrape one ABS TVD release page.

    Returns
    -------
    datetime.date | None
        Original release date, or ``None`` if the page 404s or the
        ``Released`` field cannot be matched.
    """
    url = _build_url(year, quarter_month)
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 — public ABS URL
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.debug("ABS TVD release page {} returned HTTP {}", url, e.code)
        return None

    match = _RELEASED_RE.search(body)
    if match is None:
        logger.warning(
            "ABS TVD release page {} fetched OK but no Released field matched", url
        )
        return None
    day, month_str, year_str = match.group(1).split("/")
    return date(int(year_str), int(month_str), int(day))


def tvd_publication_date(reference_quarter_end: date) -> date | None:
    """Return the TVD publication date for ``reference_quarter_end``, or
    ``None`` if the quarter is outside the scraped window.
    """
    if reference_quarter_end in _OVERRIDES:
        return _OVERRIDES[reference_quarter_end]

    cal = build_tvd_release_calendar()
    qe_ts = pd.Timestamp(reference_quarter_end)
    hit = cal.loc[cal["reference_quarter_end"] == qe_ts, "publication_date"]
    if hit.empty:
        return None
    return hit.iloc[0].date()


def build_tvd_release_calendar() -> pd.DataFrame:
    """Load the materialised TVD release calendar.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_quarter_end`` (datetime64[ns]) — quarter-end date.
        - ``publication_date`` (datetime64[ns]) — scraped release date.
        - ``source`` (object) — one of ``abs_page``, ``archive_org``,
          ``inferred``.

    Shapes
    ------
    Returns: (n_scraped, 3). Empty frame with correct columns/dtypes if
    the CSV has not been materialised yet.
    """
    if not _CSV_PATH.exists():
        logger.warning(
            "ABS TVD release-date CSV missing at {}; returning empty calendar. "
            "Run `python -m rba.data.abs_tvd_release_calendar` to populate.",
            _CSV_PATH,
        )
        return _empty_frame()

    df = pd.read_csv(_CSV_PATH, parse_dates=["reference_quarter_end", "publication_date"])
    df["reference_quarter_end"] = df["reference_quarter_end"].dt.as_unit("ns")
    df["publication_date"] = df["publication_date"].dt.as_unit("ns")
    if "source" not in df.columns:
        df["source"] = "abs_page"
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


def _iter_scrape_targets(through_year: int | None = None) -> list[tuple[int, int]]:
    """Yield ``(year, quarter_month)`` pairs from the scrape floor through
    ``through_year`` (defaults to the current calendar year).
    """
    if through_year is None:
        through_year = datetime.now().year
    pairs: list[tuple[int, int]] = []
    for year in range(_SCRAPE_FLOOR_YEAR, through_year + 1):
        for qm in (3, 6, 9, 12):
            if year == _SCRAPE_FLOOR_YEAR and qm < _SCRAPE_FLOOR_QUARTER_MONTH:
                continue
            pairs.append((year, qm))
    return pairs


def _scrape_all(through_year: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, qm in _iter_scrape_targets(through_year=through_year):
        pub = scrape_release_date(year, qm)
        qe = quarter_end(year, qm)
        slug = _MONTH_SLUGS[qm]
        if pub is None:
            logger.info("TVD {}-{} -> no release date (skipping)", year, slug)
        else:
            rows.append(
                {
                    "reference_quarter_end": qe,
                    "publication_date": pub,
                    "source": "abs_page",
                }
            )
            logger.info("TVD {}-{} -> released {}", year, slug, pub.isoformat())
        time.sleep(_SCRAPE_DELAY_SECONDS)

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_frame()
    df["reference_quarter_end"] = pd.to_datetime(df["reference_quarter_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    df["source"] = df["source"].astype(object)
    return df.sort_values("reference_quarter_end").reset_index(drop=True)


if __name__ == "__main__":
    scraped = _scrape_all()
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(_CSV_PATH, index=False)
    logger.info(
        "Wrote ABS TVD release calendar ({} rows) to {}", len(scraped), _CSV_PATH
    )
