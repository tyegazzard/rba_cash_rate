"""ABS Wage Price Index (WPI) release calendar — scraped from ABS release pages.

The ABS WPI (Cat. 6345.0) has no clean algorithmic release rule. Verified
historical releases land between days 12-21 of the second month after the
reference quarter, mostly on a Wednesday but at least one Tuesday (Jun 2024
ref -> 13 Aug 2024, a Tuesday), and split unpredictably between the 2nd and
3rd Wednesday slots across years. The observed lag ranges from ~43 to ~52
days; that range is too wide for a flat offset to be both safe and useful.

This module instead scrapes the **original release date** from each WPI
quarterly release page on the ABS website:

::

    https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/
      wage-price-index-australia/<mon>-<yyyy>

where ``<mon>`` is one of ``mar``, ``jun``, ``sep``, ``dec`` and ``<yyyy>``
is the four-digit year of the reference quarter. The scraper persists the
result to ``data/external/wpi_release_dates.csv`` so the calendar build is
deterministic, reproducible, and offline-loadable.

We deliberately extract the visible "Released" field rather than the
``dcterms.issued`` meta tag — that meta tag tracks the most recent page
revision (e.g. Jun 2020 was revised on 2020-11-02 but originally released on
2020-08-12). Point-in-time correctness requires the original publication
date.

URL coverage
------------
The new ABS site (post-2020 redesign) hosts pages back to **2019-Q3**.
Earlier quarterly pages (1997-Q3 through 2019-Q2) live under the legacy
``/Ausstats/`` URL space with opaque NSF document IDs and are not in scope
for this scraper. Pre-2019-Q3 observations therefore have no row in this
calendar; the consuming source (``rba.data.sources.abs_wpi``) applies a
flat conservative offset to those rows. See ``abs_wpi`` module docstring.

Overrides
---------
``_OVERRIDES`` lets you patch individual quarter-ends without re-scraping
(e.g. if the ABS corrects a release date). Empty by default.

Usage
-----
- ``wpi_publication_date(quarter_end)`` — single lookup; returns ``None`` if
  the quarter is outside the scraped window.
- ``build_wpi_release_calendar()`` — DataFrame keyed by
  ``reference_quarter_end``.
- ``uv run python -m rba.data.wpi_release_calendar`` — re-scrapes the ABS
  pages and rewrites ``data/external/wpi_release_dates.csv``.
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

# Path to the materialised scrape output. Loaded by build_wpi_release_calendar()
# and rewritten by the __main__ block.
_CSV_PATH: Path = EXTERNAL_DATA_DIR / "wpi_release_dates.csv"

# Earliest scrape-able reference quarter on the post-redesign ABS site.
_SCRAPE_FLOOR_YEAR = 2019
_SCRAPE_FLOOR_QUARTER = 3  # 2019-Q3 is the first page that resolves

# Map reference quarter-end month -> URL slug component.
_MONTH_SLUGS: dict[int, str] = {3: "mar", 6: "jun", 9: "sep", 12: "dec"}

# Known manual overrides, keyed by reference-quarter-end. Empty by default;
# add entries here if the ABS issues a correction or a known scrape failure
# needs a hand-verified value.
_OVERRIDES: dict[date, date] = {}

# Visible-field regex on the WPI release pages. We deliberately use this over
# the dcterms.issued meta tag because that tag reflects the most recent page
# revision, not the original publication date.
_RELEASED_RE = re.compile(
    r'<div class="field__label">Released</div>\s*'
    r'<div class="field__item">\s*(\d{1,2}/\d{1,2}/\d{4})'
)

# Polite scraper headers + a small inter-request delay so __main__ does not
# hammer the ABS site.
_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "rba-cash-rate-prediction/0.1 (research; "
        "https://github.com/Tye-G/rba-cash-rate-prediction)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}
_SCRAPE_DELAY_SECONDS = 0.5


def quarter_end(year: int, quarter_month: int) -> date:
    """Return the calendar end-of-quarter date for the quarter whose end
    month is ``quarter_month`` (3, 6, 9, or 12).
    """
    if quarter_month == 3:
        return date(year, 3, 31)
    if quarter_month == 6:
        return date(year, 6, 30)
    if quarter_month == 9:
        return date(year, 9, 30)
    if quarter_month == 12:
        return date(year, 12, 31)
    raise ValueError(f"quarter_month must be one of 3, 6, 9, 12 (got {quarter_month})")


def _build_url(year: int, quarter_month: int) -> str:
    slug = _MONTH_SLUGS[quarter_month]
    return (
        "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
        f"wage-price-index-australia/{slug}-{year}"
    )


def scrape_release_date(year: int, quarter_month: int) -> date | None:
    """Scrape the ABS WPI release page for one quarter; return the original
    publication date, or ``None`` if the page 404s or the date cannot be
    parsed.

    Parameters
    ----------
    year
        Four-digit reference year of the quarter.
    quarter_month
        End-month of the reference quarter: 3, 6, 9, or 12.

    Returns
    -------
    datetime.date | None
        Original release date, or ``None`` if not extractable.
    """
    url = _build_url(year, quarter_month)
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 — public ABS URL
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.debug("WPI release page {} returned HTTP {}", url, e.code)
        return None

    match = _RELEASED_RE.search(body)
    if match is None:
        logger.warning("WPI release page {} fetched OK but no Released field matched", url)
        return None
    day, month, yr = match.group(1).split("/")
    return date(int(yr), int(month), int(day))


def wpi_publication_date(reference_quarter_end: date) -> date | None:
    """Return the WPI publication date for ``reference_quarter_end``, or
    ``None`` if it is outside the scraped calendar window.

    Honours ``_OVERRIDES`` first. Otherwise looks up the materialised CSV
    (``data/external/wpi_release_dates.csv``); returns ``None`` if absent.
    """
    if reference_quarter_end in _OVERRIDES:
        return _OVERRIDES[reference_quarter_end]

    cal = build_wpi_release_calendar()
    qe_ts = pd.Timestamp(reference_quarter_end)
    hit = cal.loc[cal["reference_quarter_end"] == qe_ts, "publication_date"]
    if hit.empty:
        return None
    return hit.iloc[0].date()


def build_wpi_release_calendar() -> pd.DataFrame:
    """Load the materialised WPI release calendar.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_quarter_end`` (datetime64[ns]) — quarter-end date.
        - ``publication_date`` (datetime64[ns]) — scraped original release
          date (with any ``_OVERRIDES`` substitutions applied).

    Shapes
    ------
    Returns: (n_scraped, 2). Empty frame with the correct columns/dtypes if
    the CSV has not been materialised yet (run ``python -m
    rba.data.wpi_release_calendar`` to populate it).
    """
    if not _CSV_PATH.exists():
        logger.warning(
            "WPI release-date CSV missing at {}; returning empty calendar. "
            "Run `python -m rba.data.wpi_release_calendar` to populate.",
            _CSV_PATH,
        )
        return pd.DataFrame(
            {
                "reference_quarter_end": pd.Series([], dtype="datetime64[ns]"),
                "publication_date": pd.Series([], dtype="datetime64[ns]"),
            }
        )

    df = pd.read_csv(_CSV_PATH, parse_dates=["reference_quarter_end", "publication_date"])
    df["reference_quarter_end"] = df["reference_quarter_end"].dt.as_unit("ns")
    df["publication_date"] = df["publication_date"].dt.as_unit("ns")

    if _OVERRIDES:
        df = df.copy()
        for qe, pub in _OVERRIDES.items():
            mask = df["reference_quarter_end"] == pd.Timestamp(qe)
            if mask.any():
                df.loc[mask, "publication_date"] = pd.Timestamp(pub)
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "reference_quarter_end": [pd.Timestamp(qe)],
                                "publication_date": [pd.Timestamp(pub)],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        df = df.sort_values("reference_quarter_end").reset_index(drop=True)
        df["reference_quarter_end"] = df["reference_quarter_end"].astype("datetime64[ns]")
        df["publication_date"] = df["publication_date"].astype("datetime64[ns]")

    return df


def _iter_scrape_targets(through_year: int | None = None) -> list[tuple[int, int]]:
    """Yield (year, quarter_month) pairs for every quarter from the scrape
    floor (2019-Q3) through the end of ``through_year`` (defaults to current
    calendar year).
    """
    if through_year is None:
        through_year = datetime.now().year
    pairs: list[tuple[int, int]] = []
    for year in range(_SCRAPE_FLOOR_YEAR, through_year + 1):
        for qm in (3, 6, 9, 12):
            # Skip pre-floor quarters in the floor year.
            if year == _SCRAPE_FLOOR_YEAR and qm < _SCRAPE_FLOOR_QUARTER * 3:
                continue
            pairs.append((year, qm))
    return pairs


def _scrape_all(through_year: int | None = None) -> pd.DataFrame:
    """Scrape every (year, quarter) page in the window, returning a long
    DataFrame of successful hits. 404s and parse failures are skipped with
    a debug log.
    """
    rows: list[dict[str, date]] = []
    for year, qm in _iter_scrape_targets(through_year=through_year):
        pub = scrape_release_date(year, qm)
        qe = quarter_end(year, qm)
        if pub is None:
            logger.info("WPI {}-{} -> no release date (skipping)", year, _MONTH_SLUGS[qm])
        else:
            rows.append({"reference_quarter_end": qe, "publication_date": pub})
            logger.info("WPI {}-{} -> released {}", year, _MONTH_SLUGS[qm], pub.isoformat())
        time.sleep(_SCRAPE_DELAY_SECONDS)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            {
                "reference_quarter_end": pd.Series([], dtype="datetime64[ns]"),
                "publication_date": pd.Series([], dtype="datetime64[ns]"),
            }
        )
    df["reference_quarter_end"] = pd.to_datetime(df["reference_quarter_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df.sort_values("reference_quarter_end").reset_index(drop=True)


if __name__ == "__main__":
    scraped = _scrape_all()
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(_CSV_PATH, index=False)
    logger.info(
        "Wrote WPI release calendar ({} rows) to {}",
        len(scraped),
        _CSV_PATH,
    )
