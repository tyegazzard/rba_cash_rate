"""ABS GDP (Cat. 5206.0) release calendar — scraped from ABS catalogue pages.

The ABS National Accounts: National Income, Expenditure and Product is the
slowest-released of the major macro series we ingest (~9 weeks after quarter
end). The headline rule for the post-2003 era is "first Wednesday of the
third month after the reference quarter" — verified directly against 20 ABS
release pages spanning 2019-Q2 through 2025-Q4. But that rule **does not**
cleanly extend back to 1993:

- 1993-1996: lag was often ~2 months rather than 3; release weekday varied
  (Fri/Tue/Wed observed); some quarters released as early as the last week
  of the month preceding the algorithmic target.
- 1999-2002: weekday-of-Wednesday slot was unstable (1st, 2nd, and 3rd
  Wednesdays all observed); at least two non-Wednesday releases recorded
  (Jun 1999 -> Fri, Jun 2002 -> Tue).
- 2003-present: rule has held cleanly with no exceptions in the sample.

The "first Wednesday of M+3" rule is therefore unsafe before 2003, and any
algorithmic-only implementation would silently produce wrong publication
dates for ~15+ historical inflation-targeting-era quarters. We instead
**scrape the original publication date** off each quarterly release page,
mirroring the WPI approach.

URL coverage
------------
Two URL templates are required to cover 1993-present:

- **Legacy** (1993-Q1 through 2019-Q1) — lives under the old ``/ausstats/``
  namespace::

      https://www.abs.gov.au/ausstats/abs@.nsf/
        PreviousProducts/5206.0Main%20Features1<Mon>%20<YYYY>

  where ``<Mon>`` is one of ``Mar``, ``Jun``, ``Sep``, ``Dec`` and
  ``<YYYY>`` is the four-digit reference year. The original publication
  date is extracted from the ``DC.Date.issued`` meta tag (a structured ISO
  date, matches the visible "Released at" field on every sampled page).

- **Modern** (2019-Q2 onward) — lives under the post-redesign namespace::

      https://www.abs.gov.au/statistics/economy/national-accounts/
        australian-national-accounts-national-income-expenditure-and-product/
        <mon>-<yyyy>

  where ``<mon>`` is lower-case (``mar`` / ``jun`` / ``sep`` / ``dec``). The
  date is extracted from the visible ``<div class="field__label">Released
  </div>`` field — **not** the modern ``dcterms.issued`` meta tag, which
  reflects the most recent page revision rather than original publication
  (a page revised long after release would otherwise leak a future
  publication date into a historical quarter).

The boundary is hard: the modern URL 404s for 2019-Q1, and the legacy URL
silently 30x-redirects to the modern "latest release" page for 2019-Q2
onward (returning a non-quarter-specific page). The split is therefore by
reference quarter, not by URL fallback.

Overrides
---------
``_OVERRIDES`` lets you patch individual quarter-ends without re-scraping
(e.g. if the ABS corrects a release date or a transient scrape failure
needs a hand-verified value). Empty by default.

Usage
-----
- ``gdp_publication_date(quarter_end)`` — single lookup; returns ``None`` if
  the quarter is outside the scraped window.
- ``build_gdp_release_calendar()`` — DataFrame keyed by
  ``reference_quarter_end``.
- ``uv run python -m rba.data.gdp_release_calendar`` — re-scrapes both URL
  spaces and rewrites ``data/external/gdp_release_dates.csv``.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Path to the materialised scrape output. Loaded by build_gdp_release_calendar()
# and rewritten by the __main__ block.
_CSV_PATH: Path = EXTERNAL_DATA_DIR / "gdp_release_dates.csv"

# Earliest scrape-able reference quarter — inflation-targeting era starts here
# and the legacy ABS URL pattern resolves all the way back to 1993-Q1.
_SCRAPE_FLOOR_YEAR = 1993
_SCRAPE_FLOOR_QUARTER_MONTH = 3  # 1993-Q1 ends 31 Mar

# Boundary between the two URL patterns. Legacy URL is used for quarters
# strictly before this; modern URL is used from this quarter onward.
# 2019-Q2 is the earliest reference quarter the modern page resolves to.
_MODERN_URL_FROM_YEAR = 2019
_MODERN_URL_FROM_QUARTER_MONTH = 6  # 2019-Q2 ends 30 Jun

# Quarter-end month -> URL slug components.
_LEGACY_MONTH_SLUGS: dict[int, str] = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}
_MODERN_MONTH_SLUGS: dict[int, str] = {3: "mar", 6: "jun", 9: "sep", 12: "dec"}

# Known manual overrides, keyed by reference-quarter-end. Empty by default;
# add entries here if the ABS issues a correction or a known scrape failure
# needs a hand-verified value.
_OVERRIDES: dict[date, date] = {}

# Structured ISO date from the legacy ABS page header. Matches the visible
# "Released at HH:MM (CANBERRA TIME) DD/MM/YYYY" field on every sampled
# page; preferred over the visible field because the format is canonical
# and easier to parse defensively.
_LEGACY_RELEASED_RE = re.compile(
    r'<META\s+NAME="DC\.Date\.issued"\s+SCHEME="ISO8601"\s+CONTENT="(\d{4}-\d{2}-\d{2})"',
    re.IGNORECASE,
)

# Visible-field regex on the modern WPI/GDP release pages. We deliberately
# use this over the dcterms.issued meta tag because that tag reflects the
# most recent page revision, not the original publication date.
_MODERN_RELEASED_RE = re.compile(
    r'<div class="field__label">Released</div>\s*'
    r'<div class="field__item">\s*(\d{1,2}/\d{1,2}/\d{4})'
)

# Polite scraper headers + small inter-request delay so __main__ does not
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
    raise ValueError(
        f"quarter_month must be one of 3, 6, 9, 12 (got {quarter_month})"
    )


def _uses_modern_url(year: int, quarter_month: int) -> bool:
    """True if ``(year, quarter_month)`` should be scraped from the modern
    URL pattern (2019-Q2 onward); False for the legacy pattern."""
    if year > _MODERN_URL_FROM_YEAR:
        return True
    if year < _MODERN_URL_FROM_YEAR:
        return False
    return quarter_month >= _MODERN_URL_FROM_QUARTER_MONTH


def _build_url(year: int, quarter_month: int) -> str:
    if _uses_modern_url(year, quarter_month):
        slug = _MODERN_MONTH_SLUGS[quarter_month]
        return (
            "https://www.abs.gov.au/statistics/economy/national-accounts/"
            "australian-national-accounts-national-income-expenditure-and-product/"
            f"{slug}-{year}"
        )
    slug = _LEGACY_MONTH_SLUGS[quarter_month]
    # Spaces in the legacy slug must be percent-encoded for urllib.
    encoded = urllib.parse.quote(f"5206.0Main Features1{slug} {year}", safe="")
    return f"https://www.abs.gov.au/ausstats/abs@.nsf/PreviousProducts/{encoded}"


def scrape_release_date(year: int, quarter_month: int) -> date | None:
    """Scrape the ABS GDP release page for one quarter; return the original
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
        logger.debug("GDP release page {} returned HTTP {}", url, e.code)
        return None

    if _uses_modern_url(year, quarter_month):
        match = _MODERN_RELEASED_RE.search(body)
        if match is None:
            logger.warning(
                "Modern GDP release page {} fetched OK but no Released field matched",
                url,
            )
            return None
        day, month, yr = match.group(1).split("/")
        return date(int(yr), int(month), int(day))

    match = _LEGACY_RELEASED_RE.search(body)
    if match is None:
        logger.warning(
            "Legacy GDP release page {} fetched OK but no DC.Date.issued matched",
            url,
        )
        return None
    return date.fromisoformat(match.group(1))


def gdp_publication_date(reference_quarter_end: date) -> date | None:
    """Return the GDP publication date for ``reference_quarter_end``, or
    ``None`` if it is outside the scraped calendar window.

    Honours ``_OVERRIDES`` first. Otherwise looks up the materialised CSV
    (``data/external/gdp_release_dates.csv``); returns ``None`` if absent.
    """
    if reference_quarter_end in _OVERRIDES:
        return _OVERRIDES[reference_quarter_end]

    cal = build_gdp_release_calendar()
    qe_ts = pd.Timestamp(reference_quarter_end)
    hit = cal.loc[cal["reference_quarter_end"] == qe_ts, "publication_date"]
    if hit.empty:
        return None
    return hit.iloc[0].date()


def build_gdp_release_calendar() -> pd.DataFrame:
    """Load the materialised GDP release calendar.

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
    rba.data.gdp_release_calendar`` to populate it).
    """
    if not _CSV_PATH.exists():
        logger.warning(
            "GDP release-date CSV missing at {}; returning empty calendar. "
            "Run `python -m rba.data.gdp_release_calendar` to populate.",
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
    floor (1993-Q1) through the end of ``through_year`` (defaults to current
    calendar year).
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
    """Scrape every (year, quarter) page in the window, returning a long
    DataFrame of successful hits. 404s and parse failures are skipped with
    a debug log.
    """
    rows: list[dict[str, date]] = []
    for year, qm in _iter_scrape_targets(through_year=through_year):
        pub = scrape_release_date(year, qm)
        qe = quarter_end(year, qm)
        slug = (
            _MODERN_MONTH_SLUGS[qm] if _uses_modern_url(year, qm) else _LEGACY_MONTH_SLUGS[qm]
        )
        if pub is None:
            logger.info("GDP {}-{} -> no release date (skipping)", year, slug)
        else:
            rows.append({"reference_quarter_end": qe, "publication_date": pub})
            logger.info("GDP {}-{} -> released {}", year, slug, pub.isoformat())
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
        "Wrote GDP release calendar ({} rows) to {}",
        len(scraped),
        _CSV_PATH,
    )
