"""ABS Building Approvals (Cat. 8731.0) release calendar — scraped.

Building Approvals publishes monthly with a documented release window of
"early in the month, two months after the reference month". The window is
real (every sampled release landed days 1-8 of the second following month)
but the weekday is not — across an 11-release sample spanning Sep 2020 →
Mar 2026 we observed Monday, Tuesday, Wednesday, and Thursday release
weekdays with no era pattern. The day-of-month also varies between 1 and
8 with no obvious schedule:

============================  ================  ===========
Reference month               Released          Weekday
----------------------------  ----------------  -----------
2020-09 (Sep 2020)            2020-11-02        Mon
2020-06                       2020-07-30        Thu
2020-01                       2020-03-03        Tue
2019-12                       2020-02-03        Mon
2022-03 (Mar 2022)            2022-05-05        Thu
2023-03                       2023-05-08        Mon
2023-12                       2024-02-01        Thu
2024-02                       2024-04-04        Thu
2025-07                       2025-09-01        Mon
2025-11                       2026-01-07        Wed
2025-12                       2026-02-03        Tue
2026-01                       2026-03-03        Tue
2026-02                       2026-04-01        Wed
2026-03                       2026-05-04        Mon
============================  ================  ===========

No clean algorithmic rule fits this distribution. We therefore scrape the
**original release date** from each monthly Building Approvals page on the
ABS website, mirroring the WPI and GDP calendar pattern:

::

    https://www.abs.gov.au/statistics/industry/building-and-construction/
      building-approvals-australia/<mon>-<yyyy>

where ``<mon>`` is the three-letter lower-case reference month (``jan``,
``feb``, ..., ``dec``) and ``<yyyy>`` is the four-digit reference year.
The visible "Released" field is preferred over any meta tag because the
meta tag tracks the most recent page revision rather than the original
publication date — the same defensive choice WPI and GDP make.

URL coverage
------------
The modern ABS site hosts pages back to the **Dec 2019 reference month**.
Earlier monthly pages (sampled at Jul 2019, Oct 2019, Nov 2019) all return
HTTP 404. The scrape floor is ``2019-12-31`` (Dec 2019 reference month);
the source module ``abs_building_approvals`` applies a flat conservative
offset for any observation with a pre-floor reference month.

Source provenance
-----------------
Each row in the materialised CSV carries a ``source`` column with one of:

- ``abs_page`` — date came from the ABS release-page scrape (the normal
  path).
- ``archive_org`` — reserved fallback if a live ABS page 404s but a
  cached Wayback Machine snapshot exists. Not actively populated in v1;
  the scraper logs and skips on 404. Hook reserved for future use.
- ``inferred`` — date came from a non-scrape mechanism (e.g. an
  ``_OVERRIDES`` entry or a flat-offset fallback applied at lookup time).

Overrides
---------
``_OVERRIDES`` lets you patch individual reference-month-ends without
re-scraping (e.g. ABS corrects a release date or a transient scrape
failure needs a hand-verified value). Empty by default.

Usage
-----
- ``ba_publication_date(reference_month_end)`` — single lookup; returns
  ``None`` if the month is outside the scraped window.
- ``build_ba_release_calendar()`` — DataFrame keyed by
  ``reference_month_end``.
- ``uv run python -m rba.data.abs_ba_release_calendar`` — re-scrapes the
  ABS pages and rewrites ``data/external/abs_ba_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Path to the materialised scrape output. Loaded by
# build_ba_release_calendar() and rewritten by the __main__ block.
_CSV_PATH: Path = EXTERNAL_DATA_DIR / "abs_ba_release_dates.csv"

# Earliest scrape-able reference month on the post-redesign ABS site.
# Sampling confirmed dec-2019 resolves; nov-2019 / oct-2019 / jul-2019
# all 404. Set conservatively here.
_SCRAPE_FLOOR_YEAR = 2019
_SCRAPE_FLOOR_MONTH = 12

# Three-letter lower-case month slugs used by the ABS BA URL pattern.
_MONTH_SLUGS: dict[int, str] = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}

# Hand-verified overrides keyed by reference-month-end ``date``. Empty by
# default; populate as the scraper or ABS issues corrections.
_OVERRIDES: dict[date, date] = {}

# Visible-field regex matching the "Released" field on the modern ABS
# release pages. Same pattern as wpi_release_calendar and the modern
# gdp_release_calendar branch.
_RELEASED_RE = re.compile(
    r'<div class="field__label">Released</div>\s*'
    r'<div class="field__item">\s*(\d{1,2}/\d{1,2}/\d{4})'
)

# Polite scraper headers + a small inter-request delay so __main__ does
# not hammer the ABS site.
_HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "rba-cash-rate-prediction/0.1 (research; "
        "https://github.com/Tye-G/rba-cash-rate-prediction)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}
_SCRAPE_DELAY_SECONDS = 0.5


def month_end(year: int, month: int) -> date:
    """Return the last calendar day of ``(year, month)``."""
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, last_day)


def _build_url(year: int, month: int) -> str:
    slug = _MONTH_SLUGS[month]
    return (
        "https://www.abs.gov.au/statistics/industry/building-and-construction/"
        f"building-approvals-australia/{slug}-{year}"
    )


def scrape_release_date(year: int, month: int) -> date | None:
    """Scrape one ABS Building Approvals release page.

    Parameters
    ----------
    year
        Four-digit reference year.
    month
        Reference-month number (1-12).

    Returns
    -------
    datetime.date | None
        Original release date, or ``None`` if the page 404s or the
        ``Released`` field cannot be matched.
    """
    url = _build_url(year, month)
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 — public ABS URL
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.debug("ABS BA release page {} returned HTTP {}", url, e.code)
        return None

    match = _RELEASED_RE.search(body)
    if match is None:
        logger.warning("ABS BA release page {} fetched OK but no Released field matched", url)
        return None
    day, month_str, year_str = match.group(1).split("/")
    return date(int(year_str), int(month_str), int(day))


def ba_publication_date(reference_month_end: date) -> date | None:
    """Return the BA publication date for ``reference_month_end``.

    Honours ``_OVERRIDES`` first, then looks up the materialised CSV.
    Returns ``None`` if the month is outside the scraped window (the
    consuming source module applies a flat conservative offset in that
    case).
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]

    cal = build_ba_release_calendar()
    me_ts = pd.Timestamp(reference_month_end)
    hit = cal.loc[cal["reference_month_end"] == me_ts, "publication_date"]
    if hit.empty:
        return None
    return hit.iloc[0].date()


def build_ba_release_calendar() -> pd.DataFrame:
    """Load the materialised BA release calendar.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_month_end`` (datetime64[ns]) — last day of reference
          month.
        - ``publication_date`` (datetime64[ns]) — scraped original release
          date (with any ``_OVERRIDES`` substitutions applied).
        - ``source`` (object) — provenance tag, one of ``abs_page``,
          ``archive_org``, ``inferred``.

    Shapes
    ------
    Returns: (n_scraped, 3). Empty frame with the correct columns/dtypes
    if the CSV has not been materialised yet (run ``python -m
    rba.data.abs_ba_release_calendar`` to populate it).
    """
    if not _CSV_PATH.exists():
        logger.warning(
            "ABS BA release-date CSV missing at {}; returning empty calendar. "
            "Run `python -m rba.data.abs_ba_release_calendar` to populate.",
            _CSV_PATH,
        )
        return _empty_frame()

    df = pd.read_csv(_CSV_PATH, parse_dates=["reference_month_end", "publication_date"])
    df["reference_month_end"] = df["reference_month_end"].dt.as_unit("ns")
    df["publication_date"] = df["publication_date"].dt.as_unit("ns")
    if "source" not in df.columns:
        df["source"] = "abs_page"
    df["source"] = df["source"].astype(object)

    if _OVERRIDES:
        df = df.copy()
        for me, pub in _OVERRIDES.items():
            me_ts = pd.Timestamp(me)
            pub_ts = pd.Timestamp(pub)
            mask = df["reference_month_end"] == me_ts
            if mask.any():
                df.loc[mask, "publication_date"] = pub_ts
                df.loc[mask, "source"] = "inferred"
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "reference_month_end": [me_ts],
                                "publication_date": [pub_ts],
                                "source": ["inferred"],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        df = df.sort_values("reference_month_end").reset_index(drop=True)
        df["reference_month_end"] = df["reference_month_end"].astype("datetime64[ns]")
        df["publication_date"] = df["publication_date"].astype("datetime64[ns]")

    return df


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference_month_end": pd.Series([], dtype="datetime64[ns]"),
            "publication_date": pd.Series([], dtype="datetime64[ns]"),
            "source": pd.Series([], dtype=object),
        }
    )


def _iter_scrape_targets(through: date | None = None) -> list[tuple[int, int]]:
    """Yield ``(year, month)`` pairs from the scrape floor through ``through``.

    Defaults ``through`` to today; the scraper itself will skip months
    whose page 404s.
    """
    if through is None:
        through = date.today()
    pairs: list[tuple[int, int]] = []
    year, month = _SCRAPE_FLOOR_YEAR, _SCRAPE_FLOOR_MONTH
    while (year, month) <= (through.year, through.month):
        pairs.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return pairs


def _scrape_all(through: date | None = None) -> pd.DataFrame:
    """Scrape every (year, month) page in the window, return long frame.

    404s and parse failures are skipped with a debug log.
    """
    rows: list[dict[str, object]] = []
    for year, month in _iter_scrape_targets(through=through):
        pub = scrape_release_date(year, month)
        me = month_end(year, month)
        slug = _MONTH_SLUGS[month]
        if pub is None:
            logger.info("BA {}-{} -> no release date (skipping)", year, slug)
        else:
            rows.append(
                {
                    "reference_month_end": me,
                    "publication_date": pub,
                    "source": "abs_page",
                }
            )
            logger.info("BA {}-{} -> released {}", year, slug, pub.isoformat())
        time.sleep(_SCRAPE_DELAY_SECONDS)

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_frame()
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    df["source"] = df["source"].astype(object)
    return df.sort_values("reference_month_end").reset_index(drop=True)


if __name__ == "__main__":
    scraped = _scrape_all()
    _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    scraped.to_csv(_CSV_PATH, index=False)
    logger.info("Wrote ABS BA release calendar ({} rows) to {}", len(scraped), _CSV_PATH)
