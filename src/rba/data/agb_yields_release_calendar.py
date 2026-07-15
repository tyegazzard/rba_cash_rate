"""Australian Government Bond yields (RBA F2) release calendar — algorithmic.

F2 publishes daily ACGB benchmark yields. The RBA processes each trading
session's data overnight and refreshes ``f2-data.csv`` the **next
morning** (around 11:30am AEST), so the publication-date rule is:

    publication_date = observation_date + 1 Australian business day

where "business day" means Mon-Fri excluding the conservative
AU-national + NSW public-holiday set (RBA HQ is in Sydney).

Rule (in priority order)
------------------------
1. ``_OVERRIDES`` — hand-verified exceptions win over the algorithm.
2. **Next-business-day rule** — first weekday (Mon-Fri) strictly after
   ``observation_date`` that is not an Australian national or NSW state
   public holiday. The same holiday set as
   :mod:`rba.data.rba_i2_release_calendar` is used (New Year's Day with
   Mondayisation, Australia Day Mondayised, Good Friday / Easter Monday,
   Anzac Day, Queen's / Sovereign's Birthday 2nd Mon June, NSW Labour
   Day 1st Mon Oct, Christmas / Boxing Day Mondayised).

Verified samples
----------------
Pulled from Wayback Machine snapshots of ``f2-data.csv`` (the
``Publication date`` header is compared to the last data row carrying a
non-null 10y yield):

==========================  =====================  =====================  ==========
Snapshot timestamp          Last observation date  ``Publication date``   BDay lag
--------------------------  ---------------------  ---------------------  ----------
2018-01-21 (Sun)            Thu 18-Jan-2018        Fri 19-Jan-2018        +1
2018-03-25 (Sun)            Thu 22-Mar-2018        Fri 23-Mar-2018        +1
2018-04-25 (Wed)            Mon 23-Apr-2018        Tue 24-Apr-2018        +1
2018-05-25 (Fri)            Thu 24-May-2018        Fri 25-May-2018        +1
2018-08-09 (Thu)            Wed 08-Aug-2018        Thu 09-Aug-2018        +1
==========================  =====================  =====================  ==========

Coverage and validation
-----------------------
F2 (current vintage) begins 2013-05-20. The validation floor is
``1993-01-01`` to match every other source module — but in practice every
F2 observation we ingest sits well after the floor, so no NaT
publication-date region exists for this source.

Usage
-----
- ``agb_yields_publication_date(observation_date)`` — single lookup.
- ``build_agb_yields_release_calendar(start_date, end_date)`` — full
  DataFrame keyed by ``observation_date``.
- ``uv run python -m rba.data.agb_yields_release_calendar`` —
  materialises the calendar to
  ``data/external/agb_yields_release_dates.csv``.
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR
from rba.data.rba_i2_release_calendar import _au_public_holidays

# Hand-verified deviations from the "+1 business day" rule (empty by
# default — add entries as known late / early releases are observed).
# Keyed by observation date (the trade date); value is the actual F2
# release date as reported in the ``Publication date`` header.
_OVERRIDES: dict[date, date] = {}


def _next_business_day(d: date) -> date:
    """Return the first weekday strictly after ``d`` that is not an AU/NSW
    public holiday.
    """
    candidate = d + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in _au_public_holidays(candidate.year):
        candidate += timedelta(days=1)
    return candidate


def agb_yields_publication_date(observation_date: date) -> date:
    """Return the F2 publication date for the trading-day yield observed
    on ``observation_date``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup (hand-verified exceptions).
    2. Next-business-day rule: first weekday strictly after
       ``observation_date`` that is not an AU national or NSW state
       public holiday.

    Parameters
    ----------
    observation_date
        Trade date the yield observation applies to (e.g.
        ``date(2026, 5, 20)`` for the closing yield of Wed 20 May 2026).

    Returns
    -------
    datetime.date
        Publication date of the F2 refresh that first carried the
        observation.
    """
    if observation_date in _OVERRIDES:
        return _OVERRIDES[observation_date]
    return _next_business_day(observation_date)


def build_agb_yields_release_calendar(
    start_date: date | pd.Timestamp = date(2013, 5, 20),
    end_date: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every trading-day observation to its F2
    publication date.

    The calendar is built over every weekday (Mon-Fri) in the
    ``[start_date, end_date]`` window, ignoring holidays at the
    observation-date end — F2 will simply contain a null cell for any
    holiday or weekend, and a null observation never reaches
    ``_attach_publication_dates``. Including those rows here is harmless
    (the left-merge in the source module only joins on real observation
    dates) and keeps the calendar predictable.

    Parameters
    ----------
    start_date
        First trade date to include (default ``2013-05-20`` — the start
        of the current-vintage F2 coverage).
    end_date
        Last trade date to include. Defaults to ``today + 7 days`` so
        the calendar always carries a forward buffer for any same-day
        observation that hasn't been published yet at lookup time.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``observation_date`` (datetime64[ns]) — trade date.
        - ``publication_date`` (datetime64[ns]) — algorithmic F2 release
          date from ``agb_yields_publication_date`` (honours
          ``_OVERRIDES``).

    Shapes
    ------
    Returns: (n_weekdays_in_window, 2).
    """
    start = pd.Timestamp(start_date).date()
    if end_date is None:
        end_d = date.today() + timedelta(days=7)
    else:
        end_d = pd.Timestamp(end_date).date()

    rows: list[dict[str, date]] = []
    d = start
    while d <= end_d:
        if d.weekday() < 5:
            rows.append(
                {
                    "observation_date": d,
                    "publication_date": agb_yields_publication_date(d),
                }
            )
        d += timedelta(days=1)

    df = pd.DataFrame(rows)
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_agb_yields_release_calendar()
    dest = EXTERNAL_DATA_DIR / "agb_yields_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote AGB yields release calendar ({} rows) to {}", len(calendar_df), dest)
