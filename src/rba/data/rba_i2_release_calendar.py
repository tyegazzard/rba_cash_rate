"""RBA Index of Commodity Prices (I2) release calendar — algorithmic.

I2 is published monthly. The RBA states the index "is released on the first
business day of each month". Verification against Wayback-Machine snapshots
of ``i2-data.csv`` (13 distinct ``Publication date`` headers spanning
2015-04 → 2023-06) confirms the rule, including one Easter-period
shift (2018-04-03 Tue, skipping Easter Sunday 2018-04-01 and Easter
Monday 2018-04-02).

Rule (in priority order)
------------------------
1. ``_OVERRIDES`` — hand-verified exceptions win over the algorithm.
2. **First business day rule** — first weekday (Mon-Fri) on/after the 1st
   of the month following the reference month that is not an Australian
   national public holiday or NSW state public holiday (RBA HQ is in
   Sydney). Holidays considered: New Year's Day (with Mondayisation when
   it falls on a weekend), Australia Day (Mondayised), Good Friday /
   Easter Monday, Anzac Day, Queen's / Sovereign's Birthday (2nd Mon
   June), NSW Labour Day (1st Mon Oct), Christmas Day and Boxing Day
   (both Mondayised). The conservative superset is fine because the only
   holidays that ever land on the first 1-3 days of a month are New
   Year's Day (always) and Easter-period dates (rarely, when Easter
   Sunday falls on April 1 or earlier).

Verified samples
----------------
Pulled from Wayback Machine snapshots of ``i2-data.csv``:

==========================  ====================  ============================
Snapshot timestamp          ``Publication date``  Verified rule
--------------------------  --------------------  ----------------------------
2015-04-25                  01-Apr-2015           Wed Apr 1 2015 first business day
2015-07-30                  01-Jul-2015           Wed Jul 1 2015 first business day
2016-03-11                  01-Mar-2016           Tue Mar 1 2016 first business day
2016-11-14                  01-Nov-2016           Tue Nov 1 2016 first business day
2017-03-21                  01-Mar-2017           Wed Mar 1 2017 first business day
2017-06-28                  01-Jun-2017           Thu Jun 1 2017 first business day
2018-01-21                  02-Jan-2018           Tue Jan 2 (Mon Jan 1 = New Year's Day)
2018-04-25                  03-Apr-2018           Tue Apr 3 (Sun/Mon = Easter Sun/Mon)
2018-05-25                  01-May-2018           Tue May 1 2018 first business day
2019-03-18                  01-Mar-2019           Fri Mar 1 2019 first business day
2020-06-22                  01-Jun-2020           Mon Jun 1 2020 first business day
2021-03-16                  01-Mar-2021           Mon Mar 1 2021 first business day
2023-03-21                  01-Mar-2023           Wed Mar 1 2023 first business day
2023-06-08                  01-Jun-2023           Thu Jun 1 2023 first business day
==========================  ====================  ============================

The validation floor is 1993-01-31 (inflation-targeting era start) to
match other source modules. Pre-1993 observations retain ``NaT``.

Usage
-----
- ``rba_i2_publication_date(reference_month_end)`` — single lookup.
- ``build_rba_i2_release_calendar(start_year, end_year)`` — full DataFrame
  keyed by ``reference_month_end``.
- ``uv run python -m rba.data.rba_i2_release_calendar`` — materialises the
  calendar to ``data/external/rba_i2_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Hand-verified deviations from the "first business day of following month"
# rule (empty by default — add entries here as known late / early releases
# are observed). Keyed by reference-month-end; value is the actual release
# date as reported by the RBA.
_OVERRIDES: dict[date, date] = {}


def _easter_sunday(year: int) -> date:
    """Return the date of Easter Sunday in the given Gregorian year.

    Uses the anonymous Gregorian algorithm (Meeus / Jones / Butcher).
    Used here only to derive Good Friday and Easter Monday, which are
    the only Easter-period dates that can fall on the 1st-3rd of a
    publication month and thus shift the first business day.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _mondayise(d: date) -> date:
    """Shift a date forward to the next Monday if it falls on Sat/Sun.

    Used for the "in lieu" version of holidays that always sit on a
    fixed calendar date (New Year's Day, Australia Day, Christmas,
    Boxing Day, Anzac Day for NSW from 2010 onward).
    """
    if d.weekday() == 5:  # Saturday → Monday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday → Monday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence of ``weekday`` (Mon=0..Sun=6) in (year, month)."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _au_public_holidays(year: int) -> set[date]:
    """Return the set of national + NSW public-holiday dates for ``year``.

    Includes the small superset needed to decide whether the first
    business day of a month is shifted by a holiday: New Year's Day,
    Australia Day, Good Friday + Easter Monday, Anzac Day, Queen's /
    Sovereign's Birthday (2nd Mon June), NSW Labour Day (1st Mon Oct),
    Christmas Day and Boxing Day. Mondayisation is applied to the
    fixed-date holidays.
    """
    easter = _easter_sunday(year)
    good_friday = easter - timedelta(days=2)
    easter_monday = easter + timedelta(days=1)
    return {
        _mondayise(date(year, 1, 1)),     # New Year's Day
        _mondayise(date(year, 1, 26)),    # Australia Day
        good_friday,
        easter_monday,
        date(year, 4, 25),                # Anzac Day (national; some states Mondayise — superset OK)
        _nth_weekday(year, 6, 0, 2),      # Queen's / Sovereign's Birthday — 2nd Mon June
        _nth_weekday(year, 10, 0, 1),     # NSW Labour Day — 1st Mon Oct
        _mondayise(date(year, 12, 25)),   # Christmas Day
        _mondayise(date(year, 12, 26)),   # Boxing Day
    }


def first_business_day_of_month(year: int, month: int) -> date:
    """Return the first Mon-Fri in ``(year, month)`` that is not an AU/NSW
    public holiday.

    Parameters
    ----------
    year
        Four-digit calendar year.
    month
        Month number (1-12).

    Returns
    -------
    datetime.date
        Date of the first weekday in that month that is not in
        ``_au_public_holidays(year)``.
    """
    holidays = _au_public_holidays(year)
    d = date(year, month, 1)
    while d.weekday() >= 5 or d in holidays:
        d += timedelta(days=1)
    return d


def rba_i2_publication_date(reference_month_end: date) -> date:
    """Return the I2 publication date for the reference month ending
    ``reference_month_end``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup (hand-verified exceptions).
    2. First-business-day rule: first weekday of the following month that
       is not an AU/NSW public holiday. December reference months roll
       over into January of the next year.

    Parameters
    ----------
    reference_month_end
        Last calendar day of the reference month (e.g. ``date(2026, 4, 30)``
        for the April 2026 release).

    Returns
    -------
    datetime.date
        Publication date of the I2 release covering that reference month.
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]

    if reference_month_end.month == 12:
        pub_year, pub_month = reference_month_end.year + 1, 1
    else:
        pub_year, pub_month = reference_month_end.year, reference_month_end.month + 1
    return first_business_day_of_month(pub_year, pub_month)


def build_rba_i2_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every reference-month-end to its I2
    publication date.

    Parameters
    ----------
    start_year
        First calendar year to include (default 1993 — start of the
        inflation-targeting era and the source module's validation floor).
    end_year
        Last calendar year to include. ``end_year + 1`` is also included so
        forward-dated months have a row for matching. Defaults to the
        current calendar year.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_month_end`` (datetime64[ns]) — last day of reference month.
        - ``publication_date`` (datetime64[ns]) — algorithmic I2 release date
          from ``rba_i2_publication_date`` (honours ``_OVERRIDES``).

    Shapes
    ------
    Returns: (12 * (end_year - start_year + 2), 2).
    """
    if end_year is None:
        end_year = date.today().year

    rows: list[dict[str, date]] = []
    for year in range(start_year, end_year + 2):
        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)
            me = date(year, month, last_day)
            rows.append(
                {
                    "reference_month_end": me,
                    "publication_date": rba_i2_publication_date(me),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_rba_i2_release_calendar()
    dest = EXTERNAL_DATA_DIR / "rba_i2_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info(
        "Wrote RBA I2 release calendar ({} rows) to {}", len(calendar_df), dest
    )
