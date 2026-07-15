"""NAB Monthly Business Survey release calendar — algorithmic.

The NAB Monthly Business Survey publishes mid-month following the
reference month. The brief's stated rule is "second Tuesday of the
month following the reference month" — empirically verified against
the May 2025 reference month (released Tue 2025-06-10 = 2nd Tuesday of
June 2025) and consistent with NAB's monthly cadence. This module
implements the algorithmic rule directly, matching the LFS calendar
pattern.

We do **not** scrape NAB pages: their site only retains a ~5-month
rolling window of release pages live (older URLs return HTTP 404), so a
scraped calendar would be unable to cover the inflation-targeting
training window and would be no more accurate than the algorithm. Where
the actual release deviates from the rule (e.g. NAB reschedules
around a public holiday), encode the exception in ``_OVERRIDES``.

Rule
----
1. ``_OVERRIDES`` — manual rescheduling exceptions win.
2. **Default** — 2nd Tuesday of the month following the reference
   month.

Manual overrides
----------------
``_OVERRIDES`` is keyed by reference-month-end ``date``, mapping to the
actual NAB release ``date``. Empty by default; populate as exceptions
are verified.

Pre-1989 / pre-1997 history
---------------------------
The NAB business survey began quarterly in 1989-Q1 and monthly from
1997-03. RBA H3 carries values back to 1965 (the RBA splices an older
business-conditions equivalent in the same column); the calendar still
emits a date for every month in the requested window, but pre-1997
``publication_date`` values are algorithmic estimates for a series
that didn't yet exist in its modern monthly form.

Usage
-----
- ``nab_publication_date(reference_month_end)`` — one month.
- ``build_nab_release_calendar(start_year, end_year)`` — full DataFrame
  keyed by ``reference_month_end``.
- ``uv run python -m rba.data.nab_release_calendar`` — materialises
  ``data/external/nab_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Known NAB rescheduling exceptions, keyed by reference-month-end.
# Empty until exceptions are verified.
_OVERRIDES: dict[date, date] = {}


def nth_tuesday_of_month(year: int, month: int, n: int) -> date:
    """Return the n-th Tuesday of ``(year, month)``.

    Parameters
    ----------
    year, month
        Calendar year and month.
    n
        Which Tuesday (1-indexed). Raises ``ValueError`` if the month
        has fewer than ``n`` Tuesdays.
    """
    weeks = calendar.monthcalendar(year, month)
    tuesdays = [w[calendar.TUESDAY] for w in weeks if w[calendar.TUESDAY] != 0]
    if not 1 <= n <= len(tuesdays):
        raise ValueError(
            f"n={n} out of range for {year}-{month:02d}: only "
            f"{len(tuesdays)} Tuesday(s) in this month."
        )
    return date(year, month, tuesdays[n - 1])


def nab_publication_date(reference_month_end: date) -> date:
    """Return the NAB publication date for ``reference_month_end``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup.
    2. Default rule: 2nd Tuesday of the month following the reference
       month.
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]
    if reference_month_end.month == 12:
        pub_year, pub_month = reference_month_end.year + 1, 1
    else:
        pub_year, pub_month = reference_month_end.year, reference_month_end.month + 1
    return nth_tuesday_of_month(pub_year, pub_month, 2)


def build_nab_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every reference-month-end to its NAB
    publication date.

    Parameters
    ----------
    start_year
        First calendar year to include. Default 1993.
    end_year
        Last calendar year to include. ``end_year + 1`` is also
        included so freshly released months always have a row to match
        against. Defaults to ``date.today().year``.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_month_end`` (datetime64[ns]) — last day of
          reference month.
        - ``publication_date`` (datetime64[ns]) — algorithmic release
          date from ``nab_publication_date`` (honours ``_OVERRIDES``).

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
                    "publication_date": nab_publication_date(me),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_nab_release_calendar()
    dest = EXTERNAL_DATA_DIR / "nab_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote NAB release calendar ({} rows) to {}", len(calendar_df), dest)
