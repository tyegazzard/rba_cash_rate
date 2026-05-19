"""ABS CPI release calendar — algorithmic publication-date computation.

The ABS publishes the quarterly Consumer Price Index (Cat. 6401.0) on the
**last Wednesday of the month following** the reference quarter. This module
computes those dates so that the ``abs_cpi`` source can stamp accurate
publication dates onto every observation, avoiding the look-ahead leakage risk
of a fixed-day-offset heuristic.

Worked example: Q1 (Jan-Mar) ends 31 March. Publication month is April; the
last Wednesday of April 2024 is the 24th. A naive ``observation_date + 35d``
heuristic would round forward to 5 May, falsely marking the value as unseen
at any RBA meeting between 24 Apr and 5 May.

Public-holiday exceptions
-------------------------
On the rare occasion the algorithmic date conflicts with a public holiday or
the ABS reschedules a release, record the override in ``_OVERRIDES`` below —
keyed by quarter-end ``date``, mapping to the actual ABS release ``date``.
``_OVERRIDES`` is empty by default; add entries as historical exceptions are
confirmed against the ABS release calendar.

Usage
-----
- ``cpi_publication_date(quarter_end)`` — one quarter.
- ``build_cpi_release_calendar(start_year, end_year)`` — full DataFrame keyed
  by ``reference_quarter_end``.
- ``uv run python -m rba.data.cpi_release_calendar`` — materialises the
  calendar to ``data/external/cpi_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Known ABS rescheduling / public-holiday exceptions, keyed by quarter-end.
# Format: {date(year, month, day): date(year, month, day)}
# Empty until exceptions are verified against the ABS release-calendar archive.
_OVERRIDES: dict[date, date] = {}


def last_wednesday_of_month(year: int, month: int) -> date:
    """Return the last Wednesday of ``(year, month)``.

    Parameters
    ----------
    year
        Four-digit calendar year.
    month
        Month number (1-12).

    Returns
    -------
    datetime.date
        The date of the final Wednesday in that month.
    """
    weeks = calendar.monthcalendar(year, month)
    for week in reversed(weeks):
        day = week[calendar.WEDNESDAY]
        if day != 0:
            return date(year, month, day)
    raise RuntimeError(
        f"Unreachable: calendar.monthcalendar({year}, {month}) contained no "
        "Wednesday — every Gregorian month spans ≥4 weeks."
    )


def cpi_publication_date(quarter_end: date) -> date:
    """Return the ABS CPI publication date for the quarter ending ``quarter_end``.

    Algorithm: last Wednesday of the calendar month *following* the quarter
    end. Honours ``_OVERRIDES`` for known exceptions.

    Parameters
    ----------
    quarter_end
        Calendar end-of-quarter date (e.g. ``date(2024, 3, 31)`` for Q1 2024).

    Returns
    -------
    datetime.date
        Publication date of the CPI release covering that quarter.
    """
    if quarter_end.month == 12:
        next_year, next_month = quarter_end.year + 1, 1
    else:
        next_year, next_month = quarter_end.year, quarter_end.month + 1
    algorithmic = last_wednesday_of_month(next_year, next_month)
    return _OVERRIDES.get(quarter_end, algorithmic)


def build_cpi_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every quarter-end to its CPI publication date.

    Parameters
    ----------
    start_year
        First calendar year to include (default 1993 — start of inflation
        targeting).
    end_year
        Last calendar year to include. ``end_year + 1`` is also included so
        forward-dated quarters have a row for matching. Defaults to
        ``date.today().year``.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_quarter_end`` (datetime64[ns]) — quarter end date.
        - ``publication_date`` (datetime64[ns]) — CPI release date computed by
          ``cpi_publication_date`` (honours ``_OVERRIDES``).

    Shapes
    ------
    Returns: (4 * (end_year - start_year + 2), 2).
    """
    if end_year is None:
        end_year = date.today().year

    rows: list[dict[str, date]] = []
    # +2 in range to include end_year + 1, giving a forward buffer so freshly
    # released quarters always have a calendar row to merge against.
    for year in range(start_year, end_year + 2):
        for month, day in [(3, 31), (6, 30), (9, 30), (12, 31)]:
            qe = date(year, month, day)
            rows.append(
                {
                    "reference_quarter_end": qe,
                    "publication_date": cpi_publication_date(qe),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_quarter_end"] = pd.to_datetime(df["reference_quarter_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_cpi_release_calendar()
    dest = EXTERNAL_DATA_DIR / "cpi_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote CPI release calendar ({} rows) to {}", len(calendar_df), dest)
