"""RBA Financial Aggregates (D1 / D2) release calendar — algorithmic.

D1 (Growth in Selected Financial Aggregates) and D2 (Lending and Credit
Aggregates) publish to a single monthly schedule documented by the RBA as
"Last business day of each month, 11:30 am" — covering data for the
*preceding* month.

Verification window
-------------------
The rule was verified against 14 ABS-archived release pages spanning
2001-07 → 2026-03 (the earliest reference month with a scrapable release
page is 2001; ``/statistics/frequency/fin-agg/<YYYY>/`` returns HTTP 404
for all pre-2001 years). The algorithmic rule "last weekday of the month
following the reference month" matched every sampled release with the
exceptions listed in ``_OVERRIDES`` below.

Pre-2001 release pages are not accessible online but the rule itself is
RBA-documented and stable; the inflation-targeting era starts 1993 and
this is the validation floor used by the source module (pre-1993
observations retain ``NaT``).

Rules applied (in priority order)
---------------------------------
1. ``_OVERRIDES`` — hand-verified exceptions win over the algorithm.
2. **Last weekday rule** — publication is the last Mon-Fri of the month
   following the reference month.

Known overrides
---------------
- ``2013-02-28 → 2013-03-28`` and ``2024-02-29 → 2024-03-28`` are Good
  Friday collisions: in both years 31 March was a Sunday, 30 March a
  Saturday, and the algorithmic last weekday was 29 March (Good Friday),
  so the RBA released a day earlier. (Note: in 2002 the RBA released on
  Good Friday itself rather than shifting; the algorithmic rule already
  picks that date, so no override is needed for 2002. The override list
  reflects what actually happened, not a holiday calculation.)
- ``2025-11-30 → 2025-12-19`` is an early release during the Direct to
  APRA (D2A) reporting-system decommissioning announced by the RBA in
  late 2025. Add further overrides here as that transition resolves and
  more delays / early releases are observed.

Usage
-----
- ``rba_d_publication_date(reference_month_end)`` — single lookup.
- ``build_rba_d_release_calendar(start_year, end_year)`` — full DataFrame
  keyed by ``reference_month_end``.
- ``uv run python -m rba.data.rba_d_release_calendar`` — materialises the
  calendar to ``data/external/rba_d_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Hand-verified deviations from the "last weekday of following month" rule.
# Keyed by reference-month-end; value is the actual ABS-reported release date.
# See module docstring for the why behind each entry.
_OVERRIDES: dict[date, date] = {
    date(2013, 2, 28): date(2013, 3, 28),
    date(2024, 2, 29): date(2024, 3, 28),
    date(2025, 11, 30): date(2025, 12, 19),
}


def last_weekday_of_month(year: int, month: int) -> date:
    """Return the last Monday-Friday of ``(year, month)``.

    Parameters
    ----------
    year
        Four-digit calendar year.
    month
        Month number (1-12).

    Returns
    -------
    datetime.date
        Date of the last weekday (Mon-Fri) in that month. If the calendar
        last day is a Saturday, returns the preceding Friday; if Sunday,
        the preceding Friday; otherwise the last day itself.
    """
    _, last_day = calendar.monthrange(year, month)
    d = date(year, month, last_day)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def rba_d_publication_date(reference_month_end: date) -> date:
    """Return the D1/D2 publication date for the reference month ending
    ``reference_month_end``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup (hand-verified exceptions).
    2. Last-weekday rule: last Mon-Fri of the following month, with year
       rollover for December reference months.

    Parameters
    ----------
    reference_month_end
        Last calendar day of the reference month (e.g. ``date(2024, 2, 29)``
        for the February 2024 release covering Feb 2024 data).

    Returns
    -------
    datetime.date
        Publication date of the D1/D2 release covering that reference month.
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]

    if reference_month_end.month == 12:
        pub_year, pub_month = reference_month_end.year + 1, 1
    else:
        pub_year, pub_month = reference_month_end.year, reference_month_end.month + 1
    return last_weekday_of_month(pub_year, pub_month)


def build_rba_d_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every reference-month-end to its D1/D2
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
        - ``publication_date`` (datetime64[ns]) — algorithmic D1/D2 release
          date from ``rba_d_publication_date`` (honours ``_OVERRIDES``).

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
                    "publication_date": rba_d_publication_date(me),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_rba_d_release_calendar()
    dest = EXTERNAL_DATA_DIR / "rba_d_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote RBA D1/D2 release calendar ({} rows) to {}", len(calendar_df), dest)
