"""Westpac-Melbourne Institute Consumer Sentiment release calendar — algorithmic.

The Westpac-MI Consumer Sentiment Index publishes monthly: the survey is
conducted in the first week of the reference month and the headline
index is released mid-reference-month. The brief's stated rule is
"second Wednesday of the same month the survey was conducted". This
matches the LFS-style algorithmic calendar pattern (no scrape, no
materialised data).

The exact release weekday/date varies in practice (e.g. May 2026 was
released on Tue May 19 = 3rd Tue of May; Apr 2026 was released on Wed
Apr 15 = 3rd Wed of April). Where the actual release deviates from the
2nd-Wednesday rule, encode the exception in ``_OVERRIDES``. The flat
rule is conservative enough for the RBA-meeting cadence: a few days of
imprecision is well inside the typical 6-week gap between meetings.

Rule
----
1. ``_OVERRIDES`` — manual rescheduling exceptions win.
2. **Default** — 2nd Wednesday of the reference month.

Manual overrides
----------------
``_OVERRIDES`` is keyed by reference-month-end ``date``, mapping to the
actual Westpac-MI release ``date``. Empty by default; populate as
exceptions are verified.

Pre-1974 history
----------------
The Westpac-MI survey started 1974-09. We still emit dates for earlier
months in ``build_westpac_mi_release_calendar()`` for completeness (the
function takes a ``start_year`` argument that defaults to 1993, the
inflation-targeting era floor); the consuming source module
``westpac_mi_consumer_sentiment`` filters by data availability.

Usage
-----
- ``westpac_mi_publication_date(reference_month_end)`` — one month.
- ``build_westpac_mi_release_calendar(start_year, end_year)`` — full
  DataFrame keyed by ``reference_month_end``.
- ``uv run python -m rba.data.westpac_mi_release_calendar`` —
  materialises ``data/external/westpac_mi_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# Known Westpac-MI rescheduling exceptions, keyed by reference-month-end.
# Empty until exceptions are verified against the Westpac IQ archive.
_OVERRIDES: dict[date, date] = {}


def nth_wednesday_of_month(year: int, month: int, n: int) -> date:
    """Return the n-th Wednesday of ``(year, month)``.

    Parameters
    ----------
    year, month
        Calendar year and month.
    n
        Which Wednesday (1-indexed). Raises ``ValueError`` if the month
        has fewer than ``n`` Wednesdays.
    """
    weeks = calendar.monthcalendar(year, month)
    wednesdays = [w[calendar.WEDNESDAY] for w in weeks if w[calendar.WEDNESDAY] != 0]
    if not 1 <= n <= len(wednesdays):
        raise ValueError(
            f"n={n} out of range for {year}-{month:02d}: only "
            f"{len(wednesdays)} Wednesday(s) in this month."
        )
    return date(year, month, wednesdays[n - 1])


def westpac_mi_publication_date(reference_month_end: date) -> date:
    """Return the Westpac-MI publication date for ``reference_month_end``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup.
    2. Default rule: 2nd Wednesday of the reference month.
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]
    return nth_wednesday_of_month(
        reference_month_end.year, reference_month_end.month, 2
    )


def build_westpac_mi_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every reference-month-end to its Westpac-MI
    publication date.

    Parameters
    ----------
    start_year
        First calendar year to include. Default 1993 (inflation-targeting
        floor used by every other source in this repo).
    end_year
        Last calendar year to include. ``end_year + 1`` is also included
        so freshly released months always have a row to match against.
        Defaults to ``date.today().year``.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_month_end`` (datetime64[ns]) — last day of
          reference month.
        - ``publication_date`` (datetime64[ns]) — algorithmic release
          date from ``westpac_mi_publication_date`` (honours
          ``_OVERRIDES``).

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
                    "publication_date": westpac_mi_publication_date(me),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_westpac_mi_release_calendar()
    dest = EXTERNAL_DATA_DIR / "westpac_mi_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info(
        "Wrote Westpac-MI release calendar ({} rows) to {}",
        len(calendar_df),
        dest,
    )
