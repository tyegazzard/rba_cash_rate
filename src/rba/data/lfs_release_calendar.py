"""ABS Labour Force (LFS) release calendar — algorithmic publication-date computation.

The ABS publishes monthly Labour Force (Cat. 6202.0) on a Thursday in the
month following the reference month. The exact Thursday is determined by
three rules applied in priority order:

1. ``_OVERRIDES`` — manual rescheduling / public-holiday exceptions win.
2. **December exception** — if the reference month is December, publication is
   the **4th Thursday of the following January** (longer collection window
   over Christmas/New Year; 46 days vs the usual 39). Applies in both eras.
3. **Era rule** —

   - Reference month **1993 – Dec 2015**: 2nd Thursday of the following month.
   - Reference month **Jan 2016 – present**: 3rd Thursday of the following
     month.

Era change details
------------------
The schedule changed from the January 2016 reference month following the 2014
McCarthy Review recommendation (announced in the September 2015 detailed
release). The change pushed the headline LFS release back by one week to give
the ABS more processing time.

Era boundary is keyed on the *reference* month, not the publication month:
- Reference month Dec 2015 → December exception → 4th Thursday of Jan 2016.
- Reference month Jan 2016 → new-era rule → 3rd Thursday of Feb 2016.

Pre-1993 dates are outside the verification window and not emitted by default.

Manual overrides
----------------
``_OVERRIDES`` is keyed by reference-month-end ``date``, mapping to the actual
ABS release ``date``. Empty by default; populate as historical / future
exceptions are verified against the ABS release-calendar archive.

Not implemented: temporary one-week delay
-----------------------------------------
A temporary one-week delay was introduced for the **Apr–Aug 2026 reference
months** due to LFS survey modernisation. This sits outside the training data
window and is deliberately NOT implemented here. If it becomes load-bearing,
encode it as ``_OVERRIDES`` entries (one per reference month).

Usage
-----
- ``lfs_publication_date(reference_month_end)`` — one month.
- ``build_lfs_release_calendar(start_year, end_year)`` — full DataFrame keyed
  by ``reference_month_end``.
- ``uv run python -m rba.data.lfs_release_calendar`` — materialises the
  calendar to ``data/external/lfs_release_dates.csv``.
"""

from __future__ import annotations

import calendar
from datetime import date

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# From the January 2016 reference month onward, LFS publishes on the 3rd
# Thursday of the following month (vs the 2nd Thursday previously).
_THIRD_THURSDAY_ERA_START = date(2016, 1, 1)

# Known ABS rescheduling / public-holiday exceptions, keyed by reference-month-end.
# Format: {date(year, month, last_day_of_month): date(year, month, day)}
# Empty until exceptions are verified against the ABS release-calendar archive.
_OVERRIDES: dict[date, date] = {}


def nth_thursday_of_month(year: int, month: int, n: int) -> date:
    """Return the n-th Thursday of ``(year, month)``.

    Parameters
    ----------
    year
        Four-digit calendar year.
    month
        Month number (1-12).
    n
        Which Thursday to return, 1-indexed (1 = first, 2 = second, ...).

    Returns
    -------
    datetime.date
        Date of the n-th Thursday in that month.

    Raises
    ------
    ValueError
        If ``n`` is outside the range of Thursdays in the month (most months
        have 4 Thursdays, some have 5; passing ``n=5`` on a 4-Thursday month
        raises rather than silently returning the last Thursday).
    """
    weeks = calendar.monthcalendar(year, month)
    thursdays = [w[calendar.THURSDAY] for w in weeks if w[calendar.THURSDAY] != 0]
    if not 1 <= n <= len(thursdays):
        raise ValueError(
            f"n={n} out of range for {year}-{month:02d}: only "
            f"{len(thursdays)} Thursday(s) in this month."
        )
    return date(year, month, thursdays[n - 1])


def lfs_publication_date(reference_month_end: date) -> date:
    """Return the ABS LFS publication date for the reference month ending
    ``reference_month_end``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup (manual rescheduling exceptions).
    2. December exception (always 4th Thursday of the following January).
    3. Era rule (2nd Thursday pre-2016 reference months; 3rd Thursday from
       Jan 2016 onward).

    Parameters
    ----------
    reference_month_end
        Last calendar day of the reference month (e.g. ``date(2024, 2, 29)``
        for the February 2024 release).

    Returns
    -------
    datetime.date
        Publication date of the LFS release covering that reference month.
    """
    if reference_month_end in _OVERRIDES:
        return _OVERRIDES[reference_month_end]

    if reference_month_end.month == 12:
        pub_year, pub_month = reference_month_end.year + 1, 1
        nth = 4  # December exception applies in both eras.
    else:
        pub_year, pub_month = reference_month_end.year, reference_month_end.month + 1
        if reference_month_end >= _THIRD_THURSDAY_ERA_START:
            nth = 3
        else:
            nth = 2

    return nth_thursday_of_month(pub_year, pub_month, nth)


def build_lfs_release_calendar(
    start_year: int = 1993,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Build a DataFrame mapping every reference-month-end to its LFS
    publication date.

    Parameters
    ----------
    start_year
        First calendar year to include (default 1993 — start of the
        verification window).
    end_year
        Last calendar year to include. ``end_year + 1`` is also included so
        forward-dated months have a row for matching. Defaults to
        ``date.today().year``.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``reference_month_end`` (datetime64[ns]) — last day of reference month.
        - ``publication_date`` (datetime64[ns]) — algorithmic LFS release date
          from ``lfs_publication_date`` (honours ``_OVERRIDES``).

    Shapes
    ------
    Returns: (12 * (end_year - start_year + 2), 2).
    """
    if end_year is None:
        end_year = date.today().year

    rows: list[dict[str, date]] = []
    # +2 in range to include end_year + 1, giving a forward buffer so freshly
    # released months always have a calendar row to merge against.
    for year in range(start_year, end_year + 2):
        for month in range(1, 13):
            _, last_day = calendar.monthrange(year, month)
            me = date(year, month, last_day)
            rows.append(
                {
                    "reference_month_end": me,
                    "publication_date": lfs_publication_date(me),
                }
            )

    df = pd.DataFrame(rows)
    df["reference_month_end"] = pd.to_datetime(df["reference_month_end"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df


if __name__ == "__main__":
    calendar_df = build_lfs_release_calendar()
    dest = EXTERNAL_DATA_DIR / "lfs_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote LFS release calendar ({} rows) to {}", len(calendar_df), dest)
