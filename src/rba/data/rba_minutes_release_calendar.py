"""RBA Board minutes release calendar — algorithmic.

The RBA publishes the minutes of each monetary-policy Board meeting on the
**second Tuesday strictly after the meeting**. For the regular cadence (the
Board meets on a Tuesday — historically the first Tuesday of the month, from
2024 the second day of a Monday-Tuesday meeting) this is exactly
``meeting_date + 14 days``. For a non-Tuesday meeting (the out-of-cycle COVID
meeting on Thursday 2020-03-19) the rule still resolves to the second Tuesday
on the calendar after the meeting (2020-03-31, +12 days).

Why algorithmic, not scraped
----------------------------
Minutes pages carry no ``itemprop="datePublished"`` field (unlike RBA media
releases). The only per-page date marker is ``<meta name="dc.date">``, which
is unreliable: of 195 regular-regime pages probed (2008-02 → 2026-03), ~19
carry CMS-artifact values (``dc.date`` stamped at the meeting date, or even
*before* it). The robust signal is therefore the algorithmic rule, verified
against the clean ``dc.date`` majority. :mod:`rba.data.sources.rba_minutes`
scrapes ``dc.date`` only as a warn-only soft cross-check, never as the
canonical value.

Rule (in priority order)
------------------------
1. ``_OVERRIDES`` — hand-verified exceptions win over the algorithm.
2. **Second-Tuesday-after rule** — the second Tuesday on the calendar strictly
   after the meeting date.

Verified samples
----------------
175 of 195 regular-regime ``dc.date`` headers equal ``meeting_date + 14 days``
on a Tuesday exactly; the COVID out-of-cycle meeting (Thu 2020-03-19) resolves
to Tue 2020-03-31 (+12 days), also ``dc.date``-confirmed. The remaining ~19
divergent ``dc.date`` values are CMS artifacts (lag 0 or negative) that the
algorithm correctly overrides. No public-holiday shift of a publication
Tuesday was observed in the probed window, so no holiday logic is encoded —
add an ``_OVERRIDES`` entry if one is ever observed.

Validation floor
----------------
The companion source module floors minutes at 2008-02-05 (start of the regular
publication regime). The 2006-10 → 2007-11 minutes were backfilled and only
published ~2007-12-05, so the second-Tuesday-after rule does not describe them;
they are excluded by the source module's floor rather than handled here.

Usage
-----
- ``rba_minutes_publication_date(meeting_date)`` — single lookup.
- ``build_rba_minutes_release_calendar(meeting_dates)`` — full DataFrame keyed
  by ``meeting_date``.
- ``uv run python -m rba.data.rba_minutes_release_calendar`` — materialises the
  calendar (driven by F11 meeting dates) to
  ``data/external/rba_minutes_release_dates.csv``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# date.weekday(): Mon=0 .. Sun=6.
_TUESDAY = 1

# Hand-verified deviations from the second-Tuesday-after rule, keyed by
# meeting_date (the Board meeting's announcement / final day); value is the
# actual minutes release date. Empty by default — the rule matched every
# regular-regime meeting probed (2008-02 → 2026-03), including the COVID
# out-of-cycle Thursday meeting 2020-03-19 (→ 2020-03-31). Populate as
# deviations are observed.
_OVERRIDES: dict[date, date] = {}


def second_tuesday_after(meeting_date: date) -> date:
    """Return the second Tuesday strictly after ``meeting_date``.

    For a Tuesday meeting (the regular RBA cadence) this is
    ``meeting_date + 14 days``. For a non-Tuesday meeting (e.g. the out-of-cycle
    COVID meeting on Thursday 2020-03-19) it is the second Tuesday on the
    calendar after the meeting (2020-03-31, +12 days).

    Parameters
    ----------
    meeting_date
        The Board meeting's announcement / final day.

    Returns
    -------
    datetime.date
        Date of the second Tuesday strictly after ``meeting_date``.
    """
    days_to_first = (_TUESDAY - meeting_date.weekday()) % 7
    if days_to_first == 0:
        days_to_first = 7
    return meeting_date + timedelta(days=days_to_first + 7)


def rba_minutes_publication_date(meeting_date: date) -> date:
    """Return the minutes release date for a Board meeting on ``meeting_date``.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup (hand-verified exceptions).
    2. Second-Tuesday-after rule (:func:`second_tuesday_after`).

    Parameters
    ----------
    meeting_date
        The Board meeting's announcement / final day.

    Returns
    -------
    datetime.date
        Publication date of the minutes for that meeting.
    """
    if meeting_date in _OVERRIDES:
        return _OVERRIDES[meeting_date]
    return second_tuesday_after(meeting_date)


def build_rba_minutes_release_calendar(meeting_dates: Iterable[date]) -> pd.DataFrame:
    """Build a DataFrame mapping each meeting date to its minutes release date.

    Parameters
    ----------
    meeting_dates
        Iterable of Board meeting dates (announcement / final days). Unlike the
        monthly RBA calendars, the minutes schedule is keyed to the irregular
        Board meeting frame, so the meeting dates are supplied by the caller
        (the source module pulls them from F11) rather than enumerated here.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``meeting_date`` (datetime64[ns]) — Board meeting announcement day.
        - ``publication_date`` (datetime64[ns]) — algorithmic minutes release
          date from :func:`rba_minutes_publication_date` (honours
          ``_OVERRIDES``).

        Rows sorted ascending by ``meeting_date``.

    Shapes
    ------
    Returns: (n_meetings, 2).
    """
    rows = [
        {"meeting_date": md, "publication_date": rba_minutes_publication_date(md)}
        for md in meeting_dates
    ]
    df = pd.DataFrame(rows, columns=["meeting_date", "publication_date"])
    if not df.empty:
        df["meeting_date"] = pd.to_datetime(df["meeting_date"]).dt.as_unit("ns")
        df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df.sort_values("meeting_date").reset_index(drop=True)


if __name__ == "__main__":
    from rba.data.sources import rba_f11

    f11 = rba_f11.fetch(force_download=False)
    floor = pd.Timestamp("2008-02-05")
    mask = (f11["publication_date"] >= floor) & f11["minutes_url"].notna()
    meeting_dates = [
        d.date() for d in pd.to_datetime(f11.loc[mask, "publication_date"]).sort_values()
    ]
    calendar_df = build_rba_minutes_release_calendar(meeting_dates)
    dest = EXTERNAL_DATA_DIR / "rba_minutes_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info("Wrote RBA minutes release calendar ({} rows) to {}", len(calendar_df), dest)
