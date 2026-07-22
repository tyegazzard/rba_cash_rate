"""RBA Monetary Policy Board meeting schedule — the forward-looking meeting dates.

The prediction pipeline (:mod:`rba.data.align`) builds its meeting frame purely
from *historical* ``rba_f11`` decisions, so it has no knowledge of the **next**,
not-yet-held Board meeting. This module supplies that missing fact: the RBA's
published schedule of upcoming meeting **announcement dates** — the second,
decision day of each Monday–Tuesday meeting, the same convention align uses for
``meeting_date ≡ rba_f11.publication_date``.

Why a checked-in list, not a scrape
-----------------------------------
The RBA publishes its meeting schedule roughly once a year as a media release
(e.g. the 2026 dates in ``mr-25-02``), and the schedule page sits behind the same
WAF that 403s the other RBA fetchers, so there is no reliable machine source. The
dates are few (8/year since the 2024 cadence change), stable, and change rarely,
so they are hand-verified and checked in here — mirroring the ``_OVERRIDES``
convention of the ``*_release_calendar`` modules. :func:`validate_schedule`
reconciles the *past* portion of this list against the F11 ground truth so a typo
in a historical date can't survive.

Keeping current
---------------
When the RBA publishes a new year's dates, append them to
:data:`_SCHEDULED_MEETINGS` (verified against
``rba.gov.au/schedules-events/board-meeting-schedules.html``) and extend the
tests. Until the following year is published, :func:`next_meeting_date` raises
:class:`LookupError` once the known schedule is exhausted — an honest "unknown"
rather than a guess.

Usage
-----
- ``next_meeting_date(after=None)`` — the next announcement date strictly after
  ``after`` (default today).
- ``upcoming_meeting_dates(after=None, limit=...)`` — the forward slice.
- ``scheduled_meeting_dates()`` — the full sorted, override-applied tuple.
- ``uv run python -m rba.data.meeting_schedule`` — validate against cached F11,
  print the next meeting, and materialise ``data/external/meeting_schedule.csv``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from datetime import date, datetime

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR

# date.weekday(): Mon=0 .. Sun=6. The RBA announces its decision at 2:30pm on the
# second (Tuesday) day of each Mon–Tue meeting, so scheduled announcement dates
# are Tuesdays (the historical out-of-cycle COVID meeting was a Thursday, but no
# future scheduled meeting is expected off-Tuesday).
_TUESDAY = 1

# Hand-verified RBA Monetary Policy Board meeting ANNOUNCEMENT dates — the second
# / decision day of each Mon–Tue meeting (== F11 ``publication_date``). Source:
# RBA media release "2026 Monetary Policy Board Meeting Dates" (``mr-25-02``) and
# the rba.gov.au board-meeting schedule. The first four 2026 entries are already
# DECIDED (present in F11); they are retained so :func:`validate_schedule` can
# reconcile the list against the F11 ground truth.
#
# VERIFY + APPEND each year: the 2027 schedule is published late in 2026 and is
# not yet included here.
_SCHEDULED_MEETINGS: tuple[date, ...] = (
    date(2026, 2, 3),
    date(2026, 3, 17),
    date(2026, 5, 5),
    date(2026, 6, 16),
    date(2026, 8, 11),
    date(2026, 9, 29),
    date(2026, 11, 3),
    date(2026, 12, 8),
)

# Hand-verified corrections to :data:`_SCHEDULED_MEETINGS`, keyed old→new date.
# Empty by default — populate if a scheduled date is later moved by the RBA (e.g.
# a public-holiday shift) rather than editing the tuple in place, so the change is
# auditable. Mirrors the ``*_release_calendar`` override convention.
_OVERRIDES: dict[date, date] = {}


def scheduled_meeting_dates() -> tuple[date, ...]:
    """Return the full schedule as a sorted, unique, override-applied tuple.

    The single accessor every other function reads from: applies
    :data:`_OVERRIDES`, de-duplicates, and sorts ascending.

    Returns
    -------
    tuple[datetime.date, ...]
        Every known RBA announcement date, ascending.
    """
    resolved = {_OVERRIDES.get(d, d) for d in _SCHEDULED_MEETINGS}
    return tuple(sorted(resolved))


def next_meeting_date(after: date | None = None) -> date:
    """Return the next scheduled announcement date strictly after ``after``.

    Parameters
    ----------
    after
        Exclusive lower bound. Defaults to today (:meth:`datetime.date.today`) —
        so the bare call returns the next upcoming meeting. A meeting *on*
        ``after`` is treated as already held (strict ``>``).

    Returns
    -------
    datetime.date
        The earliest scheduled date ``> after``.

    Raises
    ------
    LookupError
        If the known schedule has no date after ``after`` (the checked-in
        schedule needs extending — see the module docstring).
    """
    cutoff = after if after is not None else date.today()
    for meeting in scheduled_meeting_dates():
        if meeting > cutoff:
            return meeting
    raise LookupError(
        f"No scheduled RBA meeting after {cutoff.isoformat()}; the checked-in "
        f"schedule ends {scheduled_meeting_dates()[-1].isoformat()}. Append the "
        "next year's dates to rba.data.meeting_schedule._SCHEDULED_MEETINGS."
    )


def upcoming_meeting_dates(after: date | None = None, *, limit: int | None = None) -> list[date]:
    """Return every scheduled announcement date strictly after ``after``.

    Parameters
    ----------
    after
        Exclusive lower bound. Defaults to today.
    limit
        Maximum number of dates to return (the earliest ``limit``). ``None``
        returns all remaining.

    Returns
    -------
    list[datetime.date]
        The forward slice of the schedule, ascending (possibly empty).
    """
    cutoff = after if after is not None else date.today()
    upcoming = [m for m in scheduled_meeting_dates() if m > cutoff]
    return upcoming[:limit] if limit is not None else upcoming


def validate_schedule(f11_meeting_dates: Iterable[date] | None = None) -> None:
    """Assert the checked-in schedule is internally consistent and matches F11.

    Structural checks (always): the resolved schedule is strictly ascending and
    unique, and every date falls on the expected announcement weekday (Tuesday) —
    a non-Tuesday is *warned*, not raised, so an ``_OVERRIDES`` correction can
    introduce one if the RBA ever schedules off-cycle.

    Reconciliation (when ``f11_meeting_dates`` is given): every scheduled date on
    or before the last F11 meeting must equal an actual F11 ``meeting_date``. This
    is the guard against a typo in the historical portion of the list — the past
    is ground truth, so the schedule must agree with it exactly.

    Parameters
    ----------
    f11_meeting_dates
        The actual (historical) F11 meeting dates, e.g. from
        :func:`rba.data.align.load_meeting_frame`. ``None`` skips reconciliation
        and runs only the structural checks.

    Raises
    ------
    ValueError
        If the schedule is not strictly ascending / unique, or if a past
        scheduled date has no matching F11 meeting.
    """
    schedule = scheduled_meeting_dates()
    raw = [_OVERRIDES.get(d, d) for d in _SCHEDULED_MEETINGS]
    if len(raw) != len(set(raw)):
        raise ValueError("meeting_schedule: duplicate dates in _SCHEDULED_MEETINGS.")
    if list(schedule) != sorted(schedule):
        raise ValueError("meeting_schedule: schedule is not sorted ascending.")

    off_tuesday = [m for m in schedule if m.weekday() != _TUESDAY]
    if off_tuesday:
        logger.warning(
            "meeting_schedule: {} date(s) not on a Tuesday (announcement day): {}.",
            len(off_tuesday),
            [d.isoformat() for d in off_tuesday],
        )

    if f11_meeting_dates is None:
        logger.debug("validate_schedule: structural checks passed ({} dates).", len(schedule))
        return

    f11_set = {d for d in f11_meeting_dates}
    if not f11_set:
        logger.warning("validate_schedule: empty F11 date set; skipping reconciliation.")
        return
    last_actual = max(f11_set)
    past = [m for m in schedule if m <= last_actual]
    unmatched = [m for m in past if m not in f11_set]
    if unmatched:
        raise ValueError(
            f"meeting_schedule: {len(unmatched)} past scheduled date(s) absent from F11: "
            f"{[d.isoformat() for d in unmatched]}. The historical schedule must match the "
            "F11 ground truth exactly — fix _SCHEDULED_MEETINGS or add an _OVERRIDES entry."
        )
    logger.info(
        "validate_schedule: {} past date(s) reconciled against F11; {} upcoming.",
        len(past),
        len(schedule) - len(past),
    )


def build_meeting_schedule_frame(after: date | None = None) -> pd.DataFrame:
    """Return the full schedule as a DataFrame, flagging past vs upcoming.

    Parameters
    ----------
    after
        The boundary that splits "decided" from "upcoming" for the ``is_upcoming``
        flag. Defaults to today.

    Returns
    -------
    pandas.DataFrame
        Columns ``meeting_date`` (datetime64[ns]) and ``is_upcoming`` (bool),
        sorted ascending.
    """
    cutoff = after if after is not None else date.today()
    schedule = scheduled_meeting_dates()
    frame = pd.DataFrame(
        {
            "meeting_date": pd.to_datetime(list(schedule)).as_unit("ns"),
            "is_upcoming": [m > cutoff for m in schedule],
        }
    )
    return frame.sort_values("meeting_date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.data.meeting_schedule`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.data.meeting_schedule",
        description="Validate the RBA meeting schedule and report the next meeting.",
    )
    parser.add_argument(
        "--after",
        metavar="YYYY-MM-DD",
        default=None,
        help="Report the next meeting strictly after this date (default: today).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip reconciliation of the past schedule against cached F11.",
    )
    return parser


def _parse_after(raw: str | None) -> date | None:
    """Parse a ``--after`` string to a date, or ``None``."""
    if raw is None:
        return None
    return datetime.strptime(raw, "%Y-%m-%d").date()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    after = _parse_after(args.after)

    if not args.no_validate:
        try:
            from rba.data.align import load_meeting_frame

            f11 = load_meeting_frame()
            validate_schedule(pd.to_datetime(f11["meeting_date"]).dt.date.tolist())
        except FileNotFoundError:
            logger.warning("No cached F11 snapshot; running structural validation only.")
            validate_schedule()
        except ValueError:
            logger.error("Schedule validation FAILED against F11 — fix _SCHEDULED_MEETINGS.")
            raise

    try:
        upcoming = next_meeting_date(after)
        logger.success("Next scheduled RBA meeting after {}: {}", after or date.today(), upcoming)
    except LookupError as exc:
        logger.error(str(exc))
        return 1

    frame = build_meeting_schedule_frame(after)
    dest = EXTERNAL_DATA_DIR / "meeting_schedule.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    logger.info("Wrote meeting schedule ({} rows) to {}", len(frame), dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
