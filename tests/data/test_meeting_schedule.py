"""Tests for ``rba.data.meeting_schedule`` — the forward meeting-date source.

No network — the schedule is a checked-in tuple, so every case is deterministic.
"""

from __future__ import annotations

from datetime import date

import pytest

from rba.data import meeting_schedule as ms


# -----------------------------------------------------------------------------
# Accessors.
# -----------------------------------------------------------------------------
def test_scheduled_dates_sorted_and_unique() -> None:
    dates = ms.scheduled_meeting_dates()
    assert list(dates) == sorted(dates)
    assert len(dates) == len(set(dates))
    assert date(2026, 8, 11) in dates  # the next meeting after the 2026-06-16 decision


def test_next_meeting_date_is_strictly_after() -> None:
    # A meeting *on* the cutoff is already held (strict >).
    assert ms.next_meeting_date(date(2026, 6, 16)) == date(2026, 8, 11)
    assert ms.next_meeting_date(date(2026, 7, 1)) == date(2026, 8, 11)
    assert ms.next_meeting_date(date(2026, 8, 11)) == date(2026, 9, 29)


def test_next_meeting_date_exhausted_raises() -> None:
    with pytest.raises(LookupError):
        ms.next_meeting_date(date(2099, 1, 1))


def test_upcoming_meeting_dates_slice_and_limit() -> None:
    assert ms.upcoming_meeting_dates(date(2026, 7, 1), limit=2) == [
        date(2026, 8, 11),
        date(2026, 9, 29),
    ]
    assert ms.upcoming_meeting_dates(date(2099, 1, 1)) == []


def test_overrides_take_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_OVERRIDES", {date(2026, 8, 11): date(2026, 8, 18)})
    dates = ms.scheduled_meeting_dates()
    assert date(2026, 8, 18) in dates
    assert date(2026, 8, 11) not in dates


# -----------------------------------------------------------------------------
# Validation / reconciliation against F11.
# -----------------------------------------------------------------------------
def test_validate_schedule_structural_ok() -> None:
    ms.validate_schedule()  # no F11 → structural checks only, must not raise


def test_validate_schedule_reconciles_past_with_f11() -> None:
    # Every past scheduled date is present in F11 → passes.
    f11 = [date(2026, 2, 3), date(2026, 3, 17), date(2026, 5, 5), date(2026, 6, 16)]
    ms.validate_schedule(f11)


def test_validate_schedule_raises_on_f11_mismatch() -> None:
    # 2026-05-05 is a past scheduled date (<= max f11 2026-06-16) absent from F11 → raise.
    f11 = [date(2026, 2, 3), date(2026, 3, 17), date(2026, 6, 16)]
    with pytest.raises(ValueError, match="absent from F11"):
        ms.validate_schedule(f11)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def test_parser_defaults() -> None:
    args = ms.build_parser().parse_args([])
    assert args.after is None
    assert args.no_validate is False


def test_cli_reports_next_and_materialises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(ms, "EXTERNAL_DATA_DIR", tmp_path)
    assert ms.main(["--after", "2026-07-01", "--no-validate"]) == 0
    assert (tmp_path / "meeting_schedule.csv").exists()


def test_cli_returns_1_when_schedule_exhausted(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(ms, "EXTERNAL_DATA_DIR", tmp_path)
    assert ms.main(["--after", "2099-01-01", "--no-validate"]) == 1
