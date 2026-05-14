"""Tests for ``rba.data.validate_decisions``.

The network check (``check_urls=True``) is not exercised in unit tests; the
end-to-end smoke against the live RBA page is the integration check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data.validate_decisions import cross_check_media_releases


def _decision_row(
    *,
    observation_date: str,
    rate_change_bps: int,
    statement_url: object,
    new_rate_pct: float = 4.00,
) -> dict[str, object]:
    return {
        "observation_date": pd.Timestamp(observation_date),
        "rate_change_bps": rate_change_bps,
        "new_rate_pct": new_rate_pct,
        "statement_url": statement_url,
    }


def _frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_returns_empty_when_all_changes_have_statement_urls() -> None:
    decisions = _frame(
        _decision_row(
            observation_date="2026-05-06",
            rate_change_bps=25,
            statement_url="/media-releases/2026/mr-26-12.html",
        ),
        _decision_row(
            observation_date="2025-12-10",
            rate_change_bps=0,
            statement_url="/media-releases/2025/mr-25-33.html",
        ),
    )
    result = cross_check_media_releases(decisions)
    assert result.empty


def test_holds_without_statement_url_do_not_fail() -> None:
    decisions = _frame(
        _decision_row(
            observation_date="1995-06-07",
            rate_change_bps=0,
            statement_url=None,
        ),
    )
    result = cross_check_media_releases(decisions)
    assert result.empty


def test_flags_changes_missing_statement_url() -> None:
    decisions = _frame(
        _decision_row(
            observation_date="2026-05-06",
            rate_change_bps=25,
            statement_url=None,
        ),
    )
    result = cross_check_media_releases(decisions)
    assert len(result) == 1
    assert result["failure_reason"].iloc[0] == "rate change has no statement_url"


def test_failure_frame_preserves_input_columns() -> None:
    decisions = _frame(
        _decision_row(
            observation_date="2026-05-06",
            rate_change_bps=25,
            statement_url=None,
        ),
    )
    result = cross_check_media_releases(decisions)
    assert "observation_date" in result.columns
    assert "rate_change_bps" in result.columns
    assert "new_rate_pct" in result.columns


def test_raises_when_input_missing_required_columns() -> None:
    bad = pd.DataFrame({"observation_date": [pd.Timestamp("2026-05-06")]})
    with pytest.raises(ValueError, match="missing required columns"):
        cross_check_media_releases(bad)
