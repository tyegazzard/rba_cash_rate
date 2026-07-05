"""Consolidated §5 contract tests for the numerical feature builders.

The per-builder files (``test_lags`` / ``test_rolling`` / ``test_changes`` /
``test_surprises``) cover each builder in depth. This file enforces the three
checklist invariants — **shape**, **NaN handling at series start**, and
**leakage** — *uniformly* across all four, so the contract is asserted in one
place and a future builder added to :data:`BUILDERS` is automatically held to it.

Every numerical builder is a pure ``build_*(master) -> DataFrame`` that:
- returns a frame keyed on ``meeting_date`` (first column), one row per meeting,
  sorted ascending, same length as the input;
- emits only *new* float feature columns (no source / companion / regime /
  metadata column echoed back);
- never lets a later meeting's value influence an earlier meeting's feature
  (point-in-time / Invariant #1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from rba.features.changes import build_changes
from rba.features.lags import build_lags
from rba.features.rolling import build_rolling
from rba.features.surprises import build_surprises

# A value no legitimate observation would take — if it surfaces at an earlier
# meeting, a future row leaked.
_FUTURE_SENTINEL = 9_999_999.0

# Level columns the builders operate on, and every non-level column that must be
# left untouched (companions / regime / metadata / the consensus side).
_LEVEL_COLUMNS = ("cpi", "fx")
_INJECT_COLUMNS = ("cpi", "fx", "cpi_consensus")


def _master(n: int = 24) -> pd.DataFrame:
    """A meeting-indexed master-like frame with levels, companions, regime, meta."""
    idx = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "meeting_date": pd.date_range("2021-01-01", periods=n, freq="MS"),
            # Meeting metadata / target columns (must never be transformed).
            "rate_change_bps": np.where(idx % 3 == 0, 25.0, 0.0),
            "gap_days_since_last_meeting": pd.array([30] * n, dtype="Int64"),
            # Level series the builders act on.
            "cpi": 100.0 + 1.5 * idx,
            "fx": 0.70 + 0.001 * idx,
            # A consensus column so surprises can activate.
            "cpi_consensus": 100.0 + 1.4 * idx,
            # Companions + a regime dummy (must never be transformed).
            "cpi_age_days": pd.array([20] * n, dtype="Int64"),
            "cpi_is_missing": [0] * n,
            "regime_covid": np.where((idx >= 2) & (idx <= 6), 1, 0),
        }
    )


# Each entry: (id, builder-call). Explicit kwargs so the contract does not depend
# on features.yaml. Add a new builder here and it inherits every contract test.
BUILDERS: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("lags", lambda m: build_lags(m, columns=list(_LEVEL_COLUMNS), horizons=[1, 3, 12])),
    (
        "rolling",
        lambda m: build_rolling(
            m,
            columns=list(_LEVEL_COLUMNS),
            windows=[3, 6],
            stats=["mean", "std", "zscore", "min", "max", "ewma"],
        ),
    ),
    (
        "changes",
        lambda m: build_changes(
            m,
            abs_diff_columns=["cpi"],
            pct_change_columns=["fx"],
            yoy_columns=["cpi"],
            horizons=[1, 3],
        ),
    ),
    ("surprises", lambda m: build_surprises(m, pairs={"cpi": "cpi_consensus"})),
]

# Builders that produce a genuine leading NaN (a horizon/window has no
# predecessor at the series start). Surprises is excluded: a fully-present pair
# is defined from row 0 (asserted separately to be finite, not zero-filled).
NAN_AT_START_BUILDERS = [entry for entry in BUILDERS if entry[0] != "surprises"]


@pytest.fixture
def master() -> pd.DataFrame:
    return _master()


# -----------------------------------------------------------------------------
# Shape / dtype / keying / output-isolation — uniform across builders.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("name,builder", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_output_is_keyed_sorted_and_same_length(
    name: str, builder: Callable[[pd.DataFrame], pd.DataFrame], master: pd.DataFrame
) -> None:
    out = builder(master)
    assert out.columns[0] == "meeting_date"
    assert len(out) == len(master)
    assert out["meeting_date"].is_monotonic_increasing
    assert out["meeting_date"].tolist() == sorted(master["meeting_date"].tolist())


@pytest.mark.parametrize("name,builder", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_feature_columns_are_float(
    name: str, builder: Callable[[pd.DataFrame], pd.DataFrame], master: pd.DataFrame
) -> None:
    out = builder(master)
    feature_cols = [c for c in out.columns if c != "meeting_date"]
    assert feature_cols, f"{name} produced no feature columns"
    for col in feature_cols:
        assert pd.api.types.is_float_dtype(out[col]), f"{name}:{col} is not float"


@pytest.mark.parametrize("name,builder", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_output_isolation_no_source_or_companion_echoed(
    name: str, builder: Callable[[pd.DataFrame], pd.DataFrame], master: pd.DataFrame
) -> None:
    """Only new columns come back — no source / companion / regime / meta echoed."""
    out = builder(master)
    echoed = (set(out.columns) - {"meeting_date"}) & set(master.columns)
    assert echoed == set(), f"{name} echoed input columns: {sorted(echoed)}"
    # And no transform ever targeted a non-level column.
    for banned in ("cpi_age_days", "cpi_is_missing", "regime_covid",
                   "rate_change_bps", "gap_days_since_last_meeting"):
        assert not any(c.startswith(f"{banned}_") or c == banned for c in out.columns), (
            f"{name} transformed non-level column {banned}"
        )
    assert len(out.columns) == len(set(out.columns)), f"{name} has duplicate columns"


# -----------------------------------------------------------------------------
# NaN handling at series start — never zero-filled.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,builder", NAN_AT_START_BUILDERS, ids=[b[0] for b in NAN_AT_START_BUILDERS]
)
def test_leading_nan_not_zero_at_series_start(
    name: str, builder: Callable[[pd.DataFrame], pd.DataFrame], master: pd.DataFrame
) -> None:
    out = builder(master)
    feature_cols = [c for c in out.columns if c != "meeting_date"]
    # At least one feature must be NaN (not 0) at the very first meeting, and no
    # feature may be silently zero-filled where it should be NaN.
    first_row = out.loc[0, feature_cols]
    assert first_row.isna().any(), f"{name} has no leading NaN at row 0"
    # Every feature that is NaN at the start must become populated later (i.e. the
    # NaN is a genuine start-of-series gap, not a wholly-broken column).
    for col in feature_cols:
        if pd.isna(out.loc[0, col]):
            assert out[col].notna().any(), f"{name}:{col} is all-NaN"


def test_surprises_defined_from_first_row_not_zero_filled(master: pd.DataFrame) -> None:
    """A fully-present surprise pair is defined at row 0 (no start NaN, not zero)."""
    out = build_surprises(master, pairs={"cpi": "cpi_consensus"})
    s = out["cpi_surprise"]
    assert pd.notna(s.iloc[0])
    # cpi (100) − cpi_consensus (100) == 0 at row 0 by construction; ensure the
    # column is genuinely computed (varies), not a zero column.
    assert s.abs().sum() > 0


# -----------------------------------------------------------------------------
# Leakage — a future meeting's value never changes an earlier feature.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("name,builder", BUILDERS, ids=[b[0] for b in BUILDERS])
def test_future_injection_never_changes_earlier_rows(
    name: str, builder: Callable[[pd.DataFrame], pd.DataFrame], master: pd.DataFrame
) -> None:
    """Inject a sentinel into the LAST meeting; every earlier output row is identical."""
    injected = master.copy()
    for col in _INJECT_COLUMNS:
        injected.loc[injected.index[-1], col] = _FUTURE_SENTINEL

    out_base = builder(master)
    out_inj = builder(injected)

    # Every row except the last must be byte-identical across all feature columns.
    pd.testing.assert_frame_equal(out_base.iloc[:-1], out_inj.iloc[:-1])
    # And the sentinel must never surface at an earlier meeting.
    earlier = out_inj.iloc[:-1].drop(columns="meeting_date")
    assert not (earlier == _FUTURE_SENTINEL).to_numpy().any(), (
        f"{name} leaked the future sentinel into an earlier meeting"
    )
