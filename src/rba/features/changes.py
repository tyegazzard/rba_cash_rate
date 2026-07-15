"""Meeting-space change / growth-rate features for the master frame.

Third ``§5 Feature engineering → Numerical features`` builder (after
:mod:`rba.features.lags` and :mod:`rba.features.rolling`). Emits three families
of change features from the meeting-indexed master frame:

- ``<sid>_diff_<h>`` — absolute Δ over ``h`` meetings (``master[<sid>].diff(h)``).
- ``<sid>_pct_<h>`` — % Δ over ``h`` meetings (``(x - x[-h]) / x[-h]``).
- ``<sid>_yoy_pct`` — **calendar** year-over-year % change.

Meeting-space vs calendar YoY
-----------------------------
Δ and %Δ are **meeting-space** (a horizon is a fixed number of *decisions*),
consistent with :mod:`rba.features.lags`. YoY is deliberately **calendar-anchored**
instead: for each meeting ``t`` it compares the current value against the as-of
value from the most recent meeting on/before ``t − window_days`` (default 365).
A fixed meeting count would silently stop meaning "one year" after the Feb-2024
cadence change (11/yr → 8/yr) — the exact Common-pitfall trap — so YoY is pinned
to wall-clock days and stays a true year-over-year across the break.

Leakage safety (Invariant #1)
-----------------------------
All three compare the current row against a *strictly earlier* row, and every
master level is already point-in-time (``align.as_of_levels`` guarantees
``publication_date < meeting_date``), so no future information can enter. The
first ``h`` meetings of each diff/pct horizon — and any meeting with no ≥1-year-
old predecessor — are ``NaN``, never zero (Common pitfall). Division by a zero
base yields ``NaN`` rather than ``±inf``.

Configuration mirrors the sibling builders: columns / horizons come from
``features.yaml`` ``changes:`` (reconciled to real align series_ids), absent
columns are warn-skipped, companion / ``regime_*`` / metadata columns are never
transformed, and the pure function returns ``meeting_date`` + only the new columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import load_features_config

# Default Δ / %Δ horizons in MEETINGS (shared by abs_diff + pct_change).
# Overridable via ``features.yaml`` ``changes.horizons_meetings``.
DEFAULT_CHANGE_HORIZONS_MEETINGS: tuple[int, ...] = (1, 3, 6, 12)

# Default YoY look-back in calendar days. Overridable via ``changes.yoy.window_days``.
DEFAULT_YOY_DAYS: int = 365

_MEETING_KEY = "meeting_date"

# Companion suffixes / prefixes marking a column as NOT a source level — never
# transformed (kept local per the codebase's "no centralised constant" style).
_AGE_SUFFIX = "_age_days"
_MISSING_SUFFIX = "_is_missing"
_REGIME_PREFIX = "regime_"
_META_COLUMNS: frozenset[str] = frozenset(
    {
        "meeting_date",
        "effective_date",
        "rate_change_bps",
        "new_rate_pct",
        "prior_rate_pct",
        "gap_days_since_last_meeting",
        "statement_url",
        "minutes_url",
    }
)


def _is_level_column(column: str) -> bool:
    """Whether ``column`` is a genuine source level eligible for change features."""
    if column in _META_COLUMNS:
        return False
    if column.startswith(_REGIME_PREFIX):
        return False
    return not column.endswith((_AGE_SUFFIX, _MISSING_SUFFIX))


# -----------------------------------------------------------------------------
# Config accessors (mirror rba.features.lags / rba.features.rolling).
# -----------------------------------------------------------------------------
def config_change_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Δ / %Δ horizons (in meetings) from config, or the module default."""
    section = config.get("changes", {}) or {}
    horizons = section.get("horizons_meetings")
    if not horizons:
        return DEFAULT_CHANGE_HORIZONS_MEETINGS
    return tuple(int(h) for h in horizons)


def _subsection_columns(config: Mapping[str, Any], key: str) -> list[str]:
    section = config.get("changes", {}) or {}
    sub = section.get(key, {}) or {}
    return [str(c) for c in (sub.get("columns", []) or [])]


def config_abs_diff_columns(config: Mapping[str, Any]) -> list[str]:
    """Columns for absolute Δ, from ``changes.abs_diff.columns`` (empty if absent)."""
    return _subsection_columns(config, "abs_diff")


def config_pct_change_columns(config: Mapping[str, Any]) -> list[str]:
    """Columns for %Δ, from ``changes.pct_change.columns`` (empty if absent)."""
    return _subsection_columns(config, "pct_change")


def config_yoy_columns(config: Mapping[str, Any]) -> list[str]:
    """Columns for calendar %YoY, from ``changes.yoy.columns`` (empty if absent)."""
    return _subsection_columns(config, "yoy")


def config_yoy_days(config: Mapping[str, Any]) -> int:
    """YoY look-back in calendar days, from ``changes.yoy.window_days`` or default."""
    section = config.get("changes", {}) or {}
    sub = section.get("yoy", {}) or {}
    return int(sub.get("window_days", DEFAULT_YOY_DAYS))


def _resolve_columns(master: pd.DataFrame, requested: Sequence[str], kind: str) -> list[str]:
    """Filter ``requested`` to columns present in ``master`` and level-eligible."""
    present = set(master.columns)
    resolved: list[str] = []
    for col in requested:
        if col not in present:
            logger.warning(
                "Changes[{}]: configured column {!r} absent from frame; skipping.", kind, col
            )
            continue
        if not _is_level_column(col):
            logger.warning("Changes[{}]: column {!r} is not a level series; skipping.", kind, col)
            continue
        resolved.append(col)
    return resolved


# -----------------------------------------------------------------------------
# Transforms.
# -----------------------------------------------------------------------------
def _pct_change(series: pd.Series, horizon: int) -> pd.Series:
    """% change over ``horizon`` meetings; a zero base yields NaN (not ±inf)."""
    prev = series.shift(horizon)
    return (series - prev) / prev.where(prev != 0)


def _yoy_pct(ordered: pd.DataFrame, column: str, window_days: int) -> np.ndarray:
    """Calendar YoY % change: current vs the as-of value ~``window_days`` earlier.

    For each meeting, find the most recent meeting on/before ``meeting_date −
    window_days`` (backward as-of) and return ``(now − then) / then``. Meetings
    with no ≥``window_days``-old predecessor, or a zero base, get ``NaN``.
    """
    dates = ordered[_MEETING_KEY].to_numpy()  # datetime64[ns], sorted ascending
    values = ordered[column].to_numpy(dtype="float64")
    targets = dates - np.timedelta64(int(window_days), "D")
    # Rightmost index with dates[idx] <= target (backward, exact match allowed).
    idx = np.searchsorted(dates, targets, side="right") - 1
    prev = np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        yoy = (values - prev) / np.where(prev == 0, np.nan, prev)
    return np.asarray(yoy, dtype=np.float64)


def build_changes(
    master: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    horizons: Sequence[int] | None = None,
    abs_diff_columns: Sequence[str] | None = None,
    pct_change_columns: Sequence[str] | None = None,
    yoy_columns: Sequence[str] | None = None,
    yoy_days: int | None = None,
) -> pd.DataFrame:
    """Build change / growth-rate features from the master frame.

    Emits, keyed on ``meeting_date``:

    - ``<sid>_diff_<h>`` for each ``abs_diff`` column and horizon ``h`` (meetings);
    - ``<sid>_pct_<h>`` for each ``pct_change`` column and horizon ``h``;
    - ``<sid>_yoy_pct`` for each ``yoy`` column (calendar year-over-year %).

    Parameters
    ----------
    master
        The meeting-indexed master frame (``rba.data.align.build_master`` output).
    config
        Pre-loaded ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` only when a default is needed.
    horizons
        Δ / %Δ horizons in meetings. Defaults to ``changes.horizons_meetings``
        (else :data:`DEFAULT_CHANGE_HORIZONS_MEETINGS`).
    abs_diff_columns, pct_change_columns, yoy_columns
        Per-transform series_ids. Default to the reconciled ``features.yaml``
        ``changes:`` sub-sections. Absent / non-level columns are warn-skipped.
    yoy_days
        YoY calendar look-back in days. Defaults to ``changes.yoy.window_days``
        (else :data:`DEFAULT_YOY_DAYS`).

    Returns
    -------
    pandas.DataFrame
        Keyed on ``meeting_date`` (one row per meeting, ascending), carrying only
        the new change columns — diff block, then pct block, then yoy block; each
        block column-major then horizon. Just ``meeting_date`` if nothing resolves.

    Shapes
    ------
    Returns: (n_meetings, 1 + (n_diff + n_pct) * len(horizons) + n_yoy).
    """
    need_config = config is None and (
        horizons is None
        or abs_diff_columns is None
        or pct_change_columns is None
        or yoy_columns is None
        or yoy_days is None
    )
    if need_config:
        config = load_features_config()

    if horizons is None:
        horizons = config_change_horizons(config)  # type: ignore[arg-type]
    if abs_diff_columns is None:
        abs_diff_columns = config_abs_diff_columns(config)  # type: ignore[arg-type]
    if pct_change_columns is None:
        pct_change_columns = config_pct_change_columns(config)  # type: ignore[arg-type]
    if yoy_columns is None:
        yoy_columns = config_yoy_columns(config)  # type: ignore[arg-type]
    if yoy_days is None:
        yoy_days = config_yoy_days(config)  # type: ignore[arg-type]

    ordered = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    diff_cols = _resolve_columns(ordered, abs_diff_columns, "abs_diff")
    pct_cols = _resolve_columns(ordered, pct_change_columns, "pct_change")
    yoy_cols = _resolve_columns(ordered, yoy_columns, "yoy")

    out = ordered[[_MEETING_KEY]].copy()
    for col in diff_cols:
        for h in horizons:
            out[f"{col}_diff_{h}"] = ordered[col].diff(h)
    for col in pct_cols:
        for h in horizons:
            out[f"{col}_pct_{h}"] = _pct_change(ordered[col], h)
    for col in yoy_cols:
        out[f"{col}_yoy_pct"] = _yoy_pct(ordered, col, yoy_days)

    n_new = len(out.columns) - 1
    logger.info(
        "Built {} change columns ({} diff × {} + {} pct × {} + {} yoy) over {} meetings.",
        n_new,
        len(diff_cols),
        len(horizons),
        len(pct_cols),
        len(horizons),
        len(yoy_cols),
        len(out),
    )
    return out
