"""Meeting-space rolling-statistic features for the master frame.

Second ``§5 Feature engineering → Numerical features`` builder (after
:mod:`rba.features.lags`). For each configured series ``<sid>``, window ``w``
(in meetings) and statistic ``stat``, emits ``<sid>_roll_<w>_<stat>`` — a
past-only rolling statistic over the last ``w`` meetings.

Why meeting-space + why inclusive windows?
------------------------------------------
The master frame (:mod:`rba.data.align`) is **meeting-indexed** and carries only
the as-of-latest value per series, so a rolling window spans the last ``w``
*meetings* (``master[<sid>].rolling(w)``), not ``w`` native months/days. The
window ends **inclusively** at the current meeting, which is leakage-safe
(Invariant #1 / #3): ``align.as_of_levels`` already guarantees each row's level
was published *strictly before* that meeting, so a window ending at the current
row uses only pre-meeting information. ``pandas`` rolling is past-only by default
— **never** ``center=True`` (Invariant #3). The first ``w-1`` meetings of each
window are ``NaN``, never zero (Common pitfall).

Statistics
----------
``mean`` / ``std`` / ``min`` / ``max`` / ``zscore`` / ``ewma``:

- ``std`` uses ``ddof=1`` (sample std).
- ``zscore`` = ``(x - rolling_mean) / rolling_std`` — a **rolling** z-score, never
  a whole-series one (Invariant #3). Where the window's std is 0 (a flat window),
  the z-score is ``NaN`` rather than ``±inf``.
- ``ewma`` is the exponentially-weighted mean with ``span=w`` (past-only by
  construction); unlike the windowed stats it yields a value from the first
  meeting (EWMA of one point), so it has no leading-``NaN`` window.

Configuration mirrors :mod:`rba.features.lags`: columns / windows / stats come
from ``features.yaml`` ``rolling:`` (reconciled to real align series_ids), absent
columns are warn-skipped, and companion / ``regime_*`` / metadata columns are
never rolled. Pure function returning ``meeting_date`` + only the new columns.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import load_features_config

# Default rolling windows in MEETINGS (not native months/days). 3/6/12 meetings
# ≈ recent / half-year / full-year under the pre-2024 cadence. Overridable via
# ``features.yaml`` ``rolling.windows_meetings``.
DEFAULT_ROLLING_WINDOWS_MEETINGS: tuple[int, ...] = (3, 6, 12)

# Statistics emitted per (series, window). Overridable via ``rolling.stats``.
DEFAULT_ROLLING_STATS: tuple[str, ...] = ("mean", "std", "zscore", "min", "max", "ewma")

_MEETING_KEY = "meeting_date"

# Companion suffixes / prefixes that mark a column as NOT a source level — these
# are never rolled (kept local per the codebase's "no centralised constant"
# convention; mirrors align + lags).
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
    """Whether ``column`` is a genuine source level eligible for rolling stats."""
    if column in _META_COLUMNS:
        return False
    if column.startswith(_REGIME_PREFIX):
        return False
    return not column.endswith((_AGE_SUFFIX, _MISSING_SUFFIX))


# -----------------------------------------------------------------------------
# Per-statistic builders. Each takes a level series + window and returns a series
# aligned to the input index, computed past-only.
# -----------------------------------------------------------------------------
def _roll_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _roll_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()


def _roll_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()


def _roll_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()


def _roll_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    # Flat window (std == 0) → undefined z-score → NaN, not ±inf.
    return (series - mean) / std.where(std != 0)


def _roll_ewma(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=True).mean()


_STAT_BUILDERS: dict[str, Callable[[pd.Series, int], pd.Series]] = {
    "mean": _roll_mean,
    "std": _roll_std,
    "min": _roll_min,
    "max": _roll_max,
    "zscore": _roll_zscore,
    "ewma": _roll_ewma,
}


# -----------------------------------------------------------------------------
# Config accessors (mirror rba.features.lags).
# -----------------------------------------------------------------------------
def config_rolling_columns(config: Mapping[str, Any]) -> list[str]:
    """Union of configured ``monthly_columns`` + ``daily_columns``, in order.

    De-duplicated preserving first appearance (monthly before daily). Empty if
    the ``rolling:`` section is absent.
    """
    section = config.get("rolling", {}) or {}
    ordered: list[str] = []
    seen: set[str] = set()
    for key in ("monthly_columns", "daily_columns"):
        for col in section.get(key, []) or []:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def config_rolling_windows(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Rolling windows (in meetings) from config, or the module default."""
    section = config.get("rolling", {}) or {}
    windows = section.get("windows_meetings")
    if not windows:
        return DEFAULT_ROLLING_WINDOWS_MEETINGS
    return tuple(int(w) for w in windows)


def config_rolling_stats(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Statistic names from config, or the module default."""
    section = config.get("rolling", {}) or {}
    stats = section.get("stats")
    if not stats:
        return DEFAULT_ROLLING_STATS
    return tuple(str(s) for s in stats)


def _resolve_columns(master: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    """Filter ``requested`` to columns present in ``master`` and level-eligible."""
    present = set(master.columns)
    resolved: list[str] = []
    for col in requested:
        if col not in present:
            logger.warning("Rolling: configured column {!r} absent from frame; skipping.", col)
            continue
        if not _is_level_column(col):
            logger.warning("Rolling: column {!r} is not a level series; skipping.", col)
            continue
        resolved.append(col)
    return resolved


def _validate_stats(stats: Sequence[str]) -> list[str]:
    """Return ``stats`` unchanged, raising on any unknown statistic name."""
    unknown = [s for s in stats if s not in _STAT_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown rolling statistic(s): {unknown}. Valid stats: {sorted(_STAT_BUILDERS)}."
        )
    return list(stats)


def build_rolling(
    master: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    windows: Sequence[int] | None = None,
    stats: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build meeting-space rolling-statistic features from the master frame.

    For every resolved column ``<sid>``, window ``w`` and statistic ``stat``,
    emits ``<sid>_roll_<w>_<stat>`` computed past-only over the last ``w``
    meetings (window inclusive of the current meeting; see module docstring).

    Parameters
    ----------
    master
        The meeting-indexed master frame (``rba.data.align.build_master`` output)
        — must carry ``meeting_date`` plus the level columns to roll.
    columns
        Series_ids to roll. Defaults to the reconciled ``features.yaml``
        ``rolling:`` union. Absent / non-level columns are warn-skipped.
    windows
        Rolling windows in meetings. Defaults to ``rolling.windows_meetings``
        (else :data:`DEFAULT_ROLLING_WINDOWS_MEETINGS`).
    stats
        Statistic names (subset of ``mean``/``std``/``zscore``/``min``/``max``/
        ``ewma``). Defaults to ``rolling.stats`` (else
        :data:`DEFAULT_ROLLING_STATS`). An unknown name raises ``ValueError``.
    config
        Pre-loaded ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` only when a default is needed.

    Returns
    -------
    pandas.DataFrame
        Keyed on ``meeting_date`` (one row per meeting, ascending), carrying only
        the new ``<sid>_roll_<w>_<stat>`` float columns — ordered column-major,
        then window, then statistic. Just ``meeting_date`` if nothing resolves.

    Raises
    ------
    ValueError
        If ``stats`` names an unknown statistic.

    Shapes
    ------
    Returns: (n_meetings, 1 + n_resolved_columns * len(windows) * len(stats)).
    """
    if columns is None or windows is None or stats is None:
        config = config if config is not None else load_features_config()
    if columns is None:
        columns = config_rolling_columns(config)  # type: ignore[arg-type]
    if windows is None:
        windows = config_rolling_windows(config)  # type: ignore[arg-type]
    if stats is None:
        stats = config_rolling_stats(config)  # type: ignore[arg-type]

    stat_names = _validate_stats(stats)
    ordered = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    resolved = _resolve_columns(ordered, columns)

    out = ordered[[_MEETING_KEY]].copy()
    for col in resolved:
        for w in windows:
            for stat in stat_names:
                out[f"{col}_roll_{w}_{stat}"] = _STAT_BUILDERS[stat](ordered[col], w)

    logger.info(
        "Built {} rolling columns ({} series × {} windows × {} stats) over {} meetings.",
        len(resolved) * len(windows) * len(stat_names),
        len(resolved),
        len(windows),
        len(stat_names),
        len(out),
    )
    return out
