"""Meeting-space lag features for the meeting-indexed master frame.

This is the first ``§5 Feature engineering → Numerical features`` builder. It
takes the master frame produced by :mod:`rba.data.align` (one row per RBA Board
meeting, carrying the as-of-latest value of each source series) and emits, for
each configured series ``<sid>`` and horizon ``h``, the column
``<sid>_lag_<h>`` = the value that series read **as of ``h`` meetings ago**.

Why meeting-space?
------------------
The master frame is **meeting-indexed** and carries only the as-of-latest value
per series (no native monthly/daily history), so lags are computed as
``master[<sid>].shift(h)`` over meeting rows sorted ascending by
``meeting_date``. Consequences:

- **Leakage-safe by construction** (Invariant #1). A positive ``shift`` copies a
  value from an *earlier* meeting row forward; row ``t`` can only ever see rows
  with ``meeting_date < t``. There is no calendar-date join to get wrong.
- **Cadence-robust** (Common pitfall: the Feb-2024 11/yr → 8/yr cadence change).
  Because time is indexed by meeting rather than by calendar days, a horizon is a
  fixed number of *decisions* regardless of the wall-clock gap between them; the
  gap itself is already carried separately by ``gap_days_since_last_meeting``.
- **NaN at the series start**, never zero (Common pitfall: "lags should produce
  NaNs, not zeros") — the first ``h`` meetings have no ``h``-ago predecessor.

Configuration
-------------
Columns and horizons come from ``features.yaml`` ``lags:`` (see
:func:`rba.config.load_features_config`), reconciled to the real
``align.as_of_levels`` series_ids. The builder lags the union of
``monthly_columns`` + ``daily_columns`` and **warn-skips** any configured column
absent from the frame it is handed (e.g. the FRED global block, never refreshed
into the master), never raising. Companion columns (``_age_days`` /
``_is_missing``), ``regime_*`` dummies, and meeting-metadata columns are never
lagged.

The builder is a **pure function** (per CONTEXT.md "Adding a new feature"): it
returns a frame keyed on ``meeting_date`` carrying *only* the new lag columns, so
a later ``features/build.py`` can compose it with the other numerical builders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import load_features_config

# Default lag horizons in MEETINGS (not native months/days — see module docstring).
# 1–3 meetings capture recent momentum; 6 ≈ half a year, 12 ≈ a full year under
# the pre-2024 cadence. Overridable via ``features.yaml`` ``lags.horizons_meetings``.
DEFAULT_LAG_HORIZONS_MEETINGS: tuple[int, ...] = (1, 2, 3, 6, 12)

_MEETING_KEY = "meeting_date"

# Companion-column suffixes emitted per series by ``align.as_of_levels`` /
# ``align.add_missing_indicators``. A level column is the thing WITHOUT one of
# these suffixes; these companions are themselves never lagged.
_AGE_SUFFIX = "_age_days"
_MISSING_SUFFIX = "_is_missing"
_REGIME_PREFIX = "regime_"

# Meeting-metadata / target columns carried by the master frame — never lagged
# (mirrors align._MEETING_COLUMNS; kept local to avoid depending on a private).
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


def _is_lag_eligible(column: str) -> bool:
    """Whether ``column`` is a genuine level series eligible for lagging.

    Excludes the ``_age_days`` / ``_is_missing`` companions, ``regime_*`` dummies,
    and the meeting-metadata columns — everything that is not a source level.
    """
    if column in _META_COLUMNS:
        return False
    if column.startswith(_REGIME_PREFIX):
        return False
    return not column.endswith((_AGE_SUFFIX, _MISSING_SUFFIX))


def config_lag_columns(config: Mapping[str, Any]) -> list[str]:
    """Union of the configured ``monthly_columns`` + ``daily_columns``, in order.

    Parameters
    ----------
    config
        A parsed ``features.yaml`` mapping (see
        :func:`rba.config.load_features_config`).

    Returns
    -------
    list[str]
        The configured lag target series_ids, de-duplicated preserving first
        appearance (monthly before daily). Empty if the ``lags:`` section is
        absent.
    """
    section = config.get("lags", {}) or {}
    ordered: list[str] = []
    seen: set[str] = set()
    for key in ("monthly_columns", "daily_columns"):
        for col in section.get(key, []) or []:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def config_lag_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Lag horizons (in meetings) from config, or the module default.

    Reads ``lags.horizons_meetings``; falls back to
    :data:`DEFAULT_LAG_HORIZONS_MEETINGS` when the key is absent.
    """
    section = config.get("lags", {}) or {}
    horizons = section.get("horizons_meetings")
    if not horizons:
        return DEFAULT_LAG_HORIZONS_MEETINGS
    return tuple(int(h) for h in horizons)


def _resolve_columns(master: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    """Filter ``requested`` to columns present in ``master`` and lag-eligible.

    Warn-skips (never raises) each requested column that is absent from the frame
    or is a companion / regime / metadata column, so a config that references a
    series not refreshed into this particular frame degrades gracefully.
    """
    present = set(master.columns)
    resolved: list[str] = []
    for col in requested:
        if col not in present:
            logger.warning("Lag: configured column {!r} absent from frame; skipping.", col)
            continue
        if not _is_lag_eligible(col):
            logger.warning("Lag: column {!r} is not a level series; skipping.", col)
            continue
        resolved.append(col)
    return resolved


def build_lags(
    master: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build meeting-space lag features from the master frame.

    For every resolved column ``<sid>`` and horizon ``h``, emits
    ``<sid>_lag_<h>`` = ``master[<sid>].shift(h)`` over meetings sorted ascending
    by ``meeting_date`` — the value that series read as of ``h`` meetings before
    each meeting. The first ``h`` meetings get ``NaN`` (no ``h``-ago predecessor).

    Parameters
    ----------
    master
        The meeting-indexed master frame (``rba.data.align.build_master`` output).
        Must carry a ``meeting_date`` column plus the level columns to lag.
    columns
        Series_ids to lag. Defaults to the reconciled ``features.yaml``
        ``lags:`` union (``config_lag_columns``). Any column absent from
        ``master`` — or a companion / regime / metadata column — is warn-skipped.
    horizons
        Lag horizons in meetings. Defaults to ``features.yaml``
        ``lags.horizons_meetings`` (else :data:`DEFAULT_LAG_HORIZONS_MEETINGS`).
    config
        A pre-loaded ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` only when ``columns`` or
        ``horizons`` needs a default.

    Returns
    -------
    pandas.DataFrame
        Keyed on ``meeting_date`` (one row per meeting, ascending), carrying only
        the new ``<sid>_lag_<h>`` float columns — ordered column-major then
        horizon-major (all lags of the first series, then the next). Empty of lag
        columns (just ``meeting_date``) if nothing resolves.

    Shapes
    ------
    Returns: (n_meetings, 1 + n_resolved_columns * len(horizons)).
    """
    if columns is None or horizons is None:
        config = config if config is not None else load_features_config()
    if columns is None:
        columns = config_lag_columns(config)  # type: ignore[arg-type]
    if horizons is None:
        horizons = config_lag_horizons(config)  # type: ignore[arg-type]

    ordered = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    resolved = _resolve_columns(ordered, columns)

    out = ordered[[_MEETING_KEY]].copy()
    for col in resolved:
        for h in horizons:
            out[f"{col}_lag_{h}"] = ordered[col].shift(h)

    logger.info(
        "Built {} lag columns ({} series × {} horizons) over {} meetings.",
        len(resolved) * len(horizons),
        len(resolved),
        len(horizons),
        len(out),
    )
    return out
