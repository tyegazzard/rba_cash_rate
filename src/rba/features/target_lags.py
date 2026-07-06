"""Meeting-space target-lag features for the meeting-indexed master frame.

A ``§5 Feature engineering`` builder that lags the **target itself** — the RBA's
own past decisions used as features (a standard, strong signal in policy-rate
classification: the last decision, and the level it left the rate at, condition
the next one). For each target column ``<col>`` and horizon ``h`` it emits
``<col>_lag_<h>`` = the value that column held ``h`` meetings ago.

Why a separate builder from :mod:`rba.features.lags`?
----------------------------------------------------
The generic :mod:`rba.features.lags` builder deliberately **excludes** the
meeting-metadata / target columns (``new_rate_pct``, ``rate_change_bps``, …) from
its eligible set — they are the thing being predicted, not source levels. Target
lags are the one place we intentionally lag those columns, so they live here with
their own explicit column list and ``target_lags:`` config group.

Meeting-space + leakage safety (Invariant #1)
---------------------------------------------
Identical semantics to :mod:`rba.features.lags`: the master frame is
**meeting-indexed** and a horizon is a fixed number of *decisions*, so a lag is
``master[<col>].shift(h)`` over meeting rows sorted ascending by ``meeting_date``.
This is leakage-safe by construction — a positive ``shift`` copies a value from an
*earlier* meeting forward, so row ``t`` can only ever see rows with
``meeting_date < t`` — and cadence-robust across the Feb-2024 11/yr → 8/yr change
(time is indexed by meeting, not calendar days). The first ``h`` meetings get
``NaN``, never zero (Common pitfall).

Configuration
-------------
Horizons come from ``features.yaml`` ``target_lags.horizons_meetings`` (see
:func:`rba.config.load_features_config`); the target columns are a module constant
(:data:`DEFAULT_TARGET_COLUMNS`) rather than a config list, since "the target"
is fixed by the meeting frame's schema. A configured target column absent from the
frame it is handed is **warn-skipped**, never fatal. Pure function returning
``meeting_date`` + only the new lag columns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import load_features_config

# Default target-lag horizons in MEETINGS (``features.yaml`` target_lags.
# horizons_meetings). 1–3 meetings capture the immediate decision history — "what
# did the Board just do, and the two decisions before that".
DEFAULT_TARGET_LAG_HORIZONS_MEETINGS: tuple[int, ...] = (1, 2, 3)

# The target columns to lag, from the meeting frame's schema (rba.data.align):
# ``rate_change_bps`` is the last *decision* (signed bps move); ``new_rate_pct``
# is the *level* it left the cash rate at. ``prior_rate_pct`` is redundant with a
# 1-meeting lag of ``new_rate_pct`` and is deliberately not re-lagged.
DEFAULT_TARGET_COLUMNS: tuple[str, ...] = ("rate_change_bps", "new_rate_pct")

_MEETING_KEY = "meeting_date"


def config_target_lag_horizons(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Target-lag horizons (in meetings) from config, or the module default.

    Reads ``target_lags.horizons_meetings``; falls back to
    :data:`DEFAULT_TARGET_LAG_HORIZONS_MEETINGS` when the key is absent.
    """
    section = config.get("target_lags", {}) or {}
    horizons = section.get("horizons_meetings")
    if not horizons:
        return DEFAULT_TARGET_LAG_HORIZONS_MEETINGS
    return tuple(int(h) for h in horizons)


def config_target_lags_enabled(config: Mapping[str, Any]) -> bool:
    """Whether the ``target_lags:`` group is enabled (for the pipeline orchestrator).

    :func:`rba.features.build.build_features` reads group ``enabled`` flags
    generically, so it does not depend on this; provided for symmetry with
    :func:`rba.features.surprises.config_surprises_enabled` and testability.
    """
    section = config.get("target_lags", {}) or {}
    return bool(section.get("enabled", False))


def _resolve_columns(master: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    """Filter ``requested`` target columns to those present in ``master``.

    Warn-skips (never raises) each requested column absent from the frame, so a
    frame missing a target column (unusual, but possible for a partial fixture)
    degrades gracefully.
    """
    present = set(master.columns)
    resolved: list[str] = []
    for col in requested:
        if col not in present:
            logger.warning("Target lag: column {!r} absent from frame; skipping.", col)
            continue
        resolved.append(col)
    return resolved


def build_target_lags(
    master: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build meeting-space target-lag features from the master frame.

    For every resolved target column ``<col>`` and horizon ``h``, emits
    ``<col>_lag_<h>`` = ``master[<col>].shift(h)`` over meetings sorted ascending
    by ``meeting_date`` — the value that target held ``h`` meetings before each
    meeting. The first ``h`` meetings get ``NaN`` (no ``h``-ago predecessor).

    Parameters
    ----------
    master
        The meeting-indexed master frame (``rba.data.align.build_master`` output).
        Must carry ``meeting_date`` plus the target columns to lag.
    columns
        Target columns to lag. Defaults to :data:`DEFAULT_TARGET_COLUMNS`
        (``rate_change_bps`` / ``new_rate_pct``). Any absent from ``master`` is
        warn-skipped.
    horizons
        Lag horizons in meetings. Defaults to ``features.yaml``
        ``target_lags.horizons_meetings`` (else
        :data:`DEFAULT_TARGET_LAG_HORIZONS_MEETINGS`).
    config
        A pre-loaded ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` only when ``horizons`` needs a
        default.

    Returns
    -------
    pandas.DataFrame
        Keyed on ``meeting_date`` (one row per meeting, ascending), carrying only
        the new ``<col>_lag_<h>`` float columns — ordered column-major then
        horizon-major. Just ``meeting_date`` if nothing resolves.

    Shapes
    ------
    Returns: (n_meetings, 1 + n_resolved_columns * len(horizons)).
    """
    if horizons is None:
        config = config if config is not None else load_features_config()
        horizons = config_target_lag_horizons(config)
    if columns is None:
        columns = DEFAULT_TARGET_COLUMNS

    ordered = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    resolved = _resolve_columns(ordered, columns)

    out = ordered[[_MEETING_KEY]].copy()
    for col in resolved:
        for h in horizons:
            out[f"{col}_lag_{h}"] = ordered[col].shift(h).astype("float64")

    logger.info(
        "Built {} target-lag columns ({} target(s) × {} horizons) over {} meetings.",
        len(resolved) * len(horizons),
        len(resolved),
        len(horizons),
        len(out),
    )
    return out
