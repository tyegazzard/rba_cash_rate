"""Surprise features (realised − consensus) for the master frame.

Fourth and final ``§5 Feature engineering → Numerical features`` builder. A
*surprise* is how far a realised data print landed from what was expected:

    <realised_sid>_surprise = master[<realised_sid>] − master[<consensus_sid>]

A positive CPI surprise (realised above consensus) is a hawkish signal; a
negative one dovish. The absolute (unit-preserving) form is emitted — a
standardised "surprise index" variant can be composed later via
:mod:`rba.features.rolling` if wanted.

Inert until a consensus source lands
------------------------------------
**No consensus/forecast source exists in this project yet** — the master frame
(:mod:`rba.data.align`) carries only *realised* as-of levels. This builder is the
ready interface: it emits a surprise column only where **both** the realised and
consensus columns are present in the frame, so on today's consensus-free master
every configured pair is skipped (info-logged), never crashing. It activates
automatically once a consensus source is added and its columns appear. The
``surprises:`` group is correspondingly ``enabled: false`` in ``features.yaml``.

Pairing is **explicit** — ``features.yaml`` ``surprises.pairs`` maps each real
realised ``series_id`` to its (future) consensus ``series_id`` — rather than a
naming convention, so it imposes no naming constraint on the not-yet-built
consensus source and documents each pairing.

Point-in-time correctness (Invariant #1)
----------------------------------------
The surprise is a row-wise difference of two columns that are *already* both
point-in-time in the master (``align.as_of_levels`` guarantees each carries only
values with ``publication_date < meeting_date``), so it inherits leakage safety
with no extra shifting. **Reference-period correctness is the consensus source's
responsibility**: each consensus series must be aligned to the same reference
period as its realised counterpart (see ``features.yaml`` and the CONTRACT note
there), otherwise realised − consensus is not a true surprise. A row where either
side is missing yields ``NaN`` (never zero).

Pure function returning ``meeting_date`` + only the new ``_surprise`` columns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import load_features_config

_MEETING_KEY = "meeting_date"
_SURPRISE_SUFFIX = "_surprise"

# Companion suffixes / prefixes marking a column as NOT a source level — neither
# side of a surprise pair may be one (kept local per the codebase's convention).
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
    """Whether ``column`` is a genuine source level (eligible as a surprise side)."""
    if column in _META_COLUMNS:
        return False
    if column.startswith(_REGIME_PREFIX):
        return False
    return not column.endswith((_AGE_SUFFIX, _MISSING_SUFFIX))


def config_surprise_pairs(config: Mapping[str, Any]) -> dict[str, str]:
    """The ``{realised_sid: consensus_sid}`` mapping from ``surprises.pairs``.

    Empty if the ``surprises:`` section or its ``pairs`` key is absent.
    """
    section = config.get("surprises", {}) or {}
    pairs = section.get("pairs", {}) or {}
    return {str(realised): str(consensus) for realised, consensus in pairs.items()}


def config_surprises_enabled(config: Mapping[str, Any]) -> bool:
    """Whether the ``surprises:`` group is enabled (for the pipeline orchestrator).

    The pure :func:`build_surprises` does not gate on this — it is naturally inert
    when consensus columns are absent — but ``features/build.py`` can consult it to
    skip the group entirely.
    """
    section = config.get("surprises", {}) or {}
    return bool(section.get("enabled", False))


def build_surprises(
    master: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    pairs: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build ``realised − consensus`` surprise features from the master frame.

    For every ``{realised: consensus}`` pair where **both** columns are present
    (and are genuine level series), emits ``<realised>_surprise = master[realised]
    − master[consensus]``. A pair whose consensus column is not (yet) in the frame
    is skipped with an info log — the expected state until a consensus source is
    added (see module docstring). A pair whose *realised* column is missing or
    whose either side is a companion / regime / metadata column is warn-skipped.

    Parameters
    ----------
    master
        The meeting-indexed master frame (``rba.data.align.build_master`` output).
    config
        Pre-loaded ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` only when ``pairs`` is not given.
    pairs
        ``{realised_sid: consensus_sid}`` overriding the config
        (:func:`config_surprise_pairs`).

    Returns
    -------
    pandas.DataFrame
        Keyed on ``meeting_date`` (one row per meeting, ascending), carrying only
        the new ``<realised>_surprise`` float columns, in the pairs' iteration
        order. Just ``meeting_date`` when no pair resolves (e.g. today's
        consensus-free master).

    Shapes
    ------
    Returns: (n_meetings, 1 + n_resolved_pairs).
    """
    if pairs is None:
        config = config if config is not None else load_features_config()
        pairs = config_surprise_pairs(config)

    ordered = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    present = set(ordered.columns)

    out = ordered[[_MEETING_KEY]].copy()
    n_resolved = 0
    for realised, consensus in pairs.items():
        if realised not in present or not _is_level_column(realised):
            logger.warning(
                "Surprise: realised column {!r} absent or not a level series; skipping pair.",
                realised,
            )
            continue
        if consensus not in present:
            logger.info(
                "Surprise: consensus column {!r} for {!r} not present "
                "(consensus not sourced yet); skipping pair.",
                consensus,
                realised,
            )
            continue
        if not _is_level_column(consensus):
            logger.warning(
                "Surprise: consensus column {!r} is not a level series; skipping pair.",
                consensus,
            )
            continue
        out[f"{realised}{_SURPRISE_SUFFIX}"] = ordered[realised] - ordered[consensus]
        n_resolved += 1

    logger.info(
        "Built {} surprise column(s) from {} configured pair(s) over {} meetings.",
        n_resolved,
        len(pairs),
        len(out),
    )
    return out
