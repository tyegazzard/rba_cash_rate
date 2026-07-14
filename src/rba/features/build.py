"""Feature-pipeline orchestrator — assemble every enabled feature group.

This is the ``§5 Feature engineering → Feature pipeline`` keystone: it takes the
meeting-indexed **master frame** (:mod:`rba.data.align`) and appends every enabled
derived-feature group to it, driven by ``features.yaml``. It is the single entry
point a downstream model layer calls to get the full feature matrix.

Architecture (mirrors :mod:`rba.data.align`)
--------------------------------------------
A **pure, no-network core** — :func:`build_features` — takes the master frame plus
a parsed config and returns the assembled feature frame; it never touches disk.
Keeping it pure makes the leakage / determinism tests trivial (inject a frame,
assert the output).

The orchestration around it — :func:`build_features_from_cache` + the
``python -m rba.features.build`` CLI — reads ``data/processed/master.parquet`` (or
falls back to :func:`rba.data.align.build_master_from_cache` when the cached frame
is absent), then writes ``data/processed/features.parquet`` and a local provenance
record ``data/processed/features.meta.json`` (SHA-256 / shape / meeting span /
timestamp / git commit / enabled groups), exactly mirroring
:func:`rba.data.align.write_master` + :func:`~rba.data.align.write_master_manifest`
(Invariant #5). No network anywhere.

Group gating (from ``features.yaml``)
-------------------------------------
Each group carries an ``enabled: true/false`` flag. The orchestrator reads it
**generically** — only ``surprises`` exposes a dedicated accessor — and:

- ``lags`` / ``rolling`` / ``changes`` / ``surprises`` / ``target_lags`` map to
  their pure builders (:data:`GROUP_BUILDERS`); an enabled group is built and its
  columns merged onto the meeting frame on ``meeting_date`` (a column collision
  **raises**, exactly like :func:`rba.data.align.build_master`).
- ``surprises`` is ``enabled: false`` (no consensus source yet) and is skipped
  entirely — the builder stays inert.
- ``levels`` and ``regime`` are **not** builders: the point-in-time as-of level
  columns (+ ``_age_days`` / ``_is_missing``) and the ``regime_*`` dummies are
  already materialised in the master frame by :mod:`rba.data.align`, so they pass
  through untouched (:data:`PASSTHROUGH_GROUPS`).
- ``text`` point-in-time meeting-frame alignment is **deferred** to a later
  feature-pipeline turn (:data:`DEFERRED_GROUPS`); the per-document LM scores
  remain the standalone artifact produced by :mod:`rba.features.text.lexicon`.
- ``google_trends`` / ``embeddings`` / ``fine_tuned_sentiment`` are disabled and
  have no builder, so they are simply absent from the registry.

Any configured column absent from the frame is warn-skipped by the individual
builders (never fatal), so a partially-refreshed master degrades gracefully.

``pipeline.feature_version_hash`` is intentionally left ``null`` — populating it
(a hash of config + code) is the separate "feature versioning" checklist item.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import PROCESSED_DATA_DIR, load_features_config
from rba.data import align
from rba.data.targets import build_target
from rba.features import versioning
from rba.features.changes import build_changes
from rba.features.lags import build_lags
from rba.features.rolling import build_rolling
from rba.features.surprises import build_surprises
from rba.features.target_lags import build_target_lags

_MEETING_KEY = "meeting_date"

# A pure feature builder: ``build_*(master, *, config=...) -> meeting_date + new``.
FeatureBuilder = Callable[..., pd.DataFrame]

# Config group → its pure builder, in the order their columns are appended. Each
# builder self-resolves its columns / horizons from the injected ``config`` and
# returns ``meeting_date`` + only its new columns.
GROUP_BUILDERS: dict[str, FeatureBuilder] = {
    "lags": build_lags,
    "rolling": build_rolling,
    "changes": build_changes,
    "surprises": build_surprises,
    "target_lags": build_target_lags,
}

# Groups already materialised in the master frame by ``rba.data.align`` — they
# pass through untouched (no builder). ``levels`` = the as-of level columns;
# ``regime`` = the ``regime_*`` dummies.
PASSTHROUGH_GROUPS: tuple[str, ...] = ("levels", "regime")

# Groups whose builder is deliberately deferred to a later turn. ``text``
# point-in-time meeting alignment (strict-``<`` most-recent-prior join of the LM
# scores) is tracked separately; the per-document scores remain a standalone
# artifact until then.
DEFERRED_GROUPS: tuple[str, ...] = ("text",)


# -----------------------------------------------------------------------------
# The model-input feature matrix — the canonical leakage-free X definition.
# -----------------------------------------------------------------------------
# Meeting-metadata columns that are never model inputs (mostly non-numeric, named
# explicitly for robustness): the meeting key, the effective date, and the two
# text-source URL columns.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {_MEETING_KEY, "effective_date", "statement_url", "minutes_url"}
)

# The CONTEMPORANEOUS meeting-outcome columns — the decision itself. Including any
# of these in X is trivial self-leakage (a model recovers ``sign(rate_change_bps)``
# exactly). This is the union of every ``source_columns`` entry in ``targets.yaml``
# (``rate_change_bps`` for the classification / Δ targets, ``new_rate_pct`` for the
# level target); ``tests/features/test_feature_matrix.py`` guards that this set
# stays in sync with ``targets.yaml``. NOTE: ``prior_rate_pct`` (the standing rate
# going *into* the meeting) and the ``*_lag_*`` past-decision columns are known
# before the meeting and are RETAINED as legitimate features.
OUTCOME_COLUMNS: frozenset[str] = frozenset({"rate_change_bps", "new_rate_pct"})

# Everything the feature matrix excludes: metadata + contemporaneous outcomes.
NON_MODEL_COLUMNS: frozenset[str] = NON_FEATURE_COLUMNS | OUTCOME_COLUMNS


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Canonical leakage-free model-input columns of a meeting frame.

    Returns every **numeric** column except the meeting metadata
    (:data:`NON_FEATURE_COLUMNS` — dates / URLs, also non-numeric) and the
    contemporaneous outcome columns (:data:`OUTCOME_COLUMNS` — the decision
    itself, self-leakage). ``prior_rate_pct`` and the ``*_lag_*`` past-decision
    columns are known before the meeting and are retained.

    This is the **single source of truth** for "what is X" across the §7/§8
    evaluation + tuning harness: callers must build ``X`` from here rather than an
    ad-hoc ``select_dtypes`` so no outcome column can silently leak in. The §5
    :func:`rba.features.importance.candidate_features` delegates here too.

    Parameters
    ----------
    frame
        A meeting-indexed frame (``master.parquet`` / ``features.parquet`` or any
        frame carrying the same columns).

    Returns
    -------
    list[str]
        The feature column names, in ``frame`` order.

    Shapes
    ------
    frame: (n_meetings, n_columns) -> list of length n_features.
    """
    return [
        col
        for col in frame.columns
        if col not in NON_MODEL_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The leakage-free feature sub-frame ``frame[feature_columns(frame)]`` (a copy)."""
    return frame.loc[:, feature_columns(frame)].copy()


def build_xy(
    frame: pd.DataFrame,
    target_cfg: Mapping[str, Any],
    *,
    int_labels: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the aligned, leakage-free ``(X, y)`` for a target.

    Builds ``y`` via :func:`rba.data.targets.build_target` (which drops rows the
    target can't be formed on), selects the leakage-free :func:`feature_columns`,
    and aligns ``X`` to the surviving ``y.index``. This is the canonical entry
    point the evaluation / tuning harness uses to turn a feature frame + a
    ``targets.yaml`` entry into model inputs — no ad-hoc column selection, so an
    outcome column cannot slip into ``X``.

    Parameters
    ----------
    frame
        Meeting-indexed feature frame (``features.parquet`` or ``master.parquet``).
    target_cfg
        A target entry from :func:`rba.config.load_target_config`.
    int_labels
        Forwarded to :func:`~rba.data.targets.build_target` (classification only —
        return the integer label encoding instead of string labels).

    Returns
    -------
    (X, y) : tuple[pandas.DataFrame, pandas.Series]
        ``X`` carries only leakage-free numeric features, aligned to ``y.index``.

    Shapes
    ------
    frame: (n_meetings, n_columns) -> X: (n_kept, n_features), y: (n_kept,).
    """
    y = build_target(frame, target_cfg, int_labels=int_labels)
    x = frame.loc[y.index, feature_columns(frame)].copy()
    return x, y


# -----------------------------------------------------------------------------
# Pure core.
# -----------------------------------------------------------------------------
def group_enabled(config: Mapping[str, Any], group: str) -> bool:
    """Whether ``config[group].enabled`` is truthy (generic, per-group flag).

    Reads the ``enabled`` flag out of any feature group generically — only
    ``surprises`` ships a dedicated accessor, so the orchestrator gates every
    group this way. A group absent from the config, or lacking ``enabled``, is
    treated as disabled.
    """
    section = config.get(group, {}) or {}
    return bool(section.get("enabled", False))


def build_features(
    master: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Assemble the full feature frame from the master frame (pure, no network).

    Starts from the master frame (which already carries the meeting metadata /
    target columns, the point-in-time ``levels`` + companions, and the ``regime``
    dummies) and appends every **enabled** derived-feature group in
    :data:`GROUP_BUILDERS` order, merging each builder's output onto the meeting
    frame on ``meeting_date``.

    Parameters
    ----------
    master
        The meeting-indexed master frame (:func:`rba.data.align.build_master` /
        ``rba.data.align.build_master_from_cache`` output). Must carry
        ``meeting_date``.
    config
        A parsed ``features.yaml`` mapping (injectable for tests). Loaded via
        :func:`rba.config.load_features_config` when omitted.

    Returns
    -------
    pandas.DataFrame
        The master frame plus every enabled group's columns, one row per meeting,
        sorted ascending by ``meeting_date``.

    Raises
    ------
    ValueError
        If a builder contributes a column already present in the frame (mirrors
        :func:`rba.data.align.build_master`'s collision guard).

    Shapes
    ------
    Returns: (n_meetings, n_master_columns + Σ per-enabled-group new columns).
    """
    if config is None:
        config = load_features_config()

    base = master.sort_values(_MEETING_KEY).reset_index(drop=True)
    features = base.copy()
    seen: set[str] = set(features.columns)

    for group in PASSTHROUGH_GROUPS:
        state = "enabled" if group_enabled(config, group) else "disabled"
        logger.debug("Group {!r} passes through from master ({}).", group, state)
    for group in DEFERRED_GROUPS:
        if group_enabled(config, group):
            logger.info("Group {!r} enabled but its builder is deferred; skipping.", group)

    for group, builder in GROUP_BUILDERS.items():
        if not group_enabled(config, group):
            logger.info("Group {!r} disabled; skipping.", group)
            continue
        addition = builder(base, config=config)
        features = _merge_group(features, addition, group, seen)

    logger.info(
        "Assembled feature frame: {} meetings × {} columns ({} from master).",
        len(features),
        features.shape[1],
        master.shape[1],
    )
    return features


def _merge_group(
    features: pd.DataFrame,
    addition: pd.DataFrame,
    group: str,
    seen: set[str],
) -> pd.DataFrame:
    """Merge one builder's output onto ``features`` on ``meeting_date``.

    Guards against column collisions (a builder must never re-emit a column
    already present) exactly like :func:`rba.data.align.build_master`, and updates
    ``seen`` in place.
    """
    new_cols = [c for c in addition.columns if c != _MEETING_KEY]
    collisions = seen.intersection(new_cols)
    if collisions:
        raise ValueError(
            f"Feature group {group!r} contributes columns already present: "
            f"{sorted(collisions)}. Feature columns must be unique across groups."
        )
    if not new_cols:
        logger.debug("Group {!r} produced no columns; nothing to merge.", group)
        return features
    merged = features.merge(addition, on=_MEETING_KEY, how="left")
    seen.update(new_cols)
    logger.debug("Merged group {!r}: +{} columns.", group, len(new_cols))
    return merged


# -----------------------------------------------------------------------------
# Orchestration: cache → feature frame → data/processed/features.parquet.
# -----------------------------------------------------------------------------
FEATURES_PARQUET_PATH = PROCESSED_DATA_DIR / "features.parquet"
FEATURES_META_PATH = PROCESSED_DATA_DIR / "features.meta.json"


def load_master(path: Path = align.MASTER_PARQUET_PATH) -> pd.DataFrame:
    """Load the cached master frame, or rebuild it from raw as a fallback.

    Reads ``data/processed/master.parquet`` when present (the fast path — no
    re-parsing of raw); otherwise warns and rebuilds via
    :func:`rba.data.align.build_master_from_cache` (read-only over ``data/raw``,
    no network).

    Returns
    -------
    pandas.DataFrame
        The meeting-indexed master frame.
    """
    if path.exists():
        master = pd.read_parquet(path)
        logger.info(
            "Loaded master frame from {} ({} meetings × {} cols).",
            path,
            len(master),
            master.shape[1],
        )
        return master
    logger.warning("Master parquet absent at {}; rebuilding from cached raw.", path)
    return align.build_master_from_cache()


def build_features_from_cache(
    *,
    config: Mapping[str, Any] | None = None,
    master_path: Path = align.MASTER_PARQUET_PATH,
) -> pd.DataFrame:
    """End-to-end feature frame from cached artifacts (read-only, no network).

    Loads the master frame (:func:`load_master`) and the ``features.yaml`` config,
    then runs the pure :func:`build_features` core.

    Returns
    -------
    pandas.DataFrame
        The assembled feature frame.
    """
    config = config if config is not None else load_features_config()
    master = load_master(master_path)
    return build_features(master, config=config)


def write_features(features: pd.DataFrame, path: Path = FEATURES_PARQUET_PATH) -> str:
    """Write the feature frame to Parquet and return its SHA-256 (Invariant #5).

    Mirrors :func:`rba.data.align.write_master`.

    Returns
    -------
    str
        The hex SHA-256 digest of the written Parquet bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    logger.success(
        "Wrote {} ({} meetings × {} cols); sha256={}",
        path,
        len(features),
        features.shape[1],
        digest,
    )
    return digest


def write_features_manifest(
    features: pd.DataFrame,
    config: Mapping[str, Any],
    parquet_path: Path,
    digest: str,
    path: Path = FEATURES_META_PATH,
) -> dict[str, object]:
    """Write a local provenance record for the feature frame (Invariant #5).

    Mirrors :func:`rba.data.align.write_master_manifest`: a dependency-free
    stand-in for MLflow artifact logging that records the frame's SHA-256, shape,
    meeting span, generation timestamp, git commit — plus the list of feature
    groups that were enabled, so any downstream run can verify exactly which
    feature frame it consumed and how it was configured.

    It also records the **feature-version hash** (:mod:`rba.features.versioning`)
    — a reproducibility digest over the config + builder code — alongside its
    ``config_hash`` / ``code_hash`` components. The tracked
    ``features.yaml`` ``pipeline.feature_version_hash`` field is left as a null
    pin slot; if it is pinned to a non-null value that disagrees with the computed
    hash, :func:`rba.features.versioning.resolve_feature_version` logs a drift
    warning here.

    Returns
    -------
    dict
        The manifest that was written.
    """
    enabled = [g for g in GROUP_BUILDERS if group_enabled(config, g)]
    config_hash = versioning.hash_config(config)
    code_hash = versioning.hash_code()
    feature_version = versioning.resolve_feature_version(config)
    manifest: dict[str, object] = {
        "artifact": parquet_path.name,
        "sha256": digest,
        "n_meetings": int(len(features)),
        "n_columns": int(features.shape[1]),
        "first_meeting": str(features[_MEETING_KEY].min().date()),
        "last_meeting": str(features[_MEETING_KEY].max().date()),
        "enabled_groups": enabled,
        "feature_version_hash": feature_version,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": align._git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.success("Wrote provenance manifest {}", path)
    return manifest


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.features.build`` argument parser."""
    return argparse.ArgumentParser(
        prog="rba.features.build",
        description="Assemble the feature frame from the cached master frame.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    build_parser().parse_args(argv)

    config = load_features_config()
    features = build_features_from_cache(config=config)
    digest = write_features(features)
    write_features_manifest(features, config, FEATURES_PARQUET_PATH, digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
