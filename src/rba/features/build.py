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
        logger.info("Loaded master frame from {} ({} meetings × {} cols).", path, len(master), master.shape[1])
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
    ``pipeline.feature_version_hash`` (a hash of config + code) is a separate
    checklist item and is not populated here.

    Returns
    -------
    dict
        The manifest that was written.
    """
    enabled = [g for g in GROUP_BUILDERS if group_enabled(config, g)]
    manifest: dict[str, object] = {
        "artifact": parquet_path.name,
        "sha256": digest,
        "n_meetings": int(len(features)),
        "n_columns": int(features.shape[1]),
        "first_meeting": str(features[_MEETING_KEY].min().date()),
        "last_meeting": str(features[_MEETING_KEY].max().date()),
        "enabled_groups": enabled,
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
