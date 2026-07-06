"""Feature versioning — a reproducibility hash of config + builder code.

Implements the ``§5 Feature engineering → Feature pipeline`` "feature versioning"
item (Invariant #5, reproducibility): a deterministic SHA-256 over

1. the ``features.yaml`` **config** that drives the pipeline, and
2. the **source** of every feature-builder module that determines the output,

so the same config + the same code always produce the same hash, and any change
to either is detectable. The hash is recorded in
``data/processed/features.meta.json`` by
:func:`rba.features.build.write_features_manifest`; it is **not** written back into
the tracked ``features.yaml`` (that would churn a version-controlled file on every
build). The YAML ``pipeline.feature_version_hash`` field stays a null *pin* slot:
set it to a known hash and the build logs a **drift warning** when the freshly
computed hash disagrees (config or code changed under a pinned expectation).

What feeds the hash
-------------------
- **Config** — the whole parsed ``features.yaml`` mapping, canonicalised to sorted
  JSON, with the ``pipeline.feature_version_hash`` field itself removed first (so
  the stamp never depends on its own recorded value).
- **Code** — the bytes of each builder module in :data:`FEATURE_CODE_PATHS`, keyed
  by its path relative to this package so the hash is stable across machines. The
  set is the modules :func:`rba.features.build.build_features` actually invokes;
  the text-alignment modules join it when text features are wired into the
  pipeline (currently deferred).

Pure and network-free: hashing reads only local source files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

# This package directory — builder sources are resolved relative to it, and the
# relative path (not the absolute one) is what feeds the hash, so the digest is
# reproducible across checkouts / machines.
CODE_DIR = Path(__file__).resolve().parent

# Builder modules whose source determines the assembled feature frame — i.e. the
# modules ``build_features`` invokes. A change to any of them yields a new feature
# version. The text-lexicon modules (``text/lexicon.py`` / ``text/lexicon_data``)
# are intentionally excluded until the deferred text-alignment step is wired into
# ``build_features`` — until then they do not affect the produced features.
_BUILDER_MODULES: tuple[str, ...] = (
    "build.py",
    "lags.py",
    "rolling.py",
    "changes.py",
    "surprises.py",
    "target_lags.py",
)
FEATURE_CODE_PATHS: tuple[Path, ...] = tuple(CODE_DIR / rel for rel in _BUILDER_MODULES)

# The config field excluded from the config hash so the stamp is independent of
# whatever value is (optionally) pinned there.
_PIPELINE_KEY = "pipeline"
_VERSION_FIELD = "feature_version_hash"

_ENCODING = "utf-8"


def hash_config(config: Mapping[str, Any]) -> str:
    """SHA-256 of the config, canonicalised and stripped of the version field.

    The config is round-tripped through JSON (a deep, order-independent copy that
    also rejects anything non-serialisable), the ``pipeline.feature_version_hash``
    field is removed, and the result is dumped with sorted keys before hashing —
    so two configs that differ *only* in that pinned field hash identically.

    Parameters
    ----------
    config
        A parsed ``features.yaml`` mapping.

    Returns
    -------
    str
        The 64-char hex SHA-256 digest.
    """
    payload = json.loads(json.dumps(config, default=str))
    pipeline = payload.get(_PIPELINE_KEY)
    if isinstance(pipeline, dict):
        pipeline.pop(_VERSION_FIELD, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode(_ENCODING)).hexdigest()


def hash_code(paths: Sequence[Path] = FEATURE_CODE_PATHS) -> str:
    """SHA-256 over the builder module sources, keyed by relative path.

    Each file contributes its relative key, byte length, and raw bytes to the
    running digest (length-prefixed so a filename / content boundary can't be
    forged), processed in sorted key order for determinism. A missing source file
    is warn-logged and hashed as empty rather than raising, so the version stays
    computable on a partial checkout.

    Parameters
    ----------
    paths
        Builder source files to hash (defaults to :data:`FEATURE_CODE_PATHS`).

    Returns
    -------
    str
        The 64-char hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=_code_key):
        key = _code_key(path)
        try:
            content = path.read_bytes()
        except OSError:
            logger.warning("Feature version: builder source {} missing; hashing as empty.", path)
            content = b""
        digest.update(f"{key}\0{len(content)}\0".encode(_ENCODING))
        digest.update(content)
    return digest.hexdigest()


def _code_key(path: Path) -> str:
    """Path key for hashing: relative to :data:`CODE_DIR` when possible, else name."""
    try:
        return path.resolve().relative_to(CODE_DIR).as_posix()
    except ValueError:
        return path.name


def combine(config_hash: str, code_hash: str) -> str:
    """Combine a config hash and a code hash into the single feature-version hash."""
    combined = f"config:{config_hash}\ncode:{code_hash}"
    return hashlib.sha256(combined.encode(_ENCODING)).hexdigest()


def compute_feature_version(
    config: Mapping[str, Any],
    *,
    code_paths: Sequence[Path] = FEATURE_CODE_PATHS,
) -> str:
    """The feature-version hash = ``combine(hash_config, hash_code)``.

    Deterministic: identical ``config`` + identical builder sources always yield
    the same hash.
    """
    return combine(hash_config(config), hash_code(code_paths))


def pinned_version(config: Mapping[str, Any]) -> str | None:
    """The pinned ``pipeline.feature_version_hash`` from config, or ``None``.

    Returns ``None`` when the field is absent, ``null``, or empty — i.e. no pin.
    """
    pipeline = config.get(_PIPELINE_KEY) or {}
    value = pipeline.get(_VERSION_FIELD)
    return str(value) if value else None


def resolve_feature_version(
    config: Mapping[str, Any],
    *,
    code_paths: Sequence[Path] = FEATURE_CODE_PATHS,
) -> str:
    """Compute the feature version and warn if it drifts from a pinned value.

    If ``features.yaml`` pins ``pipeline.feature_version_hash`` and it disagrees
    with the freshly computed hash, a warning is logged (config or builder code
    changed under a pinned expectation). Returns the computed hash regardless.
    """
    computed = compute_feature_version(config, code_paths=code_paths)
    pinned = pinned_version(config)
    if pinned and pinned != computed:
        logger.warning(
            "Feature version drift: features.yaml pins {} but build computed {}.",
            pinned,
            computed,
        )
    return computed
