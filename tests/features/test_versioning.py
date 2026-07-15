"""Tests for ``rba.features.versioning`` — the feature-version reproducibility hash.

No network — configs are inline dicts and code hashing runs over tmp files so the
tests never depend on the real builder sources' current bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rba.features.versioning import (
    FEATURE_CODE_PATHS,
    combine,
    compute_feature_version,
    hash_code,
    hash_config,
    pinned_version,
    resolve_feature_version,
)


@pytest.fixture
def loguru_messages():
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _config(**overrides) -> dict:
    base = {
        "lags": {"enabled": True, "horizons_meetings": [1, 2]},
        "pipeline": {"feature_version_hash": None},
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# hash_config.
# -----------------------------------------------------------------------------
def test_hash_config_is_deterministic() -> None:
    assert hash_config(_config()) == hash_config(_config())


def test_hash_config_key_order_independent() -> None:
    a = {"a": 1, "b": {"x": 1, "y": 2}}
    b = {"b": {"y": 2, "x": 1}, "a": 1}
    assert hash_config(a) == hash_config(b)


def test_hash_config_changes_with_content() -> None:
    assert hash_config(_config(lags={"enabled": True, "horizons_meetings": [1]})) != hash_config(
        _config()
    )


def test_hash_config_ignores_pinned_version_field() -> None:
    """Two configs differing only in the pinned version field hash identically."""
    unpinned = _config()
    pinned = _config(pipeline={"feature_version_hash": "deadbeef"})
    assert hash_config(unpinned) == hash_config(pinned)


# -----------------------------------------------------------------------------
# hash_code.
# -----------------------------------------------------------------------------
def _write_code(dir_: Path, files: dict[str, str]) -> list[Path]:
    paths = []
    for name, content in files.items():
        path = dir_ / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def test_hash_code_deterministic_and_order_independent(tmp_path: Path) -> None:
    paths = _write_code(tmp_path, {"a.py": "print(1)", "b.py": "print(2)"})
    assert hash_code(paths) == hash_code(list(reversed(paths)))


def test_hash_code_changes_when_a_file_changes(tmp_path: Path) -> None:
    paths = _write_code(tmp_path, {"a.py": "print(1)", "b.py": "print(2)"})
    before = hash_code(paths)
    (tmp_path / "a.py").write_text("print(999)", encoding="utf-8")
    assert hash_code(paths) != before


def test_hash_code_missing_file_warns_and_is_stable(tmp_path: Path, loguru_messages: list) -> None:
    missing = tmp_path / "gone.py"
    first = hash_code([missing])
    second = hash_code([missing])
    assert first == second  # deterministic even when absent
    assert any("missing" in m and "gone.py" in m for m in loguru_messages)


def test_default_builder_sources_all_exist() -> None:
    for path in FEATURE_CODE_PATHS:
        assert path.exists(), f"builder source missing: {path}"


# -----------------------------------------------------------------------------
# combine / compute_feature_version.
# -----------------------------------------------------------------------------
def test_combine_is_deterministic_and_sensitive() -> None:
    assert combine("aa", "bb") == combine("aa", "bb")
    assert combine("aa", "bb") != combine("bb", "aa")


def test_compute_feature_version_deterministic(tmp_path: Path) -> None:
    paths = _write_code(tmp_path, {"a.py": "x = 1"})
    v1 = compute_feature_version(_config(), code_paths=paths)
    v2 = compute_feature_version(_config(), code_paths=paths)
    assert v1 == v2
    assert len(v1) == 64


def test_compute_feature_version_reacts_to_config_and_code(tmp_path: Path) -> None:
    paths = _write_code(tmp_path, {"a.py": "x = 1"})
    base = compute_feature_version(_config(), code_paths=paths)
    # Config change moves the version.
    assert compute_feature_version(_config(lags={"enabled": False}), code_paths=paths) != base
    # Code change moves the version.
    (tmp_path / "a.py").write_text("x = 2", encoding="utf-8")
    assert compute_feature_version(_config(), code_paths=paths) != base


# -----------------------------------------------------------------------------
# pinned_version / drift.
# -----------------------------------------------------------------------------
def test_pinned_version_reads_field() -> None:
    assert pinned_version(_config(pipeline={"feature_version_hash": "abc"})) == "abc"
    assert pinned_version(_config()) is None  # null pin
    assert pinned_version({}) is None


def test_resolve_warns_on_drift(tmp_path: Path, loguru_messages: list) -> None:
    paths = _write_code(tmp_path, {"a.py": "x = 1"})
    config = _config(pipeline={"feature_version_hash": "not_the_real_hash"})
    computed = resolve_feature_version(config, code_paths=paths)
    assert computed == compute_feature_version(config, code_paths=paths)
    assert any("drift" in m for m in loguru_messages)


def test_resolve_no_warn_when_unpinned(tmp_path: Path, loguru_messages: list) -> None:
    paths = _write_code(tmp_path, {"a.py": "x = 1"})
    resolve_feature_version(_config(), code_paths=paths)
    assert not any("drift" in m for m in loguru_messages)


def test_resolve_no_warn_when_pin_matches(tmp_path: Path, loguru_messages: list) -> None:
    paths = _write_code(tmp_path, {"a.py": "x = 1"})
    correct = compute_feature_version(_config(), code_paths=paths)
    config = _config(pipeline={"feature_version_hash": correct})
    resolve_feature_version(config, code_paths=paths)
    assert not any("drift" in m for m in loguru_messages)
