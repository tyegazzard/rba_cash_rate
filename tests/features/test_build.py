"""Tests for ``rba.features.build`` — the feature-pipeline orchestrator.

No network — the master frame is built inline and the config is injected. Covers
generic group gating (enabled builds / disabled skips / passthrough / deferred),
the collision guard, config-injection, determinism, graceful warn-skip of absent
inputs, and the disk orchestration (load_master fallback, write + manifest).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rba.features.build import (
    GROUP_BUILDERS,
    build_features,
    build_features_from_cache,
    group_enabled,
    load_master,
    write_features,
    write_features_manifest,
)


# -----------------------------------------------------------------------------
# Fixtures.
# -----------------------------------------------------------------------------
@pytest.fixture
def loguru_messages():
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _master(n: int = 15) -> pd.DataFrame:
    """A master-like frame: meta / target / level / consensus / companions / regime."""
    idx = list(range(n))
    return pd.DataFrame(
        {
            "meeting_date": pd.date_range("2021-01-01", periods=n, freq="MS"),
            # meeting metadata / target
            "rate_change_bps": [25.0 if i % 3 == 0 else 0.0 for i in idx],
            "new_rate_pct": [1.0 + 0.1 * i for i in idx],
            # level + its consensus (so surprises can activate when enabled)
            "cpi": [100.0 + 1.5 * i for i in idx],
            "cpi_consensus": [100.0 + 1.4 * i for i in idx],
            # companions + regime dummy (must pass through untouched)
            "cpi_age_days": pd.array([20] * n, dtype="Int64"),
            "cpi_is_missing": [0] * n,
            "regime_covid": [1 if 2 <= i <= 6 else 0 for i in idx],
        }
    )


def _config(**overrides) -> dict:
    """A features.yaml-shaped config with every group off, then overrides applied."""
    base = {
        "levels": {"enabled": True},
        "regime": {"enabled": True},
        "lags": {"enabled": False},
        "rolling": {"enabled": False},
        "changes": {"enabled": False},
        "surprises": {"enabled": False},
        "target_lags": {"enabled": False},
        "text": {"enabled": False},
    }
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# group_enabled — the generic gate.
# -----------------------------------------------------------------------------
def test_group_enabled_reads_flag_generically() -> None:
    config = _config(lags={"enabled": True})
    assert group_enabled(config, "lags") is True
    assert group_enabled(config, "rolling") is False
    assert group_enabled(config, "missing_group") is False
    assert group_enabled({}, "lags") is False


# -----------------------------------------------------------------------------
# Enabled groups contribute their expected columns.
# -----------------------------------------------------------------------------
def test_each_enabled_group_contributes_columns() -> None:
    config = _config(
        lags={"enabled": True, "horizons_meetings": [1, 2], "monthly_columns": ["cpi"]},
        rolling={"enabled": True, "windows_meetings": [3], "stats": ["mean"], "monthly_columns": ["cpi"]},
        changes={
            "enabled": True,
            "horizons_meetings": [1],
            "abs_diff": {"columns": ["cpi"]},
            "pct_change": {"columns": []},
            "yoy": {"columns": []},
        },
        target_lags={"enabled": True, "horizons_meetings": [1]},
    )
    out = build_features(_master(), config=config)

    for col in (
        "cpi_lag_1",
        "cpi_lag_2",
        "cpi_roll_3_mean",
        "cpi_diff_1",
        "new_rate_pct_lag_1",
        "rate_change_bps_lag_1",
    ):
        assert col in out.columns
    # Master columns are all retained (feature frame = master + derived).
    for col in _master().columns:
        assert col in out.columns
    assert len(out) == len(_master())


def test_disabled_group_contributes_nothing() -> None:
    """surprises is off — no ``_surprise`` column even with a consensus pair present."""
    config = _config(
        surprises={"enabled": False, "pairs": {"cpi": "cpi_consensus"}},
        target_lags={"enabled": True, "horizons_meetings": [1]},
    )
    out = build_features(_master(), config=config)
    assert not any(c.endswith("_surprise") for c in out.columns)


def test_enabled_surprises_activates_when_consensus_present() -> None:
    """Gating works both ways: enabling surprises with a present pair adds the column."""
    config = _config(surprises={"enabled": True, "pairs": {"cpi": "cpi_consensus"}})
    out = build_features(_master(), config=config)
    assert "cpi_surprise" in out.columns
    # cpi (100 + 1.5i) − cpi_consensus (100 + 1.4i) == 0.1i
    assert out.loc[5, "cpi_surprise"] == pytest.approx(0.5)


def test_passthrough_groups_untouched() -> None:
    """levels / regime are already in the master — passed through unchanged."""
    master = _master()
    out = build_features(master, config=_config())
    pd.testing.assert_series_equal(out["regime_covid"], master["regime_covid"])
    pd.testing.assert_series_equal(out["cpi"], master["cpi"])
    # Nothing enabled → the feature frame equals the master frame exactly.
    pd.testing.assert_frame_equal(out, master.reset_index(drop=True))


def test_deferred_text_logs_and_adds_no_columns(loguru_messages: list) -> None:
    config = _config(text={"enabled": True})
    out = build_features(_master(), config=config)
    assert list(out.columns) == list(_master().columns)  # nothing added
    assert any("text" in m and "deferred" in m for m in loguru_messages)


# -----------------------------------------------------------------------------
# Collision guard.
# -----------------------------------------------------------------------------
def test_column_collision_raises() -> None:
    """A builder re-emitting an existing column raises (mirrors build_master)."""
    master = _master()
    master["cpi_lag_1"] = 0.0  # pre-existing column a lag builder would re-emit
    config = _config(lags={"enabled": True, "horizons_meetings": [1], "monthly_columns": ["cpi"]})
    with pytest.raises(ValueError, match="already present"):
        build_features(master, config=config)


# -----------------------------------------------------------------------------
# Graceful warn-skip of absent inputs.
# -----------------------------------------------------------------------------
def test_absent_configured_column_warn_skips(loguru_messages: list) -> None:
    """A lag column not in the frame is warn-skipped by the builder — no crash."""
    config = _config(lags={"enabled": True, "horizons_meetings": [1], "monthly_columns": ["not_a_column"]})
    out = build_features(_master(), config=config)
    assert list(out.columns) == list(_master().columns)  # nothing added
    assert any("not_a_column" in m and "absent" in m for m in loguru_messages)


# -----------------------------------------------------------------------------
# Config-injection + determinism.
# -----------------------------------------------------------------------------
def test_config_injection_selects_groups() -> None:
    """Only the injected-enabled group contributes (disk config not consulted)."""
    config = _config(changes={
        "enabled": True,
        "horizons_meetings": [1],
        "abs_diff": {"columns": ["cpi"]},
        "pct_change": {"columns": []},
        "yoy": {"columns": []},
    })
    out = build_features(_master(), config=config)
    assert "cpi_diff_1" in out.columns
    assert not any(c.endswith("_lag_1") for c in out.columns)  # lags stayed off


def test_determinism_same_input_same_output() -> None:
    config = _config(
        lags={"enabled": True, "horizons_meetings": [1, 2], "monthly_columns": ["cpi"]},
        target_lags={"enabled": True, "horizons_meetings": [1]},
    )
    master = _master()
    out_a = build_features(master, config=config)
    out_b = build_features(master, config=config)
    pd.testing.assert_frame_equal(out_a, out_b)


def test_unsorted_master_is_sorted() -> None:
    master = _master(10)
    shuffled = master.iloc[[9, 0, 5, 2, 7, 1, 8, 3, 6, 4]].reset_index(drop=True)
    out = build_features(shuffled, config=_config())
    assert out["meeting_date"].is_monotonic_increasing


# -----------------------------------------------------------------------------
# Disk orchestration.
# -----------------------------------------------------------------------------
def test_load_master_reads_parquet(tmp_path: Path) -> None:
    master = _master()
    path = tmp_path / "master.parquet"
    master.to_parquet(path, index=False)
    loaded = load_master(path)
    pd.testing.assert_frame_equal(loaded, master)


def test_load_master_falls_back_when_absent(tmp_path: Path, monkeypatch) -> None:
    sentinel = _master(3)
    monkeypatch.setattr("rba.features.build.align.build_master_from_cache", lambda: sentinel)
    loaded = load_master(tmp_path / "missing.parquet")
    pd.testing.assert_frame_equal(loaded, sentinel)


def test_build_from_cache_uses_master_path_and_config(tmp_path: Path) -> None:
    master = _master()
    path = tmp_path / "master.parquet"
    master.to_parquet(path, index=False)
    config = _config(target_lags={"enabled": True, "horizons_meetings": [1]})
    out = build_features_from_cache(config=config, master_path=path)
    assert "new_rate_pct_lag_1" in out.columns


def test_write_features_and_manifest_roundtrip(tmp_path: Path) -> None:
    config = _config(target_lags={"enabled": True, "horizons_meetings": [1]})
    features = build_features(_master(), config=config)
    parquet_path = tmp_path / "features.parquet"
    meta_path = tmp_path / "features.meta.json"

    digest = write_features(features, parquet_path)
    manifest = write_features_manifest(features, config, parquet_path, digest, meta_path)

    assert parquet_path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), features)
    assert manifest["sha256"] == digest
    assert manifest["n_meetings"] == len(features)
    assert manifest["enabled_groups"] == ["target_lags"]
    # Feature-version reproducibility fields (rba.features.versioning).
    from rba.features import versioning

    assert manifest["config_hash"] == versioning.hash_config(config)
    assert manifest["code_hash"] == versioning.hash_code()
    assert manifest["feature_version_hash"] == versioning.combine(
        manifest["config_hash"], manifest["code_hash"]
    )
    on_disk = json.loads(meta_path.read_text())
    assert on_disk["sha256"] == digest
    assert on_disk["feature_version_hash"] == manifest["feature_version_hash"]


def test_manifest_warns_on_pinned_version_drift(tmp_path: Path, loguru_messages: list) -> None:
    config = _config(pipeline={"feature_version_hash": "a_stale_pinned_hash"})
    features = build_features(_master(), config=config)
    write_features_manifest(
        features, config, tmp_path / "features.parquet", "digest", tmp_path / "features.meta.json"
    )
    assert any("drift" in m for m in loguru_messages)


def test_group_builders_registry_covers_expected_groups() -> None:
    assert set(GROUP_BUILDERS) == {"lags", "rolling", "changes", "surprises", "target_lags"}
