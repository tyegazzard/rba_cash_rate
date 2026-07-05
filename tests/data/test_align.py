"""Tests for ``rba.data.align`` orchestration: regime dummies + master IO.

No network. Point-in-time join correctness lives in ``test_no_leakage.py``; this
file covers the regime-indicator feature, the master-frame write/hash, and the
read-only loader's graceful behaviour on an empty raw tree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from rba.data import align
from rba.data.align import (
    add_regime_dummies,
    build_master,
    build_meeting_frame,
    load_meeting_frame,
    load_source_frames,
    write_master,
    write_master_manifest,
)


def _meeting_frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"meeting_date": pd.to_datetime(dates)})


# -----------------------------------------------------------------------------
# Regime dummies.
# -----------------------------------------------------------------------------
def test_governor_eras_partition_every_meeting() -> None:
    """Exactly one Governor dummy is set per meeting across the full history."""
    frame = _meeting_frame(
        ["1994-06-07", "2000-05-02", "2010-11-02", "2020-06-02", "2025-02-18"]
    )
    out = add_regime_dummies(frame)
    gov_cols = [c for c in out.columns if c.startswith("regime_gov_")]
    assert (out[gov_cols].sum(axis=1) == 1).all()


def test_governor_boundary_is_successor_first_day() -> None:
    """The transition date belongs to the incoming Governor (inclusive start)."""
    frame = _meeting_frame(["2006-09-17", "2006-09-18"])
    out = add_regime_dummies(frame)
    assert out.loc[0, "regime_gov_macfarlane"] == 1
    assert out.loc[0, "regime_gov_stevens"] == 0
    assert out.loc[1, "regime_gov_stevens"] == 1
    assert out.loc[1, "regime_gov_macfarlane"] == 0


def test_crisis_and_cadence_windows() -> None:
    frame = _meeting_frame(
        [
            "2007-01-01",  # pre-GFC
            "2008-11-04",  # GFC
            "2020-04-07",  # COVID + forward guidance
            "2021-12-07",  # COVID, forward guidance already ended (2021-11-02)
            "2024-02-06",  # post-2024 cadence
        ]
    )
    out = add_regime_dummies(frame).set_index(frame["meeting_date"])
    assert out["regime_gfc"].tolist() == [0, 1, 0, 0, 0]
    assert out["regime_covid"].tolist() == [0, 0, 1, 1, 0]
    assert out["regime_forward_guidance"].tolist() == [0, 0, 1, 0, 0]
    assert out["regime_post2024_cadence"].tolist() == [0, 0, 0, 0, 1]


def test_regime_dummies_do_not_mutate_input() -> None:
    frame = _meeting_frame(["2020-06-02"])
    original_cols = list(frame.columns)
    add_regime_dummies(frame)
    assert list(frame.columns) == original_cols


# -----------------------------------------------------------------------------
# master.parquet write + hash.
# -----------------------------------------------------------------------------
def test_write_master_hashes_written_bytes(tmp_path: Path) -> None:
    master = pd.DataFrame(
        {"meeting_date": pd.to_datetime(["2020-01-01", "2020-02-01"]), "x": [1.0, 2.0]}
    )
    path = tmp_path / "master.parquet"
    digest = write_master(master, path=path)

    assert path.exists()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(pd.read_parquet(path), master)


# -----------------------------------------------------------------------------
# Read-only loader — graceful on an empty raw tree (no network).
# -----------------------------------------------------------------------------
def test_load_source_frames_skips_missing_sources(tmp_path: Path) -> None:
    """An empty raw tree yields no frames and raises nothing (all skipped)."""
    frames = load_source_frames(raw_root=tmp_path)
    assert frames == {}


def test_load_meeting_frame_raises_without_f11_cache(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rba_f11"):
        load_meeting_frame(raw_root=tmp_path)


# -----------------------------------------------------------------------------
# Compose: regime dummies + as-of levels coexist in one master frame.
# -----------------------------------------------------------------------------
def test_master_carries_regime_and_level_columns() -> None:
    f11 = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2020-06-03", "2024-02-07"]),
            "publication_date": pd.to_datetime(["2020-06-02", "2024-02-06"]),
            "change_raw": ["0.00", "+0.25"],
            "new_cash_rate_raw": ["0.25", "4.35"],
            "statement_url": ["/a", "/b"],
            "minutes_url": ["/m1", "/m2"],
        }
    )
    meeting_frame = add_regime_dummies(build_meeting_frame(f11))
    long_df = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2020-01-31"]),
            "publication_date": pd.to_datetime(["2020-02-20"]),
            "series_id": "demo",
            "value": [3.3],
        }
    )
    master = build_master(meeting_frame, {"src": long_df})

    assert "regime_covid" in master.columns
    assert "demo" in master.columns and "demo_age_days" in master.columns
    # 2020-06 meeting is COVID-era and sees the Feb-2020 reading; 2024 meeting not.
    assert master.loc[0, "regime_covid"] == 1
    assert master.loc[0, "demo"] == 3.3


def test_module_exposes_master_path() -> None:
    assert align.MASTER_PARQUET_PATH.name == "master.parquet"


def test_master_manifest_records_provenance(tmp_path: Path) -> None:
    master = pd.DataFrame(
        {
            "meeting_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "x": [1.0, 2.0],
        }
    )
    parquet_path = tmp_path / "master.parquet"
    digest = write_master(master, path=parquet_path)
    meta_path = tmp_path / "master.meta.json"
    manifest = write_master_manifest(master, parquet_path, digest, path=meta_path)

    assert meta_path.exists()
    assert manifest["sha256"] == digest
    assert manifest["n_meetings"] == 2
    assert manifest["n_columns"] == 2
    assert manifest["first_meeting"] == "2020-01-01"
    assert manifest["last_meeting"] == "2020-02-01"
    # git_commit is best-effort: a 40-hex SHA in this repo, or None elsewhere.
    commit = manifest["git_commit"]
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)
