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
    add_missing_indicators,
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
    frame = _meeting_frame(["1994-06-07", "2000-05-02", "2010-11-02", "2020-06-02", "2025-02-18"])
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


# -----------------------------------------------------------------------------
# _is_missing indicators (§4 / Invariant #4).
# -----------------------------------------------------------------------------
def _demo_f11() -> pd.DataFrame:
    """Two-meeting F11 frame (announcements 2020-02-03 and 2020-06-02)."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2020-02-04", "2020-06-03"]),
            "publication_date": pd.to_datetime(["2020-02-03", "2020-06-02"]),
            "change_raw": ["0.00", "-0.25"],
            "new_cash_rate_raw": ["0.75", "0.25"],
            "statement_url": ["/a", "/b"],
            "minutes_url": ["/m1", "/m2"],
        }
    )


def _demo_long() -> pd.DataFrame:
    """One 'demo' observation published 2020-03-20 — visible to 2020-06 only."""
    return pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2020-03-01"]),
            "publication_date": pd.to_datetime(["2020-03-20"]),
            "series_id": "demo",
            "value": [3.3],
        }
    )


def test_is_missing_matches_level_isna() -> None:
    """``<sid>_is_missing`` is exactly the level column's NaN mask, per meeting."""
    master = build_master(build_meeting_frame(_demo_f11()), {"src": _demo_long()})
    assert "demo_is_missing" in master.columns
    assert master["demo_is_missing"].tolist() == master["demo"].isna().astype(int).tolist()
    # Concretely: the 2020-02 meeting predates the reading (missing), 2020-06 sees it.
    assert master.loc[0, "demo_is_missing"] == 1
    assert master.loc[1, "demo_is_missing"] == 0


def test_every_level_column_gets_an_indicator_meta_columns_do_not() -> None:
    """Every ``<sid>`` (identified by its ``_age_days`` companion) gets a flag."""
    meeting_frame = add_regime_dummies(build_meeting_frame(_demo_f11()))
    master = build_master(meeting_frame, {"src": _demo_long()})
    level_ids = [c[: -len("_age_days")] for c in master.columns if c.endswith("_age_days")]
    assert level_ids == ["demo"]
    for sid in level_ids:
        assert f"{sid}_is_missing" in master.columns
    # Meeting-metadata and regime dummies are not level columns → no indicator.
    assert "regime_covid_is_missing" not in master.columns
    assert "meeting_date_is_missing" not in master.columns
    assert "demo_age_days_is_missing" not in master.columns


def test_missing_indicators_are_int_zero_one_flags() -> None:
    master = build_master(build_meeting_frame(_demo_f11()), {"src": _demo_long()})
    flags = master["demo_is_missing"]
    assert pd.api.types.is_integer_dtype(flags)
    assert set(flags.unique()) <= {0, 1}


def test_build_master_column_count_is_meta_plus_three_per_series() -> None:
    """Each series contributes exactly level + _age_days + _is_missing."""
    meeting_frame = build_meeting_frame(_demo_f11())
    n_in = meeting_frame.shape[1]
    src_a = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2019-01-01"]),
            "publication_date": pd.to_datetime(["2019-01-20"]),
            "series_id": "a_series",
            "value": [1.1],
        }
    )
    src_b = src_a.assign(series_id="b_series", value=[2.2])
    master = build_master(meeting_frame, {"a": src_a, "b": src_b})
    assert master.shape[1] == n_in + 3 * 2


def test_add_missing_indicators_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "meeting_date": pd.to_datetime(["2020-01-01"]),
            "x": [1.0],
            "x_age_days": pd.array([5], dtype="Int64"),
        }
    )
    original_cols = list(df.columns)
    add_missing_indicators(df)
    assert list(df.columns) == original_cols


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
