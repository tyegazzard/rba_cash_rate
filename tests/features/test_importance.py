"""Tests for ``rba.features.importance`` — the feature-importance report.

No network — a synthetic feature frame is built inline. Covers target derivation,
candidate selection (targets/metadata excluded), the MI + correlation core
(signal ranks above noise, constant / sparse handling, determinism), grouping,
and the Markdown/CSV report writer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from rba.features.importance import (
    DEFAULT_MIN_OBS_FOR_MI,
    candidate_features,
    derive_targets,
    feature_importance,
    load_features,
    render_markdown,
    write_importance_report,
)


def _cell(table: pd.DataFrame, feature: str, column: str) -> float:
    """One numeric cell of the importance table as a plain ``float``.

    ``DataFrame.loc[label, label]`` is typed as pandas' broad scalar union
    (``str | date | complex | …``), which supports none of ``>`` / ``abs`` /
    ``math.isnan``. The importance columns (``pearson_r`` / ``spearman_r`` /
    ``mutual_info``) are float-valued by construction, so narrowing to ``float``
    is provably correct — the cast is a no-op at runtime.
    """
    return cast(float, table.loc[feature, column])


# -----------------------------------------------------------------------------
# Fixtures.
# -----------------------------------------------------------------------------
def _features(n: int = 60) -> pd.DataFrame:
    """A feature-frame-shaped fixture: targets, metadata, and varied features."""
    rng = np.random.default_rng(12)
    idx = np.arange(n)
    # A 3-class Δrate pattern (cut / hold / hike all well represented).
    change = np.where(idx % 5 == 0, -25.0, np.where(idx % 5 == 2, 25.0, 0.0))
    frame = pd.DataFrame(
        {
            "meeting_date": pd.date_range("2015-01-01", periods=n, freq="MS"),
            "effective_date": pd.date_range("2015-01-02", periods=n, freq="MS"),
            "statement_url": [f"/s/{i}" for i in idx],
            # Targets / outcomes (must be excluded as features).
            "rate_change_bps": change,
            "new_rate_pct": 2.0 + 0.25 * np.cumsum(change) / 100.0,
            # Legitimate features.
            "prior_rate_pct": 2.0,
            "signal": change + rng.normal(scale=1.0, size=n),  # strongly related
            "noise": rng.normal(size=n),  # unrelated
            "cpi": 100.0 + rng.normal(size=n),  # unrelated level, with NaNs below
            "const": 1.0,  # zero-variance
            "regime_covid": np.where((idx >= 10) & (idx <= 20), 1, 0),
            "cpi_is_missing": np.where(idx % 7 == 0, 1, 0),
            "gap_days_since_last_meeting": pd.array([30] * n, dtype="Int64"),
        }
    )
    frame.loc[:4, "cpi"] = np.nan  # 5 NaNs → n_obs 55
    # A sparse feature: only 10 non-NaN observations (below the MI floor).
    sparse = np.full(n, np.nan)
    sparse[:10] = rng.normal(size=10)
    frame["sparse"] = sparse
    return frame


# -----------------------------------------------------------------------------
# Target derivation.
# -----------------------------------------------------------------------------
def test_derive_targets_sign_encoding() -> None:
    frame = pd.DataFrame({"rate_change_bps": [25.0, 0.0, -25.0, np.nan]})
    targets = derive_targets(frame)
    assert targets["change_bps"].tolist()[:3] == [25.0, 0.0, -25.0]
    assert targets["direction"].tolist()[:3] == [1, 0, -1]
    assert pd.isna(targets["direction"].iloc[3])  # NaN Δ → undefined direction


# -----------------------------------------------------------------------------
# Candidate selection.
# -----------------------------------------------------------------------------
def test_candidate_features_excludes_targets_and_metadata() -> None:
    cols = candidate_features(_features())
    for excluded in ("meeting_date", "effective_date", "statement_url", "rate_change_bps", "new_rate_pct"):
        assert excluded not in cols
    for included in ("prior_rate_pct", "signal", "noise", "cpi", "regime_covid", "cpi_is_missing"):
        assert included in cols


# -----------------------------------------------------------------------------
# Core: signal vs noise, constant, sparse, determinism.
# -----------------------------------------------------------------------------
def test_signal_ranks_above_noise() -> None:
    table = feature_importance(_features())
    imp = table.set_index("feature")
    assert abs(_cell(imp, "signal", "pearson_r")) > abs(_cell(imp, "noise", "pearson_r"))
    assert _cell(imp, "signal", "mutual_info") > _cell(imp, "noise", "mutual_info")
    # The table is sorted by mutual information descending.
    mi = table["mutual_info"].dropna().to_numpy()
    assert np.all(np.diff(mi) <= 1e-12)


def test_target_columns_never_appear_as_features() -> None:
    table = feature_importance(_features())
    assert "rate_change_bps" not in table["feature"].tolist()
    assert "new_rate_pct" not in table["feature"].tolist()


def test_n_obs_reflects_non_nan() -> None:
    table = feature_importance(_features()).set_index("feature")
    assert table.loc["cpi", "n_obs"] == 55  # 5 leading NaNs
    assert table.loc["signal", "n_obs"] == 60


def test_constant_feature_zero_mi_nan_corr() -> None:
    table = feature_importance(_features()).set_index("feature")
    assert table.loc["const", "mutual_info"] == 0.0
    assert math.isnan(_cell(table, "const", "pearson_r"))


def test_sparse_feature_below_mi_floor_is_nan() -> None:
    table = feature_importance(_features()).set_index("feature")
    assert table.loc["sparse", "n_obs"] == 10  # below DEFAULT_MIN_OBS_FOR_MI
    assert math.isnan(_cell(table, "sparse", "mutual_info"))
    assert not math.isnan(_cell(table, "sparse", "pearson_r"))  # corr still computed


def test_determinism_same_input_same_table() -> None:
    a = feature_importance(_features())
    b = feature_importance(_features())
    pd.testing.assert_frame_equal(a, b)


def test_group_labels() -> None:
    frame = _features()
    frame["cpi_lag_1"] = frame["cpi"]
    frame["cpi_roll_3_mean"] = frame["cpi"]
    frame["cpi_yoy_pct"] = frame["cpi"]
    frame["cpi_age_days"] = pd.array([10] * len(frame), dtype="Int64")
    table = feature_importance(frame).set_index("feature")
    assert table.loc["regime_covid", "group"] == "regime"
    assert table.loc["cpi_is_missing", "group"] == "missing_flag"
    assert table.loc["cpi_lag_1", "group"] == "lag"
    assert table.loc["cpi_roll_3_mean", "group"] == "rolling"
    assert table.loc["cpi_yoy_pct", "group"] == "change"
    assert table.loc["cpi_age_days", "group"] == "staleness"
    assert table.loc["cpi", "group"] == "level"


def test_empty_when_no_target_rows() -> None:
    frame = _features()
    frame["rate_change_bps"] = np.nan
    table = feature_importance(frame)
    assert table.empty
    assert list(table.columns) == ["feature", "group", "n_obs", "pearson_r", "spearman_r", "mutual_info"]


def test_min_obs_override_enables_mi() -> None:
    table = feature_importance(_features(), min_obs_for_mi=5).set_index("feature")
    assert not math.isnan(_cell(table, "sparse", "mutual_info"))  # now above the lowered floor


# -----------------------------------------------------------------------------
# Report writer.
# -----------------------------------------------------------------------------
def _meta(table: pd.DataFrame) -> dict[str, object]:
    return {
        "n_features": len(table),
        "n_meetings": 60,
        "feature_version_hash": "abc123",
        "git_commit": "deadbeef",
    }


def test_render_markdown_deterministic_and_has_top_feature() -> None:
    table = feature_importance(_features())
    meta = _meta(table)
    md_a = render_markdown(table, meta, top_n=5)
    md_b = render_markdown(table, meta, top_n=5)
    assert md_a == md_b  # no timestamp → byte-identical
    assert "# Feature importance report" in md_a
    assert "abc123" in md_a  # provenance line
    assert f"`{table.iloc[0]['feature']}`" in md_a  # top feature listed


def test_write_report_creates_md_and_full_csv(tmp_path: Path) -> None:
    table = feature_importance(_features())
    md_path = tmp_path / "feature_importance.md"
    csv_path = tmp_path / "feature_importance.csv"
    write_importance_report(table, _meta(table), md_path=md_path, csv_path=csv_path, top_n=5)

    assert md_path.exists() and csv_path.exists()
    on_disk = pd.read_csv(csv_path)
    assert len(on_disk) == len(table)  # CSV holds ALL features, not just top-N
    assert "# Feature importance report" in md_path.read_text(encoding="utf-8")


def test_load_features_reads_parquet(tmp_path: Path) -> None:
    frame = _features()
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    loaded = load_features(path)
    assert len(loaded) == len(frame)


def test_default_mi_floor_constant() -> None:
    assert DEFAULT_MIN_OBS_FOR_MI == 30
