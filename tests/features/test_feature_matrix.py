"""Tests for the canonical leakage-free feature matrix (``rba.features.build``).

The §7 "Feature matrix & leakage guard": proves :func:`feature_columns` /
:func:`feature_matrix` / :func:`build_xy` never admit a contemporaneous outcome
column into ``X`` (the trivial self-leakage that lets a tree recover
``sign(rate_change_bps)`` exactly), that :data:`OUTCOME_COLUMNS` stays in sync
with ``targets.yaml``, and — the regression guard — that a greedy model trained
on the canonical matrix does **not** hit trivially-perfect held-out accuracy,
whereas the same model *would* if an outcome column leaked back in.

No network; a synthetic feature-frame fixture inline. The real-data smoke skips
when ``features.parquet`` has not been materialised (e.g. in CI, where ``data/``
is gitignored).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from rba.config import load_target_config, load_targets_config
from rba.data.targets import build_target
from rba.features.build import (
    FEATURES_PARQUET_PATH,
    OUTCOME_COLUMNS,
    build_xy,
    feature_columns,
    feature_matrix,
)


def _frame(n: int = 150) -> pd.DataFrame:
    """A feature-frame-shaped fixture: metadata, outcomes, and legitimate features.

    ``rate_change_bps`` is drawn i.i.d. among hike / hold / cut so the target is
    balanced and perfectly recoverable *from that column* — the whole point of the
    leakage guard — yet **not** recoverable from any retained feature: the noise
    columns, the constant standing rate, the regime step, and the lag of an i.i.d.
    series all carry zero signal about the current draw. So a model handed the
    canonical (outcome-free) matrix cannot beat chance, while one handed the
    outcome column is perfect.
    """
    rng = np.random.default_rng(12)
    idx = np.arange(n)
    change = rng.choice([25.0, 0.0, -25.0], size=n)
    return pd.DataFrame(
        {
            # Metadata (non-numeric / never a feature).
            "meeting_date": pd.date_range("2010-01-01", periods=n, freq="MS"),
            "effective_date": pd.date_range("2010-01-02", periods=n, freq="MS"),
            "statement_url": [f"/s/{i}" for i in idx],
            "minutes_url": [f"/m/{i}" for i in idx],
            # Contemporaneous outcomes (must be excluded).
            "rate_change_bps": change,
            "new_rate_pct": 3.0 + np.cumsum(change) / 100.0,
            # Legitimate pre-meeting features.
            "prior_rate_pct": 3.0,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "regime_covid": (idx > n // 2).astype(int),
            "rate_change_bps_lag_1": np.roll(change, 1),  # PAST decision — legit
        }
    )


# -----------------------------------------------------------------------------
# feature_columns / feature_matrix — exclusion + retention.
# -----------------------------------------------------------------------------
def test_feature_columns_exclude_outcomes_and_metadata() -> None:
    cols = feature_columns(_frame())
    for excluded in (
        "meeting_date",
        "effective_date",
        "statement_url",
        "minutes_url",
        "rate_change_bps",
        "new_rate_pct",
    ):
        assert excluded not in cols
    for kept in ("prior_rate_pct", "noise1", "noise2", "regime_covid", "rate_change_bps_lag_1"):
        assert kept in cols


def test_feature_matrix_is_numeric_and_outcome_free() -> None:
    fm = feature_matrix(_frame())
    assert list(fm.columns) == feature_columns(_frame())
    assert fm.select_dtypes(exclude="number").shape[1] == 0
    assert OUTCOME_COLUMNS.isdisjoint(fm.columns)


# -----------------------------------------------------------------------------
# build_xy — alignment + NaN-target drop + leakage-free X.
# -----------------------------------------------------------------------------
def test_build_xy_aligns_and_is_leakage_free() -> None:
    X, y = build_xy(_frame(), load_target_config("three_class"))
    assert X.index.equals(y.index)
    assert len(X) == len(y)
    assert OUTCOME_COLUMNS.isdisjoint(X.columns)


def test_build_xy_drops_rows_where_target_is_undefined() -> None:
    frame = _frame()
    frame.loc[0, "rate_change_bps"] = np.nan  # target undefined for this meeting
    X, y = build_xy(frame, load_target_config("three_class"))
    assert len(X) == len(y) == len(frame) - 1
    assert 0 not in y.index and 0 not in X.index


def test_build_xy_int_labels_flow_through() -> None:
    _, y = build_xy(_frame(), load_target_config("three_class"), int_labels=True)
    assert set(np.unique(y)).issubset({-1, 0, 1})


# -----------------------------------------------------------------------------
# OUTCOME_COLUMNS stays in sync with targets.yaml (the single source of truth).
# -----------------------------------------------------------------------------
def test_outcome_columns_match_targets_yaml_source_columns() -> None:
    """Every targets.yaml source column is a contemporaneous outcome we exclude."""
    cfg = load_targets_config()
    declared: set[str] = set()
    for name, entry in cfg.items():
        if name == "default" or not isinstance(entry, dict):
            continue
        declared.update(entry.get("source_columns") or [])
    assert declared == set(OUTCOME_COLUMNS), (
        "targets.yaml source_columns drifted from build.OUTCOME_COLUMNS — update "
        "OUTCOME_COLUMNS (and confirm the new column is truly an outcome to exclude)."
    )


# -----------------------------------------------------------------------------
# The regression guard: no trivially-perfect accuracy on the clean matrix.
# -----------------------------------------------------------------------------
def _holdout_accuracy(X: pd.DataFrame, y: pd.Series, n_test: int = 40) -> float:
    """Greedy DecisionTree, last-``n_test`` holdout — the canonical leak detector."""
    n_train = len(y) - n_test
    model = DecisionTreeClassifier(random_state=12).fit(X.iloc[:n_train], y.iloc[:n_train])
    return float((np.asarray(y.iloc[n_train:]) == model.predict(X.iloc[n_train:])).mean())


def test_clean_matrix_has_no_trivial_accuracy_but_leak_would() -> None:
    """The guard is meaningful: with the outcome column a tree is perfect; without it, not."""
    frame = _frame(n=180)
    y = build_target(frame, load_target_config("three_class"))
    clean = frame.loc[y.index, feature_columns(frame)]
    leaky = frame.loc[y.index, [*feature_columns(frame), "rate_change_bps"]]

    leaky_acc = _holdout_accuracy(leaky, y)
    clean_acc = _holdout_accuracy(clean, y)

    # The leak IS trivially exploitable — this validates the detector itself.
    assert leaky_acc > 0.98, f"expected the leaked column to be recoverable; got {leaky_acc:.3f}"
    # The canonical matrix removes it → no trivially-perfect accuracy.
    assert clean_acc < 0.90, (
        f"clean matrix hit {clean_acc:.3f} — an outcome column may have leaked"
    )


@pytest.mark.skipif(
    not FEATURES_PARQUET_PATH.exists(),
    reason="features.parquet not materialised (data/ is gitignored)",
)
def test_no_trivial_accuracy_on_real_feature_matrix() -> None:
    """Real-data regression guard: the shipped feature matrix carries no outcome leak."""
    frame = pd.read_parquet(FEATURES_PARQUET_PATH)
    y = build_target(frame, load_target_config("three_class"))
    X = frame.loc[y.index, feature_columns(frame)]
    acc = _holdout_accuracy(X, y)
    assert acc < 0.98, (
        f"a greedy tree hit {acc:.3f} held-out accuracy on the 'clean' real feature "
        "matrix — an outcome/target column has likely leaked into feature_columns()."
    )
