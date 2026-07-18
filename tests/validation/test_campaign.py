"""Tests for ``rba.validation.campaign`` — the §8 dev-only tuning campaign.

Fast synthetic data, tiny Optuna budgets. Covers: tuning a model produces a
buildable tuned config + per-class threshold weights; a model whose search fails
(empty search space) falls back to the default config with ``tuned=False`` rather
than crashing; the record round-trips through JSON; and ``run_campaign`` writes
one ``<model>.json`` per model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rba.models import build_model
from rba.validation import WalkForwardSplit
from rba.validation import campaign as camp


def _xy(n: int = 70, p: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(12)
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    score = X["f0"] - X["f1"]
    y = pd.Series(np.where(score > 0.5, "hike", np.where(score < -0.5, "cut", "hold")))
    return X, y


def _frame(n: int = 70, p: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    dates = pd.date_range(end="2023-04-01", periods=n, freq="MS")
    rate_change = rng.choice([-25, 0, 25], size=n, p=[0.2, 0.6, 0.2]).astype(float)
    frame = pd.DataFrame({"meeting_date": dates, "rate_change_bps": rate_change})
    for j in range(p):
        frame[f"f{j}"] = rng.normal(size=n)
    return frame


def _splitter() -> WalkForwardSplit:
    return WalkForwardSplit(initial_train_size=30, test_size=1, step=5)


def test_tune_model_produces_buildable_config_and_weights() -> None:
    X, y = _xy()
    record = camp.tune_model(
        "logistic_regression",
        X,
        y,
        splitter=_splitter(),
        n_trials=3,
        timeout=None,
        threshold_trials=20,
    )
    assert record.tuned is True
    assert record.model_name == "logistic_regression"
    # Weights align to the class set and are positive + finite (geomean-1 normalised).
    assert len(record.class_weights) == len(record.classes)
    assert all(w > 0 and np.isfinite(w) for w in record.class_weights)
    # best_model_cfg builds a real model.
    model = build_model(record.best_model_cfg)
    assert model.fit(X, y) is model


def test_tune_model_falls_back_when_search_space_empty() -> None:
    """majority_class has an empty search_space → run_search raises → graceful fallback."""
    X, y = _xy()
    record = camp.tune_model(
        "majority_class",
        X,
        y,
        splitter=_splitter(),
        n_trials=3,
        timeout=None,
        threshold_trials=20,
    )
    assert record.tuned is False
    assert record.n_trials == 0
    assert np.isnan(record.best_value)
    # Thresholds are still tuned on the default config's OOF proba.
    assert len(record.class_weights) == len(record.classes)


def test_tuned_record_round_trips_through_json(tmp_path) -> None:
    X, y = _xy()
    record = camp.tune_model(
        "logistic_regression",
        X,
        y,
        splitter=_splitter(),
        n_trials=2,
        timeout=None,
        threshold_trials=20,
    )
    path = tmp_path / "logistic_regression.json"
    camp.save_tuned(record, path)
    loaded = camp.load_tuned("logistic_regression", tmp_path)
    assert loaded.model_name == record.model_name
    assert loaded.class_weights == record.class_weights
    assert loaded.best_model_cfg == record.best_model_cfg


def test_run_campaign_writes_one_json_per_model(tmp_path) -> None:
    frame = _frame()
    result = camp.run_campaign(
        frame,
        models=["logistic_regression"],
        output_dir=tmp_path,
        initial_train_size=30,
        step=5,
        n_trials=2,
        timeout=None,
    )
    assert len(result.tuned) == 1
    assert (tmp_path / "logistic_regression.json").exists()
    loaded = camp.load_tuned("logistic_regression", tmp_path)
    assert loaded.model_name == "logistic_regression"
