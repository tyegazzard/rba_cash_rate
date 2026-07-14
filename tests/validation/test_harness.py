"""Tests for :func:`rba.validation.harness.evaluate` — the §7 wrapper.

No network. Synthetic data throughout — the master frame isn't materialised in
CI. Exercises the classification path (with proba + calibration), the
regression path (Taylor rule against a level target), the market-baseline
comparison path (hit-rate-vs-market populates when a classifier baseline is
provided), the persistence-to-disk contract (manifest.json + predictions.parquet
+ calibration.parquet in ``output_dir/run_id/``), the graceful mlflow degrade
(the current stack is upstream-blocked; harness must not crash), and the
error paths (task mismatch, empty splitter).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rba.validation import EvaluationResult, WalkForwardSplit, evaluate


# =============================================================================
# Fixtures.
# =============================================================================
@pytest.fixture
def classification_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Chronological meeting frame with columns every baseline can read."""
    n = 40
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        {
            "cpi_headline_yoy": rng.uniform(1.5, 4.5, size=n),
            "output_gap": rng.uniform(-1.5, 1.5, size=n),
            "asx_30d_implied_rate": rng.uniform(3.0, 5.0, size=n),
            "prior_rate_pct": rng.uniform(3.0, 5.0, size=n),
        }
    )
    # Slight class imbalance mirroring the RBA reality.
    y = pd.Series(rng.choice(["cut", "hold", "hike"], size=n, p=[0.15, 0.65, 0.20]))
    return X, y


@pytest.fixture
def regression_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame for Taylor-rule regression on a level target."""
    n = 40
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        {
            "cpi_headline_yoy": rng.uniform(1.5, 4.5, size=n),
            "output_gap": rng.uniform(-1.5, 1.5, size=n),
        }
    )
    y = X["cpi_headline_yoy"] * 0.5 + rng.normal(3.0, 0.1, size=n)
    return X, y


# =============================================================================
# Classification path — MajorityClass + persistence.
# =============================================================================
def test_evaluate_majority_class_returns_result(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert isinstance(result, EvaluationResult)
    assert result.model_name == "majority_class"
    assert result.target_name == "three_class"
    assert result.task == "classification"
    assert result.n_folds == 30
    assert result.n_samples_scored == 30


def test_evaluate_writes_manifest_and_predictions_parquet(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.predictions_path is not None and result.predictions_path.exists()
    with result.manifest_path.open() as fh:
        manifest = json.load(fh)
    # Manifest carries the summary but not the heavy payloads.
    assert manifest["model_name"] == "majority_class"
    assert manifest["n_folds"] == 30
    assert "predictions" not in manifest
    assert "calibration" not in manifest
    # Predictions parquet round-trips a row per scored sample.
    preds = pd.read_parquet(result.predictions_path)
    assert len(preds) == 30
    assert {"sample_index", "fold_index", "y_true", "y_pred"} <= set(preds.columns)


def test_evaluate_classification_writes_calibration_parquet_with_probas(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """MajorityClass has predict_proba → per-class reliability data lands on disk."""
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.calibration_path is not None and result.calibration_path.exists()
    calibration = pd.read_parquet(result.calibration_path)
    assert "class_label" in calibration.columns
    assert set(calibration["class_label"].unique()) == {"cut", "hike", "hold"}


def test_evaluate_persistence_baseline_carries_metrics_forward(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """Persistence has classes_ + proba → dispatcher runs the full metric suite."""
    X, y = classification_frame
    result = evaluate(
        model_name="persistence",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "confusion_matrix",
    } <= set(result.metrics_overall)
    # Per-fold metrics carry the fold_index tag.
    assert all("fold_index" in entry for entry in result.metrics_per_fold)


# =============================================================================
# Regression path — TaylorRule.
# =============================================================================
def test_evaluate_taylor_rule_regression_runs_full_pipeline(
    regression_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = regression_frame
    result = evaluate(
        model_name="taylor_rule",
        target_name="level_regression",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.task == "regression"
    assert set(result.metrics_overall) == {"rmse", "mae", "r2", "directional_accuracy"}
    # No calibration for regression targets.
    assert result.calibration is None
    assert result.calibration_path is None
    assert result.hit_rate_vs_market_result is None


def test_evaluate_rejects_market_baseline_against_regression_target(
    regression_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = regression_frame
    with pytest.raises(ValueError, match="market_model_name"):
        evaluate(
            model_name="taylor_rule",
            target_name="level_regression",
            X=X,
            y=y,
            splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
            market_model_name="market_implied",
            output_dir=tmp_path,
            use_mlflow=False,
        )


# =============================================================================
# Market-baseline integration — hit-rate-vs-market populates the manifest.
# =============================================================================
def test_evaluate_with_market_baseline_populates_hit_rate(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        market_model_name="market_implied",
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.hit_rate_vs_market_result is not None
    hitrate = result.hit_rate_vs_market_result
    assert {"model_hit_rate", "market_hit_rate", "lift", "n_compared"} <= set(hitrate)
    assert hitrate["n_compared"] > 0
    # Predictions parquet has the y_pred_market column.
    preds = pd.read_parquet(result.predictions_path)  # type: ignore[arg-type]
    assert "y_pred_market" in preds.columns


def test_evaluate_market_baseline_survives_missing_columns(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """When X lacks the market columns entirely, per-row fallback fills NaN and hit-rate returns NaN."""
    X, y = classification_frame
    X_lean = X.drop(columns=["asx_30d_implied_rate", "prior_rate_pct"])
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X_lean,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        market_model_name="market_implied",
        output_dir=tmp_path,
        use_mlflow=False,
    )
    # Market predictions are all NaN → hit-rate reports NaN metrics + n_compared=0.
    assert result.hit_rate_vs_market_result is not None
    assert result.hit_rate_vs_market_result["n_compared"] == 0.0


# =============================================================================
# MLflow — graceful degrade contract.
# =============================================================================
def test_evaluate_mlflow_import_failure_does_not_crash(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    """Current stack: mlflow 1.27 fails to import on protobuf 7 — must not crash the run."""
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=True,
    )
    # Local persistence unaffected.
    assert result.manifest_path is not None and result.manifest_path.exists()
    # Status string starts with "import_failed" (current state) or "logged" (once upstream unblocks) or "logging_failed".
    assert result.mlflow_status.startswith(("import_failed", "logged", "logging_failed"))


def test_evaluate_mlflow_disabled_by_caller(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.mlflow_status == "disabled_by_caller"


# =============================================================================
# Error paths.
# =============================================================================
def test_evaluate_rejects_len_mismatch(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    with pytest.raises(ValueError, match="len\\(X\\)"):
        evaluate(
            model_name="majority_class",
            target_name="three_class",
            X=X,
            y=y.iloc[:5],
            splitter=WalkForwardSplit(initial_train_size=3, test_size=1),
            output_dir=tmp_path,
            use_mlflow=False,
        )


def test_evaluate_rejects_splitter_yielding_zero_folds(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    with pytest.raises(ValueError, match="0 folds"):
        evaluate(
            model_name="majority_class",
            target_name="three_class",
            X=X,
            y=y,
            splitter=WalkForwardSplit(initial_train_size=len(X) + 10, test_size=1),
            output_dir=tmp_path,
            use_mlflow=False,
        )


# =============================================================================
# Run identifier + reproducibility of persistence layout.
# =============================================================================
def test_evaluate_default_run_id_encodes_model_target_and_timestamp(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.run_id.startswith("majority_class_three_class_")
    # Run folder lives under output_dir.
    assert result.manifest_path is not None
    assert result.manifest_path.parent == tmp_path / result.run_id


def test_evaluate_explicit_run_id_wins(
    classification_frame: tuple[pd.DataFrame, pd.Series], tmp_path: Path
) -> None:
    X, y = classification_frame
    result = evaluate(
        model_name="majority_class",
        target_name="three_class",
        X=X,
        y=y,
        splitter=WalkForwardSplit(initial_train_size=10, test_size=1),
        run_id="custom_run_123",
        output_dir=tmp_path,
        use_mlflow=False,
    )
    assert result.run_id == "custom_run_123"
    assert (tmp_path / "custom_run_123" / "manifest.json").exists()
