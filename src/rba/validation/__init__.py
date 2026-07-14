"""Validation harness — walk-forward CV, metrics, calibration.

The §7 evaluation harness lives here. :class:`WalkForwardSplit` is the canonical
time-series splitter (see CONTEXT.md Invariant #2); metric / calibration modules
land alongside it as they're implemented.
"""

from __future__ import annotations

from rba.validation.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    plot_all_classes,
    plot_reliability_diagram,
    reliability_diagram_data,
)
from rba.validation.cv import WalkForwardSplit
from rba.validation.harness import EvaluationResult, evaluate
from rba.validation.metrics import (
    accuracy,
    balanced_accuracy,
    brier_score,
    compute_classification_metrics,
    compute_regression_metrics,
    confusion_matrix,
    directional_accuracy,
    hit_rate_vs_market,
    log_loss,
    macro_f1,
    mae,
    r2,
    rmse,
)

__all__ = [
    "EvaluationResult",
    "WalkForwardSplit",
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "confusion_matrix",
    "directional_accuracy",
    "evaluate",
    "expected_calibration_error",
    "hit_rate_vs_market",
    "log_loss",
    "macro_f1",
    "mae",
    "maximum_calibration_error",
    "plot_all_classes",
    "plot_reliability_diagram",
    "r2",
    "reliability_diagram_data",
    "rmse",
]
