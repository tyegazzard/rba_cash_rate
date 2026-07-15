"""Tests for :mod:`rba.validation.metrics` — the §7 classification metrics.

No network; tiny hand-built ``(y_true, y_pred, y_proba)`` fixtures. Covers each
metric's canonical anchors (perfect / all-wrong / imbalanced), the
prediction-vs-probability signature split, the dispatcher's inclusion rules
(skips proba metrics when ``y_proba`` absent, raises when ``y_proba`` given
without ``labels``), and the paired :func:`hit_rate_vs_market` comparator
(both-right / both-wrong / only-model / only-market cells, NaN-in-market
filtering, length-mismatch error). One integration test wires
:class:`~rba.models.baselines.MajorityClass` through :func:`compute_classification_metrics`
so the harness's actual downstream shape is exercised end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from rba.models.baselines import MajorityClass
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


# =============================================================================
# accuracy.
# =============================================================================
def test_accuracy_perfect_predictions() -> None:
    assert accuracy(["cut", "hold", "hike"], ["cut", "hold", "hike"]) == 1.0


def test_accuracy_all_wrong_is_zero() -> None:
    assert accuracy(["cut", "hold"], ["hike", "cut"]) == 0.0


def test_accuracy_partial_correct() -> None:
    y = ["hold", "hold", "hold", "hike"]
    y_hat = ["hold", "hold", "hike", "hike"]
    assert accuracy(y, y_hat) == pytest.approx(0.75)


def test_accuracy_returns_python_float_not_numpy() -> None:
    """Downstream loggers want a primitive ``float`` — no ``numpy.float64``."""
    result = accuracy(["hold"], ["hold"])
    assert type(result) is float


# =============================================================================
# balanced_accuracy — the honest floor under class imbalance.
# =============================================================================
def test_balanced_accuracy_of_predict_hold_is_one_third() -> None:
    """MajorityClass on RBA data scores ~0.77 accuracy but 1/3 balanced accuracy."""
    y_true = ["hold"] * 8 + ["hike", "cut"]
    y_pred = ["hold"] * 10
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(1 / 3)


def test_balanced_accuracy_perfect_predictions_is_one() -> None:
    assert balanced_accuracy(["cut", "hold", "hike"], ["cut", "hold", "hike"]) == 1.0


# =============================================================================
# macro_f1 — cycle-turn signal metric.
# =============================================================================
def test_macro_f1_perfect_predictions_is_one() -> None:
    assert macro_f1(["cut", "hold", "hike"], ["cut", "hold", "hike"]) == 1.0


def test_macro_f1_predict_hold_only_matches_expected() -> None:
    """Predict-hold classifier: hold F1 > 0, hike / cut F1 = 0."""
    y_true = ["hold", "hold", "hold", "hike", "cut"]
    y_pred = ["hold"] * 5
    # hold: precision = 3/5, recall = 1.0 -> F1 = 0.75
    # hike / cut: precision = 0, recall = 0 -> F1 = 0
    # Macro = (0.75 + 0 + 0) / 3 = 0.25
    assert macro_f1(y_true, y_pred, labels=["cut", "hold", "hike"]) == pytest.approx(0.25)


def test_macro_f1_labels_pins_absent_classes_to_zero() -> None:
    """A class listed in labels but absent from y still gets an F1=0 slot in the average."""
    y_true = ["hold", "hike", "hold", "hike"]
    y_pred = ["hold", "hike", "hold", "hike"]
    # Without cut in the mix: perfect on hold + hike → macro = 1.0
    assert macro_f1(y_true, y_pred) == 1.0
    # Passing labels=[cut, hold, hike]: cut F1 = 0, so macro = (0 + 1 + 1)/3 = 2/3.
    assert macro_f1(y_true, y_pred, labels=["cut", "hold", "hike"]) == pytest.approx(2 / 3)


# =============================================================================
# log_loss.
# =============================================================================
def test_log_loss_matches_sklearn_for_hand_crafted_input() -> None:
    """Cross-check against sklearn's raw call — same clipping, same labels handling."""
    y_true = ["hold", "hike", "cut", "hold"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.array(
        [
            [0.05, 0.10, 0.85],  # hold, correct
            [0.05, 0.85, 0.10],  # hike, correct
            [0.85, 0.10, 0.05],  # cut, correct
            [0.10, 0.10, 0.80],  # hold, correct
        ]
    )
    expected = sk_log_loss(y_true, y_proba, labels=labels)
    assert log_loss(y_true, y_proba, labels=labels) == pytest.approx(expected)


def test_log_loss_persistence_style_one_hot_wrong_call_is_bounded() -> None:
    """A one-hot on the wrong class hits sklearn's clip — large but finite, not inf."""
    y_true = ["hike"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.array([[1.0, 0.0, 0.0]])  # wrong one-hot
    result = log_loss(y_true, y_proba, labels=labels)
    assert np.isfinite(result)
    assert result > 30  # sklearn clips at float64 eps; log(eps) ≈ -34.


# =============================================================================
# brier_score.
# =============================================================================
def test_brier_score_perfect_one_hot_is_zero() -> None:
    """One-hot on the correct class: (1 - 1)² + 0² + 0² = 0."""
    y_true = ["hold"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.array([[0.0, 0.0, 1.0]])
    assert brier_score(y_true, y_proba, labels=labels) == 0.0


def test_brier_score_wrong_one_hot_is_two() -> None:
    """One-hot on the wrong class: (1 - 0)² + 0² + (0 - 1)² = 2."""
    y_true = ["hold"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.array([[1.0, 0.0, 0.0]])
    assert brier_score(y_true, y_proba, labels=labels) == 2.0


def test_brier_score_uniform_prior_matches_manual() -> None:
    """Uniform prior (1/3 each) on any true label: (2/3)² + (1/3)² + (1/3)² = 6/9."""
    y_true = ["cut", "hold", "hike"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (3, 1))
    expected = (2 / 3) ** 2 + (1 / 3) ** 2 + (1 / 3) ** 2
    assert brier_score(y_true, y_proba, labels=labels) == pytest.approx(expected)


def test_brier_score_bounded_below_by_zero_above_by_two() -> None:
    """Range invariant across any valid probability distribution."""
    rng = np.random.default_rng(12)
    y_proba = rng.dirichlet(alpha=[1, 1, 1], size=50)
    y_true = rng.choice(["cut", "hold", "hike"], size=50)
    score = brier_score(y_true, y_proba, labels=["cut", "hike", "hold"])
    assert 0.0 <= score <= 2.0


def test_brier_score_rejects_shape_mismatch() -> None:
    y_true = ["hold"]
    y_proba = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="labels"):
        brier_score(y_true, y_proba, labels=["cut", "hike", "hold"])


def test_brier_score_rejects_1d_proba() -> None:
    y_true = ["hold"]
    y_proba = np.array([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="2-D"):
        brier_score(y_true, y_proba, labels=["cut", "hike", "hold"])


# =============================================================================
# confusion_matrix.
# =============================================================================
def test_confusion_matrix_row_is_true_column_is_predicted() -> None:
    """Convention: cm[i, j] = count of true=labels[i] predicted as labels[j]."""
    labels = ["cut", "hike", "hold"]
    y_true = ["cut", "cut", "hold", "hold", "hike"]
    y_pred = ["cut", "hold", "hold", "hike", "hike"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # True=cut (2 rows): predicted cut once, hold once.
    np.testing.assert_array_equal(cm[0], np.array([1, 0, 1]))
    # True=hike (1 row): predicted hike once.
    np.testing.assert_array_equal(cm[1], np.array([0, 1, 0]))
    # True=hold (2 rows): predicted hike once, hold once.
    np.testing.assert_array_equal(cm[2], np.array([0, 1, 1]))


def test_confusion_matrix_labels_pins_shape_across_folds() -> None:
    """A label absent from y_true/y_pred still gets a zero row/column."""
    labels = ["cut", "hike", "hold"]
    cm = confusion_matrix(["hold", "hold"], ["hold", "hold"], labels=labels)
    assert cm.shape == (3, 3)
    # Only the hold-hold cell (index [2, 2]) has count 2, everything else 0.
    assert cm[2, 2] == 2
    assert cm.sum() == 2


# =============================================================================
# compute_classification_metrics — the harness dispatcher.
# =============================================================================
def test_dispatcher_returns_all_metrics_when_proba_and_labels_given() -> None:
    y_true = ["cut", "hold", "hike", "hold"]
    y_pred = ["hold", "hold", "hike", "hold"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.array(
        [
            [0.10, 0.10, 0.80],
            [0.05, 0.10, 0.85],
            [0.05, 0.85, 0.10],
            [0.10, 0.20, 0.70],
        ]
    )
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba, labels=labels)
    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "confusion_matrix",
        "log_loss",
        "brier_score",
    }
    assert isinstance(metrics["confusion_matrix"], np.ndarray)
    assert all(
        isinstance(metrics[k], float)
        for k in ["accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score"]
    )


def test_dispatcher_skips_proba_metrics_when_y_proba_absent() -> None:
    """A classifier without calibrated probabilities can still be scored on the four preds-based metrics."""
    metrics = compute_classification_metrics(
        ["cut", "hold", "hike"],
        ["cut", "hold", "hold"],
        labels=["cut", "hike", "hold"],
    )
    assert set(metrics) == {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"}


def test_dispatcher_raises_when_proba_given_without_labels() -> None:
    """``labels`` is required whenever ``y_proba`` is provided (it pins the column order)."""
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="labels"):
        compute_classification_metrics(["a", "b"], ["a", "b"], y_proba=y_proba)


# =============================================================================
# End-to-end: MajorityClass through the dispatcher.
# =============================================================================
def test_dispatcher_end_to_end_with_majority_class() -> None:
    """The natural §7 harness call shape: model.classes_ → labels; predict / predict_proba → inputs."""
    X = pd.DataFrame({"f0": np.arange(10, dtype=float)})
    y_train = pd.Series(["hold"] * 7 + ["hike"] * 2 + ["cut"] * 1)
    model = MajorityClass().fit(X, y_train)

    X_test = pd.DataFrame({"f0": np.arange(4, dtype=float)})
    y_test = pd.Series(["hold", "hold", "hike", "cut"])
    metrics = compute_classification_metrics(
        y_test,
        model.predict(X_test),
        y_proba=model.predict_proba(X_test),
        labels=model.classes_,
    )
    # MajorityClass predicts "hold" for every row — 2/4 correct.
    assert metrics["accuracy"] == pytest.approx(0.5)
    # Predict-hold classifier's balanced accuracy is 1/3 (one class perfect, two class zero).
    assert metrics["balanced_accuracy"] == pytest.approx(1 / 3)
    # Log-loss / Brier are finite and non-negative.
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["brier_score"] <= 2
    # Confusion matrix follows model.classes_ order: [cut, hike, hold].
    assert metrics["confusion_matrix"].shape == (3, 3)


# =============================================================================
# hit_rate_vs_market — paired accuracy against MarketImplied.
# =============================================================================
def test_hit_rate_vs_market_model_beats_market_completely() -> None:
    """Every meeting: model right, market wrong → lift 1, only_model_correct 1."""
    y_true = ["cut", "hold", "hike"]
    y_model = ["cut", "hold", "hike"]
    y_market = ["hold", "hike", "cut"]  # every one wrong
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["model_hit_rate"] == pytest.approx(1.0)
    assert result["market_hit_rate"] == pytest.approx(0.0)
    assert result["lift"] == pytest.approx(1.0)
    assert result["only_model_correct"] == pytest.approx(1.0)
    assert result["only_market_correct"] == pytest.approx(0.0)
    assert result["both_correct"] == pytest.approx(0.0)
    assert result["both_wrong"] == pytest.approx(0.0)
    assert result["n_compared"] == pytest.approx(3.0)


def test_hit_rate_vs_market_both_right_gives_zero_lift() -> None:
    """Meetings where both signals were right — lift is zero, both_correct is one."""
    y_true = ["cut", "hold", "hike"]
    y_model = ["cut", "hold", "hike"]
    y_market = ["cut", "hold", "hike"]
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["lift"] == 0.0
    assert result["both_correct"] == 1.0


def test_hit_rate_vs_market_both_wrong_signals_hard_meetings() -> None:
    """Genuine surprises — both signals miss. Neither is dominated."""
    y_true = ["hike", "hike", "hike"]
    y_model = ["hold", "hold", "hold"]
    y_market = ["cut", "cut", "cut"]
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["both_wrong"] == 1.0
    assert result["lift"] == 0.0


def test_hit_rate_vs_market_paired_disagreement_breakdown_sums_to_one() -> None:
    """The four disagreement cells always partition the sample space."""
    y_true = ["hike", "hike", "hold", "hold", "cut", "cut", "hold", "hike"]
    y_model = ["hike", "hold", "hold", "hike", "cut", "hold", "hold", "cut"]
    y_market = ["cut", "hike", "hold", "hold", "hike", "cut", "hike", "hike"]
    result = hit_rate_vs_market(y_true, y_model, y_market)
    total = (
        result["both_correct"]
        + result["both_wrong"]
        + result["only_model_correct"]
        + result["only_market_correct"]
    )
    assert total == pytest.approx(1.0)


def test_hit_rate_vs_market_manually_computed_mixed_case() -> None:
    """Hand-computed anchor: 4 meetings, one per disagreement quadrant."""
    y_true = ["hike", "hike", "hike", "hike"]
    y_model = ["hike", "hike", "hold", "hold"]  # right, right, wrong, wrong
    y_market = ["hike", "hold", "hike", "hold"]  # right, wrong, right, wrong
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["both_correct"] == 0.25
    assert result["only_model_correct"] == 0.25
    assert result["only_market_correct"] == 0.25
    assert result["both_wrong"] == 0.25
    assert result["model_hit_rate"] == 0.5
    assert result["market_hit_rate"] == 0.5
    assert result["lift"] == 0.0


def test_hit_rate_vs_market_drops_rows_where_market_is_nan() -> None:
    """Pre-2022 meetings without market coverage drop from the paired comparison."""
    y_true = ["cut", "hold", "hike", "hold"]
    y_model = ["cut", "hold", "hike", "cut"]  # 3/4 right on all 4 rows
    # dtype=object keeps the NaNs as floats (a plain list mixes float+str →
    # list[object], which is not ArrayLike; an object array is and preserves NaN).
    y_market = np.array([np.nan, np.nan, "hike", "hold"], dtype=object)  # only last 2 rows covered
    result = hit_rate_vs_market(y_true, y_model, y_market)
    # Only rows 2 & 3 survive. Model correct on row 2, wrong on row 3 → 1/2.
    # Market correct on row 2, correct on row 3 → 2/2.
    assert result["n_compared"] == 2.0
    assert result["model_hit_rate"] == 0.5
    assert result["market_hit_rate"] == 1.0
    assert result["lift"] == -0.5


def test_hit_rate_vs_market_all_market_nan_returns_nan_metrics() -> None:
    """No paired rows survive → every metric is NaN, n_compared is 0."""
    y_true = ["cut", "hold"]
    y_model = ["cut", "hold"]
    y_market = [np.nan, np.nan]
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["n_compared"] == 0.0
    for key in (
        "model_hit_rate",
        "market_hit_rate",
        "lift",
        "both_correct",
        "both_wrong",
        "only_model_correct",
        "only_market_correct",
    ):
        assert np.isnan(result[key]), f"{key} should be NaN when no rows survive."


def test_hit_rate_vs_market_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        hit_rate_vs_market(["cut", "hold"], ["cut"], ["cut", "hold"])


def test_hit_rate_vs_market_int_labels_work_too() -> None:
    """The int-encoded three_class target (``class_to_int``) should behave the same."""
    y_true = [-1, 0, 1]
    y_model = [-1, 0, 1]
    y_market = [0, 0, 0]  # always predicts hold → right once (middle row)
    result = hit_rate_vs_market(y_true, y_model, y_market)
    assert result["model_hit_rate"] == pytest.approx(1.0)
    assert result["market_hit_rate"] == pytest.approx(1 / 3)
    assert result["lift"] == pytest.approx(2 / 3)


def test_hit_rate_vs_market_returns_python_floats() -> None:
    """Downstream loggers want ``float``, not ``numpy.float64``."""
    result = hit_rate_vs_market(["cut", "hold"], ["cut", "hold"], ["cut", "hold"])
    for key, value in result.items():
        assert type(value) is float, f"{key} is {type(value).__name__}, expected float."


def test_hit_rate_vs_market_ignores_row_index_uses_positional_alignment() -> None:
    """Pandas Series with a wonky index shouldn't break positional alignment."""
    y_true = pd.Series(["cut", "hold", "hike"], index=[10, 20, 30])
    y_model = pd.Series(["cut", "hold", "hike"], index=[100, 200, 300])
    y_market = pd.Series(["cut", "hold", "hike"], index=[0, 1, 2])
    result = hit_rate_vs_market(y_true, y_model, y_market)
    # Positional: all three arrays match at every position → both_correct = 1.
    assert result["both_correct"] == 1.0


# =============================================================================
# Regression metrics — rmse / mae / r2 / directional_accuracy.
# =============================================================================
def test_rmse_perfect_predictions_is_zero() -> None:
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_matches_hand_computed_value() -> None:
    """Residuals [1, -1, 2] → RMSE = sqrt(mean(1 + 1 + 4)) = sqrt(2)."""
    assert rmse([1.0, 2.0, 3.0], [2.0, 1.0, 5.0]) == pytest.approx(np.sqrt(2.0))


def test_mae_matches_hand_computed_value() -> None:
    """Absolute residuals [1, 1, 2] → MAE = 4/3."""
    assert mae([1.0, 2.0, 3.0], [2.0, 1.0, 5.0]) == pytest.approx(4 / 3)


def test_r2_perfect_predictions_is_one() -> None:
    assert r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_r2_predict_mean_is_zero() -> None:
    """Always predicting the training-set mean → R² = 0."""
    y = [1.0, 2.0, 3.0, 4.0]
    y_hat = [2.5, 2.5, 2.5, 2.5]  # mean of y
    assert r2(y, y_hat) == pytest.approx(0.0)


def test_r2_can_go_negative_when_worse_than_mean() -> None:
    """A model worse than "predict the mean" gets a negative R² — that's the honest signal."""
    y = [1.0, 2.0, 3.0]
    y_hat = [3.0, 1.0, 2.0]  # rotated → worse than the mean
    assert r2(y, y_hat) < 0.0


def test_directional_accuracy_all_correct_directions() -> None:
    """Every sign matches → 1.0. Zeros count as correct when both are zero."""
    y = [-25.0, 0.0, 25.0]
    y_hat = [-10.0, 0.0, 40.0]
    assert directional_accuracy(y, y_hat) == 1.0


def test_directional_accuracy_flipped_signs_is_zero() -> None:
    y = [-25.0, 25.0]
    y_hat = [25.0, -25.0]
    assert directional_accuracy(y, y_hat) == 0.0


def test_directional_accuracy_partial() -> None:
    """2/4 signs correct → 0.5."""
    y = [-25.0, 25.0, 0.0, 25.0]
    y_hat = [-10.0, 25.0, 10.0, -25.0]  # right, right, wrong (0 vs +), wrong
    assert directional_accuracy(y, y_hat) == 0.5


def test_compute_regression_metrics_returns_all_four() -> None:
    result = compute_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert set(result) == {"rmse", "mae", "r2", "directional_accuracy"}
    assert all(isinstance(v, float) for v in result.values())
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
