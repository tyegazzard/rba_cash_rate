"""Tests for ``rba.validation.thresholds`` — per-class decision-weight tuning.

Covers :func:`predict_weighted` (uniform → plain argmax; upweighting flips the
decision), :func:`tune_class_weights` / :class:`ThresholdTuner` (recovers minority
classes an argmax buries, lifting balanced accuracy; rejects a bad metric), and
:func:`collect_oof_proba` (full-width aligned OOF probabilities that sum to 1).

No network; synthetic probabilities inline; Optuna logging silenced.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from rba.validation import WalkForwardSplit
from rba.validation.metrics import balanced_accuracy
from rba.validation.thresholds import (
    ThresholdTuner,
    collect_oof_proba,
    predict_weighted,
    tune_class_weights,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CLASSES = np.array(["cut", "hold", "hike"])


# -----------------------------------------------------------------------------
# predict_weighted.
# -----------------------------------------------------------------------------
def test_uniform_weights_are_plain_argmax() -> None:
    proba = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.1, 0.2, 0.7]])
    preds = predict_weighted(proba, np.ones(3), _CLASSES)
    np.testing.assert_array_equal(preds, np.array(["hold", "cut", "hike"]))


def test_upweighting_a_class_flips_the_decision() -> None:
    proba = np.array([[0.45, 0.5, 0.05]])  # argmax → hold
    assert predict_weighted(proba, np.ones(3), _CLASSES)[0] == "hold"
    # Upweight cut (index 0) → 0.45 * 1.3 = 0.585 > 0.5 → cut wins.
    assert predict_weighted(proba, np.array([1.3, 1.0, 1.0]), _CLASSES)[0] == "cut"


# -----------------------------------------------------------------------------
# tune_class_weights / ThresholdTuner.
# -----------------------------------------------------------------------------
def _biased_oof() -> tuple[np.ndarray, np.ndarray]:
    """OOF probs where argmax always says 'hold' but the minorities have signal."""
    rng = np.random.default_rng(12)
    rows, labels = [], []
    for _ in range(100):  # hold: confident, correct under argmax
        rows.append([0.2, 0.6, 0.2])
        labels.append("hold")
    for _ in range(25):  # cut: hold edges it out under argmax (0.5 > 0.45)
        rows.append([0.45, 0.5, 0.05])
        labels.append("cut")
    for _ in range(25):  # hike: same
        rows.append([0.05, 0.5, 0.45])
        labels.append("hike")
    proba = np.array(rows) + rng.normal(0, 0.01, (150, 3))
    return np.array(labels), np.clip(proba, 0, None)


def test_tuning_lifts_balanced_accuracy_over_argmax() -> None:
    y, proba = _biased_oof()
    uniform_bal = balanced_accuracy(y, predict_weighted(proba, np.ones(3), _CLASSES))
    weights = tune_class_weights(y, proba, _CLASSES, n_trials=200)
    tuned_bal = balanced_accuracy(y, predict_weighted(proba, weights, _CLASSES))
    assert uniform_bal < 0.4  # argmax collapses to 'hold' → ~1/3
    assert tuned_bal > uniform_bal + 0.3  # recovering the minorities is a big lift


def test_weights_are_geomean_normalised() -> None:
    y, proba = _biased_oof()
    weights = tune_class_weights(y, proba, _CLASSES, n_trials=50)
    assert weights.shape == (3,)
    np.testing.assert_allclose(np.exp(np.mean(np.log(weights))), 1.0, atol=1e-9)


def test_threshold_tuner_fit_predict_round_trip() -> None:
    y, proba = _biased_oof()
    tuner = ThresholdTuner(n_trials=150).fit(y, proba, _CLASSES)
    assert tuner.weights_.shape == (3,)
    preds = tuner.predict(proba)
    assert preds.shape == (len(y),)
    assert balanced_accuracy(y, preds) > 0.6  # much better than the 1/3 argmax floor


def test_tune_class_weights_rejects_bad_metric() -> None:
    y, proba = _biased_oof()
    with pytest.raises(ValueError, match="Unknown metric"):
        tune_class_weights(y, proba, _CLASSES, metric="f9")


# -----------------------------------------------------------------------------
# collect_oof_proba.
# -----------------------------------------------------------------------------
def test_collect_oof_proba_shape_and_alignment() -> None:
    rng = np.random.default_rng(12)
    n = 60
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(4)})
    score = X["f0"] + 0.5 * X["f1"]
    y = pd.Series(np.where(score > 0.3, "hike", np.where(score < -0.3, "cut", "hold")))
    splitter = WalkForwardSplit(initial_train_size=40, test_size=5, step=5)
    y_true, proba, classes = collect_oof_proba("logistic_regression", X, y, splitter)
    np.testing.assert_array_equal(classes, np.array(["cut", "hike", "hold"]))
    assert proba.shape == (len(y_true), 3)
    assert (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(y_true)), atol=1e-6)
