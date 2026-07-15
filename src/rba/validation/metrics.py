"""Classification metrics — the §7 harness's numerical scoring surface.

Six metrics per CHECKLIST.md §7 first bullet: :func:`accuracy`,
:func:`balanced_accuracy`, :func:`macro_f1`, :func:`log_loss`,
:func:`brier_score`, :func:`confusion_matrix`. Plus one convenience dispatcher
— :func:`compute_classification_metrics` — that a harness can call once per
fold with ``(y_true, y_pred, y_proba, labels)`` from any classifier satisfying
:class:`rba.models.Model`.

All numeric metrics return a plain ``float`` (not ``numpy.float64``) so downstream
loggers (MLflow / plain dicts) get a clean primitive. :func:`confusion_matrix`
returns ``numpy.ndarray`` because that's the natural shape.

Design
------
- Prediction-based metrics (``accuracy`` / ``balanced_accuracy`` / ``macro_f1``
  / ``confusion_matrix``) accept ``y_pred`` — the ``predict()`` output.
- Probability-based metrics (``log_loss`` / ``brier_score``) accept ``y_proba``
  — the ``predict_proba()`` output — and **require** ``labels`` (the classifier's
  ``classes_``, i.e. the column order of ``y_proba``). Without it we can't join
  the probability columns back to ``y_true`` labels.
- Multiclass Brier: ``mean_i(sum_k (P_ik − I[y_i = k])²)``. sklearn's
  ``brier_score_loss`` is binary-only (as of 1.8), so this is implemented
  directly — the definition is one line of numpy and is cross-version stable.
- Every metric that takes an optional ``labels`` argument forwards it to the
  underlying sklearn call so that a class absent from a small test window still
  contributes a zero row/column instead of being silently dropped.

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import (
    confusion_matrix as _sk_confusion_matrix,
)
from sklearn.metrics import (
    log_loss as _sk_log_loss,
)

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "confusion_matrix",
    "directional_accuracy",
    "hit_rate_vs_market",
    "log_loss",
    "macro_f1",
    "mae",
    "r2",
    "rmse",
]


# =============================================================================
# Prediction-based metrics.
# =============================================================================
def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of correctly-labelled samples — ``(y_pred == y_true).mean()``.

    Wraps :func:`sklearn.metrics.accuracy_score` for correctness on edge cases
    (empty arrays raise a clear sklearn error rather than a NaN mean).
    """
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean per-class recall — robust to class imbalance.

    Wraps :func:`sklearn.metrics.balanced_accuracy_score`. For the RBA target
    (77% hold), this is the honest classification floor: a "predict hold"
    classifier scores ``1/3`` here even though its plain accuracy is ~0.77.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
) -> float:
    """Unweighted mean of per-class F1 — the cycle-turn signal metric.

    Wraps :func:`sklearn.metrics.f1_score` with ``average="macro"`` and passes
    ``labels`` through so a class absent from a small test fold still counts as
    an F1 of zero (sklearn otherwise drops it, quietly overweighting the classes
    that appear).
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(labels) if labels is not None else None,
            average="macro",
            zero_division=0,
        )
    )


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
) -> np.ndarray:
    """Rows are true classes, columns are predicted — same order as ``labels``.

    Wraps :func:`sklearn.metrics.confusion_matrix`. Passing ``labels`` pins the
    row/column order (recommended: the classifier's ``classes_``) so matrices
    across folds are comparable.
    """
    return _sk_confusion_matrix(
        y_true, y_pred, labels=list(labels) if labels is not None else None
    )


# =============================================================================
# Probability-based metrics.
# =============================================================================
def log_loss(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    labels: Sequence[Any] | np.ndarray,
) -> float:
    """Cross-entropy — ``mean_i(-log(P_ik*))`` where ``k*`` is the true class of row ``i``.

    Wraps :func:`sklearn.metrics.log_loss`, which clips predicted probabilities
    at ``float64`` eps so a one-hot classifier (e.g. :class:`~rba.models.baselines.Persistence`)
    yields a large-but-finite score on wrong calls rather than infinity.

    Parameters
    ----------
    y_true
        True class labels aligned to rows of ``y_proba``.
    y_proba
        Shape ``(n_samples, n_classes)`` probability matrix — the classifier's
        ``predict_proba(X)`` output.
    labels
        The column order of ``y_proba`` — pass the classifier's ``classes_``.
        Required (unlike sklearn's optional ``labels``) so an unseen-class row in
        a small test window doesn't silently drop.
    """
    return float(_sk_log_loss(y_true, y_proba, labels=list(labels)))


def brier_score(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    labels: Sequence[Any] | np.ndarray,
) -> float:
    """Multiclass Brier — ``mean_i(sum_k (P_ik − I[y_i = k])²)``.

    Direct implementation (sklearn 1.8's ``brier_score_loss`` is binary-only).
    Range is ``[0, 2]``: ``0`` at perfect one-hot predictions on the correct
    class; ``2`` at a wrong-class one-hot prediction (``1² + 1² = 2``).

    Parameters
    ----------
    y_true, y_proba, labels
        Same contract as :func:`log_loss` — ``labels`` is the column order of
        ``y_proba`` and pins the true-label one-hot encoding.
    """
    labels_arr = np.asarray(list(labels))
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)
    if y_proba_arr.ndim != 2:
        raise ValueError(f"y_proba must be 2-D; got shape {y_proba_arr.shape}.")
    if y_proba_arr.shape[1] != len(labels_arr):
        raise ValueError(
            f"y_proba has {y_proba_arr.shape[1]} columns but labels has "
            f"{len(labels_arr)} entries — the two must match."
        )
    # One-hot encode y_true against ``labels`` (column ordering).
    onehot = np.zeros_like(y_proba_arr)
    for col, cls in enumerate(labels_arr):
        onehot[y_true_arr == cls, col] = 1.0
    return float(np.mean(np.sum((y_proba_arr - onehot) ** 2, axis=1)))


# =============================================================================
# Dispatcher — the one call every §7 fold runs.
# =============================================================================
def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_proba: ArrayLike | None = None,
    labels: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute every §7 classification metric that ``(y_true, y_pred, y_proba)`` supports.

    Prediction-based metrics (accuracy / balanced_accuracy / macro_f1 /
    confusion_matrix) always run. Probability-based metrics (log_loss /
    brier_score) run only when ``y_proba`` is given; they also require
    ``labels`` (the column order of ``y_proba`` — pass the classifier's
    ``classes_``).

    Parameters
    ----------
    y_true, y_pred, y_proba, labels
        Same contract as the individual metric functions.

    Returns
    -------
    dict
        Keys ``accuracy`` / ``balanced_accuracy`` / ``macro_f1`` /
        ``confusion_matrix`` always present. ``log_loss`` / ``brier_score``
        present iff ``y_proba`` given. Numeric values are ``float``;
        ``confusion_matrix`` is ``numpy.ndarray``.

    Raises
    ------
    ValueError
        If ``y_proba`` is given but ``labels`` is not.
    """
    if y_proba is not None and labels is None:
        raise ValueError(
            "compute_classification_metrics: ``labels`` is required when "
            "``y_proba`` is given (it is the column order of ``y_proba``); "
            "pass the classifier's ``classes_``."
        )
    metrics: dict[str, Any] = {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
    if y_proba is not None:
        metrics["log_loss"] = log_loss(y_true, y_proba, labels=labels)  # type: ignore[arg-type]
        metrics["brier_score"] = brier_score(y_true, y_proba, labels=labels)  # type: ignore[arg-type]
    logger.debug(
        "compute_classification_metrics: {} metrics computed on {} samples.",
        len(metrics),
        len(np.asarray(y_true)),
    )
    return metrics


# =============================================================================
# Paired comparator — hit-rate vs the market-implied baseline.
# =============================================================================
def hit_rate_vs_market(
    y_true: ArrayLike,
    y_pred_model: ArrayLike,
    y_pred_market: ArrayLike,
) -> dict[str, float]:
    """Paired accuracy comparison of a model against the market-implied baseline.

    Runs on the *intersection* of meetings where both the model and the market
    have a prediction: rows where ``y_pred_market`` is NaN are dropped from the
    comparison (ASX IB-futures coverage begins 2022-04-21 per the
    :class:`~rba.models.baselines.MarketImplied` docstring, so the model's full
    history isn't compared — just the covered window). NaN in ``y_true`` or
    ``y_pred_model`` is treated the same way (dropped) so the three arrays
    always end up aligned.

    Answers three related questions:

    1. **Lift** — how much accuracy does the model add over the market on the
       same meetings? ``model_hit_rate − market_hit_rate``.
    2. **Where** is the model beating / losing to the market? The paired
       disagreement breakdown (``only_model_correct`` vs
       ``only_market_correct``) exposes the *signed* asymmetry that raw
       accuracy hides.
    3. **How hard** was the meeting set? ``both_correct`` counts easy calls,
       ``both_wrong`` counts genuine surprises that broke both signals.

    Parameters
    ----------
    y_true
        True class labels for the compared meetings.
    y_pred_model
        Model predictions, same shape as ``y_true``.
    y_pred_market
        Market baseline predictions (typically
        :meth:`rba.models.baselines.MarketImplied.predict`). NaN rows are
        dropped from the comparison rather than treated as wrong — see above.

    Returns
    -------
    dict[str, float]
        Keys:

        - ``model_hit_rate`` — fraction of compared meetings the model got right.
        - ``market_hit_rate`` — fraction the market got right.
        - ``lift`` — ``model_hit_rate − market_hit_rate``. Positive: model
          beats market; negative: market beats model.
        - ``both_correct`` — fraction where both were right.
        - ``both_wrong`` — fraction where both were wrong.
        - ``only_model_correct`` — fraction where only the model was right
          (the "model found signal the market missed" cell).
        - ``only_market_correct`` — fraction where only the market was right
          (the "market had info the model didn't capture" cell).
        - ``n_compared`` — number of surviving rows after NaN filtering.

        All fractions sum to 1 (``both_correct + both_wrong + only_model_correct
        + only_market_correct == 1``). When no rows survive filtering, every
        float is ``NaN`` and ``n_compared`` is ``0``.

    Raises
    ------
    ValueError
        If the three arrays have different lengths.
    """
    # ``pd.Series`` on the raw input (not ``np.asarray``) preserves NaN in
    # mixed-type object lists — ``np.asarray([np.nan, "hike"])`` coerces to
    # ``"nan"`` strings and loses the NaN semantics we need to filter on.
    y_true_s = pd.Series(y_true).reset_index(drop=True)
    y_model_s = pd.Series(y_pred_model).reset_index(drop=True)
    y_market_s = pd.Series(y_pred_market).reset_index(drop=True)
    if not (len(y_true_s) == len(y_model_s) == len(y_market_s)):
        raise ValueError(
            f"hit_rate_vs_market: array lengths must match; got "
            f"y_true={len(y_true_s)}, y_pred_model={len(y_model_s)}, "
            f"y_pred_market={len(y_market_s)}."
        )
    keep = y_true_s.notna() & y_model_s.notna() & y_market_s.notna()
    n = int(keep.sum())
    if n == 0:
        logger.warning(
            "hit_rate_vs_market: no rows survive NaN filtering (n_input={}).",
            len(y_true_s),
        )
        return {
            "model_hit_rate": float("nan"),
            "market_hit_rate": float("nan"),
            "lift": float("nan"),
            "both_correct": float("nan"),
            "both_wrong": float("nan"),
            "only_model_correct": float("nan"),
            "only_market_correct": float("nan"),
            "n_compared": 0.0,
        }

    y_true_arr = y_true_s[keep].to_numpy()
    model_arr = y_model_s[keep].to_numpy()
    market_arr = y_market_s[keep].to_numpy()

    model_right = model_arr == y_true_arr
    market_right = market_arr == y_true_arr

    both_correct = float((model_right & market_right).sum() / n)
    both_wrong = float((~model_right & ~market_right).sum() / n)
    only_model_correct = float((model_right & ~market_right).sum() / n)
    only_market_correct = float((~model_right & market_right).sum() / n)
    model_hit_rate = float(model_right.mean())
    market_hit_rate = float(market_right.mean())

    result = {
        "model_hit_rate": model_hit_rate,
        "market_hit_rate": market_hit_rate,
        "lift": model_hit_rate - market_hit_rate,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "only_model_correct": only_model_correct,
        "only_market_correct": only_market_correct,
        "n_compared": float(n),
    }
    logger.debug(
        "hit_rate_vs_market: n={}, model={:.4f}, market={:.4f}, lift={:+.4f}.",
        n,
        model_hit_rate,
        market_hit_rate,
        result["lift"],
    )
    return result


# =============================================================================
# Regression metrics — rmse / mae / r2 / directional_accuracy per targets.yaml.
# =============================================================================
def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error. Wraps :func:`sklearn.metrics.mean_squared_error`."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error. Wraps :func:`sklearn.metrics.mean_absolute_error`."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination. Wraps :func:`sklearn.metrics.r2_score`.

    Can go negative when the model is worse than predicting the training-set
    mean — that's the honest signal, not an error.
    """
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of rows where ``sign(y_pred) == sign(y_true)``.

    The natural regression counterpart of hit-rate for a Δrate model — a
    prediction that gets the *direction* right is materially useful even when
    the magnitude is off (a hike-vs-cut call beats a Taylor-rule level miss).
    Both ``0`` cases (``sign(0) == sign(0)``) count as correct — a genuine "no
    change" call.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr)))


def compute_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute every §7 regression metric that ``(y_true, y_pred)`` supports.

    Returns a dict with keys ``rmse`` / ``mae`` / ``r2`` /
    ``directional_accuracy``. ``directional_accuracy`` is meaningful for
    Δrate regression (``delta_regression`` in ``targets.yaml``); for
    ``level_regression`` it degrades to "same-sign" cheaply (both true and
    predicted rate levels are typically positive), which is still valid but
    less informative — the caller can drop it downstream if desired.
    """
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    logger.debug(
        "compute_regression_metrics: n={}, rmse={:.4f}, mae={:.4f}, r2={:.4f}.",
        len(np.asarray(y_true)),
        metrics["rmse"],
        metrics["mae"],
        metrics["r2"],
    )
    return metrics
