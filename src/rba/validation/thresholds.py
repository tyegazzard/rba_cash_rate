"""§7 decision-threshold tuning — per-class weights on ``predict_proba``.

For the 77%-hold RBA target, a plain ``argmax`` over calibrated probabilities
almost always predicts *hold*: the minority hike / cut classes rarely win the
argmax even when the model assigns them real probability. The multiclass analog
of "moving a binary decision threshold" is a **per-class weight** applied before
the argmax — ``pred = classes[argmax(proba * w)]``. Upweighting the minority
classes shifts the decision boundaries so they win more often, trading a little
plain accuracy for a large balanced-accuracy / macro-F1 gain.

The weights are tuned on **out-of-fold** dev probabilities (never in-sample,
never on the held-out test window): :func:`collect_oof_proba` runs the
walk-forward CV over dev and returns the concatenated OOF probabilities, and
:func:`tune_class_weights` (or the :class:`ThresholdTuner` wrapper) optimises the
per-class weights on them. Because scoring a weight vector is a cheap argmax over
precomputed probabilities, the Optuna search here runs hundreds of trials in a
blink.

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
import numpy as np
import optuna
import pandas as pd

from rba.config import RANDOM_SEED, load_model_config
from rba.models import build_model
from rba.validation.cv import WalkForwardSplit
from rba.validation.metrics import accuracy, balanced_accuracy, macro_f1

__all__ = [
    "ThresholdTuner",
    "collect_oof_proba",
    "predict_weighted",
    "tune_class_weights",
]

# Label metrics the threshold search maximises (all argmax-based, so no proba
# alignment needed once the weights are applied).
_METRIC_FNS: dict[str, Callable[..., float]] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
}


def predict_weighted(
    proba: np.ndarray,
    weights: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Class prediction from probability-weighted argmax.

    ``classes[argmax(proba * weights)]`` per row. Uniform ``weights`` reproduce a
    plain argmax; upweighting a class lowers its effective decision threshold.

    Shapes
    ------
    proba: (n, k), weights: (k,), classes: (k,) -> (n,).
    """
    scores = np.asarray(proba, dtype=float) * np.asarray(weights, dtype=float)
    return np.asarray(classes)[np.argmax(scores, axis=1)]


def tune_class_weights(
    y_true: Any,
    proba: np.ndarray,
    classes: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    n_trials: int = 300,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Optimise per-class decision weights on (out-of-fold) probabilities.

    Searches per-class weights in ``[0.1, 10]`` (log scale) to maximise ``metric``
    on ``predict_weighted(proba, w, classes)``. The returned weights are
    normalised to geometric mean 1 (argmax is scale-invariant, so this is purely
    for readability — a weight > 1 means "predict this class more readily").

    Parameters
    ----------
    y_true
        True labels (the OOF targets), length ``n``.
    proba
        OOF probabilities, shape ``(n, k)``, columns aligned to ``classes``.
    classes
        The class labels, shape ``(k,)`` — the ``predict_proba`` column order.
    metric
        ``'balanced_accuracy'`` (default) / ``'macro_f1'`` / ``'accuracy'``.
    n_trials
        Optuna trials (cheap — scoring is an argmax over precomputed ``proba``).
    random_state
        Seed for the TPE sampler.

    Returns
    -------
    numpy.ndarray
        The tuned per-class weights, shape ``(k,)``, aligned to ``classes``.

    Raises
    ------
    ValueError
        If ``metric`` is unknown.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(_METRIC_FNS)}.")
    scorer = _METRIC_FNS[metric]
    y_arr = np.asarray(y_true)
    proba_arr = np.asarray(proba, dtype=float)
    classes_arr = np.asarray(classes)
    k = len(classes_arr)

    def objective(trial: optuna.Trial) -> float:
        w = np.array([trial.suggest_float(f"w{i}", 0.1, 10.0, log=True) for i in range(k)])
        return scorer(y_arr, predict_weighted(proba_arr, w, classes_arr))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    weights = np.array([study.best_params[f"w{i}"] for i in range(k)])
    weights = weights / np.exp(np.mean(np.log(weights)))  # geomean 1 (argmax-invariant)
    logger.debug(
        "tune_class_weights({}): weights={} (best {}={:.4f}).",
        metric,
        dict(zip(classes_arr.tolist(), weights.round(3).tolist())),
        metric,
        study.best_value,
    )
    return weights


class ThresholdTuner:
    """Fit per-class decision weights on OOF probabilities; apply at predict time.

    A thin stateful wrapper over :func:`tune_class_weights` /
    :func:`predict_weighted`: :meth:`fit` learns the weights on out-of-fold
    probabilities, :meth:`predict` applies them to a fresh probability matrix.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The class labels the weights are aligned to.
    weights_ : numpy.ndarray
        The tuned per-class weights (geometric mean 1).
    """

    def __init__(
        self,
        *,
        metric: str = "balanced_accuracy",
        n_trials: int = 300,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.metric = metric
        self.n_trials = int(n_trials)
        self.random_state = random_state

    def fit(self, y_true: Any, proba: np.ndarray, classes: np.ndarray) -> "ThresholdTuner":
        """Learn the per-class weights from OOF ``(y_true, proba)``."""
        self.classes_ = np.asarray(classes)
        self.weights_ = tune_class_weights(
            y_true,
            proba,
            self.classes_,
            metric=self.metric,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )
        return self

    def predict(self, proba: np.ndarray) -> np.ndarray:
        """Weighted-argmax prediction for a probability matrix (``(n, k) -> (n,)``)."""
        return predict_weighted(proba, self.weights_, self.classes_)


def collect_oof_proba(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplit,
    *,
    model_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward out-of-fold probabilities for a model over ``(X, y)`` (dev only).

    Fits a fresh model per fold and collects its test-fold probabilities, aligning
    each fold's columns to the global class set (``np.unique(y)``) so folds that
    miss a class still contribute a full-width, correctly-ordered row (missing
    class → probability 0). Pass the **dev** matrix only.

    Parameters
    ----------
    model_name
        A ``models.yaml`` key (used when ``model_cfg`` is not given).
    X, y
        The **dev** feature matrix and target.
    splitter
        The walk-forward splitter defining the OOF folds.
    model_cfg
        Optional pre-resolved (tuned) config, overriding
        ``load_model_config(model_name)`` — so the §8 threshold tuning runs on the
        *tuned* model's OOF probabilities rather than the default config's.

    Returns
    -------
    (y_true, proba, classes) :
        ``y_true`` (n,), ``proba`` (n, k) aligned to ``classes`` (k,), in fold order.

    Shapes
    ------
    X: (n_dev, n_features), y: (n_dev,) -> y_true: (n,), proba: (n, k), classes: (k,).
    """
    cfg = model_cfg if model_cfg is not None else load_model_config(model_name)
    classes = np.unique(np.asarray(y))
    col_of = {c: i for i, c in enumerate(classes)}
    y_parts: list[np.ndarray] = []
    proba_parts: list[np.ndarray] = []
    for train_idx, test_idx in splitter.split(X):
        model = build_model(cfg)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_proba = np.asarray(model.predict_proba(X.iloc[test_idx]), dtype=float)
        aligned = np.zeros((len(test_idx), len(classes)), dtype=float)
        for j, cls in enumerate(np.asarray(model.classes_)):  # type: ignore[attr-defined]
            aligned[:, col_of[cls]] = fold_proba[:, j]
        proba_parts.append(aligned)
        y_parts.append(y.iloc[test_idx].to_numpy())
    return np.concatenate(y_parts), np.concatenate(proba_parts), classes
