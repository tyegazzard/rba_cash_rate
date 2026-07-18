"""§7 hyperparameter search — Optuna over ``models.yaml`` search spaces, CV-only.

:func:`run_search` tunes a model's hyperparameters with Optuna, scoring every
trial by **walk-forward cross-validation** (:class:`~rba.validation.cv.WalkForwardSplit`)
over the data it is handed. That data must be the **dev** portion only
(:func:`rba.features.build.dev_test_split`) — the held-out test window
(:data:`rba.config.TEST_WINDOW_START`) is never passed here, so the search
cannot see it (CHECKLIST "Hyperparameter search ... within walk-forward CV only"
/ "Never tune on the held-out test set").

Each model's ``search_space`` in ``models.yaml`` is already written in Optuna
distribution conventions (``loguniform`` / ``uniform`` / ``int`` / ``categorical``);
:func:`run_search` maps those to ``trial.suggest_*`` calls, samples a config,
merges it over the model's ``default`` block, and evaluates it. The resolved
(human-readable) parameter set of each trial is stashed in the trial's
``user_attrs`` so a ``categorical`` over non-primitive choices (the MLP's
``hidden_layer_sizes`` list-of-lists) round-trips cleanly.

Scoring metrics
---------------
Label metrics (``accuracy`` / ``balanced_accuracy`` (default) / ``macro_f1``) are
computed on the **concatenated** out-of-fold predictions — the same aggregation
the §7 harness uses for ``metrics_overall``. The probabilistic ``log_loss`` is
averaged **per fold** (each fold self-contained with its own ``classes_``) to
avoid cross-fold class-column misalignment on the small walk-forward test folds.

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
import warnings

from loguru import logger
import numpy as np
import optuna
import pandas as pd

from rba.config import RANDOM_SEED, load_model_config
from rba.models import build_model
from rba.validation.cv import WalkForwardSplit
from rba.validation.metrics import accuracy, balanced_accuracy, log_loss, macro_f1

__all__ = ["SearchResult", "run_search"]

# Primitive types Optuna accepts directly as ``suggest_categorical`` choices.
_PRIMITIVE = (type(None), bool, int, float, str)


@dataclass(frozen=True)
class _MetricSpec:
    """A tunable metric: its scorer, optimisation direction, and proba need."""

    fn: Callable[..., float]
    direction: str  # 'maximize' | 'minimize'
    needs_proba: bool


# The metrics the search can optimise. ``balanced_accuracy`` is the default — the
# honest, imbalance-aware objective for the 77%-hold target.
_METRICS: dict[str, _MetricSpec] = {
    "accuracy": _MetricSpec(accuracy, "maximize", False),
    "balanced_accuracy": _MetricSpec(balanced_accuracy, "maximize", False),
    "macro_f1": _MetricSpec(macro_f1, "maximize", False),
    "log_loss": _MetricSpec(log_loss, "minimize", True),
}


@dataclass
class SearchResult:
    """Outcome of a :func:`run_search` run.

    Attributes
    ----------
    model_name, metric, direction :
        What was tuned and how it was scored.
    best_value :
        The best cross-validated ``metric`` achieved.
    best_params :
        The resolved winning hyper-parameters (ready to read).
    best_model_cfg :
        A ``models.yaml``-shaped config with ``best_params`` merged over the
        model's ``default`` block — pass straight to
        :func:`rba.models.build_model` to instantiate the tuned model.
    n_trials :
        Number of trials evaluated.
    trials :
        Per-trial ``{number, value, params}`` records (for logging / plots).
    """

    model_name: str
    metric: str
    direction: str
    best_value: float
    best_params: dict[str, Any]
    best_model_cfg: dict[str, Any]
    n_trials: int
    trials: list[dict[str, Any]] = field(default_factory=list)


def run_search(
    *,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplit,
    n_trials: int = 50,
    metric: str = "balanced_accuracy",
    random_state: int = RANDOM_SEED,
    timeout: float | None = None,
) -> SearchResult:
    """Tune ``model_name`` by walk-forward CV over ``(X, y)`` — the dev portion only.

    Parameters
    ----------
    model_name
        A ``models.yaml`` key with a non-empty ``search_space``.
    X, y
        The **dev** feature matrix and target (from
        :func:`rba.features.build.dev_test_split`); positional / chronological
        order, since walk-forward relies on it. **Never pass the test window.**
    splitter
        The inner-CV :class:`~rba.validation.cv.WalkForwardSplit` used to score
        each trial.
    n_trials
        Number of Optuna trials.
    metric
        One of ``accuracy`` / ``balanced_accuracy`` (default) / ``macro_f1`` /
        ``log_loss``. Direction is inferred.
    random_state
        Seed for Optuna's TPE sampler (reproducible search).
    timeout
        Optional wall-clock budget (seconds) passed to ``study.optimize``.

    Returns
    -------
    SearchResult
        Best params + a ready-to-build ``best_model_cfg``.

    Raises
    ------
    ValueError
        If the model has no ``search_space``, or ``metric`` is unknown.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(_METRICS)}.")
    spec = _METRICS[metric]

    base_cfg = load_model_config(model_name)
    search_space: Mapping[str, Any] = base_cfg.get("search_space") or {}
    if not search_space:
        raise ValueError(
            f"Model {model_name!r} has an empty search_space in models.yaml — nothing to tune."
        )

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, dist) for name, dist in search_space.items()}
        trial.set_user_attr("resolved_params", params)
        cfg = _merge_params(base_cfg, params)
        try:
            return _cv_score(cfg, X, y, splitter, spec)
        except Exception as err:  # noqa: BLE001 — an invalid param combo prunes the trial, never crashes the search.
            logger.debug(
                "run_search: trial {} pruned ({}): {}", trial.number, type(err).__name__, err
            )
            raise optuna.TrialPruned() from err

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=spec.direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError(
            f"run_search({model_name!r}): every trial failed/pruned — check the search "
            "space and that (X, y) has enough rows/classes for the splitter."
        )
    best_params = dict(study.best_trial.user_attrs["resolved_params"])
    result = SearchResult(
        model_name=model_name,
        metric=metric,
        direction=spec.direction,
        best_value=float(study.best_value),
        best_params=best_params,
        best_model_cfg=_merge_params(base_cfg, best_params),
        n_trials=len(study.trials),
        trials=[
            {
                "number": t.number,
                "value": t.value,
                "params": t.user_attrs.get("resolved_params", {}),
            }
            for t in study.trials
        ],
    )
    logger.info(
        "run_search: model={} metric={} best={:.4f} over {} trial(s); best_params={}.",
        model_name,
        metric,
        result.best_value,
        result.n_trials,
        best_params,
    )
    return result


# =============================================================================
# Internals.
# =============================================================================
def _suggest(trial: optuna.Trial, name: str, dist: Mapping[str, Any]) -> Any:
    """Map one ``models.yaml`` distribution entry to an Optuna suggestion."""
    kind = dist.get("type")
    if kind == "loguniform":
        return trial.suggest_float(name, dist["low"], dist["high"], log=True)
    if kind == "uniform":
        return trial.suggest_float(name, dist["low"], dist["high"])
    if kind == "int":
        return trial.suggest_int(name, dist["low"], dist["high"])
    if kind == "categorical":
        choices = list(dist["choices"])
        if all(isinstance(c, _PRIMITIVE) for c in choices):
            return trial.suggest_categorical(name, choices)
        # Non-primitive choices (e.g. hidden_layer_sizes list-of-lists) aren't valid
        # Optuna categoricals — suggest an index and resolve to the real value.
        idx = trial.suggest_categorical(name, list(range(len(choices))))
        return choices[int(idx)]
    raise ValueError(f"Unknown search-space distribution {kind!r} for param {name!r}.")


def _merge_params(base_cfg: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base_cfg`` with ``params`` merged over its ``default`` block."""
    cfg = dict(base_cfg)
    cfg["default"] = {**(base_cfg.get("default") or {}), **params}
    return cfg


def _cv_score(
    cfg: Mapping[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplit,
    spec: _MetricSpec,
) -> float:
    """Walk-forward CV score for one config (fresh model per fold — out-of-sample)."""
    with warnings.catch_warnings():
        # Small walk-forward folds trip inherent sklearn UserWarnings (single-label
        # fold, y_pred classes absent from y_true) — not bugs; silence the spam.
        warnings.filterwarnings("ignore", message="A single label was found", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true", category=UserWarning
        )
        if spec.needs_proba:
            per_fold: list[float] = []
            for train_idx, test_idx in splitter.split(X):
                model = build_model(cfg)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                proba = np.asarray(model.predict_proba(X.iloc[test_idx]), dtype=np.float64)
                per_fold.append(
                    spec.fn(y.iloc[test_idx].to_numpy(), proba, labels=model.classes_)  # type: ignore[attr-defined]
                )
            return float(np.mean(per_fold))

        y_true: list[np.ndarray] = []
        y_pred: list[np.ndarray] = []
        for train_idx, test_idx in splitter.split(X):
            model = build_model(cfg)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred.append(np.asarray(model.predict(X.iloc[test_idx])))
            y_true.append(y.iloc[test_idx].to_numpy())
        return float(spec.fn(np.concatenate(y_true), np.concatenate(y_pred)))
