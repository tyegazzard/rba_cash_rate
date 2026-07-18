"""Tests for ``rba.validation.tuning`` — the Optuna walk-forward search driver.

Covers :func:`run_search`: it returns a buildable tuned config whose params sit
within the ``models.yaml`` search space, is reproducible under a fixed seed,
rejects an unknown metric / an empty search space, optimises ``log_loss`` in the
minimising direction, and survives a search space with invalid param combos (the
LR ``l1`` + ``lbfgs`` clash, auto-repaired by the wrapper). And :func:`_suggest`:
each distribution kind maps to the right Optuna suggestion, and a non-primitive
``categorical`` (the MLP ``hidden_layer_sizes`` list-of-lists) resolves to a real
choice value.

No network; tiny synthetic frames; Optuna's own logging is silenced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from rba.models import Model, build_model
from rba.validation import WalkForwardSplit, run_search
from rba.validation.tuning import SearchResult, _suggest

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _clf_frame(n: int = 60, p: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """3-class target with real signal in f0/f1 so tuning has something to chase."""
    rng = np.random.default_rng(12)
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    score = X["f0"] + 0.5 * X["f1"]
    y = pd.Series(np.where(score > 0.3, "hike", np.where(score < -0.3, "cut", "hold")))
    return X, y


def _splitter() -> WalkForwardSplit:
    return WalkForwardSplit(initial_train_size=40, test_size=5, step=5)


# -----------------------------------------------------------------------------
# run_search — contract.
# -----------------------------------------------------------------------------
def test_run_search_returns_buildable_tuned_config() -> None:
    X, y = _clf_frame()
    result = run_search(
        model_name="logistic_regression", X=X, y=y, splitter=_splitter(), n_trials=6
    )
    assert isinstance(result, SearchResult)
    assert set(result.best_params) == {"C", "penalty"}  # exactly the search-space keys
    assert 0.001 <= result.best_params["C"] <= 100  # within declared bounds
    assert result.best_params["penalty"] in ("l1", "l2")
    assert 0.0 <= result.best_value <= 1.0  # balanced_accuracy
    # The tuned config builds and fits (the whole point).
    tuned = build_model(result.best_model_cfg)
    assert isinstance(tuned.fit(X, y), Model)


def test_run_search_is_reproducible_under_fixed_seed() -> None:
    X, y = _clf_frame()
    a = run_search(model_name="logistic_regression", X=X, y=y, splitter=_splitter(), n_trials=6)
    b = run_search(model_name="logistic_regression", X=X, y=y, splitter=_splitter(), n_trials=6)
    assert a.best_params == b.best_params
    assert a.best_value == b.best_value


def test_run_search_l1_penalty_does_not_crash_the_search() -> None:
    """The LR search sweeps penalty ∈ {l1, l2}; the l1+lbfgs clash must not abort it."""
    X, y = _clf_frame()
    result = run_search(
        model_name="logistic_regression", X=X, y=y, splitter=_splitter(), n_trials=8
    )
    assert result.n_trials == 8  # every trial completed (l1 auto-repaired to saga)


def test_run_search_log_loss_minimises() -> None:
    X, y = _clf_frame()
    result = run_search(
        model_name="logistic_regression",
        X=X,
        y=y,
        splitter=_splitter(),
        n_trials=5,
        metric="log_loss",
    )
    assert result.direction == "minimize"
    assert result.best_value >= 0.0


def test_run_search_unknown_metric_raises() -> None:
    X, y = _clf_frame()
    with pytest.raises(ValueError, match="Unknown metric"):
        run_search(model_name="logistic_regression", X=X, y=y, splitter=_splitter(), metric="f9")


def test_run_search_empty_search_space_raises() -> None:
    X, y = _clf_frame()
    with pytest.raises(ValueError, match="empty search_space"):
        run_search(model_name="majority_class", X=X, y=y, splitter=_splitter())


# -----------------------------------------------------------------------------
# _suggest — distribution mapping.
# -----------------------------------------------------------------------------
def _suggest_once(dist: dict[str, Any]) -> Any:
    captured: dict[str, Any] = {}

    def objective(trial: optuna.Trial) -> float:
        captured["v"] = _suggest(trial, "p", dist)
        return 0.0

    optuna.create_study().optimize(objective, n_trials=1)
    return captured["v"]


def test_suggest_loguniform_in_bounds() -> None:
    v = _suggest_once({"type": "loguniform", "low": 0.01, "high": 100})
    assert 0.01 <= v <= 100


def test_suggest_uniform_in_bounds() -> None:
    v = _suggest_once({"type": "uniform", "low": 0.5, "high": 2.5})
    assert 0.5 <= v <= 2.5


def test_suggest_int_in_bounds() -> None:
    v = _suggest_once({"type": "int", "low": 2, "high": 10})
    assert isinstance(v, int) and 2 <= v <= 10


def test_suggest_primitive_categorical() -> None:
    v = _suggest_once({"type": "categorical", "choices": ["a", "b", "c"]})
    assert v in ("a", "b", "c")


def test_suggest_non_primitive_categorical_resolves_to_list() -> None:
    """hidden_layer_sizes list-of-lists → a real list choice (not an index)."""
    v = _suggest_once({"type": "categorical", "choices": [[32], [64, 32], [128, 64, 32]]})
    assert v in ([32], [64, 32], [128, 64, 32])


def test_suggest_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError, match="Unknown search-space distribution"):
        _suggest_once({"type": "beta", "low": 0, "high": 1})
