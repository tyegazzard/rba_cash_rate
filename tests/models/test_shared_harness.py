"""Shared-harness integration test — every model drives the same eval loop.

The contract-level proof that every model registered in ``models.yaml`` — the four
§6 baselines **and** the seven fittable §7 ML models (logistic regression, random
forest, SVM, MLP, XGBoost, LightGBM, ordinal logistic) — can be swapped into a
single, shared fit → predict → predict_proba flow with the task branch driven only
by the yaml ``task`` field, zero model-specific code paths. The ``ordinal`` task
runs the same probability-validity checks as ``classification`` (mord exposes
``predict_proba``); ``xrfm_classifier`` is excluded (a deferred stub whose fit
raises — see :mod:`rba.models.xrfm`).

Kept deliberately synthetic (deterministic RNG seed): no data-source or network
dependency, so it runs in the same CI slice as the rest of ``tests/models/``. The
frame is engineered to carry every feature column any baseline needs
(``cpi_headline_yoy`` / ``output_gap`` for :class:`~rba.models.baselines.TaylorRule`,
``asx_30d_implied_rate`` / ``prior_rate_pct`` for
:class:`~rba.models.baselines.MarketImplied`); the ML models simply consume all
four numeric columns. ``n`` is sized so every class is well-populated for the ML
models' internal resampling (SVC Platt CV, MLP early-stopping split).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from rba.config import load_model_config
from rba.models import Model, build_model

# The §6 baselines PLUS every fittable §7 ML model — all pass through the same
# parametric fit → predict → proba-validity contract, driven only by the yaml
# ``task`` field. ``xrfm_classifier`` is deliberately excluded (a deferred stub
# whose fit raises NotImplementedError; see :mod:`rba.models.xrfm`).
BASELINE_NAMES: tuple[str, ...] = (
    # baselines
    "majority_class",
    "persistence",
    "taylor_rule",
    "market_implied",
    # §7 ML models
    "logistic_regression",
    "random_forest_classifier",
    "svm_classifier",
    "mlp_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "ordinal_logistic",
)


@pytest.fixture(scope="module")
def synthetic_meeting_frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """A tiny chronological meeting frame carrying every baseline's inputs.

    Returns ``(X, y_class, y_reg)`` — one classification target (3-class strings
    matching the ``targets.yaml`` ``three_class`` default) and one regression
    target (the realised cash-rate level, close to ``prior_rate_pct``). The
    ``ordinal`` task derives its integer-encoded target from ``y_class`` in
    :func:`_select_target`.

    ``n`` is sized so every class is comfortably populated (cut=11, hike=16,
    hold=33 under seed 12) — enough for the ML models' internal resampling
    (SVC's 5-fold Platt calibration, MLP's stratified early-stopping split).
    """
    n = 60
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        {
            "cpi_headline_yoy": rng.uniform(1.5, 4.5, size=n),
            "output_gap": rng.uniform(-1.5, 1.5, size=n),
            "asx_30d_implied_rate": rng.uniform(3.0, 5.0, size=n),
            "prior_rate_pct": rng.uniform(3.0, 5.0, size=n),
        }
    )
    y_class = pd.Series(rng.choice(["cut", "hold", "hike"], size=n, p=[0.15, 0.65, 0.20]))
    y_reg = X["prior_rate_pct"] + rng.normal(0.0, 0.05, size=n)
    return X, y_class, y_reg


def _select_target(task: str, y_class: pd.Series, y_reg: pd.Series) -> pd.Series:
    """Task-driven target dispatch — the pattern the §7 harness will use."""
    if task == "classification":
        return y_class
    if task == "regression":
        return y_reg
    if task == "ordinal":
        # mord needs integer-encoded ordinal labels; map the 3-class strings to
        # their signed-int ordinal encoding (cut < hold < hike).
        return y_class.map({"cut": -1, "hold": 0, "hike": 1}).astype(int)
    raise AssertionError(f"Unexpected task {task!r} — extend the shared harness.")


@pytest.mark.parametrize("baseline_name", BASELINE_NAMES)
def test_baseline_conforms_to_shared_harness_contract(
    baseline_name: str,
    synthetic_meeting_frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """Each baseline: factory-build → fit → predict → task-appropriate proba check."""
    X, y_class, y_reg = synthetic_meeting_frame
    cfg = load_model_config(baseline_name)
    model = build_model(cfg)
    y = _select_target(cfg["task"], y_class, y_reg)

    fitted = model.fit(X, y)

    assert isinstance(fitted, Model), (
        f"{baseline_name} does not satisfy the Model protocol after fit."
    )
    assert fitted.feature_names_ == list(X.columns), (
        f"{baseline_name} did not record the fitted feature names."
    )

    preds = fitted.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),), (
        f"{baseline_name}.predict returned shape {preds.shape}, expected ({len(X)},)."
    )
    assert not pd.isna(preds).any(), f"{baseline_name}.predict produced NaN(s)."

    if cfg["task"] in ("classification", "ordinal"):
        proba = fitted.predict_proba(X)
        assert proba.ndim == 2 and proba.shape[0] == len(X), (
            f"{baseline_name}.predict_proba returned shape {proba.shape}."
        )
        assert (proba >= 0).all() and (proba <= 1).all(), (
            f"{baseline_name}.predict_proba emitted values outside [0, 1]."
        )
        np.testing.assert_allclose(
            proba.sum(axis=1),
            np.ones(len(X)),
            # 1e-6, not tighter: XGBoost emits float32 probabilities whose rows
            # sum to 1 only within float32 epsilon (~1e-7) — still a valid
            # distribution; the float64 baselines clear this with room to spare.
            atol=1e-6,
            err_msg=f"{baseline_name}.predict_proba rows do not sum to 1.",
        )
    else:
        with pytest.raises(NotImplementedError):
            fitted.predict_proba(X)


def _classification_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Accuracy — the simplest classification harness metric."""
    return float((np.asarray(y_true) == y_pred).mean())


def _regression_score(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """RMSE — the simplest regression harness metric."""
    residuals = np.asarray(y_true, dtype=float) - y_pred.astype(float)
    return float(np.sqrt((residuals**2).mean()))


def test_all_baselines_drive_a_single_shared_eval_loop(
    synthetic_meeting_frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    """The four baselines can be scored by ONE dispatch loop keyed on ``task``.

    Proves the §6 checklist item literally: "All baselines pass the same
    evaluation harness as ML models" — same loop body, same X frame, same
    ``models.yaml`` factory path, task-branch only at metric selection.
    """
    X, y_class, y_reg = synthetic_meeting_frame
    metrics: dict[str, Callable[[pd.Series, np.ndarray], float]] = {
        "classification": _classification_score,
        "ordinal": _classification_score,
        "regression": _regression_score,
    }
    scores: dict[str, float] = {}
    for name in BASELINE_NAMES:
        cfg = load_model_config(name)
        model = build_model(cfg)
        y = _select_target(cfg["task"], y_class, y_reg)
        model.fit(X, y)
        preds = model.predict(X)
        scores[name] = metrics[cfg["task"]](y, preds)

    assert set(scores) == set(BASELINE_NAMES)
    for name, score in scores.items():
        assert np.isfinite(score), f"{name} produced non-finite score {score!r}."
