"""§8 Step 1 — the tuning campaign that finalises models on **dev only**.

The §7 turn built the tuning *tools* (:func:`rba.validation.run_search`,
:class:`rba.validation.ThresholdTuner`) and validated them, but never ran a
production sweep. This module is that sweep: for each model it

1. runs :func:`~rba.validation.tuning.run_search` (Optuna) over an inner
   walk-forward on the **dev** portion — the held-out test window is never passed,
   so the search cannot see it (CHECKLIST "never tune on the held-out test set");
2. threshold-tunes per-class decision weights on the tuned model's dev
   out-of-fold probabilities (:func:`~rba.validation.thresholds.collect_oof_proba`
   + :func:`~rba.validation.thresholds.tune_class_weights`); and
3. persists a :class:`TunedModel` record to ``reports/tuned/<model>.json`` — the
   tuned config the §8 held-out evaluation (:mod:`rba.validation.holdout`) then
   scores on the test window.

Everything here is dev-only and reproducible (seeded Optuna). One model failing
its search (e.g. every trial pruned) is logged and **falls back to the default
config** with ``tuned=False`` rather than aborting the campaign.

Ordinal note
------------
``ordinal_logistic`` is tuned on the ``three_class`` target with signed-int labels
(``int_labels=True`` → ``cut < hold < hike`` == sorted order, what mord needs), so
it is directly comparable to the classifiers in the same three-way decision space.

No network anywhere in this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import datetime as dt
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import PROJ_ROOT, RANDOM_SEED, load_model_config, load_target_config
from rba.features.build import dev_test_split
from rba.validation.cv import WalkForwardSplit
from rba.validation.thresholds import collect_oof_proba, tune_class_weights
from rba.validation.tuning import run_search

__all__ = [
    "TunedModel",
    "DEFAULT_MODELS",
    "TUNED_DIR",
    "load_tuned",
    "run_campaign",
    "save_tuned",
    "tune_model",
]

TUNED_DIR = PROJ_ROOT / "reports" / "tuned"

# The seven fittable §7 models tuned for §8 (xrfm is a deferred stub — skipped).
DEFAULT_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest_classifier",
    "xgboost_classifier",
    "lightgbm_classifier",
    "svm_classifier",
    "mlp_classifier",
    "ordinal_logistic",
)

# Models whose target must be the signed-int encoding (ordinal ordering).
_INT_LABEL_MODELS: frozenset[str] = frozenset({"ordinal_logistic"})


@dataclass
class TunedModel:
    """A finalised model: tuned hyper-parameters + tuned decision thresholds.

    JSON-round-trips (numpy coerced to lists/floats). ``best_model_cfg`` is a
    ``models.yaml``-shaped config ready for :func:`rba.models.build_model` /
    :func:`rba.validation.harness.evaluate`'s ``model_cfg`` override.
    """

    model_name: str
    task: str
    int_labels: bool
    tuned: bool  # False when the search failed and the default config was kept.
    search_metric: str
    best_value: float  # best dev walk-forward CV score of ``search_metric``.
    best_params: dict[str, Any]
    best_model_cfg: dict[str, Any]
    n_trials: int
    threshold_metric: str
    class_weights: list[float]
    classes: list[Any]
    baseline_dev_score: float | None = None  # dev CV score of the *default* config.
    timestamp_utc: str | None = None
    git_commit: str | None = None
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot (numpy scalars/arrays coerced)."""
        return json.loads(json.dumps(asdict(self), default=_json_default))


@dataclass
class CampaignResult:
    """Summary of a full :func:`run_campaign` run."""

    tuned: list[TunedModel] = field(default_factory=list)
    output_dir: Path | None = None


def tune_model(
    model_name: str,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    *,
    splitter: WalkForwardSplit,
    n_trials: int = 30,
    timeout: float | None = 300.0,
    search_metric: str = "balanced_accuracy",
    threshold_metric: str = "balanced_accuracy",
    threshold_trials: int = 300,
    random_state: int = RANDOM_SEED,
    int_labels: bool = False,
) -> TunedModel:
    """Tune one model on dev: Optuna hyper-search + OOF threshold weights.

    Parameters
    ----------
    model_name
        A ``models.yaml`` key with a non-empty ``search_space``.
    x_dev, y_dev
        The dev feature matrix / target (never the test window).
    splitter
        Inner walk-forward splitter scoring each trial and the OOF probabilities.
    n_trials, timeout
        Optuna budget (a per-model wall-clock ``timeout`` in seconds bounds the
        bounded sweep — the search stops at whichever of ``n_trials`` / ``timeout``
        comes first).
    search_metric
        Objective for :func:`~rba.validation.tuning.run_search`.
    threshold_metric, threshold_trials
        Objective / budget for :func:`~rba.validation.thresholds.tune_class_weights`.
    random_state
        Seed for both searches.
    int_labels
        Whether ``y_dev`` is the signed-int encoding (ordinal model).

    Returns
    -------
    TunedModel
        The finalised record. If the hyper-search fails (all trials pruned), the
        default config is kept (``tuned=False``) and thresholds are still tuned on
        it — the campaign never hard-fails on one model.
    """
    base_cfg = load_model_config(model_name)
    task = str(base_cfg.get("task", "classification"))

    tuned_ok = True
    try:
        search = run_search(
            model_name=model_name,
            X=x_dev,
            y=y_dev,
            splitter=splitter,
            n_trials=n_trials,
            metric=search_metric,
            random_state=random_state,
            timeout=timeout,
        )
        best_cfg = search.best_model_cfg
        best_params = search.best_params
        best_value = search.best_value
        n_done = search.n_trials
    except (ValueError, RuntimeError) as err:
        logger.warning(
            "tune_model({}): search failed ({}); falling back to default config.",
            model_name,
            err,
        )
        tuned_ok = False
        best_cfg = dict(base_cfg)
        best_params = dict(base_cfg.get("default") or {})
        best_value = float("nan")
        n_done = 0

    # Threshold weights on the tuned model's dev OOF probabilities.
    y_true, proba, classes = collect_oof_proba(
        model_name, x_dev, y_dev, splitter, model_cfg=best_cfg
    )
    weights = tune_class_weights(
        y_true,
        proba,
        classes,
        metric=threshold_metric,
        n_trials=threshold_trials,
        random_state=random_state,
    )

    record = TunedModel(
        model_name=model_name,
        task=task,
        int_labels=int_labels,
        tuned=tuned_ok,
        search_metric=search_metric,
        best_value=float(best_value),
        best_params=best_params,
        best_model_cfg=best_cfg,
        n_trials=int(n_done),
        threshold_metric=threshold_metric,
        class_weights=[float(w) for w in weights],
        classes=[_py(c) for c in classes],
        timestamp_utc=_now_utc_iso(),
        git_commit=None,
    )
    logger.info(
        "tune_model({}): tuned={} best {}={:.4f}, weights={}.",
        model_name,
        tuned_ok,
        search_metric,
        record.best_value,
        dict(zip(record.classes, np.round(weights, 3).tolist())),
    )
    return record


def run_campaign(
    frame: pd.DataFrame,
    target_cfg: Mapping[str, Any] | None = None,
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    output_dir: Path = TUNED_DIR,
    initial_train_size: int = 200,
    step: int = 1,
    n_trials: int = 30,
    timeout: float | None = 300.0,
    random_state: int = RANDOM_SEED,
) -> CampaignResult:
    """Tune every model on dev and persist ``reports/tuned/<model>.json``.

    Parameters
    ----------
    frame
        The meeting-indexed feature frame (``features.parquet``).
    target_cfg
        The classification target (default ``three_class``).
    models
        Which ``models.yaml`` keys to tune (default the seven §7 models).
    output_dir
        Where the ``<model>.json`` records are written.
    initial_train_size, step
        Inner walk-forward geometry over dev (``step > 1`` thins the folds to bound
        the bounded-sweep runtime).
    n_trials, timeout
        Per-model Optuna budget.
    random_state
        Seed.

    Returns
    -------
    CampaignResult
        The tuned records (also written to disk).
    """
    cfg = dict(target_cfg) if target_cfg is not None else load_target_config("three_class")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[TunedModel] = []
    for model_name in models:
        int_labels = model_name in _INT_LABEL_MODELS
        split = dev_test_split(frame, cfg, int_labels=int_labels)
        splitter = WalkForwardSplit(initial_train_size=initial_train_size, test_size=1, step=step)
        logger.info(
            "run_campaign: tuning {} (int_labels={}) on {} dev meetings.",
            model_name,
            int_labels,
            len(split.y_dev),
        )
        record = tune_model(
            model_name,
            split.x_dev,
            split.y_dev,
            splitter=splitter,
            n_trials=n_trials,
            timeout=timeout,
            random_state=random_state,
            int_labels=int_labels,
        )
        save_tuned(record, output_dir / f"{model_name}.json")
        records.append(record)
    logger.success(
        "run_campaign: tuned {}/{} models → {}.",
        sum(1 for r in records if r.tuned),
        len(records),
        output_dir,
    )
    return CampaignResult(tuned=records, output_dir=output_dir)


# =============================================================================
# Persistence.
# =============================================================================
def save_tuned(record: TunedModel, path: Path) -> None:
    """Write a :class:`TunedModel` to ``path`` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    logger.debug("save_tuned: wrote {}.", path)


def load_tuned(model_name: str, tuned_dir: Path = TUNED_DIR) -> TunedModel:
    """Load a persisted :class:`TunedModel` record for ``model_name``.

    Raises
    ------
    FileNotFoundError
        If no ``<model_name>.json`` exists under ``tuned_dir`` (run the campaign
        first).
    """
    path = tuned_dir / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No tuned config for {model_name!r} at {path} — run run_campaign first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return TunedModel(**data)


# =============================================================================
# Small helpers.
# =============================================================================
def _py(value: Any) -> Any:
    """Coerce a numpy scalar to a Python primitive for JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


# =============================================================================
# CLI.
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.validation.campaign`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.validation.campaign",
        description="Run the §8 dev-only tuning campaign and persist reports/tuned/<model>.json.",
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--initial-train-size", type=int, default=200)
    parser.add_argument("--step", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    from rba.features.build import (
        FEATURES_PARQUET_PATH,
        load_master,
    )  # local import: avoids a heavy import at module load.

    args = build_parser().parse_args(argv)
    if FEATURES_PARQUET_PATH.exists():
        frame = pd.read_parquet(FEATURES_PARQUET_PATH)
    else:
        logger.warning("features.parquet absent; rebuilding master frame from cache.")
        frame = load_master()
    run_campaign(
        frame,
        models=args.models,
        n_trials=args.n_trials,
        timeout=args.timeout,
        initial_train_size=args.initial_train_size,
        step=args.step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
