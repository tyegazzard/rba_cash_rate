"""§7 evaluation harness — the wrapper that ties every §6/§7 piece together.

:func:`evaluate` is the single call the rest of the project runs to score a
model on a target: it fits the model fold-by-fold via a
:class:`~rba.validation.cv.WalkForwardSplit`, collects predictions and — for
classifiers — probabilities across every fold, computes per-fold and overall
metrics via the :mod:`~rba.validation.metrics` dispatchers, optionally
side-runs a market baseline (typically
:class:`~rba.models.baselines.MarketImplied`) to compute
:func:`~rba.validation.metrics.hit_rate_vs_market`, and persists the whole
result to disk.

**MLflow status.** The checklist item literally says "logs everything to
MLflow." As of 2026-07, MLflow is upstream-blocked on the project's stack:
mlflow 1.27 fails to import on protobuf 7, and mlflow 3.14 (the current
release) caps ``pandas<3`` whereas the project runs pandas 3 — see
CONTEXT.md's "MLflow is wired but upstream-blocked" note. This harness
therefore treats MLflow as the *optional* layer over the local persistence
layer (matching the same graceful-degrade pattern used for
``master.meta.json`` under the same block). When ``use_mlflow=True`` (the
default), the harness attempts an ``import mlflow`` inside :func:`evaluate`;
any :class:`ImportError` / setup failure logs a warning and continues — the
JSON manifest + parquet artifacts under ``reports/runs/<run_id>/`` are always
written and are the primary tracking record until the upstream pins move.

Persistence contract
--------------------
Each :func:`evaluate` call writes three files under
``<output_dir>/<run_id>/`` (default ``reports/runs/<run_id>/``, ``run_id``
default: ``<model>_<target>_<UTC-timestamp>``):

- ``manifest.json`` — human-readable run summary: model / target / splitter
  kwargs, ``n_folds`` / ``n_samples_scored``, ``metrics_overall``,
  ``metrics_per_fold``, ``hit_rate_vs_market`` (if a market model ran),
  optional git commit + timestamp.
- ``predictions.parquet`` — one row per scored sample:
  ``sample_index`` / ``fold_index`` / ``y_true`` / ``y_pred`` / (classification
  only) one ``proba_<class>`` column per class from ``model.classes_``, and
  (if a market model ran) ``y_pred_market``.
- ``calibration.parquet`` — (classification only) per-class per-bin
  :func:`~rba.validation.calibration.reliability_diagram_data` output, with a
  ``class_label`` column added.

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import datetime as dt
import json
from pathlib import Path
import subprocess
from typing import Any
import warnings

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import PROJ_ROOT, load_model_config
from rba.models import Model, build_model
from rba.validation.calibration import reliability_diagram_data
from rba.validation.cv import WalkForwardSplit
from rba.validation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    hit_rate_vs_market,
)

__all__ = ["EvaluationResult", "evaluate"]

_DEFAULT_RUNS_DIR = PROJ_ROOT / "reports" / "runs"


@dataclass
class EvaluationResult:
    """Summary of a single :func:`evaluate` run — the return value + on-disk manifest.

    Every field is JSON-serialisable (numpy floats coerced to ``float``;
    numpy arrays stored as nested lists) so :meth:`to_dict` and the manifest
    round-trip cleanly. ``predictions`` / ``calibration`` are held as
    DataFrames in memory and written as parquet siblings of the manifest.
    """

    run_id: str
    model_name: str
    target_name: str
    task: str
    n_folds: int
    n_samples_scored: int
    splitter_kwargs: dict[str, Any]
    metrics_overall: dict[str, Any]
    metrics_per_fold: list[dict[str, Any]]
    hit_rate_vs_market_result: dict[str, float] | None = None
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame | None = None
    manifest_path: Path | None = None
    predictions_path: Path | None = None
    calibration_path: Path | None = None
    git_commit: str | None = None
    timestamp_utc: str | None = None
    mlflow_status: str = "not_attempted"

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot excluding the ``pd.DataFrame`` payloads."""
        payload = asdict(self)
        for heavy in ("predictions", "calibration"):
            payload.pop(heavy, None)
        for key in ("manifest_path", "predictions_path", "calibration_path"):
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        # confusion_matrix is a numpy array — jsonify.
        for section in (
            (payload.get("metrics_overall") or {}),
            *(payload.get("metrics_per_fold") or []),
        ):
            cm = section.get("confusion_matrix") if isinstance(section, dict) else None
            if isinstance(cm, np.ndarray):
                section["confusion_matrix"] = cm.tolist()
        return payload


# =============================================================================
# The public entry point.
# =============================================================================
def evaluate(
    *,
    model_name: str,
    target_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: WalkForwardSplit,
    market_model_name: str | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    use_mlflow: bool = True,
    experiment_name: str = "rba_eval",
) -> EvaluationResult:
    """Walk-forward-evaluate a model on ``(X, y)`` and persist the result.

    Parameters
    ----------
    model_name
        A ``models.yaml`` key; resolved via :func:`rba.config.load_model_config`
        + :func:`rba.models.build_model` (a **fresh** instance is built per
        fold so the fit is truly out-of-sample).
    target_name
        A ``targets.yaml`` key — carried into the manifest for provenance;
        does not affect the loop (the caller builds ``y`` upstream).
    X
        Feature frame, one row per meeting; positional order **must** be
        chronological (walk-forward relies on it).
    y
        Target aligned to ``X.index``; same positional order.
    splitter
        :class:`~rba.validation.cv.WalkForwardSplit` (or any object with the
        same ``split()`` / ``get_n_splits()`` protocol).
    market_model_name
        Optional ``models.yaml`` key for a market-implied baseline
        (typically ``"market_implied"``). When set, the harness fits and
        scores it in parallel with the main model over the same folds; the
        manifest gains a ``hit_rate_vs_market_result`` field. Rows where the
        market baseline can't predict (NaN in required columns → :class:`ValueError`
        raised at :meth:`~rba.models.baselines.MarketImplied.predict` per its
        coverage-guard behaviour) are recorded as ``y_pred_market = NaN`` and
        drop from :func:`~rba.validation.metrics.hit_rate_vs_market` naturally.
    output_dir
        Root directory for the run subfolder. Default ``reports/runs/``.
    run_id
        Subfolder name; default ``<model>_<target>_<UTC-timestamp>``.
    use_mlflow
        Try to log to MLflow after local persistence. Any ``ImportError``
        (mlflow currently unimportable on this stack, per the module
        docstring) or logging failure is caught and warned; the local
        manifest is unaffected.
    experiment_name
        MLflow experiment name (only used if ``use_mlflow`` succeeds).

    Returns
    -------
    EvaluationResult
        In-memory summary; the JSON manifest at
        :attr:`~EvaluationResult.manifest_path` is the durable artefact.

    Raises
    ------
    ValueError
        On task-mismatch (main model is regression but market model is set,
        or vice versa), or on empty splitter output.
    """
    if len(X) != len(y):
        raise ValueError(
            f"evaluate: len(X)={len(X)} != len(y)={len(y)} — align X to y.index first."
        )

    main_cfg = load_model_config(model_name)
    task = main_cfg.get("task", "classification")
    if task not in ("classification", "regression"):
        raise ValueError(
            f"evaluate: unsupported task {task!r} for model {model_name!r} "
            "(expected 'classification' or 'regression'; ordinal etc. not yet wired)."
        )
    market_cfg = load_model_config(market_model_name) if market_model_name else None
    if market_cfg is not None and task != "classification":
        raise ValueError(
            f"evaluate: market_model_name={market_model_name!r} only compares against "
            f"classification models; main model {model_name!r} is {task!r}."
        )

    fold_records = _run_walk_forward(
        X=X,
        y=y,
        splitter=splitter,
        main_cfg=main_cfg,
        market_cfg=market_cfg,
        task=task,
    )
    if not fold_records:
        raise ValueError(
            f"evaluate: splitter emitted 0 folds on {len(X)} rows — "
            f"initial_train_size or test_size likely too large."
        )

    predictions = _assemble_predictions_frame(fold_records)
    metrics_overall, metrics_per_fold, calibration_data, hitrate = _score_all_folds(
        fold_records=fold_records,
        task=task,
    )

    run_id = run_id or _default_run_id(model_name, target_name)
    output_dir = output_dir or _DEFAULT_RUNS_DIR
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result = EvaluationResult(
        run_id=run_id,
        model_name=model_name,
        target_name=target_name,
        task=task,
        n_folds=len(fold_records),
        n_samples_scored=int(len(predictions)),
        splitter_kwargs=_extract_splitter_kwargs(splitter),
        metrics_overall=metrics_overall,
        metrics_per_fold=metrics_per_fold,
        hit_rate_vs_market_result=hitrate,
        predictions=predictions,
        calibration=calibration_data,
        git_commit=_current_git_commit(),
        timestamp_utc=_now_utc_iso(),
    )
    _persist_local(result, run_dir=run_dir)

    if use_mlflow:
        result.mlflow_status = _try_mlflow_log(result, experiment_name=experiment_name)
    else:
        result.mlflow_status = "disabled_by_caller"

    logger.info(
        "evaluate: model={} target={} n_folds={} n_samples={} run_dir={} mlflow={}.",
        model_name,
        target_name,
        result.n_folds,
        result.n_samples_scored,
        run_dir,
        result.mlflow_status,
    )
    return result


# =============================================================================
# Walk-forward loop.
# =============================================================================
@dataclass
class _FoldRecord:
    fold_index: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None
    classes: np.ndarray | None
    y_pred_market: np.ndarray | None


def _run_walk_forward(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    splitter: Any,
    main_cfg: dict[str, Any],
    market_cfg: dict[str, Any] | None,
    task: str,
) -> list[_FoldRecord]:
    """Iterate every fold; return one :class:`_FoldRecord` per fold."""
    records: list[_FoldRecord] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fresh model per fold — the walk-forward invariant.
        model: Model = build_model(main_cfg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_proba: np.ndarray | None = None
        classes: np.ndarray | None = None
        if task == "classification":
            try:
                y_proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
                classes = np.asarray(model.classes_)  # type: ignore[attr-defined]
            except NotImplementedError:
                y_proba = None
                classes = None

        y_pred_market = (
            _predict_market_or_nan(market_cfg, X_train, y_train, X_test) if market_cfg else None
        )

        records.append(
            _FoldRecord(
                fold_index=fold_idx,
                train_idx=np.asarray(train_idx, dtype=np.int64),
                test_idx=np.asarray(test_idx, dtype=np.int64),
                y_true=np.asarray(y_test),
                y_pred=np.asarray(y_pred),
                y_proba=y_proba,
                classes=classes,
                y_pred_market=y_pred_market,
            )
        )
    return records


def _predict_market_or_nan(
    market_cfg: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Fit and score a market baseline; NaN for rows the baseline can't score.

    The :class:`~rba.models.baselines.MarketImplied` baseline raises
    :class:`ValueError` on NaN in its required columns (ASX IB-futures
    coverage floor). Handle by trying the whole test window first, and on
    failure filling those rows individually — rows that raise become NaN in
    the returned array. That preserves the alignment with ``y_pred`` /
    ``y_true`` so downstream :func:`~rba.validation.metrics.hit_rate_vs_market`
    can drop them cleanly.
    """
    market: Model = build_model(market_cfg)
    market.fit(X_train, y_train)
    try:
        return np.asarray(market.predict(X_test), dtype=object)
    except (ValueError, KeyError) as first_err:
        logger.debug(
            "market baseline failed whole-window predict ({}); falling back to per-row.",
            first_err,
        )
    out = np.full(len(X_test), np.nan, dtype=object)
    for row_pos in range(len(X_test)):
        try:
            out[row_pos] = market.predict(X_test.iloc[[row_pos]])[0]
        except (ValueError, KeyError):
            pass  # leave NaN — filters out at hit_rate_vs_market.
    return out


# =============================================================================
# Aggregation + metric computation.
# =============================================================================
def _assemble_predictions_frame(fold_records: list[_FoldRecord]) -> pd.DataFrame:
    """Concatenate per-fold arrays into one row-per-sample DataFrame."""
    frames: list[pd.DataFrame] = []
    for rec in fold_records:
        df = pd.DataFrame(
            {
                "sample_index": rec.test_idx,
                "fold_index": rec.fold_index,
                "y_true": rec.y_true,
                "y_pred": rec.y_pred,
            }
        )
        if rec.y_proba is not None and rec.classes is not None:
            for col, cls in enumerate(rec.classes):
                df[f"proba_{cls}"] = rec.y_proba[:, col]
        if rec.y_pred_market is not None:
            df["y_pred_market"] = rec.y_pred_market
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _score_all_folds(
    *,
    fold_records: list[_FoldRecord],
    task: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame | None,
    dict[str, float] | None,
]:
    """Compute overall + per-fold metrics, calibration data, and hit-rate."""
    dispatcher: Callable[..., dict[str, Any]]
    if task == "classification":
        dispatcher = compute_classification_metrics
    else:
        dispatcher = compute_regression_metrics

    metrics_per_fold: list[dict[str, Any]] = []
    # Small test folds (default ``test_size=1``) trip sklearn UserWarnings that
    # are inherent to walk-forward CV rather than a bug: single-label folds,
    # ``y_pred`` classes absent from ``y_true``, and R² undefined below 2
    # samples. The *overall* metrics (computed on the concatenation below)
    # don't hit these edge cases; suppress only the per-fold spam here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A single label was found", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true", category=UserWarning
        )
        warnings.filterwarnings("ignore", message=r"R\^2 score is not well-defined")
        for rec in fold_records:
            if task == "classification":
                entry = dispatcher(
                    rec.y_true,
                    rec.y_pred,
                    y_proba=rec.y_proba,
                    labels=rec.classes,
                )
            else:
                entry = dispatcher(rec.y_true, rec.y_pred)
            metrics_per_fold.append({"fold_index": rec.fold_index, **entry})

    all_true = np.concatenate([rec.y_true for rec in fold_records])
    all_pred = np.concatenate([rec.y_pred for rec in fold_records])

    if task == "classification":
        # Use the first fold's ``classes`` as the canonical column order —
        # every baseline in models.yaml produces a stable ``classes_`` sorted
        # order (per :func:`~rba.validation.metrics.confusion_matrix`).
        classes = fold_records[0].classes
        all_proba = None
        if all(rec.y_proba is not None for rec in fold_records):
            all_proba = np.concatenate([rec.y_proba for rec in fold_records])
        metrics_overall = dispatcher(all_true, all_pred, y_proba=all_proba, labels=classes)
        calibration_data = (
            _build_calibration_frame(all_true, all_proba, classes)
            if all_proba is not None
            else None
        )
    else:
        metrics_overall = dispatcher(all_true, all_pred)
        calibration_data = None

    hitrate: dict[str, float] | None = None
    if (
        all(rec.y_pred_market is not None for rec in fold_records)
        and fold_records[0].y_pred_market is not None
    ):
        all_market = np.concatenate([rec.y_pred_market for rec in fold_records])  # type: ignore[misc]
        hitrate = hit_rate_vs_market(all_true, all_pred, all_market)

    return metrics_overall, metrics_per_fold, calibration_data, hitrate


def _build_calibration_frame(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """One :func:`reliability_diagram_data` table per class, concatenated with a ``class_label`` column."""
    frames: list[pd.DataFrame] = []
    for cls in classes:
        table = reliability_diagram_data(y_true, y_proba, classes, cls)
        table = table.assign(class_label=cls)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Persistence — local (always) + MLflow (optional / degrades gracefully).
# =============================================================================
def _persist_local(result: EvaluationResult, *, run_dir: Path) -> None:
    """Write the manifest.json + predictions.parquet (+ calibration.parquet) triple."""
    result.manifest_path = run_dir / "manifest.json"
    result.predictions_path = run_dir / "predictions.parquet"
    if result.calibration is not None:
        result.calibration_path = run_dir / "calibration.parquet"

    result.predictions.to_parquet(result.predictions_path, index=False)
    if result.calibration is not None and result.calibration_path is not None:
        result.calibration.to_parquet(result.calibration_path, index=False)

    with result.manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_manifest_dict(), fh, indent=2, default=_json_default)


def _try_mlflow_log(result: EvaluationResult, *, experiment_name: str) -> str:
    """Best-effort MLflow logging; returns a status string for the manifest."""
    try:
        import mlflow  # noqa: PLC0415 — import inside the try so ImportError is caught here.
    except ImportError as err:
        logger.warning(
            "MLflow unavailable ({}). Local persistence at {} is the tracking record. "
            "Re-enable MLflow once a released mlflow supports pandas 3 (see CONTEXT.md).",
            err.__class__.__name__,
            result.manifest_path,
        )
        return f"import_failed:{err.__class__.__name__}"

    try:
        from rba.config import MLFLOW_TRACKING_URI  # noqa: PLC0415

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=result.run_id):
            mlflow.log_params(
                {
                    "model": result.model_name,
                    "target": result.target_name,
                    "task": result.task,
                    "n_folds": result.n_folds,
                    "n_samples_scored": result.n_samples_scored,
                    **{f"splitter_{k}": v for k, v in result.splitter_kwargs.items()},
                }
            )
            scalar_metrics = {
                k: float(v)
                for k, v in result.metrics_overall.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            mlflow.log_metrics(scalar_metrics)
            if result.hit_rate_vs_market_result is not None:
                mlflow.log_metrics(
                    {
                        f"hitrate_{k}": float(v)
                        for k, v in result.hit_rate_vs_market_result.items()
                        if isinstance(v, (int, float))
                    }
                )
            for artifact in (
                result.manifest_path,
                result.predictions_path,
                result.calibration_path,
            ):
                if artifact is not None:
                    mlflow.log_artifact(str(artifact))
        return "logged"
    except Exception as err:  # noqa: BLE001 — same graceful-degrade contract as the frame-hash step.
        logger.warning(
            "MLflow logging failed ({}): {}. Local persistence at {} is the tracking record.",
            err.__class__.__name__,
            err,
            result.manifest_path,
        )
        return f"logging_failed:{err.__class__.__name__}"


# =============================================================================
# Small helpers.
# =============================================================================
def _extract_splitter_kwargs(splitter: Any) -> dict[str, Any]:
    """Best-effort snapshot of the splitter's config for the manifest."""
    if isinstance(splitter, WalkForwardSplit):
        return {
            "kind": "WalkForwardSplit",
            "initial_train_size": splitter.initial_train_size,
            "test_size": splitter.test_size,
            "step": splitter.step,
            "expanding": splitter.expanding,
            "max_train_size": splitter.max_train_size,
        }
    return {"kind": type(splitter).__name__}


def _default_run_id(model_name: str, target_name: str) -> str:
    """``<model>_<target>_<UTC-timestamp>`` — same convention as CONTEXT.md."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_name}_{target_name}_{stamp}"


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _current_git_commit() -> str | None:
    """Return the current HEAD commit; ``None`` if not in a git repo or git fails."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJ_ROOT,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback — numpy scalars, arrays, Paths, non-string dict keys."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")
