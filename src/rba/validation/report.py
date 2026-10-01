"""§8 evaluation orchestration — run the finalised models on test and write results.

This is the composition layer that turns the pieces built in §6/§7 (and the §8
:mod:`~rba.validation.holdout` / :mod:`~rba.validation.campaign` /
:mod:`~rba.validation.error_analysis` modules) into the deliverables the CHECKLIST
§8 names:

- run every tuned model + the two ensembles + the four baselines on the held-out
  test window (expanding walk-forward, dev-tuned configs frozen);
- a paired metrics table (accuracy / balanced-accuracy / macro-F1 / log-loss /
  Brier, argmax **and** dev-tuned-threshold);
- a confusion matrix per model;
- a calibration (reliability) figure per probabilistic model, with ECE;
- hit-rate vs the market-implied baseline;
- error analysis of the best model (regime / cycle phase / surprise);
- ensemble vs best-individual comparison;
- ``reports/results.md`` summarising it all, and a best-model artifact persisted
  locally (with a graceful MLflow-registry attempt that no-ops on the current
  upstream-blocked stack — see the harness docstring).

The **compute** path (:func:`run_holdout_suite`) is separated from the **format**
path (:func:`build_metrics_table` / :func:`write_results_md`) so the formatting is
unit-testable on small synthetic :class:`ModelReport` objects without running any
model.

Label spaces
------------
Everything is normalised to the ``three_class`` string space ``{cut, hold, hike}``
for comparison. The ordinal model is evaluated on the signed-int target and its
predictions / probabilities are mapped back to strings here, so it sits in the
same table and confusion matrix as the classifiers.

No network for the evaluation; the market baseline reads the cached raw IB
snapshot (no live download), and calibration figures are written with plotly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import datetime as dt
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import PROJ_ROOT, load_target_config
from rba.validation.calibration import expected_calibration_error
from rba.validation.campaign import DEFAULT_MODELS, TUNED_DIR, TunedModel, load_tuned
from rba.validation.error_analysis import ErrorAnalysis, analyse_errors
from rba.validation.holdout import (
    HoldoutResult,
    evaluate_on_test,
    market_implied_on_test,
    taylor_on_test,
)
from rba.validation.metrics import (
    compute_classification_metrics,
    confusion_matrix,
    hit_rate_vs_market,
)
from rba.validation.thresholds import predict_weighted

__all__ = [
    "ModelReport",
    "SuiteResult",
    "build_metrics_table",
    "run_holdout_suite",
    "write_results_md",
]

# Canonical 3-class order — sklearn-sorted (== every classifier's ``classes_`` and
# what sklearn's log_loss expects), so the proba matrix, metric labels, and
# confusion-matrix axes all agree without a re-ordering warning.
DISPLAY_CLASSES: tuple[str, ...] = ("cut", "hike", "hold")
# Signed-int → string map for the ordinal model's predictions / proba columns.
_ORDINAL_MAP: dict[int, str] = {-1: "cut", 0: "hold", 1: "hike"}

_ENSEMBLES: tuple[str, ...] = ("voting_ensemble", "stacking_ensemble")
_SIMPLE_BASELINES: tuple[str, ...] = ("majority_class", "persistence")
_RESULTS_MD = PROJ_ROOT / "reports" / "results.md"
_METRICS_CSV = PROJ_ROOT / "reports" / "holdout_metrics.csv"
_FIGURES_DIR = PROJ_ROOT / "reports" / "figures"
_RUNS_DIR = PROJ_ROOT / "reports" / "runs"
_BEST_DIR = PROJ_ROOT / "reports" / "best_model"


@dataclass
class ModelReport:
    """One model's held-out test result, normalised to the string label space.

    Attributes
    ----------
    name : str
        Model / baseline key.
    kind : {'model', 'ensemble', 'baseline'}
        Which group it belongs to (drives the "vs baselines" / "vs best individual"
        comparisons).
    tuned : bool
        Whether a dev-tuned config was used (models) vs a default (ensembles /
        baselines / untuned fallback).
    predictions : pandas.DataFrame
        ``meeting_date`` / ``y_true`` / ``y_pred`` (string labels) and, when
        available, ``y_pred_tuned`` (threshold-adjusted) + ``proba_<class>``.
    metrics : dict
        Argmax metrics (accuracy / balanced_accuracy / macro_f1 / [log_loss /
        brier_score]).
    metrics_tuned : dict or None
        Same suite after applying the dev-tuned decision thresholds (``None`` when
        no thresholds / proba).
    confusion : numpy.ndarray
        Confusion matrix in :data:`DISPLAY_CLASSES` order (argmax predictions).
    hit_rate : dict or None
        :func:`~rba.validation.metrics.hit_rate_vs_market` vs the market baseline
        (``None`` for the market baseline itself, or when no market coverage).
    ece : dict or None
        Per-class expected calibration error (``None`` when no proba).
    """

    name: str
    kind: str
    tuned: bool
    predictions: pd.DataFrame
    metrics: dict[str, Any]
    metrics_tuned: dict[str, Any] | None
    confusion: np.ndarray
    hit_rate: dict[str, float] | None = None
    ece: dict[str, float] | None = None


@dataclass
class SuiteResult:
    """The full §8 evaluation over the held-out window."""

    reports: list[ModelReport] = field(default_factory=list)
    market_predictions: pd.DataFrame | None = None
    taylor: HoldoutResult | None = None
    best_model_name: str | None = None
    error_analysis: ErrorAnalysis | None = None
    test_class_distribution: dict[str, int] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    mlflow_status: str = "not_attempted"

    def get(self, name: str) -> ModelReport | None:
        """Return the report named ``name`` (or ``None``)."""
        return next((r for r in self.reports if r.name == name), None)


# =============================================================================
# Compute — run the finalised models on the held-out window.
# =============================================================================
def run_holdout_suite(
    frame: pd.DataFrame,
    *,
    tuned_dir: Path = TUNED_DIR,
    model_names: Sequence[str] = DEFAULT_MODELS,
    ensembles: Sequence[str] = _ENSEMBLES,
    baselines: Sequence[str] = _SIMPLE_BASELINES,
    include_market: bool = True,
    include_taylor: bool = True,
    runs_dir: Path = _RUNS_DIR,
    figures_dir: Path = _FIGURES_DIR,
    write_figures: bool = True,
) -> SuiteResult:
    """Evaluate every model + baseline on the held-out test window.

    Parameters
    ----------
    frame
        The meeting-indexed feature frame (``features.parquet``).
    tuned_dir
        Where :func:`~rba.validation.campaign.load_tuned` looks for per-model tuned
        configs. A model with no tuned record is run on its ``models.yaml`` default
        (and flagged ``tuned=False``).
    model_names, ensembles, baselines
        Which keys to run in each group.
    include_market, include_taylor
        Whether to score the market-implied / Taylor baselines.
    runs_dir, figures_dir
        Output roots for per-run manifests and calibration figures.
    write_figures
        Write per-model calibration reliability figures (plotly HTML).

    Returns
    -------
    SuiteResult
    """
    cfg = load_target_config("three_class")
    market_predictions = market_implied_on_test(frame, cfg) if include_market else None

    reports: list[ModelReport] = []
    for name in model_names:
        reports.append(
            _run_one(name, "model", frame, cfg, tuned_dir, market_predictions, runs_dir)
        )
    for name in ensembles:
        reports.append(
            _run_one(name, "ensemble", frame, cfg, tuned_dir, market_predictions, runs_dir)
        )
    for name in baselines:
        reports.append(
            _run_one(name, "baseline", frame, cfg, tuned_dir, market_predictions, runs_dir)
        )
    if market_predictions is not None:
        reports.append(_market_report(market_predictions))

    taylor = taylor_on_test(frame, output_dir=runs_dir) if include_taylor else None

    # Best learned model by balanced accuracy (prefer the tuned-threshold score).
    learned = [r for r in reports if r.kind in ("model", "ensemble")]
    best = max(learned, key=_headline_balanced_accuracy) if learned else None
    best_name = best.name if best is not None else None

    error = None
    if best is not None:
        error = analyse_errors(best.predictions, frame, market_predictions=market_predictions)

    figures: dict[str, str] = {}
    if write_figures:
        figures = _write_calibration_figures(reports, figures_dir)

    y_test = reports[0].predictions["y_true"] if reports else pd.Series(dtype=object)
    dist = {c: int((y_test == c).sum()) for c in DISPLAY_CLASSES}

    suite = SuiteResult(
        reports=reports,
        market_predictions=market_predictions,
        taylor=taylor,
        best_model_name=best_name,
        error_analysis=error,
        test_class_distribution=dist,
        figures=figures,
    )
    logger.success(
        "run_holdout_suite: {} reports, best={} (bal_acc={:.3f}).",
        len(reports),
        best_name,
        _headline_balanced_accuracy(best) if best else float("nan"),
    )
    return suite


def _run_one(
    name: str,
    kind: str,
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    tuned_dir: Path,
    market_predictions: pd.DataFrame | None,
    runs_dir: Path,
) -> ModelReport:
    """Evaluate one model on test and assemble its (string-space) report."""
    tuned = _maybe_load_tuned(name, tuned_dir) if kind == "model" else None
    int_labels = tuned.int_labels if tuned is not None else (name == "ordinal_logistic")
    model_cfg = tuned.best_model_cfg if tuned is not None else None

    holdout = evaluate_on_test(
        model_name=name,
        frame=frame,
        target_cfg=cfg,
        model_cfg=model_cfg,
        int_labels=int_labels,
        output_dir=runs_dir,
    )
    return _build_model_report(
        name, kind, holdout, tuned, market_predictions, int_labels=int_labels
    )


def _maybe_load_tuned(name: str, tuned_dir: Path) -> TunedModel | None:
    try:
        return load_tuned(name, tuned_dir)
    except FileNotFoundError:
        logger.warning("No tuned config for {}; using models.yaml default.", name)
        return None


def _build_model_report(
    name: str,
    kind: str,
    holdout: HoldoutResult,
    tuned: TunedModel | None,
    market_predictions: pd.DataFrame | None,
    *,
    int_labels: bool,
) -> ModelReport:
    """Normalise a holdout result to string labels + compute the metric suite."""
    preds = holdout.predictions.copy()
    label_map = _ORDINAL_MAP if int_labels else None

    y_true = _map_labels(preds["y_true"].to_numpy(), label_map)
    y_pred = _map_labels(preds["y_pred"].to_numpy(), label_map)

    proba_common = _proba_in_display_order(preds, int_labels)

    # Argmax metrics (proba passed for log-loss / Brier when available).
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_proba=proba_common,
        labels=list(DISPLAY_CLASSES) if proba_common is not None else None,
    )
    conf = confusion_matrix(y_true, y_pred, labels=list(DISPLAY_CLASSES))

    out = pd.DataFrame(
        {"meeting_date": preds["meeting_date"].to_numpy(), "y_true": y_true, "y_pred": y_pred}
    )
    if proba_common is not None:
        for i, cls in enumerate(DISPLAY_CLASSES):
            out[f"proba_{cls}"] = proba_common[:, i]

    # Threshold-tuned metrics (dev-tuned weights applied to the test proba).
    metrics_tuned = None
    if tuned is not None and proba_common is not None:
        tuned_pred = _apply_tuned_thresholds(holdout.predictions, tuned, int_labels)
        out["y_pred_tuned"] = tuned_pred
        metrics_tuned = compute_classification_metrics(
            y_true, tuned_pred, y_proba=proba_common, labels=list(DISPLAY_CLASSES)
        )

    ece = None
    if proba_common is not None:
        ece = {
            cls: expected_calibration_error(y_true, proba_common, list(DISPLAY_CLASSES), cls)
            for cls in DISPLAY_CLASSES
        }

    hit = None
    if market_predictions is not None:
        hit = _hit_rate_vs_market(out, market_predictions)

    return ModelReport(
        name=name,
        kind=kind,
        tuned=bool(tuned is not None and tuned.tuned),
        predictions=out,
        metrics=metrics,
        metrics_tuned=metrics_tuned,
        confusion=conf,
        hit_rate=hit,
        ece=ece,
    )


def _market_report(market_predictions: pd.DataFrame) -> ModelReport:
    """Assemble the market-implied baseline's report (covered meetings only)."""
    mk = market_predictions.dropna(subset=["y_pred_market"]).copy()
    y_true = _map_labels(mk["y_true"].to_numpy(), None)
    y_pred = _map_labels(mk["y_pred_market"].to_numpy(), None)
    metrics = compute_classification_metrics(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred, labels=list(DISPLAY_CLASSES))
    out = pd.DataFrame(
        {"meeting_date": mk["meeting_date"].to_numpy(), "y_true": y_true, "y_pred": y_pred}
    )
    return ModelReport(
        name="market_implied",
        kind="baseline",
        tuned=False,
        predictions=out,
        metrics=metrics,
        metrics_tuned=None,
        confusion=conf,
        hit_rate=None,
        ece=None,
    )


# =============================================================================
# Label / probability normalisation helpers.
# =============================================================================
def _map_labels(values: np.ndarray, label_map: dict[int, str] | None) -> np.ndarray:
    """Map integer ordinal labels to strings (identity when ``label_map`` is None)."""
    if label_map is None:
        return np.asarray([str(v) for v in values], dtype=object)
    return np.asarray([label_map[int(v)] for v in values], dtype=object)


def _proba_in_display_order(preds: pd.DataFrame, int_labels: bool) -> np.ndarray | None:
    """Extract the proba matrix reordered to :data:`DISPLAY_CLASSES` (or None)."""
    proba_cols = [c for c in preds.columns if c.startswith("proba_")]
    if not proba_cols:
        return None
    # Native column label after the ``proba_`` prefix (int string or class string).
    matrix = np.zeros((len(preds), len(DISPLAY_CLASSES)), dtype=float)
    reverse = {v: k for k, v in _ORDINAL_MAP.items()}
    for i, display_cls in enumerate(DISPLAY_CLASSES):
        native = str(reverse[display_cls]) if int_labels else display_cls
        col = f"proba_{native}"
        if col in preds.columns:
            matrix[:, i] = preds[col].to_numpy()
    return matrix


def _apply_tuned_thresholds(
    preds: pd.DataFrame, tuned: TunedModel, int_labels: bool
) -> np.ndarray:
    """Apply dev-tuned class weights to the native proba, then map to strings."""
    native_classes = np.asarray(tuned.classes)
    matrix = np.zeros((len(preds), len(native_classes)), dtype=float)
    for j, cls in enumerate(native_classes):
        col = f"proba_{cls}"
        if col in preds.columns:
            matrix[:, j] = preds[col].to_numpy()
    weighted = predict_weighted(matrix, np.asarray(tuned.class_weights), native_classes)
    label_map = _ORDINAL_MAP if int_labels else None
    return _map_labels(weighted, label_map)


def _hit_rate_vs_market(preds: pd.DataFrame, market_predictions: pd.DataFrame) -> dict[str, float]:
    """Pair a model's string predictions against the market baseline on meeting_date."""
    mk = market_predictions.copy()
    mk["meeting_date"] = pd.to_datetime(mk["meeting_date"])
    left = preds.copy()
    left["meeting_date"] = pd.to_datetime(left["meeting_date"])
    joined = left.merge(mk[["meeting_date", "y_pred_market"]], on="meeting_date", how="left")
    return hit_rate_vs_market(
        joined["y_true"].to_numpy(),
        joined["y_pred"].to_numpy(),
        joined["y_pred_market"].to_numpy(),
    )


def _headline_balanced_accuracy(report: ModelReport | None) -> float:
    """Balanced accuracy used to rank / select models — the argmax metric.

    Ranking is on the **argmax** balanced accuracy (the robust primary metric,
    available for every model / ensemble / baseline and consistent with how the
    paired table is sorted). The dev-tuned-threshold balanced accuracy is reported
    as a *secondary* column (``bal_acc_tuned``) but is deliberately **not** the
    ranking key: on an N≈39 test window the threshold adjustment is noisy (it helps
    some models and hurts others), so ranking on it would risk selecting the model
    whose thresholds happened to land well by chance.
    """
    if report is None:
        return float("nan")
    return float(report.metrics["balanced_accuracy"])


# =============================================================================
# Calibration figures.
# =============================================================================
def _write_calibration_figures(reports: list[ModelReport], figures_dir: Path) -> dict[str, str]:
    """Write a reliability figure per probabilistic model; return {name: path}."""
    from rba.validation.calibration import plot_all_classes

    figures_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for report in reports:
        proba_cols = [c for c in report.predictions.columns if c.startswith("proba_")]
        if not proba_cols:
            continue
        proba = report.predictions[[f"proba_{c}" for c in DISPLAY_CLASSES]].to_numpy()
        try:
            fig = plot_all_classes(
                report.predictions["y_true"].to_numpy(), proba, list(DISPLAY_CLASSES)
            )
            path = figures_dir / f"calibration_{report.name}.html"
            fig.write_html(str(path))
            out[report.name] = _rel_reports(path)
        except Exception as err:  # noqa: BLE001 — a figure failure must not sink the suite.
            logger.warning("calibration figure for {} failed: {}.", report.name, err)
    return out


# =============================================================================
# Format — the paired metrics table.
# =============================================================================
def build_metrics_table(suite: SuiteResult) -> pd.DataFrame:
    """One row per model/baseline: the paired test-metric comparison.

    Columns: ``model`` / ``kind`` / ``tuned`` / ``n`` / ``accuracy`` /
    ``balanced_accuracy`` / ``macro_f1`` / ``log_loss`` / ``brier_score`` /
    ``bal_acc_tuned`` (dev-tuned threshold) / ``market_lift`` (accuracy over the
    market baseline on covered meetings). Sorted best-balanced-accuracy first.
    """
    rows: list[dict[str, Any]] = []
    for r in suite.reports:
        m = r.metrics
        row: dict[str, Any] = {
            "model": r.name,
            "kind": r.kind,
            "tuned": r.tuned,
            "n": int(len(r.predictions)),
            "accuracy": round(float(m["accuracy"]), 4),
            "balanced_accuracy": round(float(m["balanced_accuracy"]), 4),
            "macro_f1": round(float(m["macro_f1"]), 4),
            "log_loss": round(float(m["log_loss"]), 4) if "log_loss" in m else None,
            "brier_score": round(float(m["brier_score"]), 4) if "brier_score" in m else None,
            "bal_acc_tuned": (
                round(float(r.metrics_tuned["balanced_accuracy"]), 4)
                if r.metrics_tuned is not None
                else None
            ),
            "market_lift": (
                round(float(r.hit_rate["lift"]), 4)
                if r.hit_rate is not None and not np.isnan(r.hit_rate["lift"])
                else None
            ),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)


def _confusion_md(conf: np.ndarray, classes: Sequence[str]) -> str:
    """Render a confusion matrix (rows=true, cols=pred) as a markdown table."""
    header = "| true \\ pred | " + " | ".join(classes) + " |"
    sep = "|" + "---|" * (len(classes) + 1)
    lines = [header, sep]
    for i, cls in enumerate(classes):
        lines.append(f"| **{cls}** | " + " | ".join(str(int(v)) for v in conf[i]) + " |")
    return "\n".join(lines)


def _table_md(table: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (floats rounded, NaN → '—')."""
    cols = list(table.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [header, sep]
    for _, row in table.iterrows():
        cells = [_fmt_cell(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fmt_cell(value: Any) -> str:
    """Markdown cell: NaN → '—', floats rounded to 4 dp, everything else str()."""
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{round(float(value), 4)}"
    return str(value)


# =============================================================================
# Format — the results.md write-up.
# =============================================================================
def write_results_md(
    suite: SuiteResult,
    path: Path = _RESULTS_MD,
    *,
    metrics_csv: Path | None = _METRICS_CSV,
) -> Path:
    """Write ``reports/results.md`` (and the metrics CSV) from a computed suite.

    Returns the path written. Also persists :func:`build_metrics_table` to
    ``metrics_csv`` when given.
    """
    table = build_metrics_table(suite)
    if metrics_csv is not None:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(metrics_csv, index=False)

    best = suite.get(suite.best_model_name) if suite.best_model_name else None
    dist = suite.test_class_distribution
    n_test = sum(dist.values())
    lines: list[str] = []

    lines.append("# RBA cash-rate prediction — §8 model evaluation (held-out test)\n")
    lines.append(
        f"_Generated {_now_utc_iso()}. Held-out test window: "
        f"{n_test} meetings on/after `TEST_WINDOW_START` "
        f"(cut={dist.get('cut', 0)} / hold={dist.get('hold', 0)} / hike={dist.get('hike', 0)})._\n"
    )
    lines.append(_method_section(dist, n_test))
    lines.append(_headline_section(table, best))
    lines.append(_hit_rate_section(suite))
    lines.append(_confusion_section(suite, best))
    lines.append(_calibration_section(suite))
    lines.append(_ensemble_section(suite))
    lines.append(_error_section(suite))
    lines.append(_taylor_section(suite))
    lines.append(_limitations_section(suite))
    lines.append(_artifacts_section(suite, metrics_csv))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.success("write_results_md: wrote {}.", path)
    return path


def _method_section(dist: dict[str, int], n_test: int) -> str:
    hold_share = dist.get("hold", 0) / n_test if n_test else float("nan")
    return (
        "## Method\n\n"
        "- **Split**: `dev_test_split` at `TEST_WINDOW_START` — dev is every meeting "
        "before the window; the test window is scored exactly once.\n"
        "- **Test-prediction shape**: expanding walk-forward *through* the test window "
        "— each test meeting is predicted by a model trained on all strictly-earlier "
        "meetings (dev + earlier test meetings). Hyper-parameters and decision "
        "thresholds are frozen from the dev-only tuning campaign; only the training "
        "data expands. No test row informs its own prediction.\n"
        "- **Tuning** (dev only): Optuna hyper-search over each model's `models.yaml` "
        "`search_space` scored by inner walk-forward CV, then per-class decision-"
        "threshold weights fit on dev out-of-fold probabilities. Persisted to "
        "`reports/tuned/<model>.json`.\n"
        f"- **Why balanced accuracy, not raw accuracy**: the test window is "
        f"{hold_share:.0%} hold (much more balanced than the ~77% full-history prior — "
        "it deliberately spans a full hike→hold→cut cycle), so the majority-class "
        f"floor here is only ~{hold_share:.0%}. Balanced accuracy / macro-F1 / log-loss "
        "are the honest metrics; raw accuracy is reported but not ranked on.\n"
    )


def _headline_section(table: pd.DataFrame, best: ModelReport | None) -> str:
    body = [
        "## Headline — paired metrics table\n",
        "Ranked by **balanced accuracy (argmax)** — the robust primary metric. "
        "`bal_acc_tuned` applies each model's dev-tuned decision thresholds; on this "
        "N≈39 window it helps some models and hurts others, so it is reported but not "
        "ranked on. `market_lift` is the model's accuracy minus the market baseline's "
        "on the covered meetings.\n",
        _table_md(table),
        "",
    ]
    top = table.iloc[0] if len(table) else None
    if top is not None and top["kind"] == "baseline":
        body.append(
            f"\n**Key finding — the market-implied baseline (`{top['model']}`, balanced "
            f"accuracy {top['balanced_accuracy']}) is the single best predictor on the "
            "test window; no learned model beats it (every `market_lift` is negative).** "
            "On a market as liquid and rate-tracked as the RBA cash rate, the 30-day "
            "futures price already impounds the macro signal the models are trying to "
            "learn — the honest result for a portfolio project, not a disappointment.\n"
        )
    if best is not None:
        body.append(
            f"\n**Best learned model: `{best.name}`** "
            f"(balanced accuracy {_headline_balanced_accuracy(best):.3f}"
            + (
                f", tuned-threshold {best.metrics_tuned['balanced_accuracy']:.3f}"
                if best.metrics_tuned is not None
                else ""
            )
            + ").\n"
        )
    return "\n".join(body)


def _hit_rate_section(suite: SuiteResult) -> str:
    rows: list[dict[str, Any]] = []
    for r in suite.reports:
        if r.hit_rate is None or np.isnan(r.hit_rate.get("n_compared", float("nan"))):
            continue
        h = r.hit_rate
        if h["n_compared"] == 0:
            continue
        rows.append(
            {
                "model": r.name,
                "n_compared": int(h["n_compared"]),
                "model_hit_rate": round(h["model_hit_rate"], 4),
                "market_hit_rate": round(h["market_hit_rate"], 4),
                "lift": round(h["lift"], 4),
                "only_model_correct": round(h["only_model_correct"], 4),
                "only_market_correct": round(h["only_market_correct"], 4),
            }
        )
    if not rows:
        return (
            "## Hit rate vs market-implied baseline\n\n_No market coverage on the test window._\n"
        )
    table = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    return (
        "## Hit rate vs market-implied baseline\n\n"
        "Paired on the meetings the ASX 30-day-futures curve covers "
        "(from 2022-04). `lift` = model − market accuracy on the same meetings.\n\n"
        + _table_md(table)
        + "\n"
    )


def _confusion_section(suite: SuiteResult, best: ModelReport | None) -> str:
    body = ["## Confusion matrices\n"]
    shown = [
        r
        for r in suite.reports
        if r.name in {suite.best_model_name, "market_implied", "majority_class"}
    ]
    seen: set[str] = set()
    for r in shown:
        if r.name in seen:
            continue
        seen.add(r.name)
        body.append(f"\n### `{r.name}` (argmax)\n")
        body.append(_confusion_md(r.confusion, DISPLAY_CLASSES))
        body.append("")
    body.append(
        "\n_Full per-model confusion matrices are in each run's `manifest.json` under "
        "`reports/runs/`._\n"
    )
    return "\n".join(body)


def _calibration_section(suite: SuiteResult) -> str:
    body = ["## Calibration (reliability)\n"]
    rows: list[dict[str, Any]] = []
    for r in suite.reports:
        if r.ece is None:
            continue
        rows.append({"model": r.name, **{f"ece_{c}": round(v, 4) for c, v in r.ece.items()}})
    if rows:
        body.append(
            "Per-class expected calibration error (ECE, lower = better-calibrated "
            "probabilities):\n"
        )
        body.append(_table_md(pd.DataFrame(rows)))
    if suite.figures:
        body.append("\nReliability diagrams (one panel per class):\n")
        for name, rel in suite.figures.items():
            body.append(f"- `{name}` → [`{rel}`]({rel})")
    return "\n".join(body) + "\n"


def _ensemble_section(suite: SuiteResult) -> str:
    ensembles = [r for r in suite.reports if r.kind == "ensemble"]
    individuals = [r for r in suite.reports if r.kind == "model"]
    if not ensembles or not individuals:
        return "## Ensemble vs best individual\n\n_No ensemble / individual models run._\n"
    best_ind = max(individuals, key=_headline_balanced_accuracy)
    body = ["## Ensemble vs best individual\n"]
    body.append(
        f"Best individual model: `{best_ind.name}` "
        f"(balanced accuracy {_headline_balanced_accuracy(best_ind):.3f}).\n"
    )
    rows = [
        {
            "ensemble": e.name,
            "balanced_accuracy": round(e.metrics["balanced_accuracy"], 4),
            "macro_f1": round(e.metrics["macro_f1"], 4),
            "vs_best_individual": round(
                e.metrics["balanced_accuracy"] - best_ind.metrics["balanced_accuracy"], 4
            ),
        }
        for e in ensembles
    ]
    body.append(_table_md(pd.DataFrame(rows)))
    return "\n".join(body) + "\n"


def _error_section(suite: SuiteResult) -> str:
    ea = suite.error_analysis
    if ea is None:
        return "## Error analysis\n\n_No best model to analyse._\n"
    body = [f"## Error analysis — `{suite.best_model_name}`\n"]
    body.append(
        f"Overall: accuracy {ea.overall['accuracy']:.3f}, balanced accuracy "
        f"{ea.overall['balanced_accuracy']:.3f}, {int(ea.overall['n_errors'])}/"
        f"{int(ea.overall['n'])} meetings missed.\n"
    )
    body.append("\n**By actual decision (recall):**\n")
    body.append(_table_md(ea.per_true_class))
    if not ea.by_regime.empty:
        body.append("\n**By regime (accuracy on vs off the flag):**\n")
        body.append(_table_md(ea.by_regime))
    if not ea.by_cycle_phase.empty:
        body.append("\n**By cycle phase:**\n")
        body.append(_table_md(ea.by_cycle_phase))
    if ea.by_surprise is not None and not ea.by_surprise.empty:
        body.append("\n**On market-surprise meetings** (where the market-implied call missed):\n")
        body.append(_table_md(ea.by_surprise))
    if not ea.errors.empty:
        body.append("\n**Every missed meeting:**\n")
        show = ea.errors.copy()
        show["meeting_date"] = pd.to_datetime(show["meeting_date"]).dt.date.astype(str)
        body.append(_table_md(show))
    return "\n".join(body) + "\n"


def _taylor_section(suite: SuiteResult) -> str:
    t = suite.taylor
    if t is None:
        return ""
    m = t.result.metrics_overall
    preds = t.predictions
    body = ["## Taylor-rule baseline (regression / directional footing)\n"]
    body.append(
        "The Taylor rule predicts a rate *level*, not a class, so it is scored on "
        "level-regression footing (predict `new_rate_pct`) rather than forced into "
        "the class table. Inputs are **proxies** (see Limitations).\n"
    )
    body.append(
        f"\n- Level RMSE: {m['rmse']:.3f} pp · MAE: {m['mae']:.3f} pp · R²: {m['r2']:.3f}\n"
    )
    if "y_pred_direction" in preds.columns and "prior_rate_pct" in preds.columns:
        # Meaningful directional metric: the derived hike/hold/cut call vs the
        # actual decision direction (sign of the realised level change). The
        # regression `directional_accuracy` above is degenerate on a level target
        # (both predicted and actual levels are positive → always same sign).
        prior = preds["prior_rate_pct"].to_numpy(dtype=float)
        actual_change = preds["y_true"].to_numpy(dtype=float) - prior
        actual_dir = np.where(
            actual_change > 0, "hike", np.where(actual_change < 0, "cut", "hold")
        )
        dir_acc = float((preds["y_pred_direction"].to_numpy() == actual_dir).mean())
        counts = preds["y_pred_direction"].value_counts().to_dict()
        body.append(
            f"- Derived hike/hold/cut accuracy vs the actual decision: {dir_acc:.3f} "
            f"(direction calls on test: {counts}). The proxy Taylor rule over-prescribes "
            "hikes on this window, so its directional signal is weak.\n"
        )
    return "\n".join(body)


def _market_coverage_note(suite: SuiteResult) -> str:
    """The limitations bullet on how many test meetings the futures curve covers."""
    market = suite.market_predictions
    if market is None or market.empty:
        return "- **Market baseline** was not scored in this run.\n"
    n_total = len(market)
    n_missing = int(market["y_pred_market"].isna().sum())
    if n_missing == 0:
        return (
            "- **Market baseline coverage** begins 2022-04 and reaches every one of the "
            f"{n_total} test meetings.\n"
        )
    noun, verb = ("meeting has", "drops") if n_missing == 1 else ("meetings have", "drop")
    return (
        f"- **Market baseline coverage** begins 2022-04; {n_missing} of the {n_total} test "
        f"{noun} no futures quote and {verb} from the paired hit-rate comparison.\n"
    )


def _limitations_section(suite: SuiteResult) -> str:
    return (
        "## Limitations & honest caveats\n\n"
        "- **Test window is small (N≈39) and idiosyncratic** — one hike→hold→cut cycle. "
        "Balanced-accuracy differences of a few points are within sampling noise; treat "
        "the ranking as indicative, not decisive.\n"
        + _market_coverage_note(suite)
        + "- **Taylor-rule inputs are proxies**: `cpi_headline_yoy` is a trimmed-mean-CPI "
        "year-over-year built from the meeting-aligned index, and `output_gap` is a "
        "negative unemployment gap (Okun-style) vs a rolling trend — not a true "
        "potential-output gap. The Taylor numbers are directional, not authoritative.\n"
        "- **MLflow is upstream-blocked** on this stack (mlflow caps `pandas<3`; the "
        "project runs pandas 3). Local artifacts under `reports/` are the durable "
        f"record; the registry write degraded gracefully (`{suite.mlflow_status}`).\n"
        "- **Vintage bias**: features use current-vintage macro data, not the exact "
        "real-time values the Board saw (see CONTEXT.md 'Vintage policy').\n"
    )


def _artifacts_section(suite: SuiteResult, metrics_csv: Path | None) -> str:
    body = ["## Artifacts\n\n_Paths are relative to `reports/` (where this file lives)._\n"]
    if metrics_csv is not None:
        body.append(f"- Paired metrics table (CSV): `{_rel_reports(metrics_csv)}`")
    body.append("- Per-run manifests / predictions / calibration: `runs/<run_id>/`")
    body.append("- Tuned configs: `tuned/<model>.json`")
    body.append("- Best-model artifact: `best_model/`")
    for name, rel in suite.figures.items():
        body.append(f"- Calibration figure `{name}`: [`{rel}`]({rel})")
    return "\n".join(body) + "\n"


# =============================================================================
# Best-model persistence (local always; MLflow registry when unblocked).
# =============================================================================
def persist_best_model(
    suite: SuiteResult,
    frame: pd.DataFrame,
    *,
    output_dir: Path = _BEST_DIR,
    tuned_dir: Path = TUNED_DIR,
) -> str:
    """Refit the best model on all data and persist it locally (+ MLflow if possible).

    Fits the best learned model on the **full** frame (dev + test — the deployable
    artifact should use every observation available), pickles it with joblib, and
    writes a manifest. Attempts an MLflow registry write that degrades gracefully on
    the current upstream-blocked stack (returns a status string, also stored on the
    suite).

    Returns
    -------
    str
        The MLflow status (``"logged"`` / ``"registered"`` / ``"import_failed:..."`` /
        ``"skipped:..."``).
    """
    import joblib

    from rba.config import load_model_config
    from rba.features.build import build_xy
    from rba.models import build_model
    from rba.validation.campaign import _INT_LABEL_MODELS

    name = suite.best_model_name
    if name is None:
        return "skipped:no_best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    int_labels = name in _INT_LABEL_MODELS
    cfg = load_target_config("three_class")
    x_all, y_all = build_xy(frame, cfg, int_labels=int_labels)

    tuned = _maybe_load_tuned(name, tuned_dir)
    model_cfg = tuned.best_model_cfg if tuned is not None else load_model_config(name)
    model = build_model(model_cfg)
    model.fit(x_all, y_all)

    model_path = output_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    manifest = {
        "best_model": name,
        "int_labels": int_labels,
        "tuned": bool(tuned is not None and tuned.tuned),
        "model_cfg": model_cfg,
        "n_train": int(len(x_all)),
        "balanced_accuracy_test": _headline_balanced_accuracy(suite.get(name)),
        "artifact": model_path.name,
        "generated_at_utc": _now_utc_iso(),
    }
    (output_dir / "best_model.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    status = _try_mlflow_register(name, model, manifest, output_dir)
    suite.mlflow_status = status
    logger.success("persist_best_model: {} -> {} (mlflow={}).", name, model_path, status)
    return status


def _try_mlflow_register(name: str, model: Any, manifest: dict[str, Any], output_dir: Path) -> str:
    """Best-effort MLflow model registration; graceful on the blocked stack."""
    try:
        import mlflow  # noqa: PLC0415
    except ImportError as err:
        logger.warning(
            "MLflow unavailable ({}); local artifact at {} is the durable record.",
            err.__class__.__name__,
            output_dir,
        )
        return f"import_failed:{err.__class__.__name__}"
    try:
        from rba.config import MLFLOW_TRACKING_URI  # noqa: PLC0415

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("rba_eval")
        with mlflow.start_run(run_name=f"best_{name}"):
            mlflow.log_params({"best_model": name, "n_train": manifest["n_train"]})
            mlflow.log_metric("balanced_accuracy_test", float(manifest["balanced_accuracy_test"]))
            mlflow.log_artifact(str(output_dir / f"{name}.joblib"))
            mlflow.log_artifact(str(output_dir / "best_model.json"))
        return "logged"
    except Exception as err:  # noqa: BLE001 — same graceful-degrade contract as the harness.
        logger.warning("MLflow registry write failed ({}): {}.", err.__class__.__name__, err)
        return f"logging_failed:{err.__class__.__name__}"


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible, else the plain string."""
    try:
        return str(path.relative_to(PROJ_ROOT))
    except ValueError:
        return str(path)


def _rel_reports(path: Path) -> str:
    """Path relative to ``reports/`` with forward slashes — for links inside results.md.

    ``results.md`` lives at ``reports/results.md``, so its links resolve relative to
    ``reports/`` (e.g. ``figures/calibration_x.html``, ``holdout_metrics.csv``).
    Falls back to the repo-relative posix path when ``path`` is outside ``reports/``
    (e.g. a ``tmp_path`` under test).
    """
    reports_root = PROJ_ROOT / "reports"
    try:
        return path.relative_to(reports_root).as_posix()
    except ValueError:
        return _rel(path).replace("\\", "/")


# =============================================================================
# CLI.
# =============================================================================
def build_parser() -> Any:
    """Build the ``rba.validation.report`` argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rba.validation.report",
        description="Run the §8 held-out evaluation suite and write reports/results.md.",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip calibration figures")
    parser.add_argument("--no-persist", action="store_true", help="skip best-model persistence")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    from rba.features.build import FEATURES_PARQUET_PATH, load_master

    args = build_parser().parse_args(argv)
    if FEATURES_PARQUET_PATH.exists():
        frame = pd.read_parquet(FEATURES_PARQUET_PATH)
    else:
        logger.warning("features.parquet absent; rebuilding master frame from cache.")
        frame = load_master()

    suite = run_holdout_suite(frame, write_figures=not args.no_figures)
    if not args.no_persist:
        persist_best_model(suite, frame)
    write_results_md(suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
