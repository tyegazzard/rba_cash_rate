"""Tests for ``rba.validation.report`` — the §8 orchestration + write-up.

Formatting functions (paired table, confusion markdown, results.md sections) are
tested on hand-built :class:`ModelReport` objects — no model runs. A single small
integration test drives :func:`run_holdout_suite` end-to-end on a synthetic frame
(one model + one baseline, no figures / market / taylor) to prove the compute path
assembles, ranks a best model, and produces an error analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rba.validation import report
from rba.validation.report import ModelReport, SuiteResult


def _report(name: str, kind: str, bal_acc: float, *, tuned: bool = False) -> ModelReport:
    """A minimal ModelReport with a 3-class prediction frame and metrics."""
    dates = pd.date_range("2022-05-01", periods=6, freq="MS")
    y_true = np.array(["hold", "hike", "hold", "cut", "hike", "hold"], dtype=object)
    y_pred = y_true.copy()
    preds = pd.DataFrame({"meeting_date": dates, "y_true": y_true, "y_pred": y_pred})
    for c in report.DISPLAY_CLASSES:
        preds[f"proba_{c}"] = (y_true == c).astype(float)
    metrics = {
        "accuracy": 0.8,
        "balanced_accuracy": bal_acc,
        "macro_f1": bal_acc,
        "log_loss": 0.5,
        "brier_score": 0.3,
        "confusion_matrix": np.eye(3, dtype=int),
    }
    return ModelReport(
        name=name,
        kind=kind,
        tuned=tuned,
        predictions=preds,
        metrics=metrics,
        metrics_tuned=None,
        confusion=np.array([[1, 0, 0], [0, 2, 0], [0, 1, 2]]),
        hit_rate={
            "model_hit_rate": 0.7,
            "market_hit_rate": 0.6,
            "lift": 0.1,
            "only_model_correct": 0.2,
            "only_market_correct": 0.1,
            "both_correct": 0.5,
            "both_wrong": 0.2,
            "n_compared": 6.0,
        },
        ece={c: 0.05 for c in report.DISPLAY_CLASSES},
    )


def _suite() -> SuiteResult:
    reports = [
        _report("xgboost_classifier", "model", 0.62, tuned=True),
        _report("logistic_regression", "model", 0.55),
        _report("voting_ensemble", "ensemble", 0.60),
        _report("majority_class", "baseline", 0.33),
    ]
    return SuiteResult(
        reports=reports,
        best_model_name="xgboost_classifier",
        test_class_distribution={"cut": 3, "hold": 20, "hike": 16},
    )


def test_build_metrics_table_sorts_by_balanced_accuracy() -> None:
    table = report.build_metrics_table(_suite())
    assert list(table.columns) == [
        "model",
        "kind",
        "tuned",
        "n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "brier_score",
        "bal_acc_tuned",
        "market_lift",
    ]
    # Best balanced accuracy first.
    assert table.iloc[0]["model"] == "xgboost_classifier"
    assert table["balanced_accuracy"].is_monotonic_decreasing


def test_confusion_md_renders_true_by_pred() -> None:
    conf = np.array([[1, 0, 0], [0, 2, 0], [0, 1, 2]])
    md = report._confusion_md(conf, report.DISPLAY_CLASSES)
    assert "true \\ pred" in md
    assert "cut" in md and "hold" in md and "hike" in md
    assert md.count("\n") >= 4  # header + sep + 3 class rows


def test_write_results_md_has_all_sections(tmp_path) -> None:
    suite = _suite()
    from rba.validation.error_analysis import analyse_errors

    # Give it a real error analysis so that section renders.
    frame = pd.DataFrame(
        {
            "meeting_date": suite.reports[0].predictions["meeting_date"],
            "prior_rate_pct": np.linspace(0.5, 3.5, 6),
            "regime_post2024_cadence": [0, 0, 0, 1, 1, 1],
        }
    )
    suite.error_analysis = analyse_errors(suite.reports[0].predictions, frame)

    path = report.write_results_md(suite, tmp_path / "results.md", metrics_csv=tmp_path / "m.csv")
    text = path.read_text(encoding="utf-8")
    for heading in (
        "# RBA cash-rate prediction",
        "## Method",
        "## Headline",
        "## Hit rate vs market",
        "## Confusion matrices",
        "## Calibration",
        "## Ensemble vs best individual",
        "## Error analysis",
        "## Limitations",
        "## Artifacts",
    ):
        assert heading in text, f"missing section: {heading}"
    assert (tmp_path / "m.csv").exists()


def test_market_coverage_note_counts_uncovered_meetings() -> None:
    suite = _suite()
    assert "not scored" in report._market_coverage_note(suite)

    suite.market_predictions = pd.DataFrame(
        {
            "meeting_date": pd.date_range("2024-01-01", periods=3, freq="MS"),
            "y_true": ["hold", "hike", "hold"],
            "y_pred_market": np.array(["hold", "hike", "hold"], dtype=object),
        }
    )
    assert "every one of the 3 test meetings" in report._market_coverage_note(suite)

    suite.market_predictions.loc[2, "y_pred_market"] = np.nan
    note = report._market_coverage_note(suite)
    assert "1 of the 3 test meeting has no futures quote and drops" in note

    suite.market_predictions.loc[1, "y_pred_market"] = np.nan
    note = report._market_coverage_note(suite)
    assert "2 of the 3 test meetings have no futures quote and drop " in note


def test_limitations_section_carries_the_coverage_note() -> None:
    suite = _suite()
    text = report._limitations_section(suite)
    assert "- **Market baseline** was not scored in this run." in text
    assert "- **Taylor-rule inputs are proxies**" in text
    assert "~1 test meeting" not in text


# =============================================================================
# Integration — compute path on a small synthetic frame.
# =============================================================================
def _meeting_frame(n_dev: int = 64, n_test: int = 12, p: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    n = n_dev + n_test
    dates = pd.date_range(end="2023-04-01", periods=n, freq="MS")
    rate_change = rng.choice([-25, 0, 25], size=n, p=[0.2, 0.6, 0.2]).astype(float)
    prior = 2.0 + np.cumsum(rng.normal(0.0, 0.05, size=n))
    frame = pd.DataFrame(
        {
            "meeting_date": dates,
            "rate_change_bps": rate_change,
            "prior_rate_pct": prior,
            "regime_covid": (np.arange(n) < 5).astype(int),
        }
    )
    for j in range(p):
        frame[f"f{j}"] = rng.normal(size=n)
    return frame


def test_run_holdout_suite_minimal(tmp_path) -> None:
    frame = _meeting_frame()
    suite = report.run_holdout_suite(
        frame,
        tuned_dir=tmp_path / "tuned",  # empty → default configs, tuned=False
        model_names=["logistic_regression"],
        ensembles=[],
        baselines=["majority_class"],
        include_market=False,
        include_taylor=False,
        runs_dir=tmp_path / "runs",
        write_figures=False,
    )
    names = {r.name for r in suite.reports}
    assert {"logistic_regression", "majority_class"} <= names
    assert suite.best_model_name == "logistic_regression"  # only learned model
    assert suite.error_analysis is not None
    # Metrics table builds and every model row has a balanced_accuracy.
    table = report.build_metrics_table(suite)
    assert table["balanced_accuracy"].notna().all()
