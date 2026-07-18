"""Tests for ``rba.validation.error_analysis`` — the §8 "when does it fail?" slices.

Synthetic per-meeting predictions + a frame carrying a regime dummy and the
standing rate. Covers cycle-phase labelling from the rate trajectory, the regime /
phase / per-class slices, the market-surprise partition, and the per-meeting error
table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rba.validation import error_analysis as ea


def _frame(n: int = 12) -> pd.DataFrame:
    dates = pd.date_range("2022-05-01", periods=n, freq="MS")
    # Rising then falling rate → tightening then easing.
    rate = np.concatenate([np.linspace(0.5, 3.5, n // 2), np.linspace(3.5, 1.0, n - n // 2)])
    regime = (np.arange(n) >= n // 2).astype(int)  # second half in a "new regime"
    return pd.DataFrame(
        {"meeting_date": dates, "prior_rate_pct": rate, "regime_post2024_cadence": regime}
    )


def _predictions(frame: pd.DataFrame, y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"meeting_date": frame["meeting_date"], "y_true": y_true, "y_pred": y_pred}
    )


def test_cycle_phase_detects_tightening_and_easing() -> None:
    frame = _frame(12)
    phase = ea.cycle_phase(frame["prior_rate_pct"])
    assert set(phase.unique()).issubset({"tightening", "easing", "on-hold"})
    # The rising first half should show tightening; the falling second half easing.
    assert (phase == "tightening").any()
    assert (phase == "easing").any()


def test_analyse_errors_overall_and_slices() -> None:
    frame = _frame(12)
    y_true = ["hike"] * 6 + ["cut"] * 6
    # Model gets all hikes right, misses 3 of the cuts.
    y_pred = ["hike"] * 6 + ["cut", "hold", "hold", "cut", "hold", "cut"]
    preds = _predictions(frame, y_true, y_pred)
    result = ea.analyse_errors(preds, frame)

    assert result.overall["n"] == 12
    assert result.overall["n_errors"] == 3
    # Per-class recall: hike perfect, cut degraded.
    recalls = dict(zip(result.per_true_class["true_class"], result.per_true_class["recall"]))
    assert recalls["hike"] == 1.0
    assert recalls["cut"] < 1.0
    # Regime slice present for the discriminating flag.
    assert "regime_post2024_cadence" in set(result.by_regime["regime"])
    # Cycle-phase slice covers tightening + easing.
    assert {"tightening", "easing"} <= set(result.by_cycle_phase["phase"])
    # Error table lists exactly the 3 misclassified meetings.
    assert len(result.errors) == 3
    assert (result.errors["y_true"] == "cut").all()


def test_analyse_errors_surprise_partition() -> None:
    frame = _frame(12)
    y_true = ["hike"] * 6 + ["cut"] * 6
    y_pred = ["hike"] * 6 + ["cut"] * 6  # model perfect
    preds = _predictions(frame, y_true, y_pred)
    # Market expects hold everywhere → surprised on every meeting; one uncovered (NaN).
    market = pd.DataFrame(
        {
            "meeting_date": frame["meeting_date"],
            "y_pred_market": ["hold"] * 11 + [np.nan],
        }
    )
    result = ea.analyse_errors(preds, frame, market_predictions=market)
    assert result.by_surprise is not None
    # All covered meetings are market surprises; the uncovered one is dropped.
    surprise_row = result.by_surprise[result.by_surprise["bucket"] == "market_surprise"]
    assert int(surprise_row["n"].iloc[0]) == 11
    assert float(surprise_row["model_accuracy"].iloc[0]) == 1.0
