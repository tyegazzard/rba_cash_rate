"""Tests for ``rba.validation.holdout`` — the §8 held-out test evaluation path.

Synthetic meeting frames spanning the ``TEST_WINDOW_START`` boundary; no network
(the market-implied derivation is monkeypatched so the raw IB snapshot isn't
required in CI). Covers: the expanding-walk-forward held-out shape scores only the
test meetings and maps them back to ``meeting_date``; the Taylor proxy inputs are
built leakage-safely and the Taylor baseline evaluates on level-regression footing
with a derived direction; the market baseline scores covered test meetings and
returns NaN on uncovered ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.config import load_target_config
from rba.validation import holdout


def _meeting_frame(n_dev: int = 64, n_test: int = 12, p: int = 4) -> pd.DataFrame:
    """A monthly synthetic meeting frame straddling 2022-05-01 (the test boundary)."""
    rng = np.random.default_rng(12)
    n = n_dev + n_test
    # Monthly meetings ending after the boundary so n_test land in the test window.
    dates = pd.date_range(end="2023-04-01", periods=n, freq="MS")
    rate_change = rng.choice([-25, 0, 25], size=n, p=[0.15, 0.7, 0.15]).astype(float)
    prior = 2.0 + np.cumsum(rng.normal(0.0, 0.05, size=n))
    frame = pd.DataFrame(
        {
            "meeting_date": dates,
            "rate_change_bps": rate_change,
            "prior_rate_pct": prior,
            "new_rate_pct": prior + rate_change / 100.0,
            "trimmed_mean_index": np.linspace(100.0, 130.0, n) + rng.normal(0, 0.2, n),
            "unemployment_rate_sa": 5.0 + rng.normal(0.0, 0.3, size=n),
        }
    )
    for j in range(p):
        frame[f"f{j}"] = rng.normal(size=n)
    return frame


def test_evaluate_on_test_scores_only_test_meetings(tmp_path) -> None:
    frame = _meeting_frame()
    result = holdout.evaluate_on_test(
        model_name="majority_class",
        frame=frame,
        output_dir=tmp_path,
        use_mlflow=False,
    )
    # Test window (>= 2022-05-01) is exactly the trailing 12 monthly meetings.
    n_test = int((pd.to_datetime(frame["meeting_date"]) >= pd.Timestamp("2022-05-01")).sum())
    assert len(result.predictions) == n_test
    assert result.result.n_folds == n_test
    assert {"meeting_date", "y_true", "y_pred"} <= set(result.predictions.columns)
    # Every scored meeting is in the test window, in chronological order.
    md = pd.to_datetime(result.predictions["meeting_date"])
    assert (md >= pd.Timestamp("2022-05-01")).all()
    assert md.is_monotonic_increasing
    assert {"accuracy", "balanced_accuracy", "macro_f1"} <= set(result.result.metrics_overall)


def test_evaluate_on_test_empty_window_raises() -> None:
    frame = _meeting_frame(n_dev=20, n_test=0)  # all before the boundary
    frame["meeting_date"] = pd.date_range(end="2020-01-01", periods=len(frame), freq="MS")
    with pytest.raises(ValueError, match="empty test window"):
        holdout.evaluate_on_test(model_name="majority_class", frame=frame, use_mlflow=False)


# =============================================================================
# Taylor proxy inputs + Taylor held-out evaluation.
# =============================================================================
def test_attach_taylor_inputs_builds_leakage_safe_proxies() -> None:
    frame = _meeting_frame()
    out = holdout.attach_taylor_inputs(frame)
    assert "cpi_headline_yoy" in out.columns
    assert "output_gap" in out.columns
    # YoY needs ~a year of history → early rows NaN, later rows finite.
    yoy = out["cpi_headline_yoy"]
    assert yoy.iloc[:6].isna().all()
    assert yoy.iloc[-1] == pytest.approx(yoy.iloc[-1])  # finite (not NaN) late in series
    # Output gap is a past-only rolling deviation → first few NaN, later finite.
    assert out["output_gap"].iloc[-1] == pytest.approx(out["output_gap"].iloc[-1])


def test_taylor_on_test_reports_regression_and_direction(tmp_path) -> None:
    frame = _meeting_frame()
    result = holdout.taylor_on_test(frame, output_dir=tmp_path)
    assert result.result.task == "regression"
    assert {"rmse", "mae", "r2", "directional_accuracy"} <= set(result.result.metrics_overall)
    preds = result.predictions
    assert {"y_pred", "y_pred_direction", "prior_rate_pct"} <= set(preds.columns)
    assert set(preds["y_pred_direction"].unique()).issubset({"cut", "hold", "hike"})


# =============================================================================
# Market-implied held-out scoring (derivation monkeypatched — no raw IB needed).
# =============================================================================
def test_market_implied_on_test_scores_covered_meetings(monkeypatch) -> None:
    frame = _meeting_frame()

    def _fake_attach(f: pd.DataFrame, *, lookback_business_days: int = 1) -> pd.DataFrame:
        out = f.copy()
        # Implied = prior + a small positive drift → the market "expects hikes".
        out["asx_30d_implied_rate"] = out["prior_rate_pct"] + 0.2
        # Leave the last meeting uncovered (NaN) to exercise the drop path.
        out.loc[out.index[-1], "asx_30d_implied_rate"] = np.nan
        return out

    monkeypatch.setattr(holdout, "attach_market_implied", _fake_attach)
    out = holdout.market_implied_on_test(frame, load_target_config("three_class"))
    assert {"meeting_date", "y_true", "y_pred_market"} <= set(out.columns)
    # One uncovered meeting → its market pred is NaN; the rest are hike (drift > 12.5bp).
    covered = out["y_pred_market"].notna()
    assert covered.sum() == len(out) - 1
    assert set(pd.Series(out.loc[covered, "y_pred_market"]).unique()).issubset(
        {"cut", "hold", "hike"}
    )


def test_attach_market_implied_adds_float_column() -> None:
    """The real derivation path: column is added and float-typed (values may be NaN in CI)."""
    frame = _meeting_frame()
    out = holdout.attach_market_implied(frame)
    assert "asx_30d_implied_rate" in out.columns
    assert out["asx_30d_implied_rate"].dtype == float
    assert len(out) == len(frame)
