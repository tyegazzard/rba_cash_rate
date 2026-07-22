"""Tests for ``rba.models.predict_next`` — the before-meeting serving module.

Covers: model loading, top-feature extraction, the market-implied graceful-NaN
path, result assembly, JSONL logging, the CLI, and a subprocess regression test
proving importing this module does **not** pull in ``rba.validation`` (no cycle).
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rba.models import predict_next as pn


# -----------------------------------------------------------------------------
# Stubs.
# -----------------------------------------------------------------------------
class _StubInner:
    """Stands in for the wrapped sklearn estimator (native importances)."""

    feature_importances_ = np.array([3.0, 1.0, 0.0])


class _StubModel:
    """A minimal Model-shaped stub: classes_, feature_names_, predict_proba, _model."""

    classes_ = np.array(["cut", "hike", "hold"])
    feature_names_ = ["a", "b", "c"]
    _model = _StubInner()

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([[0.1, 0.2, 0.7]])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array(["hold"])


def _stub_nmf(meeting: str = "2026-08-11") -> pn.NextMeetingFeatures:
    ts = pd.Timestamp(meeting)
    return pn.NextMeetingFeatures(
        meeting_date=ts,
        x=pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]}),
        prior_rate_pct=4.35,
        future_row=pd.DataFrame({"meeting_date": [ts], "prior_rate_pct": [4.35]}),
        vintage={
            "master_last_meeting": "2026-06-16",
            "future_gap_days": 56,
            "n_series_missing": 0,
            "feature_staleness_days": {"median": 7.0, "max": 30.0},
            "git_commit": None,
        },
    )


def _result(predicted_class: str = "hold") -> pn.PredictionResult:
    return pn.PredictionResult(
        meeting_date="2026-08-11",
        model_name="stub",
        classes=["cut", "hike", "hold"],
        probabilities={"cut": 0.1, "hike": 0.2, "hold": 0.7},
        predicted_class=predicted_class,
        prior_rate_pct=4.35,
        market_implied=None,
        top_features=None,
        vintage={
            "master_last_meeting": "2026-06-16",
            "future_gap_days": 56,
            "n_series_missing": 0,
            "feature_staleness_days": {"median": 7.0, "max": 30.0},
        },
        generated_at_utc="2026-07-22T00:00:00+00:00",
    )


# -----------------------------------------------------------------------------
# Model loading.
# -----------------------------------------------------------------------------
def test_load_best_model_roundtrip(tmp_path: Path) -> None:
    import joblib

    joblib.dump(_StubModel(), tmp_path / "stub.joblib")
    (tmp_path / "best_model.json").write_text(
        json.dumps({"best_model": "stub_model", "artifact": "stub.joblib"}), encoding="utf-8"
    )
    best = pn.load_best_model(best_dir=tmp_path)
    assert best.name == "stub_model"
    assert best.path.name == "stub.joblib"
    assert isinstance(best.model, _StubModel)


def test_load_best_model_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pn.load_best_model(best_dir=tmp_path)


# -----------------------------------------------------------------------------
# Top features.
# -----------------------------------------------------------------------------
def test_top_features_ranks_and_limits() -> None:
    feats = pn._top_features(_StubModel(), top_n=2)
    assert feats is not None
    assert len(feats) == 2
    assert [f["feature"] for f in feats] == ["a", "b"]  # 3.0, 1.0 — descending
    assert feats[0]["importance"] == 3.0
    assert feats[0]["importance_pct"] == 75.0  # 3 / (3+1+0)


def test_top_features_none_without_native_importances() -> None:
    class _NoInner:
        feature_names_ = ["a"]

    assert pn._top_features(_NoInner(), 5) is None


# -----------------------------------------------------------------------------
# Market-implied — graceful NaN / error handling.
# -----------------------------------------------------------------------------
def _future_row() -> pd.DataFrame:
    return pd.DataFrame({"meeting_date": [pd.Timestamp("2026-08-11")], "prior_rate_pct": [4.35]})


def test_market_implied_none_when_uncovered(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_attach(frame, **kwargs):
        out = frame.copy()
        out["asx_30d_implied_rate"] = np.nan
        return out

    monkeypatch.setattr("rba.validation.holdout.attach_market_implied", fake_attach)
    assert pn._market_implied(_future_row()) is None


def test_market_implied_none_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(frame, **kwargs):
        raise ValueError("no curve")

    monkeypatch.setattr("rba.validation.holdout.attach_market_implied", boom)
    assert pn._market_implied(_future_row()) is None


def test_market_implied_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_attach(frame, **kwargs):
        out = frame.copy()
        out["asx_30d_implied_rate"] = 4.10  # 25bp below prior 4.35 → a full cut
        return out

    monkeypatch.setattr("rba.validation.holdout.attach_market_implied", fake_attach)
    mi = pn._market_implied(_future_row())
    assert mi is not None
    assert mi["implied_rate_pct"] == 4.10
    assert set(mi["probabilities"]) == {"cut", "hike", "hold"}
    assert abs(sum(mi["probabilities"].values()) - 1.0) < 1e-9
    assert mi["predicted_class"] == "cut"


# -----------------------------------------------------------------------------
# Result assembly.
# -----------------------------------------------------------------------------
def test_predict_next_meeting_assembles_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pn,
        "load_best_model",
        lambda *a, **k: pn.BestModel(_StubModel(), "stub", {}, Path("stub.joblib")),
    )
    monkeypatch.setattr(pn, "build_next_meeting_features", lambda md, **k: _stub_nmf())
    monkeypatch.setattr(pn, "_market_implied", lambda row: None)

    res = pn.predict_next_meeting(meeting_date="2026-08-11", top_n=2)
    assert res.meeting_date == "2026-08-11"
    assert res.model_name == "stub"
    assert res.classes == ["cut", "hike", "hold"]
    assert res.predicted_class == "hold"  # argmax of [0.1, 0.2, 0.7]
    assert abs(sum(res.probabilities.values()) - 1.0) < 1e-9
    assert res.prior_rate_pct == 4.35
    assert res.market_implied is None
    assert res.top_features is not None and len(res.top_features) == 2
    assert res.generated_at_utc


def test_predict_next_meeting_resolves_next_when_date_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pn, "load_best_model", lambda *a, **k: pn.BestModel(_StubModel(), "stub", {}, Path("x"))
    )
    monkeypatch.setattr(pn, "build_next_meeting_features", lambda md, **k: _stub_nmf(str(md)))
    monkeypatch.setattr(pn, "_market_implied", lambda row: None)
    monkeypatch.setattr("rba.data.meeting_schedule.next_meeting_date", lambda: date(2026, 8, 11))
    res = pn.predict_next_meeting(meeting_date=None, top_n=1)
    assert res.meeting_date == "2026-08-11"


# -----------------------------------------------------------------------------
# Prediction log.
# -----------------------------------------------------------------------------
def test_log_prediction_appends_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "predictions.jsonl"
    pn.log_prediction(_result(), path=log)
    pn.log_prediction(_result(predicted_class="cut"), path=log)
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["meeting_date"] == "2026-08-11"
    assert first["predicted_class"] == "hold"
    assert json.loads(lines[1])["predicted_class"] == "cut"


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def test_parser_defaults() -> None:
    args = pn.build_parser().parse_args([])
    assert args.meeting_date is None
    assert args.refresh is False
    assert args.top_n == 10
    assert args.no_log is False


def test_cli_logs_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(pn, "predict_next_meeting", lambda **k: _result())
    monkeypatch.setattr(pn, "log_prediction", lambda r, *a, **k: calls.append(r))
    assert pn.main(["--meeting-date", "2026-08-11", "--json"]) == 0
    assert len(calls) == 1


def test_cli_no_log_skips_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(pn, "predict_next_meeting", lambda **k: _result())
    monkeypatch.setattr(pn, "log_prediction", lambda r, *a, **k: calls.append(r))
    assert pn.main(["--no-log", "--json"]) == 0
    assert calls == []


def test_cli_returns_1_on_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(pn, "predict_next_meeting", boom)
    assert pn.main(["--no-log"]) == 1


# -----------------------------------------------------------------------------
# No import cycle (models → validation is lazy).
# -----------------------------------------------------------------------------
def test_importing_predict_next_does_not_import_validation() -> None:
    code = (
        "import sys, rba.models.predict_next; "
        "leaked = sorted(m for m in sys.modules if m.startswith('rba.validation')); "
        "assert not leaked, leaked"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# -----------------------------------------------------------------------------
# End-to-end integration (skips without the cached raw + best model).
# -----------------------------------------------------------------------------
def test_build_next_meeting_features_integration() -> None:
    from rba.config import RAW_DATA_DIR

    pytest.importorskip("pyarrow")
    if not (RAW_DATA_DIR / "rba_f11" / "_metadata.json").exists():
        pytest.skip("no cached raw snapshot")
    if not (pn._BEST_DIR / "best_model.json").exists():
        pytest.skip("no persisted best model")

    from rba.data.meeting_schedule import next_meeting_date

    meeting = next_meeting_date(date(2026, 7, 1))
    nmf = pn.build_next_meeting_features(meeting)
    best = pn.load_best_model()
    x = nmf.x.reindex(columns=list(best.model.feature_names_))
    assert x.shape == (1, len(best.model.feature_names_))
    assert nmf.prior_rate_pct is not None
    proba = best.model.predict_proba(x)[0]
    assert abs(float(proba.sum()) - 1.0) < 1e-6
