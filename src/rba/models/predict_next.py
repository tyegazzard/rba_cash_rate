"""Predict the next (not-yet-decided) RBA Board meeting — the serving entry point.

This is the §9 "before-meeting predictions" keystone. Where
:mod:`rba.validation.report` *evaluates* finalised models on the historical
held-out window, this module *serves* the persisted best model on the upcoming
meeting: it builds a leakage-free feature row for the next scheduled meeting date,
loads the deployable artifact from ``reports/best_model/``, and returns a
structured, logged prediction (class probabilities, the market-implied comparison,
and the model's top global drivers).

Pipeline
--------
1. :func:`rba.data.meeting_schedule.next_meeting_date` supplies the upcoming
   announcement date (the pipeline's meeting frame is historical-only).
2. :func:`rba.data.align.append_future_meeting` extends the meeting frame with the
   undecided row; the existing point-in-time join + feature builders populate it
   from data known strictly before that date (no leakage — see that function).
3. :func:`rba.features.build.feature_columns` selects the same leakage-free 681
   columns the model was fitted on; the row is reindexed to
   ``model.feature_names_`` so a missing/extra source degrades gracefully rather
   than breaking LightGBM's column check.
4. The persisted :class:`~rba.models.base.Model` predicts; the rule-based
   :class:`~rba.models.baselines.MarketImplied` baseline is scored alongside it
   (graceful ``None`` when the meeting-month ASX futures contract isn't quoted).

Data freshness
--------------
Predictions run on the **cached** raw snapshots by default (fast, deterministic,
offline). Pass ``force_download=True`` / ``--refresh`` to pull every source live
first; a partial refresh (some RBA endpoints 403 intermittently) is non-fatal —
the prediction proceeds on whatever cache exists, with staleness reported honestly
via each feature's ``_age_days`` (surfaced in the result's ``vintage``).

Import layering
---------------
The project's dependency order is ``config < data < features < models <
validation``. This module lives in ``rba.models`` yet needs
:func:`rba.validation.holdout.attach_market_implied`; that single upward edge is
imported **function-locally** (mirroring holdout.py's own lazy imports), so
``import rba.models.predict_next`` never pulls in ``rba.validation`` and no cycle
forms. It is deliberately NOT re-exported from ``rba.models.__init__``.

CLI: ``uv run python -m rba.models.predict_next [--meeting-date … --refresh
--top-n … --json]``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import datetime as dt
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import RAW_DATA_DIR, REPORTS_DIR

__all__ = [
    "BestModel",
    "PredictionResult",
    "build_next_meeting_features",
    "load_best_model",
    "log_prediction",
    "predict_next_meeting",
]

_BEST_DIR = REPORTS_DIR / "best_model"
_PRED_LOG_PATH = REPORTS_DIR / "predictions" / "predictions.jsonl"
_MEETING_KEY = "meeting_date"


@dataclass(frozen=True)
class BestModel:
    """The deployable model loaded from ``reports/best_model/``.

    Attributes
    ----------
    model : Model
        The fitted estimator (refit on all data by
        :func:`rba.validation.report.persist_best_model`).
    name : str
        The model key from ``best_model.json`` (e.g. ``lightgbm_classifier``).
    manifest : dict
        The full ``best_model.json`` payload (int_labels / tuned / model_cfg / …).
    path : pathlib.Path
        The ``.joblib`` artifact path that was loaded.
    """

    model: Any
    name: str
    manifest: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class NextMeetingFeatures:
    """The leakage-free feature row for one upcoming meeting."""

    meeting_date: pd.Timestamp
    x: pd.DataFrame  # 1-row, columns = feature_columns(features)
    prior_rate_pct: float | None
    future_row: pd.DataFrame  # 1-row: meeting_date + prior_rate_pct (for market baseline)
    vintage: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    """A single before-meeting prediction — the logged, serialisable deliverable."""

    meeting_date: str
    model_name: str
    classes: list[str]
    probabilities: dict[str, float]
    predicted_class: str
    prior_rate_pct: float | None
    market_implied: dict[str, Any] | None
    top_features: list[dict[str, Any]] | None
    vintage: dict[str, Any]
    generated_at_utc: str
    extras: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Model loading — first consumer of reports/best_model/*.joblib.
# =============================================================================
def load_best_model(
    model_path: str | Path | None = None, *, best_dir: Path = _BEST_DIR
) -> BestModel:
    """Load the persisted best model + its manifest from ``reports/best_model/``.

    Parameters
    ----------
    model_path
        Explicit ``.joblib`` path override. ``None`` resolves the artifact named
        in ``best_dir/best_model.json``.
    best_dir
        Directory holding ``best_model.json`` + the ``.joblib`` (default
        ``reports/best_model/``; injectable for tests).

    Returns
    -------
    BestModel

    Raises
    ------
    FileNotFoundError
        If the manifest or the artifact is absent (run
        ``python -m rba.validation.report`` to persist one).
    """
    import joblib

    manifest_path = best_dir / "best_model.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No best-model manifest at {manifest_path}. Run `python -m rba.validation.report` "
            "to persist the best model first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(model_path) if model_path is not None else best_dir / manifest["artifact"]
    if not artifact.exists():
        raise FileNotFoundError(f"Best-model artifact absent at {artifact}.")
    model = joblib.load(artifact)
    name = str(manifest.get("best_model", type(model).__name__))
    logger.info("load_best_model: loaded {} from {}.", name, artifact)
    return BestModel(model=model, name=name, manifest=manifest, path=artifact)


# =============================================================================
# Feature row for the upcoming meeting.
# =============================================================================
def build_next_meeting_features(
    meeting_date: pd.Timestamp | dt.date | str,
    *,
    config: dict[str, Any] | None = None,
    raw_root: Path = RAW_DATA_DIR,
) -> NextMeetingFeatures:
    """Build the leakage-free feature row for the meeting on ``meeting_date``.

    Composes the existing pipeline: extend the historical meeting frame with the
    undecided row (:func:`rba.data.align.append_future_meeting`), run the
    point-in-time join + feature builders, and select the canonical leakage-free
    columns (:func:`rba.features.build.feature_columns`) for the future row.

    Parameters
    ----------
    meeting_date
        The upcoming meeting's announcement date. Must be after the last decided
        meeting in the cached F11 frame.
    config
        A parsed ``features.yaml`` mapping (loaded when omitted).
    raw_root
        The ``data/raw`` root (injectable for tests).

    Returns
    -------
    NextMeetingFeatures

    Raises
    ------
    ValueError
        If ``meeting_date`` is not strictly after the last cached meeting.
    """
    from rba.config import load_features_config
    from rba.data.align import (
        add_regime_dummies,
        append_future_meeting,
        build_master,
        load_meeting_frame,
        load_source_frames,
    )
    from rba.features.build import build_features, feature_columns

    cfg = config if config is not None else load_features_config()
    meeting_ts = pd.Timestamp(meeting_date)

    meeting_frame = load_meeting_frame(raw_root)
    hist_last = pd.to_datetime(meeting_frame[_MEETING_KEY]).max()
    extended = add_regime_dummies(append_future_meeting(meeting_frame, meeting_ts))
    master = build_master(extended, load_source_frames(raw_root))
    features = build_features(master, config=cfg)

    mask = pd.to_datetime(features[_MEETING_KEY]) == meeting_ts
    if not mask.any():
        raise ValueError(f"build_next_meeting_features: no feature row for {meeting_ts.date()}.")
    cols = feature_columns(features)
    x_future = features.loc[mask, cols].copy()
    prior = features.loc[mask, "prior_rate_pct"].iloc[0]
    prior_rate = float(prior) if pd.notna(prior) else None
    future_row = features.loc[mask, [_MEETING_KEY, "prior_rate_pct"]].copy()

    vintage = _vintage(features, mask, hist_last, meeting_ts)
    logger.info(
        "build_next_meeting_features: {} — {} features, {} series missing, median staleness {}d.",
        meeting_ts.date(),
        len(cols),
        vintage["n_series_missing"],
        vintage["feature_staleness_days"]["median"],
    )
    return NextMeetingFeatures(
        meeting_date=meeting_ts,
        x=x_future,
        prior_rate_pct=prior_rate,
        future_row=future_row,
        vintage=vintage,
    )


def _vintage(
    features: pd.DataFrame,
    mask: pd.Series,
    hist_last: pd.Timestamp,
    meeting_ts: pd.Timestamp,
) -> dict[str, Any]:
    """Summarise how fresh / complete the future row's data is (honesty signal)."""
    age_cols = [c for c in features.columns if c.endswith("_age_days")]
    miss_cols = [c for c in features.columns if c.endswith("_is_missing")]
    row = features.loc[mask].iloc[0]
    ages = pd.to_numeric(row[age_cols], errors="coerce").dropna()
    return {
        "master_last_meeting": hist_last.date().isoformat(),
        "future_gap_days": int((meeting_ts - hist_last).days),
        "n_series_missing": int(row[miss_cols].sum()) if miss_cols else 0,
        "feature_staleness_days": {
            "median": float(ages.median()) if len(ages) else None,
            "max": float(ages.max()) if len(ages) else None,
        },
        "git_commit": _git_commit(),
    }


# =============================================================================
# The prediction.
# =============================================================================
def predict_next_meeting(
    meeting_date: pd.Timestamp | dt.date | str | None = None,
    *,
    model_path: str | Path | None = None,
    top_n: int = 10,
    force_download: bool = False,
    config: dict[str, Any] | None = None,
    best_dir: Path = _BEST_DIR,
    raw_root: Path = RAW_DATA_DIR,
) -> PredictionResult:
    """Produce a before-meeting prediction for the next (or a given) RBA meeting.

    Parameters
    ----------
    meeting_date
        The meeting to predict. ``None`` resolves the next scheduled meeting via
        :func:`rba.data.meeting_schedule.next_meeting_date`.
    model_path
        Override for the ``.joblib`` artifact (default: the one in ``best_dir``).
    top_n
        Number of top global feature importances to include.
    force_download
        Refresh every source live before predicting (non-fatal on partial
        failure). Default ``False`` — predict on cached raw.
    config
        Parsed ``features.yaml`` (loaded when omitted).
    best_dir, raw_root
        Injectable directories for tests.

    Returns
    -------
    PredictionResult
    """
    if force_download:
        _refresh_all(raw_root)
    if meeting_date is None:
        from rba.data.meeting_schedule import next_meeting_date

        meeting_date = next_meeting_date()

    best = load_best_model(model_path, best_dir=best_dir)
    nmf = build_next_meeting_features(meeting_date, config=config, raw_root=raw_root)

    x = nmf.x.reindex(columns=list(best.model.feature_names_))
    proba = np.asarray(best.model.predict_proba(x))[0]
    classes = [str(c) for c in best.model.classes_]
    probabilities = {c: float(p) for c, p in zip(classes, proba)}
    predicted = classes[int(np.argmax(proba))]

    market = _market_implied(nmf.future_row)
    top_features = _top_features(best.model, top_n)

    result = PredictionResult(
        meeting_date=nmf.meeting_date.date().isoformat(),
        model_name=best.name,
        classes=classes,
        probabilities=probabilities,
        predicted_class=predicted,
        prior_rate_pct=nmf.prior_rate_pct,
        market_implied=market,
        top_features=top_features,
        vintage=nmf.vintage,
        generated_at_utc=_now_utc_iso(),
    )
    logger.success(
        "predict_next_meeting: {} → {} ({:.1%}).",
        result.meeting_date,
        predicted,
        probabilities[predicted],
    )
    return result


def _refresh_all(raw_root: Path) -> None:
    """Refresh every registered source live; a partial failure is non-fatal."""
    from rba.data.refresh import refresh, select_specs

    logger.info("predict_next: --refresh requested; pulling all sources live ...")
    summary = refresh(select_specs(None), force_download=True)
    if summary.failed:
        logger.warning(
            "predict_next: {} source(s) failed to refresh; predicting on the surviving cache.",
            len(summary.failed),
        )


def _market_implied(future_row: pd.DataFrame) -> dict[str, Any] | None:
    """Score the market-implied baseline on the future row (``None`` if uncovered).

    Lazily imports :func:`rba.validation.holdout.attach_market_implied` — the one
    upward (``models → validation``) edge — so importing this module never loads
    ``rba.validation`` (no cycle). Returns ``None`` when the ASX futures curve does
    not quote the meeting-month contract (the baseline raises on the NaN row).
    """
    try:
        from rba.models.baselines import MarketImplied
        from rba.validation.holdout import (
            attach_market_implied,  # noqa: PLC0415 — lazy: avoids cycle
        )

        framed = attach_market_implied(future_row.copy())
        implied = framed["asx_30d_implied_rate"].iloc[0]
        if pd.isna(implied):
            logger.info("_market_implied: no futures coverage for this meeting month; skipping.")
            return None
        market = MarketImplied()
        market.fit(framed, pd.Series(["cut", "hold", "hike"]))
        proba = market.predict_proba(framed)[0]
        classes = [str(c) for c in market.classes_]
        return {
            "implied_rate_pct": float(implied),
            "probabilities": {c: float(p) for c, p in zip(classes, proba)},
            "predicted_class": classes[int(np.argmax(proba))],
        }
    except (ValueError, KeyError, ImportError, FileNotFoundError) as err:
        logger.warning(
            "_market_implied: unavailable ({}); returning None.", err.__class__.__name__
        )
        return None


def _top_features(model: Any, top_n: int) -> list[dict[str, Any]] | None:
    """Top-``n`` GLOBAL feature importances from the model (not per-meeting SHAP).

    Reads the native importances of the wrapped estimator (``model._model`` for the
    tree wrappers). These are model-level split importances — the features that
    drive the model overall, not a decomposition of *this* prediction. Returns
    ``None`` when the model exposes no native importances (e.g. a baseline).
    """
    inner = getattr(model, "_model", None)
    importances = getattr(inner, "feature_importances_", None)
    if importances is None or not hasattr(model, "feature_names_"):
        return None
    names = list(model.feature_names_)
    imps = np.asarray(importances, dtype=float)
    total = float(imps.sum()) or 1.0
    order = np.argsort(imps)[::-1][: max(top_n, 0)]
    return [
        {
            "feature": names[i],
            "importance": float(imps[i]),
            "importance_pct": round(float(imps[i]) / total * 100.0, 2),
        }
        for i in order
    ]


# =============================================================================
# Prediction log — append-only JSONL audit trail (item §9 "log with timestamp").
# =============================================================================
def log_prediction(result: PredictionResult, path: Path = _PRED_LOG_PATH) -> Path:
    """Append one prediction to the JSONL log; return the log path.

    JSONL (one self-describing object per line) is append-only — no
    read-modify-write, tolerant of schema drift as the result grows — matching the
    repo's JSON-manifest persistence convention.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), default=str) + "\n")
    logger.success("log_prediction: appended {} to {}.", result.meeting_date, path)
    return path


# =============================================================================
# Small helpers.
# =============================================================================
def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    from rba.data.align import _git_commit as align_git_commit

    return align_git_commit()


# =============================================================================
# CLI.
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.models.predict_next`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.models.predict_next",
        description="Predict the next (not-yet-decided) RBA cash-rate decision.",
    )
    parser.add_argument(
        "--meeting-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Meeting to predict (default: next scheduled meeting).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Pull every source live before predicting (default: cached raw).",
    )
    parser.add_argument("--model-path", default=None, help="Override the .joblib artifact path.")
    parser.add_argument(
        "--top-n", type=int, default=10, help="Top global feature drivers to show."
    )
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    parser.add_argument(
        "--no-log", action="store_true", help="Do not append to the prediction log."
    )
    return parser


def _parse_meeting_date(raw: str | None) -> dt.date | None:
    if raw is None:
        return None
    return dt.datetime.strptime(raw, "%Y-%m-%d").date()


def _print_human(result: PredictionResult) -> None:
    """Print a compact human-readable summary of the prediction."""
    probs = result.probabilities
    order = sorted(probs, key=lambda c: probs[c], reverse=True)
    prob_str = "  ".join(f"{c} {probs[c]:.1%}" for c in order)
    prior = f"{result.prior_rate_pct:.2f}%" if result.prior_rate_pct is not None else "n/a"
    # ASCII-only: the Windows console (cp1252) cannot encode arrows / em-dashes.
    print(f"\nRBA cash-rate prediction - next meeting {result.meeting_date}")
    print(f"  Model:         {result.model_name}   |   Prior rate: {prior}")
    print(f"  Probabilities: {prob_str}   ->  {result.predicted_class.upper()}")
    if result.market_implied is not None:
        mp = result.market_implied["probabilities"]
        m_order = sorted(mp, key=lambda c: mp[c], reverse=True)
        m_str = "  ".join(f"{c} {mp[c]:.1%}" for c in m_order)
        print(
            f"  Market-implied:{m_str}   ->  {result.market_implied['predicted_class'].upper()}"
            f"  (implied {result.market_implied['implied_rate_pct']:.2f}%)"
        )
    else:
        print("  Market-implied: unavailable (no futures quote for the meeting month)")
    if result.top_features:
        top = ", ".join(
            f"{f['feature']} ({f['importance_pct']:.1f}%)" for f in result.top_features[:5]
        )
        print(f"  Top drivers (global importance): {top}")
    v = result.vintage
    print(
        f"  Data vintage:  master through {v['master_last_meeting']} ({v['future_gap_days']}d ago), "
        f"median staleness {v['feature_staleness_days']['median']}d, {v['n_series_missing']} series missing"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        meeting_date = _parse_meeting_date(args.meeting_date)
        result = predict_next_meeting(
            meeting_date=meeting_date,
            model_path=args.model_path,
            top_n=args.top_n,
            force_download=args.refresh,
        )
    except (FileNotFoundError, LookupError, ValueError) as exc:
        logger.error("predict_next failed: {}", exc)
        return 1

    if not args.no_log:
        log_prediction(result)
    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
