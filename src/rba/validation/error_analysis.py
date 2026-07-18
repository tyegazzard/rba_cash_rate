"""§8 error analysis — *when* does the best model fail?

Given a model's per-test-meeting predictions (from
:func:`rba.validation.holdout.evaluate_on_test`) and the feature frame, this
module slices the errors along the three axes the CHECKLIST names:

- **regime** — accuracy inside vs outside each ``regime_*`` / governor-era dummy
  that is active in the test window (COVID, the post-2024 8-meeting cadence, the
  Lowe→Bullock handover, …);
- **cycle phase** — tightening / on-hold / easing, derived from the trailing
  trajectory of the standing cash rate (``prior_rate_pct``), so we can see whether
  the model stumbles at the *turns* of the cycle;
- **surprise meetings** — meetings the *market* got wrong (market-implied call ≠
  outcome), the genuine surprises; the model's accuracy there vs on the
  market-expected meetings shows whether it adds signal exactly when it matters.

Plus a per-meeting **error table** of every misclassified test meeting with its
context, for the qualitative write-up in ``results.md``.

Everything operates on whatever label space the predictions carry (strings for
``three_class``; the report layer normalises the ordinal model's ints to strings
first). No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.validation.metrics import balanced_accuracy

__all__ = ["ErrorAnalysis", "analyse_errors", "cycle_phase"]

_MEETING_KEY = "meeting_date"


@dataclass
class ErrorAnalysis:
    """Bundle of the §8 error slices for one model.

    Attributes
    ----------
    overall : dict
        ``accuracy`` / ``balanced_accuracy`` / ``n`` / ``n_errors`` on the test set.
    per_true_class : pandas.DataFrame
        ``true_class`` / ``support`` / ``n_correct`` / ``recall`` — where the misses
        concentrate by actual decision.
    by_regime : pandas.DataFrame
        One row per active regime flag: accuracy when the flag is on vs off.
    by_cycle_phase : pandas.DataFrame
        ``phase`` / ``n`` / ``accuracy`` over tightening / on-hold / easing.
    by_surprise : pandas.DataFrame or None
        Model accuracy on market-surprise vs market-expected meetings (``None`` if
        no market predictions were supplied).
    errors : pandas.DataFrame
        One row per misclassified test meeting, with regime / phase context.
    """

    overall: dict[str, float]
    per_true_class: pd.DataFrame
    by_regime: pd.DataFrame
    by_cycle_phase: pd.DataFrame
    by_surprise: pd.DataFrame | None
    errors: pd.DataFrame


def cycle_phase(
    prior_rate: pd.Series,
    *,
    lookback: int = 3,
    threshold_pct: float = 0.05,
) -> pd.Series:
    """Label each meeting tightening / easing / on-hold from the rate trajectory.

    Compares the standing rate at each meeting to its value ``lookback`` meetings
    earlier: a rise beyond ``threshold_pct`` is **tightening**, a fall beyond it is
    **easing**, otherwise **on-hold**. Uses only past values (a backward ``diff``),
    so it is leakage-safe; the leading ``lookback`` rows (no history) are labelled
    ``on-hold``.

    Shapes
    ------
    prior_rate: (n,) -> (n,) of {'tightening', 'easing', 'on-hold'}.
    """
    rate = pd.to_numeric(prior_rate, errors="coerce").reset_index(drop=True)
    delta = rate - rate.shift(lookback)
    phase = np.where(
        delta > threshold_pct,
        "tightening",
        np.where(delta < -threshold_pct, "easing", "on-hold"),
    )
    return pd.Series(phase, index=prior_rate.index)


def analyse_errors(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    market_predictions: pd.DataFrame | None = None,
    regime_prefixes: Sequence[str] = ("regime_",),
    prior_rate_col: str = "prior_rate_pct",
) -> ErrorAnalysis:
    """Slice a model's test errors by regime, cycle phase, and market surprise.

    Parameters
    ----------
    predictions
        Per-test-meeting frame with ``meeting_date`` / ``y_true`` / ``y_pred`` (the
        :attr:`~rba.validation.holdout.HoldoutResult.predictions` shape, with
        labels already in the comparison space).
    frame
        The feature frame — supplies the ``regime_*`` dummies and ``prior_rate_col``
        joined onto the predictions on ``meeting_date``.
    market_predictions
        Optional market baseline frame (``meeting_date`` / ``y_pred_market``) for
        the surprise slice; ``None`` skips it.
    regime_prefixes
        Column-name prefixes treated as regime indicators (default ``regime_``).
    prior_rate_col
        Column carrying the standing rate, for :func:`cycle_phase`.

    Returns
    -------
    ErrorAnalysis
    """
    merged = _merge_context(predictions, frame, regime_prefixes, prior_rate_col)
    merged["correct"] = merged["y_true"].to_numpy() == merged["y_pred"].to_numpy()

    overall = {
        "accuracy": float(merged["correct"].mean()),
        "balanced_accuracy": float(
            balanced_accuracy(merged["y_true"].to_numpy(), merged["y_pred"].to_numpy())
        ),
        "n": float(len(merged)),
        "n_errors": float((~merged["correct"]).sum()),
    }

    per_true_class = _per_true_class(merged)
    by_regime = _by_regime(merged, regime_prefixes)
    by_cycle_phase = _by_cycle_phase(merged)
    by_surprise = (
        _by_surprise(merged, market_predictions) if market_predictions is not None else None
    )
    errors = _error_table(merged, regime_prefixes)

    logger.info(
        "analyse_errors: acc={:.3f} bal_acc={:.3f} on {} meetings ({} errors).",
        overall["accuracy"],
        overall["balanced_accuracy"],
        int(overall["n"]),
        int(overall["n_errors"]),
    )
    return ErrorAnalysis(
        overall=overall,
        per_true_class=per_true_class,
        by_regime=by_regime,
        by_cycle_phase=by_cycle_phase,
        by_surprise=by_surprise,
        errors=errors,
    )


# =============================================================================
# Internals.
# =============================================================================
def _merge_context(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    regime_prefixes: Sequence[str],
    prior_rate_col: str,
) -> pd.DataFrame:
    """Join regime dummies + standing rate + cycle phase onto the predictions."""
    regime_cols = [c for c in frame.columns if any(c.startswith(p) for p in regime_prefixes)]
    keep = [
        _MEETING_KEY,
        *([prior_rate_col] if prior_rate_col in frame.columns else []),
        *regime_cols,
    ]
    ctx = frame.loc[:, keep].copy()
    ctx[_MEETING_KEY] = pd.to_datetime(ctx[_MEETING_KEY])

    out = predictions.copy()
    out[_MEETING_KEY] = pd.to_datetime(out[_MEETING_KEY])
    out = out.merge(ctx, on=_MEETING_KEY, how="left").sort_values(_MEETING_KEY)
    out = out.reset_index(drop=True)
    if prior_rate_col in out.columns:
        out["cycle_phase"] = cycle_phase(out[prior_rate_col]).to_numpy()
    else:
        out["cycle_phase"] = "unknown"
    return out


def _per_true_class(merged: pd.DataFrame) -> pd.DataFrame:
    """Support / correct / recall per actual decision class."""
    rows: list[dict[str, Any]] = []
    for cls, grp in merged.groupby("y_true"):
        rows.append(
            {
                "true_class": cls,
                "support": int(len(grp)),
                "n_correct": int(grp["correct"].sum()),
                "recall": float(grp["correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("true_class").reset_index(drop=True)


def _by_regime(merged: pd.DataFrame, regime_prefixes: Sequence[str]) -> pd.DataFrame:
    """Accuracy on / off each regime flag that is active in the test window."""
    regime_cols = [c for c in merged.columns if any(c.startswith(p) for p in regime_prefixes)]
    rows: list[dict[str, Any]] = []
    for col in regime_cols:
        active = merged[col].fillna(0).astype(float) > 0
        n_active = int(active.sum())
        if n_active == 0 or n_active == len(merged):
            continue  # flag not discriminating on the test window — skip.
        rows.append(
            {
                "regime": col,
                "n_active": n_active,
                "acc_active": float(merged.loc[active, "correct"].mean()),
                "n_inactive": int((~active).sum()),
                "acc_inactive": float(merged.loc[~active, "correct"].mean()),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _by_cycle_phase(merged: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per tightening / on-hold / easing phase."""
    rows: list[dict[str, Any]] = []
    for phase, grp in merged.groupby("cycle_phase"):
        rows.append(
            {
                "phase": phase,
                "n": int(len(grp)),
                "accuracy": float(grp["correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def _by_surprise(merged: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Model accuracy on market-surprise vs market-expected meetings.

    A "market surprise" is a covered meeting where the market-implied call missed
    the outcome. Meetings the futures curve doesn't cover (NaN market pred) are
    excluded from both buckets.
    """
    mk = market.copy()
    mk[_MEETING_KEY] = pd.to_datetime(mk[_MEETING_KEY])
    joined = merged.merge(mk[[_MEETING_KEY, "y_pred_market"]], on=_MEETING_KEY, how="left")
    covered = joined["y_pred_market"].notna()
    joined = joined[covered].copy()
    if joined.empty:
        return pd.DataFrame(columns=["bucket", "n", "model_accuracy"])
    market_right = joined["y_pred_market"].to_numpy() == joined["y_true"].to_numpy()
    joined["bucket"] = np.where(market_right, "market_expected", "market_surprise")
    rows: list[dict[str, Any]] = []
    for bucket, grp in joined.groupby("bucket"):
        rows.append(
            {
                "bucket": bucket,
                "n": int(len(grp)),
                "model_accuracy": float(grp["correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def _error_table(merged: pd.DataFrame, regime_prefixes: Sequence[str]) -> pd.DataFrame:
    """Per-meeting detail of every misclassified test meeting, with context."""
    regime_cols = [c for c in merged.columns if any(c.startswith(p) for p in regime_prefixes)]
    errs = merged[~merged["correct"]].copy()
    # A compact "active regimes" label per error row for the write-up.
    active_regimes = errs[regime_cols].fillna(0).astype(float).gt(0) if regime_cols else None
    labels = (
        active_regimes.apply(lambda r: ", ".join(c for c in regime_cols if r[c]), axis=1)
        if active_regimes is not None
        else pd.Series(["" for _ in range(len(errs))], index=errs.index)
    )
    out = pd.DataFrame(
        {
            _MEETING_KEY: errs[_MEETING_KEY].to_numpy(),
            "y_true": errs["y_true"].to_numpy(),
            "y_pred": errs["y_pred"].to_numpy(),
            "cycle_phase": errs["cycle_phase"].to_numpy(),
            "active_regimes": labels.to_numpy(),
        }
    )
    return out.sort_values(_MEETING_KEY).reset_index(drop=True)
