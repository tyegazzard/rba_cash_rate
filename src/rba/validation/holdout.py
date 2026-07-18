"""§8 held-out test evaluation — score finalised models on the untouchable window.

Where :func:`rba.validation.harness.evaluate` runs walk-forward CV over whatever
``(X, y)`` it is handed, this module wires that engine into the **§8 shape**: a
model is finalised on ``dev`` (config + thresholds tuned there, never on test) and
scored on the held-out window (:data:`rba.config.TEST_WINDOW_START`, 39 meetings)
exactly once.

Test-prediction shape (documented decision)
-------------------------------------------
Per the CONTEXT.md §8 note, test predictions use an **expanding walk-forward that
ends on the test rows**: :func:`evaluate_on_test` concatenates ``dev`` then
``test`` in chronological order and drives a
:class:`~rba.validation.cv.WalkForwardSplit` whose first training window is *all*
of dev (``initial_train_size = len(dev)``), stepping one meeting at a time with
``test_size=1``. So test meeting *t* is predicted by a model trained on every
meeting strictly before *t* (all of dev plus the earlier test meetings) — each
test row is scored exactly once, never leaks future information, and the setup
mirrors realistic "retrain before each meeting" deployment. Hyper-parameters are
frozen (from the dev tuning campaign) — only the training *data* expands.

The two data-dependent baselines are handled deliberately rather than forced
through the shared ``X``:

- :func:`attach_market_implied` derives ``asx_30d_implied_rate`` per meeting from
  the raw ASX 30-day IB-futures curve (never aligned into ``features.parquet``)
  and joins it on ``meeting_date``. :func:`market_implied_on_test` then scores the
  :class:`~rba.models.baselines.MarketImplied` baseline on the covered test
  meetings — futures coverage begins 2022-04-21, so ~1 test meeting with no quote
  yields NaN and drops from the paired comparison.
- :func:`attach_taylor_inputs` builds the two Taylor-rule proxy regressors
  (``cpi_headline_yoy`` from a trimmed-mean-CPI year-over-year, ``output_gap`` from
  a negative unemployment gap) that the feature frame does not carry, so the
  Taylor baseline can be scored on ``level_regression`` footing (see
  :func:`taylor_on_test`).

No network for the model evaluation itself; :func:`attach_market_implied` reads
the cached raw IB snapshot (``force_download=False``) — no live download.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import load_model_config, load_target_config
from rba.features.build import dev_test_split, feature_columns
from rba.validation.cv import WalkForwardSplit
from rba.validation.harness import EvaluationResult, evaluate

__all__ = [
    "HoldoutResult",
    "attach_market_implied",
    "attach_taylor_inputs",
    "evaluate_on_test",
    "market_implied_on_test",
    "taylor_on_test",
]

_MEETING_KEY = "meeting_date"
_IMPLIED_COL = "asx_30d_implied_rate"


@dataclass
class HoldoutResult:
    """A single model's held-out test evaluation.

    Attributes
    ----------
    result : EvaluationResult
        The underlying :func:`~rba.validation.harness.evaluate` result — its
        ``metrics_overall`` are the **test** metrics (only the test folds were
        scored), and it carries the persisted manifest / predictions / calibration
        paths.
    predictions : pandas.DataFrame
        One row per scored test meeting: ``meeting_date`` / ``y_true`` / ``y_pred``
        and (classification / ordinal) one ``proba_<class>`` column per class.
        Indexed 0..n_test-1 in chronological order.
    """

    result: EvaluationResult
    predictions: pd.DataFrame


# =============================================================================
# Market-implied rate — derived from the raw IB curve, joined on meeting_date.
# =============================================================================
def attach_market_implied(
    frame: pd.DataFrame,
    *,
    lookback_business_days: int = 1,
) -> pd.DataFrame:
    """Return ``frame`` with an added ``asx_30d_implied_rate`` column.

    Derives the market-implied cash rate per meeting from the cached raw ASX
    30-day IB-futures curve (:func:`rba.data.sources.asx_ib_futures.fetch` with
    ``force_download=False``) via
    :func:`~rba.data.sources.asx_ib_futures.derive_meeting_implied`, and joins it
    on ``meeting_date``. Meetings before futures coverage (2022-04-21) — i.e. the
    whole dev window — get NaN, which is exactly what the
    :class:`~rba.models.baselines.MarketImplied` coverage guard expects.

    Parameters
    ----------
    frame
        A meeting-indexed feature frame carrying ``meeting_date``.
    lookback_business_days
        Trading days strictly before the meeting to read the curve from (default
        1 — the T-1 close, publicly known before the meeting; no leakage).

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with the extra ``asx_30d_implied_rate`` column. If the
        raw IB snapshot is unreadable, logs a warning and adds an all-NaN column
        (so the market baseline degrades to "no coverage" rather than crashing).

    Shapes
    ------
    frame: (n_meetings, c) -> (n_meetings, c + 1).
    """
    out = frame.copy()
    try:
        from rba.data.sources import asx_ib_futures as ib

        long_df = ib.fetch(force_download=False)
        meetings = pd.DataFrame({"observation_date": pd.to_datetime(out[_MEETING_KEY])})
        implied = ib.derive_meeting_implied(
            long_df, meetings, lookback_business_days=lookback_business_days
        )
        mapping = dict(zip(pd.to_datetime(implied[_MEETING_KEY]), implied["implied_cash_rate"]))
        out[_IMPLIED_COL] = pd.to_datetime(out[_MEETING_KEY]).map(mapping).astype(float)
        n_covered = int(out[_IMPLIED_COL].notna().sum())
        logger.info(
            "attach_market_implied: {}/{} meetings carry a futures-implied rate.",
            n_covered,
            len(out),
        )
    except (FileNotFoundError, ValueError, ImportError) as err:
        logger.warning(
            "attach_market_implied: could not derive implied rate ({}); adding all-NaN column.",
            err.__class__.__name__,
        )
        out[_IMPLIED_COL] = np.nan
    return out


def market_implied_on_test(
    frame: pd.DataFrame,
    target_cfg: Mapping[str, Any] | None = None,
    *,
    lookback_business_days: int = 1,
    labels: tuple[Any, Any, Any] = ("cut", "hold", "hike"),
) -> pd.DataFrame:
    """Score the market-implied baseline on the held-out test meetings.

    The :class:`~rba.models.baselines.MarketImplied` baseline is rule-based (no
    learning), so it is scored directly rather than through the walk-forward
    engine: it is fit once on dev (to materialise ``classes_``) and predicts each
    test meeting, with rows the futures curve doesn't cover (NaN implied rate)
    returned as NaN so they drop from the paired hit-rate comparison.

    Parameters
    ----------
    frame
        A meeting-indexed feature frame carrying ``meeting_date`` and
        ``prior_rate_pct`` (``asx_30d_implied_rate`` is derived here).
    target_cfg
        The classification target (default ``three_class``). Its labels must match
        ``labels``.
    lookback_business_days
        Passed to :func:`attach_market_implied`.
    labels
        ``(cut, hold, hike)`` label values in ``y`` — defaults to the ``three_class``
        strings. Use ``(-1, 0, 1)`` for the int-encoded target.

    Returns
    -------
    pandas.DataFrame
        Columns ``meeting_date`` / ``y_true`` / ``y_pred_market``; one row per test
        meeting (``y_pred_market`` NaN where the curve doesn't cover it).
    """
    from rba.models.baselines import MarketImplied

    cfg = dict(target_cfg) if target_cfg is not None else load_target_config("three_class")
    cut, hold, hike = labels
    frame_m = attach_market_implied(frame, lookback_business_days=lookback_business_days)
    split = dev_test_split(frame_m, cfg)

    # Rule-based baseline built directly with the requested label space (int vs string).
    market = MarketImplied(cut_label=cut, hold_label=hold, hike_label=hike)
    market.fit(split.x_dev, split.y_dev)

    x_test = split.x_test
    preds = np.full(len(x_test), np.nan, dtype=object)
    for pos in range(len(x_test)):
        try:
            preds[pos] = market.predict(x_test.iloc[[pos]])[0]
        except (ValueError, KeyError):
            pass  # uncovered meeting → NaN, drops from the paired comparison.

    meeting_dates = pd.to_datetime(frame_m.loc[x_test.index, _MEETING_KEY]).to_numpy()
    out = pd.DataFrame(
        {
            _MEETING_KEY: meeting_dates,
            "y_true": split.y_test.to_numpy(),
            "y_pred_market": preds,
        }
    )
    logger.info(
        "market_implied_on_test: scored {}/{} covered test meetings.",
        int(pd.Series(preds).notna().sum()),
        len(out),
    )
    return out


# =============================================================================
# Taylor-rule proxy inputs — the feature frame carries neither, so build them.
# =============================================================================
def attach_taylor_inputs(
    frame: pd.DataFrame,
    *,
    cpi_col: str = "trimmed_mean_index",
    unemployment_col: str = "unemployment_rate_sa",
    yoy_lookback_days: int = 365,
    gap_window: int = 60,
) -> pd.DataFrame:
    """Return ``frame`` with the two Taylor-rule proxy regressors added.

    The feature frame carries CPI as an index *level* and unemployment as a rate,
    but not the ``cpi_headline_yoy`` / ``output_gap`` the
    :class:`~rba.models.baselines.TaylorRule` baseline reads. This builds honest,
    leakage-safe proxies from the meeting-aligned as-of values:

    - ``cpi_headline_yoy`` — year-over-year % change of the trimmed-mean CPI index
      (the RBA's preferred core measure): for each meeting, the as-of index divided
      by the as-of index at the most recent meeting ``>= yoy_lookback_days`` earlier
      (a backward :func:`~pandas.merge_asof`, so only past index values are used).
    - ``output_gap`` — a **proxy** ``-(u - ū)`` where ``ū`` is a trailing
      ``gap_window``-meeting rolling mean of the unemployment rate (past-only,
      Invariant #3). The sign follows Okun's law (unemployment above trend ⇒
      negative output gap ⇒ economy running cold). This is a stand-in for a true
      potential-output gap, documented as such in ``results.md``.

    Parameters
    ----------
    frame
        Meeting-indexed feature frame carrying ``meeting_date``, ``cpi_col`` and
        ``unemployment_col``.
    cpi_col, unemployment_col
        Source column names for the two proxies.
    yoy_lookback_days
        Calendar-day lookback for the YoY denominator (default 365).
    gap_window
        Rolling window (in meetings) for the unemployment trend.

    Returns
    -------
    pandas.DataFrame
        A copy of ``frame`` with ``cpi_headline_yoy`` and ``output_gap`` added.

    Shapes
    ------
    frame: (n_meetings, c) -> (n_meetings, c + 2).
    """
    out = frame.sort_values(_MEETING_KEY).reset_index(drop=True).copy()
    dates = pd.to_datetime(out[_MEETING_KEY])

    # YoY inflation via a backward as-of self-join: match each meeting to the most
    # recent meeting >= yoy_lookback_days earlier and divide the index levels.
    left = pd.DataFrame({"date": dates, "idx_now": out[cpi_col].to_numpy()})
    right = pd.DataFrame(
        {"date": dates + pd.Timedelta(days=yoy_lookback_days), "idx_prev": out[cpi_col].to_numpy()}
    )
    joined = pd.merge_asof(
        left.sort_values("date"),
        right.sort_values("date"),
        on="date",
        direction="backward",
    )
    yoy = (joined["idx_now"] / joined["idx_prev"] - 1.0) * 100.0
    out["cpi_headline_yoy"] = yoy.to_numpy()

    # Output-gap proxy: negative unemployment gap vs a past-only rolling trend.
    u = out[unemployment_col]
    trend = u.rolling(window=gap_window, min_periods=4).mean()
    out["output_gap"] = -(u - trend).to_numpy()

    logger.debug(
        "attach_taylor_inputs: cpi_headline_yoy non-null {}/{}, output_gap non-null {}/{}.",
        int(out["cpi_headline_yoy"].notna().sum()),
        len(out),
        int(out["output_gap"].notna().sum()),
        len(out),
    )
    return out


def taylor_on_test(
    frame: pd.DataFrame,
    *,
    fit_coefficients: bool = True,
    directional_threshold_pct: float = 0.125,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> HoldoutResult:
    """Evaluate the Taylor-rule baseline on the held-out window (level regression).

    Builds the proxy inputs (:func:`attach_taylor_inputs`), then scores the
    :class:`~rba.models.baselines.TaylorRule` baseline on the ``level_regression``
    target (predict ``new_rate_pct``) via the expanding held-out walk-forward.
    Because Taylor predicts a rate *level*, its 3-class direction is derived from
    ``sign(predicted_level - prior_rate_pct)`` with a ``directional_threshold_pct``
    dead-band, so it can also sit beside the classifiers directionally.

    Returns
    -------
    HoldoutResult
        ``result.metrics_overall`` holds the regression metrics (rmse / mae / r2 /
        directional_accuracy); ``predictions`` adds ``prior_rate_pct`` and a derived
        ``y_pred_direction`` (cut/hold/hike) column alongside the level prediction.
    """
    taylor_cfg = dict(load_model_config("taylor_rule"))
    taylor_cfg["default"] = {
        **(taylor_cfg.get("default") or {}),
        "fit_coefficients": fit_coefficients,
    }
    frame_t = attach_taylor_inputs(frame)
    level_cfg = load_target_config("level_regression")

    holdout = evaluate_on_test(
        model_name="taylor_rule",
        frame=frame_t,
        target_cfg=level_cfg,
        model_cfg=taylor_cfg,
        output_dir=output_dir,
        run_id=run_id,
    )
    # Derive the directional 3-class call from the predicted level vs the standing rate.
    preds = holdout.predictions
    prior_lookup = pd.DataFrame(
        {
            _MEETING_KEY: pd.to_datetime(frame_t[_MEETING_KEY]),
            "prior_rate_pct": pd.to_numeric(frame_t["prior_rate_pct"], errors="coerce"),
        }
    )
    merged = pd.DataFrame({_MEETING_KEY: pd.to_datetime(preds[_MEETING_KEY])}).merge(
        prior_lookup, on=_MEETING_KEY, how="left"
    )
    prior = merged["prior_rate_pct"].to_numpy(dtype=float)
    delta = preds["y_pred"].to_numpy(dtype=float) - prior
    direction = np.where(
        delta > directional_threshold_pct,
        "hike",
        np.where(delta < -directional_threshold_pct, "cut", "hold"),
    )
    preds["prior_rate_pct"] = prior
    preds["y_pred_direction"] = direction
    return holdout


# =============================================================================
# The general held-out evaluation entry point.
# =============================================================================
def evaluate_on_test(
    *,
    model_name: str,
    frame: pd.DataFrame,
    target_cfg: Mapping[str, Any] | None = None,
    model_cfg: dict[str, Any] | None = None,
    int_labels: bool = False,
    initial_train_size: int | None = None,
    output_dir: Path | None = None,
    run_id: str | None = None,
    use_mlflow: bool = False,
) -> HoldoutResult:
    """Expanding-walk-forward held-out evaluation of one model on the test window.

    Splits ``frame`` into dev / test (:func:`rba.features.build.dev_test_split`),
    concatenates them chronologically, and drives
    :func:`rba.validation.harness.evaluate` with a
    :class:`~rba.validation.cv.WalkForwardSplit` starting at ``initial_train_size``
    (default ``len(dev)``) so that only the test meetings are scored, each trained
    on strictly-earlier data (see the module docstring).

    Parameters
    ----------
    model_name
        A ``models.yaml`` key (used for the run id / manifest / task lookup).
    frame
        The meeting-indexed feature frame (already carrying any baseline-specific
        columns the model reads, e.g. via :func:`attach_taylor_inputs`).
    target_cfg
        A ``targets.yaml`` entry (default ``three_class``).
    model_cfg
        Optional pre-resolved (tuned) config passed through to
        :func:`~rba.validation.harness.evaluate`'s ``model_cfg`` override.
    int_labels
        Forwarded to :func:`~rba.features.build.dev_test_split` — use ``True`` for
        the ordinal model (signed-int labels ``cut<hold<hike``).
    initial_train_size
        First training window size. Default ``len(dev)`` (the classic held-out
        shape). A smaller value would also walk through the tail of dev.
    output_dir, run_id
        Passed to :func:`~rba.validation.harness.evaluate`.
    use_mlflow
        Passed through (default ``False`` — the report layer logs once at the end).

    Returns
    -------
    HoldoutResult
        The evaluation result plus a per-test-meeting predictions frame.

    Raises
    ------
    ValueError
        If the test window is empty (no meetings on/after ``TEST_WINDOW_START``).
    """
    cfg = dict(target_cfg) if target_cfg is not None else load_target_config("three_class")
    split = dev_test_split(frame, cfg, int_labels=int_labels)
    if len(split.y_test) == 0:
        raise ValueError("evaluate_on_test: empty test window — check TEST_WINDOW_START.")

    x_full = pd.concat([split.x_dev, split.x_test])
    y_full = pd.concat([split.y_dev, split.y_test])
    n_dev = len(split.y_dev)
    train_size = initial_train_size if initial_train_size is not None else n_dev

    splitter = WalkForwardSplit(initial_train_size=train_size, test_size=1, step=1, expanding=True)
    result = evaluate(
        model_name=model_name,
        target_name=str(cfg.get("name", "three_class")),
        X=x_full,
        y=y_full,
        splitter=splitter,
        model_cfg=model_cfg,
        output_dir=output_dir,
        run_id=run_id,
        use_mlflow=use_mlflow,
    )
    predictions = _tidy_test_predictions(result.predictions, frame, split, n_dev)
    logger.info(
        "evaluate_on_test: model={} scored {} test meeting(s) (train_size={}).",
        model_name,
        len(predictions),
        train_size,
    )
    return HoldoutResult(result=result, predictions=predictions)


def _tidy_test_predictions(
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    split: Any,
    n_dev: int,
) -> pd.DataFrame:
    """Map the harness sample_index (position in X_full) back to meeting_date."""
    # Test-fold sample positions are n_dev .. n_dev + n_test - 1 in chronological
    # order; map each to its meeting date via x_test's original frame index.
    test_index = split.x_test.index
    meeting_dates = pd.to_datetime(frame.loc[test_index, _MEETING_KEY]).to_numpy()
    ordered = raw.sort_values("sample_index").reset_index(drop=True)
    positions = ordered["sample_index"].to_numpy() - n_dev
    tidy = pd.DataFrame({_MEETING_KEY: meeting_dates[positions]})
    tidy["y_true"] = ordered["y_true"].to_numpy()
    tidy["y_pred"] = ordered["y_pred"].to_numpy()
    for col in ordered.columns:
        if col.startswith("proba_"):
            tidy[col] = ordered[col].to_numpy()
    return tidy


def leakage_free_feature_names(frame: pd.DataFrame) -> list[str]:
    """Convenience re-export of :func:`rba.features.build.feature_columns`.

    Provided so §8 callers importing from :mod:`rba.validation.holdout` have a
    single place to ask "what columns are the model inputs" without reaching back
    into the features package.
    """
    return feature_columns(frame)
