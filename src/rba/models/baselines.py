"""Baseline models — the §6 non-learning floors every ML model must beat.

These are the simplest possible predictors, run so that later models are judged
against a meaningful reference rather than against zero. Each implements the
:class:`~rba.models.base.Model` contract (via :class:`~rba.models.base.BaseModel`)
so it is swappable into the §7 evaluation harness exactly like an ML model, and
each is registered in ``src/rba/config/models.yaml`` under ``rba.models.baselines``.

Holds the four baselines named in ``models.yaml``: :class:`MajorityClass`,
:class:`Persistence`, :class:`TaylorRule`, and :class:`MarketImplied`.

No network anywhere in this module.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

from rba.models.base import BaseModel


class MajorityClass(BaseModel):
    """Predict the most frequent class; ``predict_proba`` is the empirical prior.

    The simplest classification floor. :meth:`_fit` records the class distribution
    of ``y`` (optionally weighted by ``sample_weight``); :meth:`predict` returns the
    single most-frequent class for every row, and :meth:`predict_proba` returns that
    same empirical prior for every row — a constant, base-rate-calibrated vector so
    that log-loss / Brier are meaningful floors (a one-hot would be degenerate). It
    ignores ``X`` entirely (beyond its row count and, via
    :class:`~rba.models.base.BaseModel`, its column names).

    Ties for the modal class break to the lowest sorted class label
    (``np.argmax`` over the sorted classes). Deterministic — no random state.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels seen in ``y``; the column order of
        :meth:`predict_proba`.
    class_prior_ : numpy.ndarray
        Empirical (optionally weighted) class frequencies aligned to
        :attr:`classes_`; sums to 1.
    majority_class_ : Any
        The most frequent class label — ``classes_[argmax(class_prior_)]``.
    """

    classes_: np.ndarray
    class_prior_: np.ndarray
    majority_class_: Any

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Record the (optionally weighted) class distribution of ``y``.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        labels = np.asarray(y)
        classes, inverse = np.unique(labels, return_inverse=True)
        if sample_weight is None:
            counts = np.bincount(inverse, minlength=len(classes)).astype(float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            counts = np.bincount(inverse, weights=weights, minlength=len(classes))

        self.classes_ = classes
        self.class_prior_ = counts / counts.sum()
        self.majority_class_ = classes[int(np.argmax(counts))]
        logger.debug(
            "MajorityClass fitted: majority={!r}, prior={}.",
            self.majority_class_,
            dict(zip(classes.tolist(), self.class_prior_.round(4).tolist())),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the majority class for every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return np.full(len(X), self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the empirical class prior, broadcast to every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        return np.tile(self.class_prior_, (len(X), 1))


class Persistence(BaseModel):
    """Predict the last observed decision for every future row; one-hot ``predict_proba``.

    The simplest time-series classification floor: assume the next Board decision
    will be the same as the most recent one seen during training. :meth:`_fit`
    records the *last* value of ``y`` (``y.iloc[-1]``), so the caller must pass
    ``y`` in chronological order — which the walk-forward harness does by
    construction. :meth:`predict` returns that recorded label for every row of
    ``X``; :meth:`predict_proba` returns a one-hot vector on the persisted class,
    the honest representation of a deterministic classifier (persistence has no
    calibrated uncertainty — unlike :class:`MajorityClass`, which returns the
    empirical prior). Sklearn's ``log_loss`` clips at ``float64`` eps so the
    resulting score is a finite (large) upper bound rather than infinity. ``X``
    is ignored entirely beyond its row count and, via
    :class:`~rba.models.base.BaseModel`, its column names.

    ``sample_weight`` is accepted for Model-protocol conformance but ignored:
    the persistence decision is the last row's label regardless of weight.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Sorted unique class labels seen in ``y``; the column order of
        :meth:`predict_proba`.
    last_class_ : Any
        The last label of ``y`` — the value predicted for every row.
    """

    classes_: np.ndarray
    last_class_: Any

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Record ``classes_`` and the final training label ``last_class_``.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        self.last_class_ = labels[-1]
        logger.debug(
            "Persistence fitted: last_class={!r}, classes={}.",
            self.last_class_,
            self.classes_.tolist(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the persisted class for every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        return np.full(len(X), self.last_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """One-hot on the persisted class, broadcast to every row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, n_classes).
        """
        proba_row = (self.classes_ == self.last_class_).astype(float)
        return np.tile(proba_row, (len(X), 1))


class TaylorRule(BaseModel):
    """Taylor-rule regressor for the prescribed cash-rate **level** (%).

    The classic §6 monetary-economics floor (Taylor 1993). Predicts the level a
    mechanical policy rule would prescribe given current inflation and the output
    gap:

    .. math::

        \\hat{i}_t = \\alpha + \\varphi_\\pi\\, \\pi_t + \\varphi_y\\, y_t

    where the intercept ``α`` collapses ``neutral_rate`` and ``inflation_target``
    (they are not separately identified from a single regression:
    ``α = r^\\ast + \\pi^\\ast - \\varphi_\\pi \\cdot \\pi^\\ast``). Task is
    ``regression`` (see ``models.yaml``), so :meth:`predict_proba` inherits the
    :class:`BaseModel` ``NotImplementedError`` — a regressor has no class
    probabilities.

    Two modes, per ``fit_coefficients``:

    - ``False`` (**literature defaults**) — plug in Taylor's original ``(φ_π,
      φ_y) = (1.5, 0.5)`` and the RBA-appropriate ``neutral_rate = 3.0`` /
      ``inflation_target = 2.5`` from the constructor. :meth:`_fit` does no
      learning — ``y`` and ``sample_weight`` are ignored — it just materialises
      ``intercept_`` / ``inflation_weight_`` / ``output_gap_weight_`` from the
      constructor.
    - ``True`` (**fitted**) — ordinary least squares on the training window
      recovers ``(intercept_, inflation_weight_, output_gap_weight_)`` directly
      from realised cash rates: ``y ~ 1 + π + y_gap`` via :func:`numpy.linalg.lstsq`,
      with optional ``sample_weight`` folded in as :math:`\\sqrt{w}` row scaling
      (weighted least squares). Rows with NaN in the inflation column, output-gap
      column, or ``y`` are dropped before the solve — a baseline should not
      silently impute.

    Inflation and output gap are read by **column name** from ``X`` (see
    ``inflation_col`` / ``output_gap_col``), so the class is decoupled from the
    feature pipeline's naming: the user hands in a frame that carries the two
    signals under whatever names ``features/build.py`` emits. The rest of ``X``
    is ignored (though its full column list is still stored in
    :attr:`~rba.models.base.BaseModel.feature_names_`, per protocol).

    ``output_gap`` sign convention is the standard one: ``(Y − Y*)/Y*``,
    **positive when the economy runs hot** (i.e. above potential). If the user
    substitutes an unemployment-gap proxy, they must negate it (Okun's law:
    ``y_gap ≈ −Okun · (u − u*)``) before handing it in.

    Parameters
    ----------
    inflation_target : float, default 2.5
        ``π*`` — the RBA target midpoint. Ignored under ``fit_coefficients=True``
        (absorbed into the fitted intercept).
    inflation_weight : float, default 1.5
        Taylor's ``φ_π``. The Taylor principle requires ``> 1``. Ignored under
        ``fit_coefficients=True`` (replaced by the OLS slope on ``π``).
    output_gap_weight : float, default 0.5
        Taylor's ``φ_y``. Ignored under ``fit_coefficients=True`` (replaced by
        the OLS slope on ``y_gap``).
    neutral_rate : float, default 3.0
        The **nominal** neutral cash rate ``r* + π*``. Ignored under
        ``fit_coefficients=True`` (absorbed into the fitted intercept).
    fit_coefficients : bool, default False
        Toggle between literature-defaults mode (``False``) and OLS-fitted mode
        (``True``).
    inflation_col : str, default ``"cpi_headline_yoy"``
        Column of ``X`` carrying current YoY headline inflation, in percent.
    output_gap_col : str, default ``"output_gap"``
        Column of ``X`` carrying the current output gap, as a percent of
        potential output.

    Attributes
    ----------
    intercept_ : float
        ``α`` after fit — ``neutral_rate − inflation_weight · inflation_target``
        under literature defaults; the OLS intercept when fitted.
    inflation_weight_ : float
        ``φ_π`` after fit.
    output_gap_weight_ : float
        ``φ_y`` after fit.
    """

    intercept_: float
    inflation_weight_: float
    output_gap_weight_: float

    def __init__(
        self,
        *,
        inflation_target: float = 2.5,
        inflation_weight: float = 1.5,
        output_gap_weight: float = 0.5,
        neutral_rate: float = 3.0,
        fit_coefficients: bool = False,
        inflation_col: str = "cpi_headline_yoy",
        output_gap_col: str = "output_gap",
    ) -> None:
        self._inflation_target = float(inflation_target)
        self._inflation_weight = float(inflation_weight)
        self._output_gap_weight = float(output_gap_weight)
        self._neutral_rate = float(neutral_rate)
        self._fit_coefficients = bool(fit_coefficients)
        self._inflation_col = inflation_col
        self._output_gap_col = output_gap_col

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Materialise ``intercept_`` / weights — from constants or OLS.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        if not self._fit_coefficients:
            self.intercept_ = self._neutral_rate - self._inflation_weight * self._inflation_target
            self.inflation_weight_ = self._inflation_weight
            self.output_gap_weight_ = self._output_gap_weight
            logger.debug(
                "TaylorRule (literature defaults): intercept={:.4f}, phi_pi={:.4f}, phi_y={:.4f}.",
                self.intercept_,
                self.inflation_weight_,
                self.output_gap_weight_,
            )
            return

        pi = X[self._inflation_col].to_numpy(dtype=float)
        y_gap = X[self._output_gap_col].to_numpy(dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = ~(np.isnan(pi) | np.isnan(y_gap) | np.isnan(y_arr))
        n_clean = int(mask.sum())
        if n_clean < 3:
            raise ValueError(
                f"TaylorRule with fit_coefficients=True needs at least 3 rows with "
                f"non-NaN {self._inflation_col!r}, {self._output_gap_col!r} and y; "
                f"got {n_clean}."
            )
        design = np.column_stack([np.ones(n_clean), pi[mask], y_gap[mask]])
        y_clean = y_arr[mask]
        if sample_weight is None:
            coef, *_ = np.linalg.lstsq(design, y_clean, rcond=None)
        else:
            w = np.asarray(sample_weight, dtype=float)[mask]
            sqrt_w = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y_clean * sqrt_w, rcond=None)
        self.intercept_ = float(coef[0])
        self.inflation_weight_ = float(coef[1])
        self.output_gap_weight_ = float(coef[2])
        logger.debug(
            "TaylorRule (fitted, n={}): intercept={:.4f}, phi_pi={:.4f}, phi_y={:.4f}.",
            n_clean,
            self.intercept_,
            self.inflation_weight_,
            self.output_gap_weight_,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the prescribed cash-rate level for each row of ``X``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        pi = X[self._inflation_col].to_numpy(dtype=float)
        y_gap = X[self._output_gap_col].to_numpy(dtype=float)
        return self.intercept_ + self.inflation_weight_ * pi + self.output_gap_weight_ * y_gap


class MarketImplied(BaseModel):
    """Market-implied 3-class baseline from ASX 30-day interbank cash-rate futures.

    Reads the market-implied cash rate (naïve `100 − settlement_price` of the
    meeting-month contract as of T−1, per
    :func:`rba.data.sources.asx_ib_futures.derive_meeting_implied`) and the
    current cash rate directly from ``X``, computes
    ``Δ = implied_rate − current_rate`` (percentage points), and maps that gap
    to a hike/hold/cut probability via the standard single-move decomposition:

    .. math::

        P(\\mathrm{hike}) = \\mathrm{clip}(\\Delta / m,\\,0,\\,1),\\quad
        P(\\mathrm{cut})  = \\mathrm{clip}(-\\Delta / m,\\,0,\\,1),\\quad
        P(\\mathrm{hold}) = 1 - P(\\mathrm{hike}) - P(\\mathrm{cut})

    where ``m`` is the assumed single-move size (default 0.25 pp = 25 bp).
    :meth:`predict` returns ``argmax``, so the decision boundary is at
    ``|Δ| = m/2`` — hold when ``|Δ| < 12.5 bp``, hike/cut when ``|Δ| ≥ 12.5 bp``.
    This is the "Rate Tracker"-style interpretation the ASX Rate Indicator page
    uses, without the pre-/post-meeting day-weighting decomposition (which is a
    downstream feature transform per the ``asx_ib_futures`` module docstring).

    :meth:`_fit` is rule-based — no learning happens. It materialises
    :attr:`classes_` (sorted, sklearn-convention) from the constructor's three
    label kwargs and caches the index of each semantic label so
    :meth:`predict_proba` can populate columns correctly regardless of whether
    the sort order of the labels matches the semantic order. ``y`` is inspected
    only for a soft warning when configured labels aren't seen in the training
    set (a mismatch with the target's encoding); ``sample_weight`` is ignored.

    NaN in ``X[implied_rate_col]`` or ``X[current_rate_col]`` raises
    :class:`ValueError` at :meth:`predict` — ASX IB-futures coverage begins
    2022-04-21, so pre-2022 meetings have no market-implied signal and the
    harness must window this baseline to the covered range rather than silently
    scoring garbage.

    Parameters
    ----------
    implied_rate_col : str, default ``"asx_30d_implied_rate"``
        Column in ``X`` carrying the market-implied cash rate (percent).
    current_rate_col : str, default ``"prior_rate_pct"``
        Column in ``X`` carrying the cash rate going *into* the meeting (percent).
        Matches ``rba.data.preprocess_target`` output.
    move_size_pct : float, default 0.25
        Assumed single-move size in percentage points (25 bp is the RBA's
        historical unit).
    cut_label, hold_label, hike_label : Any, defaults ``"cut"`` / ``"hold"`` / ``"hike"``
        The values in ``y`` that mean each direction. Use ``(-1, 0, 1)`` for the
        ``class_to_int`` int-encoded ``three_class`` target from ``targets.yaml``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The three labels in sorted order — the column order of
        :meth:`predict_proba`, sklearn convention.
    """

    classes_: np.ndarray

    def __init__(
        self,
        *,
        implied_rate_col: str = "asx_30d_implied_rate",
        current_rate_col: str = "prior_rate_pct",
        move_size_pct: float = 0.25,
        cut_label: Any = "cut",
        hold_label: Any = "hold",
        hike_label: Any = "hike",
    ) -> None:
        if move_size_pct <= 0:
            raise ValueError(f"move_size_pct must be positive; got {move_size_pct!r}.")
        self._implied_rate_col = implied_rate_col
        self._current_rate_col = current_rate_col
        self._move_size_pct = float(move_size_pct)
        self._cut_label = cut_label
        self._hold_label = hold_label
        self._hike_label = hike_label

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Materialise ``classes_`` and cache the per-direction column indices.

        Shapes
        ------
        X: (n_samples, n_features), y: (n_samples,),
        sample_weight: (n_samples,) or ``None``.
        """
        labels = [self._cut_label, self._hold_label, self._hike_label]
        # ``np.unique`` gives the sklearn-convention sorted-by-dtype order and
        # also enforces the "three distinct labels" invariant (duplicates
        # collapse, so we detect them by counting the unique array).
        self.classes_ = np.unique(np.array(labels))
        if len(self.classes_) != 3:
            raise ValueError(
                f"cut_label / hold_label / hike_label must all differ; got "
                f"cut={self._cut_label!r}, hold={self._hold_label!r}, "
                f"hike={self._hike_label!r}."
            )
        self._cut_idx = int(np.where(self.classes_ == self._cut_label)[0][0])
        self._hold_idx = int(np.where(self.classes_ == self._hold_label)[0][0])
        self._hike_idx = int(np.where(self.classes_ == self._hike_label)[0][0])

        y_arr = np.asarray(y)
        if len(y_arr) > 0:
            seen = set(y_arr.tolist())
            unseen = [label for label in labels if label not in seen]
            if unseen:
                logger.warning(
                    "MarketImplied.fit: configured labels {!r} not seen in training y — "
                    "check cut_label / hold_label / hike_label match your target encoding.",
                    unseen,
                )
        logger.debug(
            "MarketImplied fitted: classes={} (cut@{}, hold@{}, hike@{}).",
            self.classes_.tolist(),
            self._cut_idx,
            self._hold_idx,
            self._hike_idx,
        )

    def _implied_change_pct(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``implied − current`` for each row (percent), raising on NaN."""
        implied = X[self._implied_rate_col].to_numpy(dtype=float)
        current = X[self._current_rate_col].to_numpy(dtype=float)
        missing = np.isnan(implied) | np.isnan(current)
        if missing.any():
            n_missing = int(missing.sum())
            raise ValueError(
                f"MarketImplied cannot predict on {n_missing} row(s) with NaN in "
                f"{self._implied_rate_col!r} or {self._current_rate_col!r} — ASX IB-"
                "futures coverage begins 2022-04-21; filter uncovered meetings before "
                "calling this baseline."
            )
        return implied - current

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row ``[P(class)]`` from the single-move decomposition.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples, 3).
        """
        change = self._implied_change_pct(X)
        p_hike = np.clip(change / self._move_size_pct, 0.0, 1.0)
        p_cut = np.clip(-change / self._move_size_pct, 0.0, 1.0)
        p_hold = 1.0 - p_hike - p_cut
        proba = np.zeros((len(X), 3), dtype=float)
        proba[:, self._cut_idx] = p_cut
        proba[:, self._hold_idx] = p_hold
        proba[:, self._hike_idx] = p_hike
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``classes_[argmax(predict_proba(X))]``.

        Shapes
        ------
        X: (n_samples, n_features) -> (n_samples,).
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
