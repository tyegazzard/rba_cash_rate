"""Tests for ``rba.models.baselines`` — the §6 baseline floors.

No network; tiny inline ``(X, y)`` fixtures. Covers :class:`MajorityClass`: it
satisfies the ``Model`` protocol, predicts the modal class for every row, returns
the empirical class prior from ``predict_proba``, honours ``sample_weight``, breaks
ties to the lowest label, and inherits the ``BaseModel`` feature-name storage.
Also covers :class:`Persistence`: it predicts the last training label for every
row, returns a one-hot ``predict_proba`` on the persisted class, ignores
``sample_weight``, and preserves label dtype. And :class:`TaylorRule`: under
literature defaults it evaluates ``α + φ_π π + φ_y y_gap`` exactly (with the
canonical ``i = 3.0`` at ``π = π* = 2.5`` / ``y_gap = 0`` sanity anchor); under
``fit_coefficients=True`` OLS recovers the coefficients on synthetic data (with
and without ``sample_weight``); it drops NaN rows before the solve, raises on a
missing column, and inherits the regressor ``predict_proba → NotImplementedError``.
And :class:`MarketImplied`: hike/hold/cut probabilities from the linear ASX-
futures decomposition ``P(hike) = clip(Δ / m, 0, 1)`` etc. — sanity anchors at
``Δ ∈ {0, ±m, ±m/2}``, sklearn-sorted ``classes_``, correct proba column
placement under both string and int label sets, ``predict``/``predict_proba``
argmax consistency, custom column names and move-size flow-through, and
``ValueError`` on NaN inputs at predict time (ASX coverage floor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rba.models.base import Model
from rba.models.baselines import MajorityClass, MarketImplied, Persistence, TaylorRule


def _X(n: int) -> pd.DataFrame:
    """A dummy feature frame — MajorityClass ignores its values, uses only shape."""
    return pd.DataFrame({"f0": np.arange(n, dtype=float), "f1": np.arange(n, dtype=float)})


def _imbalanced() -> tuple[pd.DataFrame, pd.Series]:
    """7 hold / 2 hike / 1 cut → majority 'hold', prior [0.1, 0.2, 0.7] over sorted classes."""
    y = pd.Series(["hold"] * 7 + ["hike"] * 2 + ["cut"] * 1, name="target")
    return _X(len(y)), y


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_majority_class_is_a_model_when_fitted() -> None:
    X, y = _imbalanced()
    assert isinstance(MajorityClass().fit(X, y), Model)


def test_fit_returns_self() -> None:
    X, y = _imbalanced()
    model = MajorityClass()
    assert model.fit(X, y) is model


def test_feature_names_are_stored() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.feature_names_ == ["f0", "f1"]


# -----------------------------------------------------------------------------
# predict — the modal class for every row.
# -----------------------------------------------------------------------------
def test_predict_returns_majority_for_every_row() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.majority_class_ == "hold"
    preds = model.predict(_X(5))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5,)
    assert (preds == "hold").all()


def test_predict_uses_the_argument_row_count_not_the_training_count() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    assert model.predict(_X(3)).shape == (3,)


# -----------------------------------------------------------------------------
# predict_proba — the empirical class prior, constant across rows.
# -----------------------------------------------------------------------------
def test_predict_proba_is_the_empirical_prior() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    # sorted classes: ['cut', 'hike', 'hold'] -> counts 1, 2, 7
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))
    proba = model.predict_proba(_X(4))
    assert proba.shape == (4, 3)
    expected = np.array([0.1, 0.2, 0.7])
    np.testing.assert_allclose(proba, np.tile(expected, (4, 1)))
    # Every row is a valid distribution.
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_class_prior_matches_predict_proba_row() -> None:
    X, y = _imbalanced()
    model = MajorityClass().fit(X, y)
    np.testing.assert_allclose(model.predict_proba(_X(1))[0], model.class_prior_)


# -----------------------------------------------------------------------------
# sample_weight + tie-breaking + integer labels.
# -----------------------------------------------------------------------------
def test_sample_weight_can_flip_the_majority() -> None:
    X = _X(2)
    y = pd.Series([0, 1])
    # Unweighted it's a tie -> lowest label (0); weighting class 1 makes it the majority.
    weighted = MajorityClass().fit(X, y, sample_weight=np.array([1.0, 5.0]))
    assert weighted.majority_class_ == 1
    np.testing.assert_allclose(weighted.class_prior_, np.array([1 / 6, 5 / 6]))


def test_tie_breaks_to_lowest_label() -> None:
    X = _X(2)
    y = pd.Series([1, 0])  # one each -> tie
    model = MajorityClass().fit(X, y)
    assert model.majority_class_ == 0


def test_integer_labels_predict_dtype_preserved() -> None:
    X = _X(4)
    y = pd.Series([-1, 0, 0, 1])  # three_class int encoding (cut/hold/hike)
    model = MajorityClass().fit(X, y)
    preds = model.predict(_X(3))
    assert model.majority_class_ == 0
    assert np.issubdtype(preds.dtype, np.integer)
    assert (preds == 0).all()


def test_predict_proba_raises_before_fit() -> None:
    """Unfitted access fails loudly (fitted attributes are absent)."""
    with pytest.raises(AttributeError):
        MajorityClass().predict_proba(_X(2))


# =============================================================================
# Persistence — predict last training decision for every row.
# =============================================================================
def _chronological() -> tuple[pd.DataFrame, pd.Series]:
    """4 hold / 2 hike / 1 cut in that order → persisted class is the final 'cut'."""
    y = pd.Series(["hold"] * 4 + ["hike"] * 2 + ["cut"] * 1, name="target")
    return _X(len(y)), y


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_persistence_is_a_model_when_fitted() -> None:
    X, y = _chronological()
    assert isinstance(Persistence().fit(X, y), Model)


def test_persistence_fit_returns_self() -> None:
    X, y = _chronological()
    model = Persistence()
    assert model.fit(X, y) is model


def test_persistence_feature_names_are_stored() -> None:
    X, y = _chronological()
    model = Persistence().fit(X, y)
    assert model.feature_names_ == ["f0", "f1"]


# -----------------------------------------------------------------------------
# predict — the last observed class for every row, regardless of X.
# -----------------------------------------------------------------------------
def test_persistence_predict_returns_last_class_for_every_row() -> None:
    X, y = _chronological()
    model = Persistence().fit(X, y)
    assert model.last_class_ == "cut"
    preds = model.predict(_X(5))
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5,)
    assert (preds == "cut").all()


def test_persistence_predict_uses_argument_row_count_not_training_count() -> None:
    X, y = _chronological()
    model = Persistence().fit(X, y)
    assert model.predict(_X(3)).shape == (3,)


def test_persistence_reads_last_row_by_position_not_index() -> None:
    """A non-monotonic pd.Index must not scramble the 'last decision' — .iloc[-1] wins."""
    X = _X(3)
    y = pd.Series(["hike", "hold", "cut"], index=[10, 5, 42])
    model = Persistence().fit(X, y)
    assert model.last_class_ == "cut"


# -----------------------------------------------------------------------------
# predict_proba — one-hot on the persisted class.
# -----------------------------------------------------------------------------
def test_persistence_predict_proba_is_one_hot_on_last_class() -> None:
    X, y = _chronological()
    model = Persistence().fit(X, y)
    # sorted classes: ['cut', 'hike', 'hold']; last class 'cut' -> first column.
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))
    proba = model.predict_proba(_X(4))
    assert proba.shape == (4, 3)
    expected = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(proba, np.tile(expected, (4, 1)))
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_persistence_predict_and_predict_proba_agree() -> None:
    """argmax of predict_proba must recover the same label predict returns."""
    X, y = _chronological()
    model = Persistence().fit(X, y)
    proba = model.predict_proba(_X(3))
    argmax_labels = model.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(argmax_labels, model.predict(_X(3)))


# -----------------------------------------------------------------------------
# sample_weight is ignored + integer labels + unfitted access.
# -----------------------------------------------------------------------------
def test_persistence_sample_weight_is_ignored() -> None:
    """Persistence looks only at the last row; weights on earlier rows change nothing."""
    X = _X(3)
    y = pd.Series(["hike", "hold", "cut"])
    unweighted = Persistence().fit(X, y)
    weighted = Persistence().fit(X, y, sample_weight=np.array([100.0, 100.0, 0.001]))
    assert unweighted.last_class_ == "cut"
    assert weighted.last_class_ == "cut"


def test_persistence_integer_labels_predict_dtype_preserved() -> None:
    X = _X(4)
    y = pd.Series([-1, 0, 0, 1])  # three_class int encoding (cut/hold/hike)
    model = Persistence().fit(X, y)
    preds = model.predict(_X(3))
    assert model.last_class_ == 1
    assert np.issubdtype(preds.dtype, np.integer)
    assert (preds == 1).all()


def test_persistence_predict_proba_raises_before_fit() -> None:
    """Unfitted access fails loudly (fitted attributes are absent)."""
    with pytest.raises(AttributeError):
        Persistence().predict_proba(_X(2))


# =============================================================================
# TaylorRule — prescribed cash-rate level regressor.
# =============================================================================
def _taylor_frame(pi: list[float], y_gap: list[float]) -> pd.DataFrame:
    """Feature frame with the two Taylor-rule columns plus a distractor."""
    assert len(pi) == len(y_gap)
    return pd.DataFrame(
        {
            "cpi_headline_yoy": pi,
            "output_gap": y_gap,
            "distractor": np.arange(len(pi), dtype=float),
        }
    )


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract (literature-defaults mode).
# -----------------------------------------------------------------------------
def test_taylor_rule_is_a_model_when_fitted() -> None:
    X = _taylor_frame([2.5, 3.0, 1.5], [0.0, 1.0, -1.0])
    y = pd.Series([3.0, 4.5, 2.0])
    assert isinstance(TaylorRule().fit(X, y), Model)


def test_taylor_rule_fit_returns_self() -> None:
    X = _taylor_frame([2.5], [0.0])
    y = pd.Series([3.0])
    model = TaylorRule()
    assert model.fit(X, y) is model


def test_taylor_rule_feature_names_are_stored() -> None:
    X = _taylor_frame([2.5], [0.0])
    y = pd.Series([3.0])
    model = TaylorRule().fit(X, y)
    assert model.feature_names_ == ["cpi_headline_yoy", "output_gap", "distractor"]


# -----------------------------------------------------------------------------
# Literature-defaults formula.
# -----------------------------------------------------------------------------
def test_taylor_rule_at_target_and_zero_gap_predicts_neutral_rate() -> None:
    """At π = π* = 2.5 and output gap = 0, the prescribed rate is the neutral rate."""
    X = _taylor_frame([2.5], [0.0])
    y = pd.Series([0.0])  # ignored under literature defaults
    model = TaylorRule().fit(X, y)
    np.testing.assert_allclose(model.predict(X), np.array([3.0]))


def test_taylor_rule_uses_classic_coefficients() -> None:
    """+1pp above target → +1.5pp; +2pp output gap → +1.0pp; combined additive."""
    X = _taylor_frame(pi=[3.5, 2.5, 3.5], y_gap=[0.0, 2.0, 2.0])
    y = pd.Series([0.0, 0.0, 0.0])  # ignored
    model = TaylorRule().fit(X, y)
    # 3.0 + 1.5*(π - 2.5) + 0.5*y_gap
    expected = np.array([3.0 + 1.5, 3.0 + 1.0, 3.0 + 1.5 + 1.0])
    np.testing.assert_allclose(model.predict(X), expected)


def test_taylor_rule_intercept_matches_neutral_and_target() -> None:
    """intercept_ = neutral_rate − φ_π · π* under literature defaults."""
    model = TaylorRule().fit(_taylor_frame([2.5], [0.0]), pd.Series([0.0]))
    assert model.intercept_ == pytest.approx(3.0 - 1.5 * 2.5)
    assert model.inflation_weight_ == pytest.approx(1.5)
    assert model.output_gap_weight_ == pytest.approx(0.5)


def test_taylor_rule_ignores_y_and_sample_weight_under_defaults() -> None:
    """y and sample_weight must not shift the literature-defaults coefficients."""
    X = _taylor_frame([2.5, 3.0], [0.0, 1.0])
    plain = TaylorRule().fit(X, pd.Series([0.0, 0.0]))
    perturbed = TaylorRule().fit(X, pd.Series([99.0, -99.0]), sample_weight=np.array([1e6, 1e-6]))
    assert plain.intercept_ == perturbed.intercept_
    assert plain.inflation_weight_ == perturbed.inflation_weight_
    assert plain.output_gap_weight_ == perturbed.output_gap_weight_


def test_taylor_rule_custom_constructor_params() -> None:
    """A non-default (r*, π*, φ_π, φ_y) must flow through the intercept + slope."""
    model = TaylorRule(
        inflation_target=2.0,
        inflation_weight=1.75,
        output_gap_weight=0.25,
        neutral_rate=2.5,
    ).fit(_taylor_frame([2.0], [0.0]), pd.Series([0.0]))
    # At π = π* = 2.0 and y_gap = 0, predicted rate == neutral rate.
    np.testing.assert_allclose(model.predict(_taylor_frame([2.0], [0.0])), np.array([2.5]))
    assert model.inflation_weight_ == pytest.approx(1.75)
    assert model.output_gap_weight_ == pytest.approx(0.25)


def test_taylor_rule_predict_ignores_non_taylor_columns() -> None:
    """The distractor column's values must not affect the prediction."""
    X_a = _taylor_frame([3.0], [1.0])
    X_b = X_a.copy()
    X_b["distractor"] = X_b["distractor"] + 1000.0
    model = TaylorRule().fit(X_a, pd.Series([0.0]))
    np.testing.assert_allclose(model.predict(X_a), model.predict(X_b))


# -----------------------------------------------------------------------------
# Fitted-coefficient mode (OLS).
# -----------------------------------------------------------------------------
def test_taylor_rule_fit_recovers_coefficients_from_synthetic_data() -> None:
    """OLS on i = 2.0 + 1.8*π + 0.4*y_gap must recover the true coefficients."""
    rng = np.random.default_rng(12)
    n = 200
    pi = rng.uniform(0.5, 6.0, size=n)
    y_gap = rng.uniform(-3.0, 3.0, size=n)
    true_alpha, true_phi_pi, true_phi_y = 2.0, 1.8, 0.4
    i = true_alpha + true_phi_pi * pi + true_phi_y * y_gap + rng.normal(0, 0.05, size=n)
    X = _taylor_frame(list(pi), list(y_gap))
    model = TaylorRule(fit_coefficients=True).fit(X, pd.Series(i))
    assert model.intercept_ == pytest.approx(true_alpha, abs=0.05)
    assert model.inflation_weight_ == pytest.approx(true_phi_pi, abs=0.02)
    assert model.output_gap_weight_ == pytest.approx(true_phi_y, abs=0.02)


def test_taylor_rule_fit_with_sample_weight_downweights_outliers() -> None:
    """A high-weight clean subset should dominate the OLS fit over a noisy subset."""
    rng = np.random.default_rng(12)
    # Clean subset following the true rule.
    n_clean = 200
    pi_clean = rng.uniform(0.5, 6.0, size=n_clean)
    y_gap_clean = rng.uniform(-3.0, 3.0, size=n_clean)
    i_clean = 2.0 + 1.8 * pi_clean + 0.4 * y_gap_clean
    # Noisy subset following a different rule.
    n_noise = 200
    pi_noise = rng.uniform(0.5, 6.0, size=n_noise)
    y_gap_noise = rng.uniform(-3.0, 3.0, size=n_noise)
    i_noise = -5.0 + 0.1 * pi_noise + 3.0 * y_gap_noise
    pi = np.concatenate([pi_clean, pi_noise])
    y_gap = np.concatenate([y_gap_clean, y_gap_noise])
    y = np.concatenate([i_clean, i_noise])
    weights = np.concatenate([np.full(n_clean, 1.0), np.full(n_noise, 1e-6)])
    X = _taylor_frame(list(pi), list(y_gap))
    model = TaylorRule(fit_coefficients=True).fit(X, pd.Series(y), sample_weight=weights)
    # With weights ~1e6× on the clean subset, we should recover the clean rule.
    assert model.intercept_ == pytest.approx(2.0, abs=0.05)
    assert model.inflation_weight_ == pytest.approx(1.8, abs=0.05)
    assert model.output_gap_weight_ == pytest.approx(0.4, abs=0.05)


def test_taylor_rule_fit_drops_nan_rows() -> None:
    """A NaN in any of π, y_gap, or y must be dropped before the OLS solve."""
    pi = [2.0, np.nan, 3.0, 4.0, 5.0]
    y_gap = [0.0, 0.0, 1.0, np.nan, -1.0]
    y = pd.Series([3.6, 3.6, 4.8, 4.8, 5.5])
    X = _taylor_frame(pi, y_gap)
    # Rows (0, 2, 4) survive → clean sub-frame passes to OLS.
    model = TaylorRule(fit_coefficients=True).fit(X, y)
    # Sanity: predictions on the clean rows are close to y for those rows.
    clean_X = _taylor_frame([2.0, 3.0, 5.0], [0.0, 1.0, -1.0])
    preds = model.predict(clean_X)
    np.testing.assert_allclose(preds, np.array([3.6, 4.8, 5.5]), atol=0.1)


def test_taylor_rule_fit_too_few_clean_rows_raises() -> None:
    """Fewer than 3 non-NaN training rows is unfittable — raise clearly."""
    X = _taylor_frame([2.0, np.nan], [0.0, 0.0])
    y = pd.Series([3.0, 3.0])
    with pytest.raises(ValueError, match="at least 3 rows"):
        TaylorRule(fit_coefficients=True).fit(X, y)


# -----------------------------------------------------------------------------
# Error paths.
# -----------------------------------------------------------------------------
def test_taylor_rule_missing_column_raises_at_fit_when_fitting_coefficients() -> None:
    """fit_coefficients=True needs the columns at fit time — miss fails fast."""
    X = pd.DataFrame({"output_gap": [0.0, 1.0, -1.0], "other": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        TaylorRule(fit_coefficients=True).fit(X, pd.Series([3.0, 3.6, 2.4]))


def test_taylor_rule_missing_column_raises_at_predict_under_literature_defaults() -> None:
    """Literature-defaults fit doesn't read X — the miss surfaces at predict."""
    X_fit = _taylor_frame([2.5], [0.0])
    model = TaylorRule().fit(X_fit, pd.Series([0.0]))
    X_predict = pd.DataFrame({"output_gap": [0.0], "other": [1.0]})
    with pytest.raises(KeyError):
        model.predict(X_predict)


def test_taylor_rule_custom_column_names() -> None:
    """Non-default column names must flow through fit and predict."""
    X = pd.DataFrame({"inf": [2.5, 3.5], "gap": [0.0, 0.0]})
    model = TaylorRule(inflation_col="inf", output_gap_col="gap").fit(X, pd.Series([0.0, 0.0]))
    np.testing.assert_allclose(model.predict(X), np.array([3.0, 4.5]))


def test_taylor_rule_predict_proba_raises_not_implemented() -> None:
    """Regressor semantics — no class probabilities."""
    X = _taylor_frame([2.5], [0.0])
    model = TaylorRule().fit(X, pd.Series([0.0]))
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


# =============================================================================
# MarketImplied — ASX 30-day IB futures → hike/hold/cut probabilities.
# =============================================================================
def _mi_frame(implied: list[float], current: list[float]) -> pd.DataFrame:
    """Feature frame with the two MarketImplied columns plus a distractor."""
    assert len(implied) == len(current)
    return pd.DataFrame(
        {
            "asx_30d_implied_rate": implied,
            "prior_rate_pct": current,
            "distractor": np.arange(len(implied), dtype=float),
        }
    )


def _mi_training_y() -> pd.Series:
    """A tiny training target that contains all three default string labels."""
    return pd.Series(["cut", "hold", "hike", "hold"])


# -----------------------------------------------------------------------------
# Protocol conformance + fit contract.
# -----------------------------------------------------------------------------
def test_market_implied_is_a_model_when_fitted() -> None:
    X = _mi_frame([4.35, 4.35, 4.35, 4.35], [4.35, 4.35, 4.35, 4.35])
    assert isinstance(MarketImplied().fit(X, _mi_training_y()), Model)


def test_market_implied_fit_returns_self() -> None:
    X = _mi_frame([4.35], [4.35])
    y = pd.Series(["hold"])
    model = MarketImplied()
    assert model.fit(X, y) is model


def test_market_implied_feature_names_are_stored() -> None:
    X = _mi_frame([4.35], [4.35])
    model = MarketImplied().fit(X, pd.Series(["hold"]))
    assert model.feature_names_ == ["asx_30d_implied_rate", "prior_rate_pct", "distractor"]


def test_market_implied_classes_are_sorted_default_string_labels() -> None:
    """Sorted lex: 'cut' < 'hike' < 'hold'."""
    model = MarketImplied().fit(_mi_frame([4.35], [4.35]), pd.Series(["hold"]))
    np.testing.assert_array_equal(model.classes_, np.array(["cut", "hike", "hold"]))


# -----------------------------------------------------------------------------
# The decision boundary — no move, full move, half-move corner.
# -----------------------------------------------------------------------------
def test_market_implied_no_change_predicts_hold() -> None:
    """Δ = 0 → P(hold) = 1, argmax → hold."""
    X = _mi_frame([4.35], [4.35])
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    # classes_ = ['cut', 'hike', 'hold'] — hold is column index 2.
    np.testing.assert_allclose(proba, np.array([[0.0, 0.0, 1.0]]))
    assert model.predict(X) == np.array(["hold"])


def test_market_implied_full_move_up_predicts_hike() -> None:
    """Δ = +25 bp → P(hike) = 1, argmax → hike."""
    X = _mi_frame([4.60], [4.35])
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.array([[0.0, 1.0, 0.0]]))
    assert model.predict(X) == np.array(["hike"])


def test_market_implied_full_move_down_predicts_cut() -> None:
    """Δ = −25 bp → P(cut) = 1, argmax → cut."""
    X = _mi_frame([4.10], [4.35])
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.array([[1.0, 0.0, 0.0]]))
    assert model.predict(X) == np.array(["cut"])


def test_market_implied_half_move_up_is_hike_hold_tie() -> None:
    """Δ = +12.5 bp → P(hike) = P(hold) = 0.5; argmax breaks to hike (higher index)."""
    X = _mi_frame([4.475], [4.35])
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.array([[0.0, 0.5, 0.5]]))
    # np.argmax returns the FIRST maximum; classes_ = [cut, hike, hold] so hike wins.
    assert model.predict(X) == np.array(["hike"])


def test_market_implied_beyond_full_move_clips_at_one() -> None:
    """Δ = +50 bp exceeds one 25 bp move — probabilities clip cleanly to a valid distribution."""
    X = _mi_frame([4.85], [4.35])
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.array([[0.0, 1.0, 0.0]]))
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(1))


def test_market_implied_batch_probabilities_are_valid_distributions() -> None:
    """Across a fan of Δ values every row is a probability distribution."""
    X = _mi_frame([4.20, 4.30, 4.35, 4.40, 4.50], [4.35] * 5)
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    assert proba.shape == (5, 3)
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5))


def test_market_implied_predict_matches_argmax_of_predict_proba() -> None:
    X = _mi_frame([4.20, 4.30, 4.35, 4.40, 4.60], [4.35] * 5)
    model = MarketImplied().fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_array_equal(model.predict(X), model.classes_[np.argmax(proba, axis=1)])


# -----------------------------------------------------------------------------
# Custom label encoding + column names + move-size.
# -----------------------------------------------------------------------------
def test_market_implied_int_labels_populate_proba_columns_correctly() -> None:
    """With (-1, 0, +1), classes_ = [-1, 0, 1]; a hike row should have P(1) = 1."""
    X = _mi_frame([4.60, 4.35, 4.10], [4.35, 4.35, 4.35])
    y = pd.Series([-1, 0, 1, 0])
    X_fit = _mi_frame([4.35] * 4, [4.35] * 4)
    model = MarketImplied(cut_label=-1, hold_label=0, hike_label=1).fit(X_fit, y)
    np.testing.assert_array_equal(model.classes_, np.array([-1, 0, 1]))
    proba = model.predict_proba(X)
    # Rows: hike / hold / cut → P(1) / P(0) / P(-1) each = 1.
    np.testing.assert_allclose(
        proba,
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(model.predict(X), np.array([1, 0, -1]))


def test_market_implied_custom_column_names_flow_through() -> None:
    X = pd.DataFrame({"implied": [4.60], "current": [4.35]})
    model = MarketImplied(implied_rate_col="implied", current_rate_col="current").fit(
        X, _mi_training_y()
    )
    assert model.predict(X) == np.array(["hike"])


def test_market_implied_custom_move_size_shifts_boundary() -> None:
    """With a 50 bp assumed move, a +25 bp Δ is a half-move (hike/hold tie)."""
    X = _mi_frame([4.60], [4.35])
    model = MarketImplied(move_size_pct=0.50).fit(X, _mi_training_y())
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.array([[0.0, 0.5, 0.5]]))


def test_market_implied_rejects_zero_or_negative_move_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        MarketImplied(move_size_pct=0)
    with pytest.raises(ValueError, match="positive"):
        MarketImplied(move_size_pct=-0.25)


def test_market_implied_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="must all differ"):
        MarketImplied(cut_label="hold", hold_label="hold", hike_label="hike").fit(
            _mi_frame([4.35], [4.35]), pd.Series(["hold"])
        )


# -----------------------------------------------------------------------------
# Error paths.
# -----------------------------------------------------------------------------
def test_market_implied_nan_implied_rate_raises_at_predict() -> None:
    """NaN in the implied column → clear ValueError, not a silent argmax on NaN."""
    model = MarketImplied().fit(_mi_frame([4.35], [4.35]), _mi_training_y())
    X_bad = _mi_frame([np.nan, 4.35], [4.35, 4.35])
    with pytest.raises(ValueError, match="NaN"):
        model.predict(X_bad)
    with pytest.raises(ValueError, match="NaN"):
        model.predict_proba(X_bad)


def test_market_implied_nan_current_rate_raises_at_predict() -> None:
    model = MarketImplied().fit(_mi_frame([4.35], [4.35]), _mi_training_y())
    X_bad = _mi_frame([4.35, 4.35], [np.nan, 4.35])
    with pytest.raises(ValueError, match="NaN"):
        model.predict(X_bad)


def test_market_implied_missing_column_raises_at_predict() -> None:
    """Fit doesn't read the market cols; the miss surfaces at predict — same
    literature-defaults pattern as TaylorRule."""
    model = MarketImplied().fit(_mi_frame([4.35], [4.35]), _mi_training_y())
    X_bad = pd.DataFrame({"other": [1.0]})
    with pytest.raises(KeyError):
        model.predict(X_bad)


def test_market_implied_warns_when_configured_labels_unseen_in_y(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mismatch between configured labels and the actual target encoding should warn."""
    import logging

    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(caplog.handler, format="{message}", level="WARNING")
    try:
        with caplog.at_level(logging.WARNING):
            MarketImplied().fit(
                _mi_frame([4.35, 4.35], [4.35, 4.35]),
                # y uses int encoding but model expects strings — every label unseen.
                pd.Series([0, 1]),
            )
    finally:
        loguru_logger.remove(handler_id)
    assert any("not seen in training y" in rec.message for rec in caplog.records)
