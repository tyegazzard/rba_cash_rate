"""Tests for :mod:`rba.validation.calibration` — reliability diagrams + ECE/MCE + plots.

No network. Numeric tests cover the bin-table anchors (perfect calibration,
overconfident wrong one-hot, empty bins, quantile vs uniform strategies) and
the ECE / MCE reductions. Plotting tests keep it lightweight: import plotly
locally, assert the returned object is a :class:`~plotly.graph_objects.Figure`
with the expected trace count / axis ranges, no rendering to a real backend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from rba.validation.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    plot_all_classes,
    plot_reliability_diagram,
    reliability_diagram_data,
)


# =============================================================================
# reliability_diagram_data — the bin-level artefact.
# =============================================================================
def _proba_for_class(p_class: np.ndarray, labels: list[str], class_label: str) -> np.ndarray:
    """Build a (n, 3) proba matrix where the given class column holds ``p_class``.

    The other two columns split ``1 - p_class`` evenly. Not necessarily realistic
    but keeps the row a valid probability distribution.
    """
    idx = labels.index(class_label)
    other = (1.0 - p_class) / (len(labels) - 1)
    proba = np.tile(other[:, None], (1, len(labels)))
    proba[:, idx] = p_class
    return proba


def test_data_returns_expected_columns_and_shape() -> None:
    y_true = ["hike", "hold", "cut", "hold"]
    labels = ["cut", "hike", "hold"]
    y_proba = np.tile([1 / 3, 1 / 3, 1 / 3], (4, 1))
    data = reliability_diagram_data(y_true, y_proba, labels, "hike", n_bins=10)
    assert set(data.columns) == {
        "bin_index",
        "bin_low",
        "bin_high",
        "count",
        "mean_pred",
        "observed_freq",
    }
    assert len(data) == 10


def test_data_uniform_bin_edges_span_zero_to_one() -> None:
    data = reliability_diagram_data(
        ["hold"], [[0.0, 0.0, 1.0]], ["cut", "hike", "hold"], "hold", n_bins=5
    )
    np.testing.assert_allclose(data["bin_low"].to_numpy(), [0.0, 0.2, 0.4, 0.6, 0.8])
    np.testing.assert_allclose(data["bin_high"].to_numpy(), [0.2, 0.4, 0.6, 0.8, 1.0])


def test_data_perfect_calibration_gives_matching_mean_pred_and_observed_freq() -> None:
    """A 'hike prob = 0.7 → hike 70% of the time' anchor: two full bins, perfectly calibrated."""
    labels = ["cut", "hike", "hold"]
    # 10 samples: p(hike)=0.3, 3 are actually hike; 10 samples: p(hike)=0.7, 7 are actually hike.
    n_each = 10
    p_lower = np.full(n_each, 0.3)
    p_upper = np.full(n_each, 0.7)
    y_true_lower = ["hike"] * 3 + ["hold"] * 7
    y_true_upper = ["hike"] * 7 + ["hold"] * 3
    y_true = y_true_lower + y_true_upper
    y_proba = np.vstack(
        [_proba_for_class(p_lower, labels, "hike"), _proba_for_class(p_upper, labels, "hike")]
    )
    data = reliability_diagram_data(y_true, y_proba, labels, "hike", n_bins=10)
    # p=0.3 falls in the [0.3, 0.4) bin, p=0.7 falls in [0.7, 0.8).
    lower_bin = data[(data["bin_low"] <= 0.3) & (data["bin_high"] > 0.3)].iloc[0]
    upper_bin = data[(data["bin_low"] <= 0.7) & (data["bin_high"] > 0.7)].iloc[0]
    assert lower_bin["count"] == 10
    assert upper_bin["count"] == 10
    assert lower_bin["mean_pred"] == pytest.approx(0.3)
    assert lower_bin["observed_freq"] == pytest.approx(0.3)
    assert upper_bin["mean_pred"] == pytest.approx(0.7)
    assert upper_bin["observed_freq"] == pytest.approx(0.7)


def test_data_empty_bins_have_zero_count_and_nan_stats() -> None:
    """A bin with no samples surfaces cleanly — count 0, NaN mean_pred / observed_freq."""
    data = reliability_diagram_data(
        ["hold"] * 5,
        _proba_for_class(np.full(5, 0.05), ["cut", "hike", "hold"], "hold"),
        ["cut", "hike", "hold"],
        "hold",
        n_bins=10,
    )
    # Only the [0, 0.1) bin has samples; the other 9 bins are empty.
    empty = data[data["count"] == 0]
    assert len(empty) == 9
    assert empty["mean_pred"].isna().all()
    assert empty["observed_freq"].isna().all()


def test_data_probability_of_one_falls_into_last_bin() -> None:
    """The nudge into the top bin: a p=1.0 sample must not be dropped."""
    data = reliability_diagram_data(
        ["hold"],
        [[0.0, 0.0, 1.0]],
        ["cut", "hike", "hold"],
        "hold",
        n_bins=10,
    )
    assert data.loc[data["bin_index"] == 9, "count"].item() == 1


def test_data_quantile_strategy_yields_equal_count_bins() -> None:
    """Quantile bins give approximately equal counts across surviving bins."""
    rng = np.random.default_rng(12)
    n = 100
    p_hike = rng.uniform(0.1, 0.9, size=n)
    labels = ["cut", "hike", "hold"]
    y_proba = _proba_for_class(p_hike, labels, "hike")
    y_true = rng.choice(labels, size=n)
    data = reliability_diagram_data(
        y_true, y_proba, labels, "hike", n_bins=10, strategy="quantile"
    )
    # 100 samples across 10 quantile bins → 10 per bin (± 1 for edge effects).
    non_empty = data[data["count"] > 0]
    assert all(9 <= c <= 12 for c in non_empty["count"])


def test_data_quantile_collapses_duplicated_edges() -> None:
    """Highly-concentrated probabilities → fewer unique quantile edges → fewer bins."""
    # 100 rows all with p(hike) = 0.5. Every quantile edge lands on 0.5 → collapses to 1 bin.
    labels = ["cut", "hike", "hold"]
    y_proba = _proba_for_class(np.full(100, 0.5), labels, "hike")
    y_true = ["hold"] * 100
    data = reliability_diagram_data(
        y_true, y_proba, labels, "hike", n_bins=10, strategy="quantile"
    )
    assert len(data) < 10


def test_data_rejects_unknown_class_label() -> None:
    y_true = ["hold"]
    y_proba = [[1.0, 0.0]]
    labels = ["hold", "hike"]
    with pytest.raises(ValueError, match="class_label"):
        reliability_diagram_data(y_true, y_proba, labels, "cut")


def test_data_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        reliability_diagram_data(["hold"], [[1.0, 0.0]], ["hold", "hike"], "hold", strategy="oops")


def test_data_rejects_shape_mismatch() -> None:
    y_true = ["hold"]
    y_proba = [[0.5, 0.5]]
    with pytest.raises(ValueError, match="labels"):
        reliability_diagram_data(y_true, y_proba, ["cut", "hike", "hold"], "hold")


def test_data_rejects_zero_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        reliability_diagram_data(["hold"], [[1.0, 0.0]], ["hold", "hike"], "hold", n_bins=0)


# =============================================================================
# ECE / MCE reductions.
# =============================================================================
def test_ece_perfect_calibration_is_zero() -> None:
    """Same fixture as the perfect-calibration data test — ECE reduces to 0."""
    labels = ["cut", "hike", "hold"]
    p_lower = np.full(10, 0.3)
    p_upper = np.full(10, 0.7)
    y_true = ["hike"] * 3 + ["hold"] * 7 + ["hike"] * 7 + ["hold"] * 3
    y_proba = np.vstack(
        [_proba_for_class(p_lower, labels, "hike"), _proba_for_class(p_upper, labels, "hike")]
    )
    assert expected_calibration_error(y_true, y_proba, labels, "hike") == pytest.approx(0.0)


def test_ece_wrong_one_hot_is_one() -> None:
    """P(hike)=1 but every sample is actually hold: single bin, gap = 1."""
    labels = ["cut", "hike", "hold"]
    y_proba = _proba_for_class(np.full(20, 1.0), labels, "hike")
    y_true = ["hold"] * 20
    assert expected_calibration_error(y_true, y_proba, labels, "hike") == pytest.approx(1.0)


def test_ece_is_sample_weighted_across_bins() -> None:
    """A big-count perfectly-calibrated bin should dilute a small-count miscalibrated bin."""
    labels = ["hold", "hike"]
    # Bin A: 90 samples, p(hike)=0.5, actual hike rate 50% → perfect (gap 0).
    p_a = np.full(90, 0.5)
    y_a = ["hike"] * 45 + ["hold"] * 45
    # Bin B: 10 samples, p(hike)=0.9, actual hike rate 0 → gap 0.9.
    p_b = np.full(10, 0.9)
    y_b = ["hold"] * 10
    y_proba = np.vstack(
        [_proba_for_class(p_a, labels, "hike"), _proba_for_class(p_b, labels, "hike")]
    )
    y_true = y_a + y_b
    # ECE = (90/100)*0 + (10/100)*0.9 = 0.09.
    assert expected_calibration_error(y_true, y_proba, labels, "hike") == pytest.approx(0.09)


def test_mce_is_worst_bin_regardless_of_count() -> None:
    """Same fixture — MCE ignores the weight, reports the worst bin's 0.9 gap."""
    labels = ["hold", "hike"]
    p_a = np.full(90, 0.5)
    y_a = ["hike"] * 45 + ["hold"] * 45
    p_b = np.full(10, 0.9)
    y_b = ["hold"] * 10
    y_proba = np.vstack(
        [_proba_for_class(p_a, labels, "hike"), _proba_for_class(p_b, labels, "hike")]
    )
    y_true = y_a + y_b
    assert maximum_calibration_error(y_true, y_proba, labels, "hike") == pytest.approx(0.9)


def test_ece_and_mce_agree_when_only_one_bin_has_data() -> None:
    """Single non-empty bin: ECE and MCE both collapse to that bin's gap."""
    labels = ["hold", "hike"]
    y_proba = _proba_for_class(np.full(50, 0.4), labels, "hike")
    y_true = ["hike"] * 25 + ["hold"] * 25  # actual hike rate 0.5 → gap 0.1
    ece = expected_calibration_error(y_true, y_proba, labels, "hike")
    mce = maximum_calibration_error(y_true, y_proba, labels, "hike")
    assert ece == pytest.approx(0.1)
    assert mce == pytest.approx(0.1)


# =============================================================================
# Plots — smoke tests for plotly Figure structure.
# =============================================================================
def _sample_data() -> pd.DataFrame:
    labels = ["cut", "hike", "hold"]
    y_proba = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (12, 1))
    y_true = ["hike"] * 4 + ["hold"] * 4 + ["cut"] * 4
    return reliability_diagram_data(y_true, y_proba, labels, "hike", n_bins=5)


def test_plot_reliability_diagram_returns_plotly_figure() -> None:
    fig = plot_reliability_diagram(_sample_data(), title="hike")
    assert isinstance(fig, go.Figure)
    # Three traces expected: histogram + perfect line + model.
    assert len(fig.data) == 3
    assert fig.layout.title.text == "hike"


def test_plot_reliability_diagram_axes_span_zero_to_one() -> None:
    fig = plot_reliability_diagram(_sample_data())
    assert tuple(fig.layout.xaxis.range) == (0, 1)
    assert tuple(fig.layout.yaxis.range) == (0, 1)


def test_plot_reliability_diagram_optional_layers_reduce_trace_count() -> None:
    fig = plot_reliability_diagram(_sample_data(), show_histogram=False, show_perfect_line=False)
    # Only the model trace remains.
    assert len(fig.data) == 1


def test_plot_reliability_diagram_handles_empty_bins() -> None:
    """A bin table with only one non-empty bin still renders without errors."""
    data = pd.DataFrame(
        {
            "bin_index": [0, 1, 2],
            "bin_low": [0.0, 0.33, 0.67],
            "bin_high": [0.33, 0.67, 1.0],
            "count": [0, 5, 0],
            "mean_pred": [np.nan, 0.5, np.nan],
            "observed_freq": [np.nan, 0.4, np.nan],
        }
    )
    fig = plot_reliability_diagram(data)
    assert isinstance(fig, go.Figure)


def test_plot_all_classes_returns_figure_with_one_subplot_per_class() -> None:
    labels = ["cut", "hike", "hold"]
    y_proba = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (15, 1))
    y_true = ["cut"] * 5 + ["hike"] * 5 + ["hold"] * 5
    fig = plot_all_classes(y_true, y_proba, labels, n_bins=5)
    assert isinstance(fig, go.Figure)
    # Subplot titles carry the class labels.
    subplot_titles = [ann.text for ann in fig.layout.annotations]
    assert subplot_titles == ["cut", "hike", "hold"]
