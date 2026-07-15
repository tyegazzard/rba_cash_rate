"""Reliability diagrams and calibration metrics.

Answers the question :mod:`~rba.validation.metrics` doesn't: are the classifier's
stated probabilities honest? High log-loss punishes bad probabilities in
aggregate; the reliability diagram *shows* where they miss, and Expected /
Maximum Calibration Error summarise the gap numerically.

For each class ``c`` (one-vs-rest):

1. Take ``P(y = c)`` from every sample.
2. Bin the probabilities (default 10 equal-width bins over ``[0, 1]``; also
   supports 10 equal-count quantile bins — the two ``strategy`` values match
   :func:`sklearn.calibration.calibration_curve`).
3. In each bin, compute mean predicted probability + observed frequency of
   class ``c`` — the diagram's ``(x, y)`` points.
4. A perfectly calibrated classifier's points lie on the ``y = x`` diagonal.

Bin-count histogram overlay in the plots shows which bins carry weight — a
perfectly-calibrated point in a bin of size 1 is meaningless, one in a bin of
size 200 is a hard signal.

Design
------
- **Data / metric functions** (:func:`reliability_diagram_data`,
  :func:`expected_calibration_error`, :func:`maximum_calibration_error`) are
  pure numpy / pandas — no plotting import at module top so pulling
  :mod:`rba.validation.metrics` doesn't drag matplotlib / plotly along.
- **Plot functions** import :mod:`plotly` lazily inside the call. Plotly is the
  project's plotting library (declared in ``pyproject.toml``, used by the §9
  Streamlit dashboard); ``plot_reliability_diagram`` and
  :func:`plot_all_classes` return :class:`plotly.graph_objects.Figure` objects
  the harness can persist to ``reports/figures/`` or embed in Streamlit
  directly.
- **Multiclass reduction** — one-vs-rest per class, same treatment as sklearn
  and the standard reliability-diagram literature. Multiclass reliability at
  the top-label level (``argmax`` correctness vs ``max`` probability) is a
  different diagnostic and is not covered here.
- **Empty bins** — surface as rows with ``count = 0`` and NaN in ``mean_pred``
  / ``observed_freq``, so the caller can distinguish "we have no data for this
  bin" from "we have data and the model is calibrated at zero probability."

No network anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = [
    "expected_calibration_error",
    "maximum_calibration_error",
    "plot_all_classes",
    "plot_reliability_diagram",
    "reliability_diagram_data",
]

_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")


# =============================================================================
# Data: per-bin (count, mean_pred, observed_freq) — the diagram's numeric core.
# =============================================================================
def reliability_diagram_data(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Sequence[Any] | np.ndarray,
    class_label: Any,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Per-bin reliability table for ``P(y = class_label)`` (one-vs-rest).

    Parameters
    ----------
    y_true
        True class labels, aligned to rows of ``y_proba``.
    y_proba
        Shape ``(n_samples, n_classes)`` — the classifier's ``predict_proba(X)``.
    labels
        Column order of ``y_proba`` (pass the classifier's ``classes_``).
    class_label
        Which class in ``labels`` to build the reliability diagram for. Must
        be a member of ``labels``.
    n_bins
        Number of bins. Default 10 matches sklearn's ``calibration_curve``.
    strategy
        ``"uniform"`` — equal-width bins over ``[0, 1]``. Default; robust to
        any distribution of predicted probabilities.
        ``"quantile"`` — equal-count bins from the empirical quantiles of the
        predicted probabilities. Duplicated quantile edges are dropped, so the
        returned frame may have **fewer** than ``n_bins`` rows if the
        probabilities are highly concentrated.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_index`` / ``bin_low`` / ``bin_high`` / ``count`` /
        ``mean_pred`` / ``observed_freq``. One row per bin; empty bins have
        ``count = 0`` and NaN in ``mean_pred`` / ``observed_freq``.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of :data:`_STRATEGIES`, if ``class_label``
        is not in ``labels``, or if ``n_bins < 1``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins!r}.")
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}; got {strategy!r}.")
    labels_list = list(labels)
    if class_label not in labels_list:
        raise ValueError(f"class_label {class_label!r} not in labels {labels_list!r}.")

    col = labels_list.index(class_label)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)
    if y_proba_arr.ndim != 2:
        raise ValueError(f"y_proba must be 2-D; got shape {y_proba_arr.shape}.")
    if y_proba_arr.shape[1] != len(labels_list):
        raise ValueError(
            f"y_proba has {y_proba_arr.shape[1]} columns but labels has "
            f"{len(labels_list)} entries — the two must match."
        )
    y_proba_col = y_proba_arr[:, col]
    y_binary = (np.asarray(y_true) == class_label).astype(np.float64)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # quantile
        raw_edges = np.quantile(y_proba_col, np.linspace(0.0, 1.0, n_bins + 1))
        # ``np.unique`` collapses duplicated quantile edges (highly concentrated
        # probabilities) — the returned frame will have < n_bins rows in that case.
        bin_edges = np.unique(raw_edges)

    # ``np.digitize`` with the interior edges as bins gives i ∈ [0, n_bins-1].
    # ``right=False`` puts values equal to a boundary in the *upper* bin, which
    # matches sklearn's convention. The rightmost value (== 1.0 typically) is
    # nudged into the last bin manually.
    interior = bin_edges[1:-1]
    bin_idx = np.digitize(y_proba_col, interior, right=False)
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)

    rows: list[dict[str, Any]] = []
    for i in range(len(bin_edges) - 1):
        mask = bin_idx == i
        n = int(mask.sum())
        if n == 0:
            mean_pred: float = float("nan")
            observed_freq: float = float("nan")
        else:
            mean_pred = float(y_proba_col[mask].mean())
            observed_freq = float(y_binary[mask].mean())
        rows.append(
            {
                "bin_index": i,
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "count": n,
                "mean_pred": mean_pred,
                "observed_freq": observed_freq,
            }
        )
    logger.debug(
        "reliability_diagram_data(class={!r}, strategy={}, n_bins={}): "
        "returned {} bin(s), {} non-empty.",
        class_label,
        strategy,
        n_bins,
        len(rows),
        sum(1 for r in rows if r["count"] > 0),
    )
    return pd.DataFrame(rows)


# =============================================================================
# Calibration summaries — one number each, computed from the bin table.
# =============================================================================
def expected_calibration_error(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Sequence[Any] | np.ndarray,
    class_label: Any,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Sample-weighted mean gap between predicted and observed frequency per bin.

    ``ECE = Σ_bins (n_bin / N) · |mean_pred_bin − observed_freq_bin|``.
    Range ``[0, 1]``; 0 iff every non-empty bin's ``mean_pred`` matches its
    ``observed_freq``. Returns ``NaN`` on empty input (no samples in any bin).
    """
    data = reliability_diagram_data(
        y_true, y_proba, labels, class_label, n_bins=n_bins, strategy=strategy
    )
    return _weighted_gap(data, aggregator=np.average)


def maximum_calibration_error(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Sequence[Any] | np.ndarray,
    class_label: Any,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Worst per-bin gap between predicted and observed frequency.

    ``MCE = max_bins |mean_pred_bin − observed_freq_bin|`` over non-empty bins.
    Range ``[0, 1]``. Complements :func:`expected_calibration_error` — ECE hides
    a single badly-miscalibrated bin behind well-calibrated ones; MCE surfaces
    it.
    """
    data = reliability_diagram_data(
        y_true, y_proba, labels, class_label, n_bins=n_bins, strategy=strategy
    )
    return _weighted_gap(data, aggregator=_max)


def _weighted_gap(data: pd.DataFrame, *, aggregator: Any) -> float:
    """Shared reduce path for ECE / MCE — pick out non-empty bins, reduce."""
    valid = data[data["count"] > 0]
    if valid.empty:
        return float("nan")
    gaps = np.abs(valid["mean_pred"].to_numpy() - valid["observed_freq"].to_numpy())
    if aggregator is np.average:
        weights = valid["count"].to_numpy(dtype=np.float64)
        return float(np.average(gaps, weights=weights))
    return float(aggregator(gaps))


def _max(a: np.ndarray) -> float:
    """Named wrapper so :func:`_weighted_gap`'s aggregator dispatch is readable."""
    return float(np.max(a))


# =============================================================================
# Plots — plotly Figures (matches project plotting library / Streamlit needs).
# =============================================================================
def plot_reliability_diagram(
    data: pd.DataFrame,
    *,
    title: str | None = None,
    show_perfect_line: bool = True,
    show_histogram: bool = True,
) -> go.Figure:
    """Render a single reliability diagram from :func:`reliability_diagram_data` output.

    Bin markers on the main axis (mean predicted probability vs. observed
    frequency), a dashed y=x reference line for perfect calibration, and — on
    a secondary y-axis — a bar histogram of bin counts so viewers can weight
    the deviations by sample size.

    Parameters
    ----------
    data
        Output of :func:`reliability_diagram_data`.
    title
        Optional plot title; ``None`` leaves it blank so the caller can set
        one in :func:`plot_all_classes`'s subplot loop.
    show_perfect_line
        Overlay ``y = x`` (perfectly-calibrated reference). Default ``True``.
    show_histogram
        Overlay a bin-count bar histogram on a secondary y-axis. Default
        ``True``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    valid = data[data["count"] > 0]
    fig = go.Figure()

    if show_histogram:
        widths = (data["bin_high"] - data["bin_low"]).to_numpy()
        centers = (data["bin_low"] + widths / 2).to_numpy()
        fig.add_trace(
            go.Bar(
                x=centers,
                y=data["count"].to_numpy(),
                width=widths,
                name="samples per bin",
                marker={"color": "lightgray"},
                opacity=0.6,
                yaxis="y2",
                hovertemplate="bin center %{x:.2f}<br>count %{y:d}<extra></extra>",
            )
        )

    if show_perfect_line:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line={"dash": "dash", "color": "gray"},
                name="perfect calibration",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=valid["mean_pred"].to_numpy(),
            y=valid["observed_freq"].to_numpy(),
            mode="lines+markers",
            name="model",
            line={"color": "steelblue"},
            marker={"size": 8},
            customdata=valid["count"].to_numpy().reshape(-1, 1),
            hovertemplate=(
                "mean predicted %{x:.3f}<br>observed %{y:.3f}<br>n %{customdata[0]:d}"
                "<extra></extra>"
            ),
        )
    )

    layout: dict[str, Any] = {
        "xaxis": {"title": "mean predicted probability", "range": [0, 1]},
        "yaxis": {"title": "observed frequency", "range": [0, 1]},
        "legend": {"orientation": "h", "y": -0.2},
    }
    if show_histogram:
        layout["yaxis2"] = {
            "title": "sample count",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        }
    if title is not None:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


def plot_all_classes(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    labels: Sequence[Any],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    show_histogram: bool = True,
) -> go.Figure:
    """One reliability diagram per class, side-by-side in a single Figure.

    Convenience for the §7 harness: one call, one Figure, one plotly write
    to ``reports/figures/``. Titles per subplot are the class labels; the
    layout stacks legends into a single horizontal strip.

    Parameters
    ----------
    y_true, y_proba, labels
        Same contract as :func:`reliability_diagram_data`.
    n_bins, strategy
        Passed through to :func:`reliability_diagram_data` per subplot.
    show_histogram
        Overlay bin-count histogram in each subplot. Default ``True``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots

    labels_list = list(labels)
    n = len(labels_list)
    specs = [[{"secondary_y": show_histogram} for _ in range(n)]]
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[str(c) for c in labels_list],
        specs=specs,
        horizontal_spacing=0.08,
    )

    for col_idx, cls in enumerate(labels_list, start=1):
        data = reliability_diagram_data(
            y_true, y_proba, labels_list, cls, n_bins=n_bins, strategy=strategy
        )
        sub = plot_reliability_diagram(
            data,
            show_perfect_line=True,
            show_histogram=show_histogram,
        )
        for trace in sub.data:
            # Route the histogram trace to the secondary y-axis; all others to primary.
            secondary = show_histogram and getattr(trace, "yaxis", None) == "y2"
            fig.add_trace(trace, row=1, col=col_idx, secondary_y=secondary)
        fig.update_xaxes(title_text="mean predicted", range=[0, 1], row=1, col=col_idx)
        fig.update_yaxes(
            title_text="observed", range=[0, 1], row=1, col=col_idx, secondary_y=False
        )
        if show_histogram:
            fig.update_yaxes(
                title_text="count", showgrid=False, row=1, col=col_idx, secondary_y=True
            )

    fig.update_layout(showlegend=False)
    return fig
