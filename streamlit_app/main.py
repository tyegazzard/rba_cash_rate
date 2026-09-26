"""Streamlit dashboard for RBA cash-rate prediction.

Serves the deployable best model (``reports/best_model/``) on the **next** Board
meeting via :func:`rba.models.predict_next.predict_next_meeting`, and shows the
held-out backtest track record from the ``reports/`` artifacts. Run with::

    uv run streamlit run streamlit_app/main.py

The heavy prediction (a full point-in-time feature build, ~5s on cached data) is
memoised with ``st.cache_data`` so reruns are instant; the sidebar "refresh" pulls
sources live and clears the cache. Predictions rendered here are **not** written to
``reports/predictions/predictions.jsonl`` — only the CLI logs (so dashboard reruns
don't spam the audit trail).

Colour semantics (dataviz reference palette, diverging): a cash-rate decision is
polarity, not arbitrary identity, so **cut = blue** (easing), **hold = neutral
grey**, **hike = red** (tightening).
"""

from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rba.config import REPORTS_DIR
from rba.data.meeting_schedule import next_meeting_date, upcoming_meeting_dates
from rba.models.predict_next import predict_next_meeting

# -----------------------------------------------------------------------------
# Palette (dataviz reference instance, light surface) + chart chrome.
# -----------------------------------------------------------------------------
CLASS_COLOURS = {"cut": "#2a78d6", "hold": "#898781", "hike": "#e34948"}
KIND_COLOURS = {"model": "#2a78d6", "baseline": "#eb6834", "ensemble": "#4a3aa7"}
BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
_SEQ_BLUE = ["#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

_BEST_DIR = REPORTS_DIR / "best_model"
_LEADERBOARD_CSV = REPORTS_DIR / "holdout_metrics.csv"
_RUNS_DIR = REPORTS_DIR / "runs"
_DISPLAY_CLASSES = ("cut", "hike", "hold")  # sklearn-sorted, matches the run manifests

st.set_page_config(page_title="RBA Cash Rate Prediction", layout="wide", page_icon="🏦")


# -----------------------------------------------------------------------------
# Cached data loaders.
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Building the point-in-time feature row and predicting…")
def load_prediction(meeting_date: str | None, refresh_nonce: int) -> dict[str, Any]:
    """Run the model on ``meeting_date`` (or the next scheduled meeting).

    ``refresh_nonce`` is part of the cache key: bumping it (the sidebar refresh
    button) forces a live data pull + recompute. Returns the ``PredictionResult``
    as a plain dict so it caches cleanly.
    """
    result = predict_next_meeting(meeting_date=meeting_date, force_download=refresh_nonce > 0)
    return asdict(result)


@st.cache_data
def load_leaderboard() -> pd.DataFrame | None:
    """Load the §8 held-out metrics table (per-model backtest accuracy)."""
    if not _LEADERBOARD_CSV.exists():
        return None
    return pd.read_csv(_LEADERBOARD_CSV)


@st.cache_data
def load_confusion(best_model: str) -> tuple[list[list[int]], list[str]] | None:
    """Load the best model's held-out confusion matrix from its latest run manifest."""
    runs = sorted(_RUNS_DIR.glob(f"{best_model}_*"))
    if not runs:
        return None
    manifest = json.loads((runs[-1] / "manifest.json").read_text(encoding="utf-8"))
    cm = manifest.get("metrics_overall", {}).get("confusion_matrix")
    return (cm, list(_DISPLAY_CLASSES)) if cm else None


@st.cache_data
def load_best_model_meta() -> dict[str, Any]:
    """Load ``best_model.json`` (model name, tuned flag, n_train, test bal-acc)."""
    path = _BEST_DIR / "best_model.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# -----------------------------------------------------------------------------
# Figure builders (pure: data -> go.Figure).
# -----------------------------------------------------------------------------
def _base_layout(fig: go.Figure, *, height: int = 300, showlegend: bool = False) -> go.Figure:
    """Apply the shared chrome: transparent surface, recessive grid, system font.

    Margins are minimal and ``automargin`` is enabled on both axes so plotly grows
    them to fit tick labels (category names, feature names) instead of clipping.
    """
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=48, t=8, b=8),  # r fits outside bar labels; l/b grow via automargin
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        bargap=0.28,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True
    )
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=GRID, automargin=True)
    return fig


def probability_figure(probabilities: dict[str, float]) -> go.Figure:
    """Horizontal bars of P(cut/hold/hike), each in its diverging class colour."""
    order = ["hike", "hold", "cut"]  # bottom-to-top so cut(easing)→hike(tightening) reads up
    values = [probabilities.get(c, 0.0) for c in order]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[c.upper() for c in order],
            orientation="h",
            marker_color=[CLASS_COLOURS[c] for c in order],
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return _base_layout(fig, height=220)


def market_compare_figure(model: dict[str, float], market: dict[str, float]) -> go.Figure:
    """Grouped bars comparing model vs market-implied probabilities per outcome."""
    cats = ["cut", "hold", "hike"]
    fig = go.Figure()
    fig.add_bar(
        name="Model",
        x=[c.upper() for c in cats],
        y=[model.get(c, 0.0) for c in cats],
        marker_color=BLUE,
        text=[f"{model.get(c, 0.0):.0%}" for c in cats],
        textposition="outside",
        hovertemplate="Model %{x}: %{y:.1%}<extra></extra>",
        cliponaxis=False,
    )
    fig.add_bar(
        name="Market-implied",
        x=[c.upper() for c in cats],
        y=[market.get(c, 0.0) for c in cats],
        marker_color=KIND_COLOURS["baseline"],
        text=[f"{market.get(c, 0.0):.0%}" for c in cats],
        textposition="outside",
        hovertemplate="Market %{x}: %{y:.1%}<extra></extra>",
        cliponaxis=False,
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(barmode="group")
    return _base_layout(fig, height=300, showlegend=True)


def feature_figure(top_features: list[dict[str, Any]]) -> go.Figure:
    """Horizontal bars of the model's top global feature importances (single hue)."""
    feats = list(reversed(top_features))  # largest at top
    fig = go.Figure(
        go.Bar(
            x=[f["importance_pct"] for f in feats],
            y=[f["feature"] for f in feats],
            orientation="h",
            marker_color=BLUE,
            text=[f"{f['importance_pct']:.1f}%" for f in feats],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(ticksuffix="%")
    return _base_layout(fig, height=max(240, 26 * len(feats)))


def leaderboard_figure(df: pd.DataFrame) -> go.Figure:
    """Horizontal bars of held-out balanced accuracy per model, coloured by kind."""
    data = df.sort_values("balanced_accuracy", ascending=True)
    fig = go.Figure()
    for kind, colour in KIND_COLOURS.items():
        sub = data[data["kind"] == kind]
        if sub.empty:
            continue
        fig.add_bar(
            name=kind,
            y=sub["model"],
            x=sub["balanced_accuracy"],
            orientation="h",
            marker_color=colour,
            text=[f"{v:.2f}" for v in sub["balanced_accuracy"]],
            textposition="outside",
            hovertemplate="%{y}: balanced acc %{x:.3f}<extra></extra>",
            cliponaxis=False,
        )
    fig.update_xaxes(range=[0, 0.75], title_text="balanced accuracy (held-out test)")
    return _base_layout(fig, height=360, showlegend=True)


def confusion_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    """Heatmap of the best model's held-out confusion matrix (rows=true, cols=pred)."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[c.upper() for c in labels],
            y=[c.upper() for c in labels],
            colorscale=[[i / (len(_SEQ_BLUE) - 1), c] for i, c in enumerate(_SEQ_BLUE)],
            showscale=False,
            hovertemplate="true %{y} / pred %{x}: %{z}<extra></extra>",
        )
    )
    total = sum(sum(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            fig.add_annotation(
                x=j,
                y=i,
                text=str(v),
                showarrow=False,
                font=dict(color=INK if v < 0.5 * total else "#ffffff", size=15, family=FONT),
            )
    fig.update_layout(height=300, xaxis_title="predicted", yaxis_title="actual")
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=300)


# -----------------------------------------------------------------------------
# Layout.
# -----------------------------------------------------------------------------
def _upcoming_options() -> list[dt.date]:
    """The upcoming scheduled meetings for the sidebar selector (>= next)."""
    upcoming = upcoming_meeting_dates()
    if upcoming:
        return upcoming
    try:  # schedule exhausted for "today" — fall back to the last known next meeting
        return [next_meeting_date(dt.date(2020, 1, 1))]
    except LookupError:
        return []


def render_sidebar() -> tuple[str | None, int]:
    """Draw the sidebar; return (selected meeting_date ISO or None, refresh_nonce)."""
    st.sidebar.title("🏦 RBA cash-rate model")
    meta = load_best_model_meta()
    if meta:
        st.sidebar.caption(
            f"**Model:** `{meta.get('best_model', '—')}`  \n"
            f"**Tuned:** {meta.get('tuned', False)}  ·  "
            f"**Trained on:** {meta.get('n_train', '—')} meetings  \n"
            f"**Test balanced acc:** {meta.get('balanced_accuracy_test', float('nan')):.3f}"
        )
    st.sidebar.divider()

    options = _upcoming_options()
    meeting_iso: str | None = None
    if options:
        picked = st.sidebar.selectbox(
            "Meeting to predict",
            options,
            format_func=lambda d: d.strftime("%a %d %b %Y"),
            index=0,
        )
        meeting_iso = picked.isoformat()
    else:
        st.sidebar.warning("No upcoming meeting in the schedule — append the next year's dates.")

    if "refresh_nonce" not in st.session_state:
        st.session_state.refresh_nonce = 0
    if st.sidebar.button("↻ Refresh data (live pull)", use_container_width=True):
        st.session_state.refresh_nonce += 1
        st.cache_data.clear()
    st.sidebar.caption(
        "Predictions run on cached data by default. A live pull re-downloads every "
        "source (slow; some may be unavailable) and is only needed close to a meeting."
    )
    return meeting_iso, st.session_state.refresh_nonce


def render_headline(pred: dict[str, Any]) -> None:
    """The top row: predicted decision, confidence, prior rate, meeting date."""
    predicted = pred["predicted_class"]
    prob = pred["probabilities"][predicted]
    colour = CLASS_COLOURS[predicted]
    meeting = dt.date.fromisoformat(pred["meeting_date"]).strftime("%A %d %B %Y")

    st.markdown(f"### Next meeting — {meeting}")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            f"<div style='font-size:0.9rem;color:{MUTED}'>Predicted decision</div>"
            f"<div style='font-size:2.6rem;font-weight:700;color:{colour};line-height:1.1'>"
            f"{predicted.upper()}</div>"
            f"<div style='font-size:1rem;color:{MUTED}'>{prob:.1%} model probability</div>",
            unsafe_allow_html=True,
        )
    prior = pred.get("prior_rate_pct")
    c2.metric("Current cash rate", f"{prior:.2f}%" if prior is not None else "—")
    gap = pred["vintage"].get("future_gap_days")
    c3.metric("Days since last meeting", f"{gap}" if gap is not None else "—")


def render_market_row(pred: dict[str, Any]) -> None:
    """Two columns: model probability distribution + market-implied comparison."""
    left, right = st.columns(2)
    with left:
        st.markdown("#### Probability distribution")
        st.plotly_chart(
            probability_figure(pred["probabilities"]), use_container_width=True, theme=None
        )
    with right:
        st.markdown("#### Model vs market-implied")
        market = pred.get("market_implied")
        if market is not None:
            st.plotly_chart(
                market_compare_figure(pred["probabilities"], market["probabilities"]),
                use_container_width=True,
                theme=None,
            )
            st.caption(
                f"Market-implied cash rate {market['implied_rate_pct']:.2f}% "
                f"(ASX 30-day futures) → {market['predicted_class'].upper()}."
            )
        else:
            st.info(
                "Market-implied probabilities are unavailable — the cached ASX futures curve "
                "doesn't yet quote this meeting's contract. Run a live refresh closer to the "
                "meeting, when the front-month contract is trading."
            )


def render_features(pred: dict[str, Any]) -> None:
    """Top global feature drivers."""
    st.markdown("#### Top model drivers")
    top = pred.get("top_features")
    if top:
        st.plotly_chart(feature_figure(top[:12]), use_container_width=True, theme=None)
        st.caption(
            "Global feature importances (how much each feature drives the model overall) — "
            "not a per-meeting attribution."
        )
    else:
        st.caption("This model exposes no native feature importances.")


def render_track_record(pred: dict[str, Any]) -> None:
    """Held-out backtest: model leaderboard + best-model confusion matrix."""
    st.markdown("### Track record — held-out backtest")
    st.caption(
        "Every model + baseline scored once on an untouched test window (39 meetings spanning a "
        "full hike→hold→cut cycle). Balanced accuracy is the honest metric (majority-class floor "
        "≈0.51). **Note the market-implied baseline tops the table — on a rate-tracked market the "
        "30-day futures price already impounds the signal, and no learned model beats it.**"
    )
    df = load_leaderboard()
    meta = load_best_model_meta()
    best = meta.get("best_model", "")
    left, right = st.columns([3, 2])
    with left:
        if df is not None:
            st.plotly_chart(leaderboard_figure(df), use_container_width=True, theme=None)
        else:
            st.caption("No held-out metrics found (run `python -m rba.validation.report`).")
    with right:
        conf = load_confusion(best)
        if conf is not None:
            st.markdown(f"**Confusion — `{best}`**")
            st.plotly_chart(confusion_figure(*conf), use_container_width=True, theme=None)
        else:
            st.caption("No confusion matrix found for the best model.")


def render_footer(pred: dict[str, Any]) -> None:
    """Data-vintage line + honest caveats."""
    v = pred["vintage"]
    staleness = v.get("feature_staleness_days", {}).get("median")
    st.divider()
    st.caption(
        f"Data vintage: master through **{v.get('master_last_meeting', '—')}** · "
        f"median feature staleness **{staleness}d** · {v.get('n_series_missing', 0)} series missing · "
        f"generated {pred.get('generated_at_utc', '—')}."
    )
    with st.expander("Method & honest caveats"):
        st.markdown(
            "- **Point-in-time, leakage-free**: the upcoming-meeting feature row uses only data "
            "published strictly before the meeting date (strict `publication_date < meeting_date`).\n"
            "- **Not investment advice** — a portfolio ML project. The market-implied baseline is a "
            "better predictor than the model on this rate-tracked market.\n"
            "- **Vintage bias**: features use current-vintage macro data, not the exact real-time "
            "values the Board saw.\n"
            "- **Small test window (N≈39)**: backtest rankings are indicative, not decisive.\n"
            "- **Model loads from the local `reports/best_model/` artifact** (MLflow is "
            "upstream-blocked on this stack)."
        )


def main() -> None:
    """Compose the dashboard."""
    st.title("RBA Cash Rate Prediction")
    meeting_iso, refresh_nonce = render_sidebar()

    try:
        pred = load_prediction(meeting_iso, refresh_nonce)
    except (FileNotFoundError, ValueError, LookupError) as exc:
        st.error(f"Could not produce a prediction: {exc}")
        st.stop()

    render_headline(pred)
    st.divider()
    render_market_row(pred)
    st.divider()
    render_features(pred)
    st.divider()
    render_track_record(pred)
    render_footer(pred)


main()
