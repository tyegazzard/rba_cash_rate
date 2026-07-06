"""Feature-importance report — mutual information + correlation with the target.

The final ``§5 Feature engineering → Feature pipeline`` item: a **descriptive**
(in-sample) triage of every engineered feature against the RBA decision target, so
a modeller can see at a glance which features carry signal before any model is
fit. It is *not* a model-evaluation metric — these are whole-sample statistics for
feature triage, deliberately distinct from the walk-forward performance the §7/§8
validation harness will produce (Invariant #2/#3 govern *modelling*; a descriptive
correlation table is exempt, and this is stated in the report itself).

Two lenses, matched to the project's locked target formulations (``targets.yaml``)
---------------------------------------------------------------------------------
- **Correlation** (Pearson + Spearman) against the continuous Δrate target
  ``rate_change_bps`` (the ``delta_regression`` target, ``identity`` encoding).
- **Mutual information** against the 3-class **direction** target (the
  ``three_class`` target: ``sign_of_change`` → cut / hold / hike = ``-1 / 0 / +1``),
  via :func:`sklearn.feature_selection.mutual_info_classif`. MI captures non-linear
  and non-monotone dependence that correlation misses.

Target derivation mirrors ``targets.yaml`` inline (the full ``build_target`` helper
lands with the §6 model layer); this module needs only the two target vectors and
does not depend on a not-yet-built target builder.

What counts as a feature
------------------------
Every numeric column **except** the meeting metadata (``meeting_date`` /
``effective_date`` / URL columns — non-numeric, auto-excluded) and the two
**contemporaneous outcome** columns ``rate_change_bps`` / ``new_rate_pct`` (the
targets themselves — including them would be trivial self-leakage). ``prior_rate_pct``
(the standing rate going *into* the meeting) and the ``*_lag_*`` past-decision
features are legitimate and retained.

Reproducibility (Invariant #5)
------------------------------
MI's k-NN estimator is seeded with :data:`rba.config.RANDOM_SEED`; correlations
and the stable sort are deterministic; the Markdown report carries **no wall-clock
timestamp** (only the feature-version hash + git commit), so re-running on the same
features + code yields a byte-identical report. No network.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from rba.config import RANDOM_SEED, REPORTS_DIR, load_features_config
from rba.data import align
from rba.features import versioning
from rba.features.build import FEATURES_PARQUET_PATH, build_features_from_cache

_MEETING_KEY = "meeting_date"

# The continuous Δ-rate target (targets.yaml delta_regression) — correlation lens.
_CHANGE_TARGET = "rate_change_bps"
# The post-meeting level outcome (targets.yaml level_regression) — never a feature.
_LEVEL_OUTCOME = "new_rate_pct"

# Meeting-metadata columns that are not features (mostly non-numeric, but named
# explicitly for robustness) plus the two contemporaneous outcome columns.
_NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {_MEETING_KEY, "effective_date", "statement_url", "minutes_url"}
)
_EXCLUDED_AS_FEATURE: frozenset[str] = _NON_FEATURE_COLUMNS | {_CHANGE_TARGET, _LEVEL_OUTCOME}

# A feature is treated as discrete for MI if it is integer-valued with at most this
# many distinct values (regime dummies, ``_is_missing`` flags, small counts).
_DISCRETE_MAX_CARDINALITY = 10

# Minimum non-NaN observations before MI is estimated (k-NN MI is unreliable on a
# tiny sample); correlation only needs ≥ 2 with variance.
DEFAULT_MIN_OBS_FOR_MI = 30

# Rows shown in the Markdown top-list (the full table goes to CSV).
DEFAULT_TOP_N = 30

# Output artifacts (reports/ is checked in — a diffable deliverable).
FEATURE_IMPORTANCE_MD_PATH = REPORTS_DIR / "feature_importance.md"
FEATURE_IMPORTANCE_CSV_PATH = REPORTS_DIR / "feature_importance.csv"

_TABLE_COLUMNS = ("feature", "group", "n_obs", "pearson_r", "spearman_r", "mutual_info")


# -----------------------------------------------------------------------------
# Targets + feature selection.
# -----------------------------------------------------------------------------
def derive_targets(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Derive the two report targets from the feature frame (mirrors targets.yaml).

    Returns
    -------
    dict[str, pandas.Series]
        ``change_bps`` — the continuous Δrate (``rate_change_bps`` as float);
        ``direction`` — the 3-class ``sign_of_change`` label as nullable ``Int64``
        (``-1`` cut / ``0`` hold / ``+1`` hike), ``<NA>`` where Δrate is missing.
    """
    change = pd.to_numeric(features[_CHANGE_TARGET], errors="coerce").astype("float64")
    sign = np.sign(change.to_numpy())
    direction = pd.array(
        [pd.NA if math.isnan(s) else int(s) for s in sign], dtype="Int64"
    )
    return {
        "change_bps": change,
        "direction": pd.Series(direction, index=features.index, name="direction"),
    }


def candidate_features(features: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding metadata + the contemporaneous outcomes.

    Non-numeric columns (dates / URLs) and the two target columns
    (``rate_change_bps`` / ``new_rate_pct``) are dropped; everything else numeric
    — levels, companions, regime dummies, lags, rolling, changes, target lags,
    ``prior_rate_pct`` — is a candidate.
    """
    return [
        col
        for col in features.columns
        if col not in _EXCLUDED_AS_FEATURE and pd.api.types.is_numeric_dtype(features[col])
    ]


def _feature_group(column: str) -> str:
    """Coarse group label for a feature column (for report readability)."""
    if column.endswith("_is_missing"):
        return "missing_flag"
    if column.endswith("_age_days"):
        return "staleness"
    if column.startswith("regime_"):
        return "regime"
    if column.endswith("_surprise"):
        return "surprise"
    if "_roll_" in column:
        return "rolling"
    if "_lag_" in column:
        return "lag"
    if "_diff_" in column or "_pct_" in column or column.endswith("_yoy_pct"):
        return "change"
    return "level"


def _to_float(series: pd.Series) -> np.ndarray:
    """Coerce a numeric (possibly nullable ``Int64`` / ``bool``) column to float64."""
    return np.asarray(series.to_numpy(dtype="float64", na_value=np.nan), dtype="float64")


def _is_discrete(values: np.ndarray) -> bool:
    """Whether a feature is discrete for MI: integer-valued + low cardinality."""
    if values.size == 0:
        return False
    if not np.all(np.mod(values, 1.0) == 0.0):
        return False
    return np.unique(values).size <= _DISCRETE_MAX_CARDINALITY


# -----------------------------------------------------------------------------
# Pure core.
# -----------------------------------------------------------------------------
def feature_importance(
    features: pd.DataFrame,
    *,
    min_obs_for_mi: int = DEFAULT_MIN_OBS_FOR_MI,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute the descriptive importance table (pure, no network, deterministic).

    For every :func:`candidate_features` column, on the rows where the target is
    defined, computes Pearson + Spearman correlation against the continuous Δrate
    target and mutual information against the 3-class direction target. A feature
    with fewer than two distinct non-NaN values gets ``NaN`` correlations and
    ``mutual_info = 0`` (constant → no information); a feature with fewer than
    ``min_obs_for_mi`` non-NaN observations gets ``NaN`` mutual information.

    Parameters
    ----------
    features
        The assembled feature frame (:func:`rba.features.build.build_features`
        output) — must carry ``meeting_date`` and ``rate_change_bps``.
    min_obs_for_mi
        Minimum non-NaN observations before MI is estimated.
    random_state
        Seed for MI's k-NN estimator (:data:`rba.config.RANDOM_SEED`).

    Returns
    -------
    pandas.DataFrame
        One row per candidate feature — columns ``feature``, ``group``,
        ``n_obs``, ``pearson_r``, ``spearman_r``, ``mutual_info`` — sorted by
        ``mutual_info`` desc, then ``|pearson_r|`` desc, then ``feature``.

    Shapes
    ------
    Returns: (n_candidate_features, 6).
    """
    targets = derive_targets(features)
    change = targets["change_bps"]
    direction = targets["direction"]
    valid = (change.notna() & direction.notna()).to_numpy()
    n_valid = int(valid.sum())

    columns = candidate_features(features)
    if n_valid == 0 or not columns:
        logger.warning("Feature importance: no target-defined rows or no features; empty table.")
        return pd.DataFrame(columns=list(_TABLE_COLUMNS))

    y_change = change.to_numpy(dtype="float64")[valid]
    y_direction = direction.to_numpy(dtype="int64", na_value=0)[valid]

    records: list[dict[str, Any]] = []
    for col in columns:
        x = _to_float(features[col])[valid]
        mask = ~np.isnan(x)
        n_obs = int(mask.sum())
        x_obs = x[mask]
        distinct = int(np.unique(x_obs).size) if n_obs else 0

        pearson = spearman = math.nan
        mutual = math.nan
        if n_obs >= 2 and distinct >= 2:
            s_x = pd.Series(x_obs)
            s_y = pd.Series(y_change[mask])
            pearson = float(s_x.corr(s_y))
            spearman = float(s_x.corr(s_y, method="spearman"))
        if n_obs >= min_obs_for_mi:
            if distinct < 2:
                mutual = 0.0  # constant feature carries no information
            else:
                mutual = float(
                    mutual_info_classif(
                        x_obs.reshape(-1, 1),
                        y_direction[mask],
                        discrete_features=[_is_discrete(x_obs)],
                        random_state=random_state,
                    )[0]
                )

        records.append(
            {
                "feature": col,
                "group": _feature_group(col),
                "n_obs": n_obs,
                "pearson_r": pearson,
                "spearman_r": spearman,
                "mutual_info": mutual,
            }
        )

    table = pd.DataFrame.from_records(records, columns=list(_TABLE_COLUMNS))
    table["_abs_pearson"] = table["pearson_r"].abs()
    table = (
        table.sort_values(
            by=["mutual_info", "_abs_pearson", "feature"],
            ascending=[False, False, True],
            na_position="last",
        )
        .drop(columns="_abs_pearson")
        .reset_index(drop=True)
    )
    logger.info(
        "Feature importance: scored {} features over {} target-defined meetings.",
        len(table),
        n_valid,
    )
    return table


# -----------------------------------------------------------------------------
# Orchestration: load features → report → reports/feature_importance.{md,csv}.
# -----------------------------------------------------------------------------
def load_features(path: Path = FEATURES_PARQUET_PATH) -> pd.DataFrame:
    """Load the cached feature frame, or rebuild it from cache as a fallback.

    Reads ``data/processed/features.parquet`` when present; otherwise warns and
    rebuilds via :func:`rba.features.build.build_features_from_cache` (read-only,
    no network).
    """
    if path.exists():
        features = pd.read_parquet(path)
        logger.info("Loaded feature frame from {} ({} meetings × {} cols).", path, len(features), features.shape[1])
        return features
    logger.warning("Feature parquet absent at {}; rebuilding from cache.", path)
    return build_features_from_cache()


def _report_meta(table: pd.DataFrame, n_meetings: int) -> dict[str, object]:
    """Provenance header for the report (feature version + git commit; no timestamp)."""
    try:
        version = versioning.compute_feature_version(load_features_config())
    except OSError as exc:  # config unreadable — degrade, don't crash the report
        logger.warning("Feature importance: could not compute feature version: {}", exc)
        version = "unavailable"
    return {
        "n_features": int(len(table)),
        "n_meetings": int(n_meetings),
        "feature_version_hash": version,
        "git_commit": align._git_commit() or "unknown",
    }


def render_markdown(
    table: pd.DataFrame,
    meta: Mapping[str, object],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> str:
    """Render the Markdown feature-importance report (top-``top_n`` table)."""
    lines = [
        "# Feature importance report",
        "",
        "Descriptive, **in-sample** importance of each engineered feature against the",
        "RBA cash-rate decision target. These are whole-sample statistics for feature",
        "triage — **not** a model-evaluation metric and not walk-forward performance",
        "(see CONTEXT.md Invariants #2/#3, which govern *modelling*).",
        "",
        "## Targets",
        "",
        "- **Correlation** (Pearson / Spearman) vs `rate_change_bps` — the continuous",
        "  Δrate target (`targets.yaml` `delta_regression`).",
        "- **Mutual information** vs the 3-class direction target"
        " (`targets.yaml` `three_class`: sign of Δ → cut `-1` / hold `0` / hike `+1`),",
        "  which captures non-linear dependence correlation misses.",
        "",
        "## Provenance",
        "",
        f"- Features scored: **{meta['n_features']}**",
        f"- Target-defined meetings: **{meta['n_meetings']}**",
        f"- Feature version: `{meta['feature_version_hash']}`",
        f"- Git commit: `{meta['git_commit']}`",
        f"- Full table: `{FEATURE_IMPORTANCE_CSV_PATH.name}` (all features)",
        "",
        f"## Top {top_n} features by mutual information",
        "",
        "| rank | feature | group | n_obs | pearson_r | spearman_r | mutual_info |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(table.head(top_n).iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['group']} | {int(row['n_obs'])} "
            f"| {_fmt(row['pearson_r'])} | {_fmt(row['spearman_r'])} | {_fmt(row['mutual_info'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """Format a float cell to 4 dp; ``NaN`` / ``None`` → an em dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.4f}"


def write_importance_report(
    table: pd.DataFrame,
    meta: Mapping[str, object],
    *,
    md_path: Path = FEATURE_IMPORTANCE_MD_PATH,
    csv_path: Path = FEATURE_IMPORTANCE_CSV_PATH,
    top_n: int = DEFAULT_TOP_N,
) -> None:
    """Write the full table to CSV and the top-``top_n`` Markdown report."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(table, meta, top_n=top_n), encoding="utf-8")
    logger.success(
        "Wrote feature-importance report: {} (top {}) + {} ({} features).",
        md_path,
        top_n,
        csv_path,
        len(table),
    )


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.features.importance`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.features.importance",
        description="Generate the feature-importance report (mutual info + correlation).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Rows shown in the Markdown top-list (default {DEFAULT_TOP_N}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    features = load_features()
    table = feature_importance(features)
    n_meetings = int((derive_targets(features)["direction"].notna()).sum())
    meta = _report_meta(table, n_meetings)
    write_importance_report(table, meta, top_n=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
