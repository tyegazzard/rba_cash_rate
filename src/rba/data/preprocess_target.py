"""Preprocess the raw RBA cash rate meeting history into a clean event-level frame.

Consumes the output of :func:`rba.data.sources.rba_f11.fetch` and produces the
columns that downstream target builders (driven by
``src/rba/config/targets.yaml``) expect: ``rate_change_bps`` and
``new_rate_pct``. Also carries the date discipline (``observation_date``,
``publication_date``) and useful auxiliaries (``prior_rate_pct``, related-doc
URLs) through unchanged.

Scope notes
-----------
- Pre-history-start rows are dropped. CHECKLIST locks the modelling history
  at ``1993-01`` (the start of the explicit inflation-targeting era), but the
  start date is overridable for tests and ad-hoc analyses.
- Pre-1990 rows in the RBA source carry *range* targets (e.g.
  ``"-1.00 to -1.50"`` / ``"15.00 to 15.50"``) that don't parse as floats.
  These are dropped with a warning. All rows from 1993-01 onward in the
  current vintage parse cleanly as single numerics.
"""

from __future__ import annotations

from loguru import logger
import pandas as pd

DEFAULT_HISTORY_START = "1993-01-01"

# Columns produced by ``rba_f11.fetch()`` that this module expects as input.
_REQUIRED_INPUT_COLUMNS = (
    "observation_date",
    "publication_date",
    "change_raw",
    "new_cash_rate_raw",
    "statement_url",
    "minutes_url",
)

_OUTPUT_COLUMNS = (
    "observation_date",
    "publication_date",
    "rate_change_bps",
    "new_rate_pct",
    "prior_rate_pct",
    "statement_url",
    "minutes_url",
)


def build_decisions(
    raw: pd.DataFrame,
    *,
    history_start: str = DEFAULT_HISTORY_START,
) -> pd.DataFrame:
    """Clean and enrich the raw meeting history into the canonical target frame.

    Parameters
    ----------
    raw
        Output of :func:`rba.data.sources.rba_f11.fetch`. Must contain the
        columns listed in ``_REQUIRED_INPUT_COLUMNS``.
    history_start
        ISO date (inclusive lower bound, by ``observation_date``). Rows before
        this date are dropped. Defaults to ``"1993-01-01"`` per the project
        scope decision (inflation-targeting era).

    Returns
    -------
    pandas.DataFrame
        One row per RBA Board meeting at or after ``history_start``, sorted
        ascending by ``observation_date``. Columns:

        - ``observation_date`` (datetime64[ns]) — effective date as published.
        - ``publication_date`` (datetime64[ns]) — announcement date.
        - ``rate_change_bps`` (int64) — basis-point change at this meeting;
          ``0`` for holds.
        - ``new_rate_pct`` (float64) — new cash rate target, percentage points.
        - ``prior_rate_pct`` (float64) — cash rate target prior to this
          meeting (``new_rate_pct - change_pp``).
        - ``statement_url`` (object | None) — relative URL to the media release.
        - ``minutes_url`` (object | None) — relative URL to the meeting minutes.

    Shapes
    ------
    Returns: (n_meetings_in_window, 7).
    """
    missing = set(_REQUIRED_INPUT_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"raw input is missing required columns: {sorted(missing)}")

    df = raw.copy()
    start_ts = pd.Timestamp(history_start)
    pre_window = df["observation_date"] < start_ts
    if pre_window.any():
        logger.info(
            "Dropping {} rows with observation_date < {} (pre-history window)",
            int(pre_window.sum()),
            history_start,
        )
    df = df.loc[~pre_window].copy()

    change_pp = pd.to_numeric(df["change_raw"], errors="coerce")
    new_rate_pp = pd.to_numeric(df["new_cash_rate_raw"], errors="coerce")
    unparseable = change_pp.isna() | new_rate_pp.isna()
    if unparseable.any():
        bad = df.loc[unparseable, ["observation_date", "change_raw", "new_cash_rate_raw"]]
        logger.warning(
            "Dropping {} rows with non-numeric change or rate (e.g. pre-1990 range targets):\n{}",
            int(unparseable.sum()),
            bad.to_string(index=False),
        )
        df = df.loc[~unparseable].copy()
        change_pp = change_pp.loc[~unparseable]
        new_rate_pp = new_rate_pp.loc[~unparseable]

    df["rate_change_bps"] = (change_pp * 100).round().astype("int64")
    df["new_rate_pct"] = new_rate_pp.astype("float64")
    df["prior_rate_pct"] = (new_rate_pp - change_pp).astype("float64")

    out = df[list(_OUTPUT_COLUMNS)].sort_values("observation_date").reset_index(drop=True)
    _validate(out)
    return out


def _validate(df: pd.DataFrame) -> None:
    """Sanity-check the output frame; raise on any invariant violation."""
    if df["observation_date"].isna().any():
        raise ValueError("Output contains NaT in observation_date.")
    if df["publication_date"].isna().any():
        raise ValueError("Output contains NaT in publication_date.")
    if (df["publication_date"] > df["observation_date"]).any():
        raise ValueError("publication_date must never exceed observation_date.")
    if df["new_rate_pct"].isna().any():
        raise ValueError("Output contains NaN in new_rate_pct.")
    if df["rate_change_bps"].isna().any():
        raise ValueError("Output contains NaN in rate_change_bps.")
    # Cross-check: new = prior + change (within float tolerance)
    recomputed = df["prior_rate_pct"] + df["rate_change_bps"] / 100.0
    if not (recomputed - df["new_rate_pct"]).abs().lt(1e-6).all():
        raise ValueError("new_rate_pct != prior_rate_pct + change; arithmetic drift.")
