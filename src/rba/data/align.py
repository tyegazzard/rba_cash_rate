"""Point-in-time alignment of every source onto the RBA meeting frame.

This is the keystone of ``§4 Data preprocessing``: it turns the heterogeneous
ingested sources into a single meeting-indexed master frame on which all
features and models are built — without ever leaking the future.

The meeting frame
-----------------
The prediction unit is one **Board meeting**. ``rba_f11.fetch()`` returns, per
meeting, an ``observation_date`` (the *effective* date the new rate applies) and
a ``publication_date`` (the *announcement* date — when the decision became
public). The date we predict at — and align features to — is the **announcement
date**, so throughout this module ``meeting_date ≡ rba_f11.publication_date``.

Point-in-time rule (Invariant #1)
---------------------------------
A feature observation is usable at meeting ``t`` only if it was published
**strictly before** ``t``:

    feature.publication_date < meeting_date            (strict ``<``)

This is deliberately stricter than the checklist's loose "``≤``" wording: a value
published *on* the meeting day (e.g. an EOD market print, or the post-decision
media release) is treated as **not yet known** at the 2:30pm AEST decision — the
same precedent CONTEXT.md sets for the CBOE VIX same-day close. The strict rule
is enforced by :func:`as_of_levels` via ``merge_asof(..., allow_exact_matches=
False)`` and guarded by ``tests/data/test_no_leakage.py``.

Design
------
``align.py`` is a **pure, no-network function library**. :func:`build_master`
takes the meeting frame and a mapping of ``{source_name: long_frame}`` and
returns the master frame; it never fetches. A thin orchestrator (a later
checklist item) wires the real ``fetch(force_download=False)`` calls and writes
``data/processed/master.parquet``. Keeping the join pure makes the leakage tests
trivial (inject synthetic future rows; assert they never surface).

Output schema
-------------
One row per meeting (≥ :data:`HISTORY_START`), columns:

- meeting metadata / target columns from the meeting frame
  (``meeting_date``, ``effective_date``, ``rate_change_bps``, ``new_rate_pct``,
  ``prior_rate_pct``, ``gap_days_since_last_meeting``, ``statement_url``,
  ``minutes_url``);
- for every numeric series ``<sid>`` across the joined sources: the as-of level
  ``<sid>`` (most recent value with ``publication_date < meeting_date``) and its
  staleness ``<sid>_age_days`` (calendar days between that reading's publication
  and the meeting) — the explicit guard against "stale value looks fresh".

Text sources (media releases / minutes / SoMP / speeches) do **not** fit the
long ``[observation_date, publication_date, series_id, value]`` schema and are
intentionally out of scope here; their point-in-time aggregation belongs in the
text feature layer.
"""

from __future__ import annotations

from collections.abc import Mapping

from loguru import logger
import numpy as np
import pandas as pd

# Start of the inflation-targeting era (CONTEXT.md "history start"). Module-level
# named constant, mirroring rba_f11's ``CUTOVER_EFFECTIVE_DATE`` convention.
HISTORY_START = pd.Timestamp("1993-01-01")

# Percentage-points → basis-points scale (0.25 % == 25 bps).
_BPS_PER_PCT = 100.0

# Long-frame schema every numeric source emits (the join contract).
_REQUIRED_LONG_COLUMNS = ("observation_date", "publication_date", "series_id", "value")

# Meeting-frame columns carried through to the master frame, in order.
_MEETING_COLUMNS = (
    "meeting_date",
    "effective_date",
    "rate_change_bps",
    "new_rate_pct",
    "prior_rate_pct",
    "gap_days_since_last_meeting",
    "statement_url",
    "minutes_url",
)


# -----------------------------------------------------------------------------
# Meeting frame.
# -----------------------------------------------------------------------------
def build_meeting_frame(
    f11: pd.DataFrame,
    *,
    history_start: pd.Timestamp = HISTORY_START,
) -> pd.DataFrame:
    """Derive the meeting-indexed target frame from ``rba_f11.fetch()`` output.

    Parameters
    ----------
    f11
        The frame returned by :func:`rba.data.sources.rba_f11.fetch` — columns
        ``observation_date`` (effective), ``publication_date`` (announcement),
        ``change_raw``, ``new_cash_rate_raw``, ``statement_url``,
        ``minutes_url``.
    history_start
        Earliest meeting to keep (inclusive). Defaults to
        :data:`HISTORY_START` (1993-01-01, the inflation-targeting era).

    Returns
    -------
    pandas.DataFrame
        One row per meeting on/after ``history_start``, with the columns listed
        in :data:`_MEETING_COLUMNS`. ``prior_rate_pct`` and
        ``gap_days_since_last_meeting`` are computed on the full history *before*
        filtering, so the first in-window meeting still gets its true predecessor
        rather than a NaN.

    Shapes
    ------
    Returns: (n_meetings, 8) where ``n_meetings`` ≈ 350 for the 1993→present
    window.
    """
    df = f11.copy()
    df["meeting_date"] = pd.to_datetime(df["publication_date"])
    df["effective_date"] = pd.to_datetime(df["observation_date"])
    df["rate_change_bps"] = df["change_raw"].map(_parse_change_bps)
    df["new_rate_pct"] = df["new_cash_rate_raw"].map(_parse_rate_pct)

    df = df.sort_values("meeting_date").reset_index(drop=True)
    # Derived-from-predecessor columns: compute on full history, then window.
    df["prior_rate_pct"] = df["new_rate_pct"].shift(1)
    df["gap_days_since_last_meeting"] = df["meeting_date"].diff().dt.days.astype("Int64")

    for optional in ("statement_url", "minutes_url"):
        if optional not in df.columns:
            df[optional] = pd.NA

    windowed = df.loc[df["meeting_date"] >= history_start, list(_MEETING_COLUMNS)]
    return windowed.reset_index(drop=True)


def _parse_change_bps(raw: object) -> float:
    """Parse an F11 ``change_raw`` cell to signed basis points (``NaN`` on range)."""
    value = _parse_signed_float(raw)
    return value * _BPS_PER_PCT if not np.isnan(value) else np.nan


def _parse_rate_pct(raw: object) -> float:
    """Parse an F11 ``new_cash_rate_raw`` cell to a percent level (``NaN`` on range)."""
    return _parse_signed_float(raw)


def _parse_signed_float(raw: object) -> float:
    """Coerce a verbatim HTML numeric cell to float; ``NaN`` for ranges / junk.

    Range strings (``"-1.00 to -1.50"``) appear only in pre-1990s rows and map to
    ``NaN`` (those rows are dropped by the history window anyway). Handles a
    leading ``+`` and a Unicode minus sign.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return np.nan
    text = str(raw).strip().replace("−", "-").replace("+", "")
    if "to" in text:  # range string, e.g. "-1.00 to -1.50"
        return np.nan
    try:
        return float(text)
    except ValueError:
        logger.warning("Unparseable F11 numeric cell {!r}; mapping to NaN.", raw)
        return np.nan


# -----------------------------------------------------------------------------
# Point-in-time as-of join.
# -----------------------------------------------------------------------------
def as_of_levels(meeting_frame: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """As-of join one long source frame onto the meeting frame (strict ``<``).

    For every ``series_id`` in ``long_df`` and every meeting ``t``, take the most
    recent observation with ``publication_date < t`` (strictly before — no
    same-day leakage) and emit two columns: the level ``<series_id>`` and the
    staleness ``<series_id>_age_days`` (``t − publication_date`` in days).

    Parameters
    ----------
    meeting_frame
        Frame with a ``meeting_date`` (datetime64) column (see
        :func:`build_meeting_frame`).
    long_df
        Long source frame with :data:`_REQUIRED_LONG_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Keyed by ``meeting_date`` (one row per meeting), with one level + one
        age column per series. Meetings before a series' first publication get
        ``NaN`` for both.

    Shapes
    ------
    Returns: (n_meetings, 1 + 2 * n_series).
    """
    _validate_long(long_df)
    meetings = (
        meeting_frame[["meeting_date"]]
        .assign(meeting_date=lambda d: pd.to_datetime(d["meeting_date"]))
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )

    result = meetings.copy()
    for series_id, group in long_df.groupby("series_id", sort=True):
        right = (
            group[["publication_date", "value"]]
            .assign(publication_date=lambda d: pd.to_datetime(d["publication_date"]))
            .dropna(subset=["publication_date"])
            .sort_values("publication_date")
            .reset_index(drop=True)
        )
        merged = pd.merge_asof(
            meetings,
            right,
            left_on="meeting_date",
            right_on="publication_date",
            direction="backward",
            allow_exact_matches=False,  # strict publication_date < meeting_date
        )
        age_days = (merged["meeting_date"] - merged["publication_date"]).dt.days
        result[str(series_id)] = merged["value"].to_numpy()
        result[f"{series_id}_age_days"] = age_days.astype("Int64").to_numpy()

    return result


def build_master(
    meeting_frame: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Assemble the meeting-indexed master frame from all numeric sources.

    Parameters
    ----------
    meeting_frame
        The meeting frame from :func:`build_meeting_frame`.
    source_frames
        ``{source_name: long_frame}`` for every numeric source to align. Each
        long frame must carry :data:`_REQUIRED_LONG_COLUMNS`. Series ids must be
        globally unique across sources (a collision raises).

    Returns
    -------
    pandas.DataFrame
        The meeting frame plus every source's as-of level + age columns, one row
        per meeting, sorted by ``meeting_date``.

    Raises
    ------
    ValueError
        If two sources contribute the same series-id column.

    Shapes
    ------
    Returns: (n_meetings, len(_MEETING_COLUMNS) + 2 * total_series).
    """
    master = (
        meeting_frame.assign(meeting_date=lambda d: pd.to_datetime(d["meeting_date"]))
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )
    seen: set[str] = set(master.columns)

    for name, long_df in source_frames.items():
        levels = as_of_levels(meeting_frame, long_df)
        new_cols = [c for c in levels.columns if c != "meeting_date"]
        collisions = seen.intersection(new_cols)
        if collisions:
            raise ValueError(
                f"Source {name!r} contributes columns already present: "
                f"{sorted(collisions)}. Series ids must be globally unique."
            )
        master = master.merge(levels, on="meeting_date", how="left")
        seen.update(new_cols)
        logger.debug("Aligned {!r}: +{} columns.", name, len(new_cols))

    logger.info(
        "Built master frame: {} meetings × {} columns from {} sources.",
        len(master),
        master.shape[1],
        len(source_frames),
    )
    return master


def _validate_long(long_df: pd.DataFrame) -> None:
    """Assert a source frame carries the required long-format columns."""
    missing = [c for c in _REQUIRED_LONG_COLUMNS if c not in long_df.columns]
    if missing:
        raise ValueError(
            f"Long source frame missing required column(s): {missing}. "
            f"Expected {list(_REQUIRED_LONG_COLUMNS)}."
        )
