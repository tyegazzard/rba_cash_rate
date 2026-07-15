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
The join core is a **pure, no-network function**: :func:`build_master` takes the
meeting frame and a mapping of ``{source_name: long_frame}`` and returns the
master frame; it never fetches. Keeping it pure makes the leakage tests trivial
(inject synthetic future rows; assert they never surface).

The orchestration around it (:func:`build_master_from_cache` + the
``python -m rba.data.align`` CLI) is **read-only over ``data/raw/``**: it rebuilds
every source's long frame from the cached raw snapshots by reusing the
:mod:`rba.data.inventory` catalog's per-source parse adapters and each source's
own ``_attach_publication_dates`` — it never calls any ``fetch()``, so running
the pipeline never rewrites ``_metadata.json`` provenance (Invariant #5). It then
writes ``data/processed/master.parquet``, hashes it (SHA-256), and writes a local
provenance record ``data/processed/master.meta.json`` (hash / shape / meeting
span / timestamp / git commit).

MLflow artifact logging (``--mlflow``) is wired but currently
**upstream-blocked**: this project's stack (pandas 3, pyarrow 24, and protobuf 7
pulled in by streamlit) is newer than any released mlflow supports (mlflow caps
``pandas<3`` / ``pyarrow<24`` / ``protobuf<6``). The call degrades gracefully to a
warning, and the JSON manifest is the interim tracking record until upstream
mlflow catches up. See CONTEXT.md.

Output schema
-------------
One row per meeting (≥ :data:`HISTORY_START`), columns:

- meeting metadata / target columns from the meeting frame
  (``meeting_date``, ``effective_date``, ``rate_change_bps``, ``new_rate_pct``,
  ``prior_rate_pct``, ``gap_days_since_last_meeting``, ``statement_url``,
  ``minutes_url``);
- for every numeric series ``<sid>`` across the joined sources: the as-of level
  ``<sid>`` (most recent value with ``publication_date < meeting_date``), its
  staleness ``<sid>_age_days`` (calendar days between that reading's publication
  and the meeting) — the explicit guard against "stale value looks fresh" — and a
  missingness flag ``<sid>_is_missing`` (1 where no observation was published
  before the meeting, else 0). The indicator is deterministic (``level.isna()``,
  not fit on any statistic → no leakage) and emitted for *every* level column so
  the schema is stable as coverage changes (Invariant #4). Actual value
  imputation is deferred to the per-fold model ``Pipeline`` (§5/§7).

Text sources (media releases / minutes / SoMP / speeches) do **not** fit the
long ``[observation_date, publication_date, series_id, value]`` schema and are
intentionally out of scope here; their point-in-time aggregation belongs in the
text feature layer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd

from rba.config import MLFLOW_TRACKING_URI, PROCESSED_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from rba.data import inventory
from rba.data.sources import rba_f11

# Start of the inflation-targeting era (CONTEXT.md "history start"). Module-level
# named constant, mirroring rba_f11's ``CUTOVER_EFFECTIVE_DATE`` convention.
HISTORY_START = pd.Timestamp("1993-01-01")

# Percentage-points → basis-points scale (0.25 % == 25 bps).
_BPS_PER_PCT = 100.0

# Long-frame schema every numeric source emits (the join contract).
_REQUIRED_LONG_COLUMNS = ("observation_date", "publication_date", "series_id", "value")

# Per-series companion-column suffixes emitted alongside each as-of level column.
# ``_age_days`` marks a column as a level (used to enumerate levels for the
# ``_is_missing`` pass); ``_is_missing`` is the deterministic missingness flag.
_AGE_SUFFIX = "_age_days"
_MISSING_SUFFIX = "_is_missing"

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
        .assign(meeting_date=lambda d: _to_ns(d["meeting_date"]))
        .sort_values("meeting_date")
        .reset_index(drop=True)
    )

    result = meetings.copy()
    for series_id, group in long_df.groupby("series_id", sort=True):
        right = (
            group[["publication_date", "value"]]
            .assign(publication_date=lambda d: _to_ns(d["publication_date"]))
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
        The meeting frame plus, per source series, the as-of level, its
        ``_age_days`` staleness, and its ``_is_missing`` indicator column, one
        row per meeting, sorted by ``meeting_date``.

    Raises
    ------
    ValueError
        If two sources contribute the same series-id column.

    Shapes
    ------
    Returns: (n_in + 3 * total_series,) columns, where ``n_in`` is the incoming
    meeting-frame column count (meeting metadata + any ``regime_*`` dummies) and
    each series contributes a level, an ``_age_days``, and an ``_is_missing``
    column.
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

    master = add_missing_indicators(master)
    logger.info(
        "Built master frame: {} meetings × {} columns from {} sources.",
        len(master),
        master.shape[1],
        len(source_frames),
    )
    return master


def add_missing_indicators(master: pd.DataFrame) -> pd.DataFrame:
    """Append a ``<series_id>_is_missing`` indicator for every as-of level column.

    Per Invariant #4 ("Add ``<col>_is_missing`` indicator alongside any
    imputation"), every point-in-time level column ``<sid>`` gets a companion
    ``<sid>_is_missing`` int flag: 1 where the as-of join found no observation
    published before the meeting (the level is ``NaN``), else 0.

    Missingness is a deterministic function of the already-aligned frame
    (``master[sid].isna()``); it is *not* fit on any statistic, so computing it
    here introduces no leakage and needs no per-fold refit. This is the §4
    preprocessing step — actual value imputation is deliberately deferred to the
    per-fold model ``Pipeline`` (§5/§7), where it can be fit train-only. The flag
    is emitted for *every* level column regardless of its current NaN count, so
    the master-frame schema stays stable as data coverage changes (a
    fully-present series simply gets an all-zero indicator).

    Level columns are identified by their ``<sid>_age_days`` companion (emitted
    per series by :func:`as_of_levels`); meeting-metadata and ``regime_*``
    columns have no such companion and get no indicator.

    Parameters
    ----------
    master
        A meeting-indexed frame carrying ``<sid>`` / ``<sid>_age_days`` column
        pairs (the output of the :func:`build_master` join).

    Returns
    -------
    pandas.DataFrame
        A copy of ``master`` with one added ``<sid>_is_missing`` int column per
        level series, in the same series order as the level columns.

    Shapes
    ------
    Returns: (n_meetings, n_in + n_series) where ``n_series`` is the number of
    ``<sid>_age_days`` companion columns present in ``master``.
    """
    out = master.copy()
    level_ids = [
        col[: -len(_AGE_SUFFIX)]
        for col in master.columns
        if col.endswith(_AGE_SUFFIX) and col[: -len(_AGE_SUFFIX)] in master.columns
    ]
    for sid in level_ids:
        out[f"{sid}{_MISSING_SUFFIX}"] = master[sid].isna().astype(int)
    logger.debug("Added {} _is_missing indicator columns.", len(level_ids))
    return out


def _to_ns(values: pd.Series) -> pd.Series:
    """Coerce a datetime column to ``datetime64[ns]``.

    Sources parse to mixed datetime resolutions (``[us]`` from some pandas /
    pyarrow paths, ``[ns]`` from others); ``merge_asof`` requires the two join
    keys share one resolution, so both sides are normalised to ``[ns]``.
    """
    return pd.to_datetime(values).astype("datetime64[ns]")


def _validate_long(long_df: pd.DataFrame) -> None:
    """Assert a source frame carries the required long-format columns."""
    missing = [c for c in _REQUIRED_LONG_COLUMNS if c not in long_df.columns]
    if missing:
        raise ValueError(
            f"Long source frame missing required column(s): {missing}. "
            f"Expected {list(_REQUIRED_LONG_COLUMNS)}."
        )


# -----------------------------------------------------------------------------
# Regime dummies.
# -----------------------------------------------------------------------------
# RBA Governor eras as (label, start_inclusive, end_exclusive). ``None`` bounds
# are open. Transition dates are the successor's first day in office. Feature
# boundaries are deliberately documented here so they can be tuned in one place.
_GOVERNOR_ERAS: tuple[tuple[str, pd.Timestamp | None, pd.Timestamp | None], ...] = (
    ("fraser", None, pd.Timestamp("1996-09-18")),
    ("macfarlane", pd.Timestamp("1996-09-18"), pd.Timestamp("2006-09-18")),
    ("stevens", pd.Timestamp("2006-09-18"), pd.Timestamp("2016-09-18")),
    ("lowe", pd.Timestamp("2016-09-18"), pd.Timestamp("2023-09-18")),
    ("bullock", pd.Timestamp("2023-09-18"), None),
)

# Crisis / policy-regime windows as [start, end] inclusive. Judgment calls —
# documented and centralised so the model's regime dummies are auditable.
_GFC_WINDOW = (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-12-31"))  # Lehman → recovery
_COVID_WINDOW = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"))  # pandemic emergency
# Explicit forward guidance: 3-year yield-curve control introduced 2020-03-19,
# abandoned 2021-11-02; the "no hike before 2024" calendar guidance sat inside it.
_FORWARD_GUIDANCE_WINDOW = (pd.Timestamp("2020-03-19"), pd.Timestamp("2021-11-02"))
# 11→8 meetings/year cadence change (post-2023 RBA Review), first new-cadence
# meeting 2024-02-05/06.
_CADENCE_CHANGE_DATE = pd.Timestamp("2024-02-01")


def add_regime_dummies(meeting_frame: pd.DataFrame) -> pd.DataFrame:
    """Append regime-indicator columns to a meeting-indexed frame.

    Adds, keyed off ``meeting_date``:

    - ``regime_gov_<name>`` (one per Governor era) — 1 during that governorship.
    - ``regime_gfc`` / ``regime_covid`` / ``regime_forward_guidance`` — 1 inside
      the respective window.
    - ``regime_post2024_cadence`` — 1 for meetings under the 8/year cadence.

    Parameters
    ----------
    meeting_frame
        Frame with a ``meeting_date`` (datetime64) column.

    Returns
    -------
    pandas.DataFrame
        A copy of ``meeting_frame`` with the added ``regime_*`` int columns.

    Shapes
    ------
    Returns: (n_meetings, n_in + len(_GOVERNOR_ERAS) + 4).
    """
    out = meeting_frame.copy()
    dates = pd.to_datetime(out["meeting_date"])

    for label, start, end in _GOVERNOR_ERAS:
        mask = pd.Series(True, index=out.index)
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates < end
        out[f"regime_gov_{label}"] = mask.astype(int)

    out["regime_gfc"] = _in_window(dates, _GFC_WINDOW)
    out["regime_covid"] = _in_window(dates, _COVID_WINDOW)
    out["regime_forward_guidance"] = _in_window(dates, _FORWARD_GUIDANCE_WINDOW)
    out["regime_post2024_cadence"] = (dates >= _CADENCE_CHANGE_DATE).astype(int)
    return out


def _in_window(dates: pd.Series, window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    """1 where ``dates`` fall in the inclusive ``[start, end]`` window, else 0."""
    start, end = window
    return ((dates >= start) & (dates <= end)).astype(int)


# -----------------------------------------------------------------------------
# Read-only source loading (reuses the inventory catalog; never calls fetch()).
# -----------------------------------------------------------------------------
MASTER_PARQUET_PATH = PROCESSED_DATA_DIR / "master.parquet"
MASTER_META_PATH = PROCESSED_DATA_DIR / "master.meta.json"


def load_source_frames(raw_root: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Rebuild every numeric source's long frame from cached raw — read-only.

    Reuses :data:`rba.data.inventory.SERIES_SOURCES` (the per-source parse
    adapters) to parse each cached snapshot and re-attach publication dates via
    the source's own ``_attach_publication_dates`` — **without** calling any
    ``fetch()``, so the pipeline never rewrites ``_metadata.json`` provenance
    (Invariant #5). One source failing (e.g. never refreshed) is logged and
    skipped, not fatal.

    Parameters
    ----------
    raw_root
        The ``data/raw`` root (parameterised for tests).

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``{source_name: long_frame}`` with the :data:`_REQUIRED_LONG_COLUMNS`
        schema, for every source that loaded successfully.
    """
    by_name: dict[str, list[inventory.SeriesSource]] = defaultdict(list)
    for record in inventory.SERIES_SOURCES:
        by_name[record.name].append(record)

    frames: dict[str, pd.DataFrame] = {}
    for name, records in by_name.items():
        try:
            frames[name] = _load_source_long(name, records, raw_root)
        except Exception as exc:  # noqa: BLE001 — isolation, like refresh.py
            logger.warning("Skipping source {!r} (read-only load failed): {}", name, exc)
    logger.info("Loaded {} of {} numeric sources from cache.", len(frames), len(by_name))
    return frames


def _load_source_long(
    name: str,
    records: list[inventory.SeriesSource],
    raw_root: Path,
) -> pd.DataFrame:
    """Rebuild one source's long frame (all series) from its cached snapshots."""
    entries = inventory.read_metadata(raw_root, name)
    if not entries:
        raise ValueError("no _metadata.json (source never refreshed)")

    module = records[0].module
    cache: dict[Path, bytes] = {}
    is_splice = any(r.resolve is inventory._resolve_live_csv for r in records)

    if is_splice:
        frames = _parse_splice_series(name, records, entries, raw_root, cache)
    else:
        frames = _parse_simple_series(name, records, entries, raw_root, cache)

    if not frames:
        raise ValueError("no series parsed from cached raw")

    long_df = pd.concat(frames, ignore_index=True)
    long_df = module._attach_publication_dates(long_df)
    return long_df[list(_REQUIRED_LONG_COLUMNS)]


def _parse_simple_series(
    name: str,
    records: list[inventory.SeriesSource],
    entries: list[dict[str, object]],
    raw_root: Path,
    cache: dict[Path, bytes],
) -> list[pd.DataFrame]:
    """Parse one snapshot per spec (per-series-file / single-file / per-table)."""
    frames: list[pd.DataFrame] = []
    for record in records:
        for spec in record.series:
            entry = record.resolve(spec, entries)
            if entry is None:
                logger.warning(
                    "{}: series {!r} has no metadata row; skipping.", name, spec.series_id
                )
                continue
            raw_bytes = inventory._read_snapshot(raw_root, name, entry, cache)
            if raw_bytes is None:
                continue
            frames.append(record.parse_one(raw_bytes, spec))
    return frames


def _parse_splice_series(
    name: str,
    records: list[inventory.SeriesSource],
    entries: list[dict[str, object]],
    raw_root: Path,
    cache: dict[Path, bytes],
) -> list[pd.DataFrame]:
    """Rebuild a spliced source (aud / bbsw) across all XLS + live-CSV snapshots.

    Mirrors each source's own splice rule: parse every series from each archive
    (``.xls``, ``missing_ok=True``) and the live ``.csv``, concatenate with the
    live CSV last, then ``drop_duplicates(keep='last')`` so the live vintage wins.
    """
    module = records[0].module
    # XLS archives first, live CSV last (so keep='last' prefers the live vintage).
    ordered = sorted(entries, key=lambda e: str(e.get("snapshot_filename", "")).endswith(".csv"))
    frames: list[pd.DataFrame] = []
    for record in records:
        for spec in record.series:
            per_spec: list[pd.DataFrame] = []
            for entry in ordered:
                filename = str(entry.get("snapshot_filename", ""))
                raw_bytes = inventory._read_snapshot(raw_root, name, entry, cache)
                if raw_bytes is None:
                    continue
                if filename.endswith(".xls"):
                    per_spec.append(module._parse_xls(raw_bytes, spec=spec, missing_ok=True))
                elif filename.endswith(".csv"):
                    per_spec.append(module._parse_csv(raw_bytes, spec=spec, missing_ok=True))
            if per_spec:
                combined = pd.concat(per_spec, ignore_index=True).drop_duplicates(
                    subset=["series_id", "observation_date"], keep="last"
                )
                frames.append(combined)
    return frames


def load_meeting_frame(raw_root: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load + build the meeting frame from cached F11 raw — read-only.

    Reads the single cached ``rba_f11`` HTML snapshot and reproduces
    ``rba_f11.fetch``'s parse + date-convention steps without writing anything.
    """
    entries = inventory.read_metadata(raw_root, rba_f11.SOURCE_NAME)
    if not entries:
        raise FileNotFoundError(
            "No cached rba_f11 snapshot; run `python -m rba.data.refresh --source rba_f11`."
        )
    raw_bytes = inventory._read_snapshot(raw_root, rba_f11.SOURCE_NAME, entries[0], {})
    if raw_bytes is None:
        raise FileNotFoundError("Cached rba_f11 snapshot file missing.")
    f11 = rba_f11._apply_date_convention(rba_f11._parse(raw_bytes))
    return build_meeting_frame(f11)


# -----------------------------------------------------------------------------
# Orchestration: cache → master frame → data/processed/master.parquet.
# -----------------------------------------------------------------------------
def build_master_from_cache(raw_root: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """End-to-end master frame from cached raw (read-only, no network).

    Loads the meeting frame + every numeric source from cache, adds regime
    dummies, and runs the point-in-time :func:`build_master` join.

    Returns
    -------
    pandas.DataFrame
        The meeting-indexed master frame.
    """
    meeting_frame = add_regime_dummies(load_meeting_frame(raw_root))
    source_frames = load_source_frames(raw_root)
    return build_master(meeting_frame, source_frames)


def write_master(master: pd.DataFrame, path: Path = MASTER_PARQUET_PATH) -> str:
    """Write the master frame to Parquet and return its SHA-256 (Invariant #5).

    Parameters
    ----------
    master
        The master frame.
    path
        Destination (default ``data/processed/master.parquet``).

    Returns
    -------
    str
        The hex SHA-256 digest of the written Parquet bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    logger.success(
        "Wrote {} ({} meetings × {} cols); sha256={}",
        path,
        len(master),
        master.shape[1],
        digest,
    )
    return digest


def write_master_manifest(
    master: pd.DataFrame,
    parquet_path: Path,
    digest: str,
    path: Path = MASTER_META_PATH,
) -> dict[str, object]:
    """Write a local provenance record for the master frame (Invariant #5).

    A lightweight, dependency-free stand-in for MLflow artifact logging (which
    is currently upstream-incompatible with this project's stack — see the
    module docstring / CONTEXT.md). Persists the frame's SHA-256, shape, meeting
    span, generation timestamp, and the current git commit so any downstream run
    can verify exactly which processed frame it consumed.

    Parameters
    ----------
    master
        The master frame.
    parquet_path
        Path the frame was written to (its name is recorded).
    digest
        The SHA-256 hex digest returned by :func:`write_master`.
    path
        Destination JSON (default ``data/processed/master.meta.json``).

    Returns
    -------
    dict
        The manifest that was written.
    """
    manifest: dict[str, object] = {
        "artifact": parquet_path.name,
        "sha256": digest,
        "n_meetings": int(len(master)),
        "n_columns": int(master.shape[1]),
        "first_meeting": str(master["meeting_date"].min().date()),
        "last_meeting": str(master["meeting_date"].max().date()),
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.success("Wrote provenance manifest {}", path)
    return manifest


def _git_commit(root: Path = PROJ_ROOT) -> str | None:
    """Resolve the current git commit SHA read-only, or ``None`` if unavailable.

    Reads ``.git`` directly (no subprocess): follows ``HEAD`` to its ref and
    falls back to ``packed-refs``. A detached HEAD returns its raw SHA.
    """
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head  # detached HEAD — raw SHA
        ref = head.split(" ", 1)[1].strip()
        loose = root / ".git" / ref
        if loose.exists():
            return loose.read_text(encoding="utf-8").strip()
        packed = root / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(ref):
                    return line.split(" ", 1)[0]
        return None
    except OSError:
        return None


def log_master_to_mlflow(master: pd.DataFrame, path: Path, digest: str) -> None:
    """Log the master frame as an MLflow artifact with its hash + shape.

    MLflow is imported lazily so importing :mod:`rba.data.align` (and the test
    suite) never pays for it. Failures are logged, not raised.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("data_preprocessing")
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        with mlflow.start_run(run_name=f"master_frame_{stamp}"):
            mlflow.log_param("n_meetings", len(master))
            mlflow.log_param("n_columns", master.shape[1])
            mlflow.log_param("first_meeting", str(master["meeting_date"].min().date()))
            mlflow.log_param("last_meeting", str(master["meeting_date"].max().date()))
            mlflow.set_tag("sha256", digest)
            mlflow.log_artifact(str(path))
        logger.success("Logged master frame to MLflow (sha256={}).", digest)
    except Exception as exc:  # noqa: BLE001 — MLflow logging is best-effort
        logger.warning("MLflow logging skipped: {}", exc)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.data.align`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.data.align",
        description="Build the point-in-time master frame from cached raw data.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Also log the master frame to MLflow (currently upstream-blocked by "
        "this project's stack — see the module docstring; degrades gracefully).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    master = build_master_from_cache()
    digest = write_master(master)
    write_master_manifest(master, MASTER_PARQUET_PATH, digest)
    if args.mlflow:
        log_master_to_mlflow(master, MASTER_PARQUET_PATH, digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
