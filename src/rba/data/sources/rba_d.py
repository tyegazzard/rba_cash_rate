"""RBA Financial Aggregates — D1 (growth) + D2 (lending and credit, $bn levels).

Pulls headline lending / credit aggregate series the RBA tracks from two
RBA-published statistical tables:

================================================  ========  =============  ============================
Series                                            Table     RBA ID         Coverage
------------------------------------------------  --------  -------------  ----------------------------
Credit; Total — legacy (SA, $ bn)                 D2        ``DLCACS``     1976-09 → 2019-06
Credit; Total inc. select fin (SA, $ bn)          D2        ``DLCACSFS``   2019-07 → present
Credit; Business — legacy (SA, $ bn)              D2        ``DLCACBS``    1976-09 → 2019-06
Credit; Business inc. select fin (SA, $ bn)       D2        ``DLCACSFBS``  2019-07 → present
Credit; Owner-occupier housing (SA, $ bn)         D2        ``DLCACOHS``   1990-01 → present
Credit; Investor housing (SA, $ bn)               D2        ``DLCACIHS``   1990-01 → present
Credit; Other personal (SA, $ bn)                 D2        ``DLCACOPS``   1976-09 → present
Credit; Total; 12-m growth — legacy (SA, %)       D1        ``DGFAC12``    1977-09 → 2019-06
Credit; Total inc. fin; 12-m growth (SA, %)       D1        ``DGFACNW12``  2020-07 → present
Broad Money; 12-month growth (SA, %)              D1        ``DGFABM12``   1977-08 → present
================================================  ========  =============  ============================

2019 series-break — why "legacy" and "inc. fin" pairs?
------------------------------------------------------
The RBA changed credit-aggregate methodology effective 2019-07. The
old ``DLCACS`` / ``DLCACBS`` / ``DGFAC12`` definitions stopped being
published after the 2019-06 reference month and were replaced by new
``DLCACSFS`` / ``DLCACSFBS`` / ``DGFACNW12`` series that include
lending by select financial businesses. We expose **both pre- and
post-break series under distinct logical names** rather than silently
stitching, so any join is explicit downstream and the methodology
change is visible to the feature layer. The 12-month growth series
``DGFACNW12`` starts at 2020-07 (its first 12 months of underlying
level data ends at 2020-06; July 2020 is the first month with a
full-year base). The d2-series-breaks.csv file at
``https://www.rba.gov.au/statistics/tables/csv/d2-series-breaks.csv``
documents the RBA's break-adjusted growth-rate methodology.

Why D1 *and* D2?
----------------
D2 publishes credit and lending **levels** ($ billion). The feature layer
derives growth rates from the level series — same convention used by
``abs_cpi``, ``abs_wpi``, and ``abs_gdp``. We additionally pull
**growth-rate** series from D1 that are not derivable from D2:

- ``DGFABM12`` — Broad Money 12-month growth. D2 does not contain a
  broad-money level series, so this is the only place to get it.
- ``DGFAC12`` / ``DGFACNW12`` — Total Credit 12-month growth computed by
  the RBA with their official seasonal-adjustment and break-corrected
  methodology. These differ from a naive ``pct_change(12)`` of the
  level series because the RBA's growth-rate computation adjusts for
  series breaks and applies SA at the growth-rate level. Useful as both
  features in their own right and as validation cross-checks.

Source format
-------------
Both tables publish identical CSV layouts at
``https://www.rba.gov.au/statistics/tables/csv/d{1,2}-data.csv``:

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (e.g. ``D2 LENDING AND CREDIT AGGREGATES``)
1           ``Title,<column display names>``
2           ``Description,<descriptions>``
3-9         Frequency / Type / Units / blanks / Source
10          ``Publication date,<refresh date, all cols>``
11          ``Series ID,<RBA series codes>``
12+         ``DD/MM/YYYY,<observation values per series>``
==========  =====================================================

The "Publication date" row (10) is **the date of the latest refresh**,
not per-observation. That row is therefore not used for point-in-time
correctness — observation-level publication dates come from
``rba.data.rba_d_release_calendar`` instead.

Publication-date convention
---------------------------
D1/D2 publish to a fixed schedule documented by the RBA as "last
business day of each month, 11:30 am" — covering data for the
*preceding* month. The algorithmic rule "last weekday (Mon-Fri) of the
month following the reference month" matches every sampled ABS-archived
release page from 2001-07 through 2026-03, with three hand-verified
exceptions (two Easter collisions + one D2A-decommissioning early
release) handled by ``_OVERRIDES`` in ``rba.data.rba_d_release_calendar``.

Pre-2001 release pages 404 on the RBA site; the rule itself is
RBA-documented and stable. The validation floor is 1993-01-31
(inflation-targeting era start). Observations on/after the floor MUST
match the calendar; pre-1993 observations retain ``NaT``.

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release value. The RBA revises both D1 and D2
historical observations as more comprehensive APRA returns are
incorporated (the ``d2-series-breaks.csv`` file documents major break
events). We do not reconstruct prior vintages, so for revised months
the returned ``value`` is not exactly the number the RBA Board saw on
``publication_date``. Acknowledged limitation; see CONTEXT.md
Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference month.
- ``publication_date`` (datetime64[ns]) — algorithmic D1/D2 release date
  from ``rba.data.rba_d_release_calendar``. ``NaT`` for pre-1993
  observations.
- ``series_id`` (object) — one of the seven logical series IDs listed in
  ``SERIES`` below.
- ``value`` (float64) — observation value (level in $ bn for D2 series,
  percent for D1 growth series).

Side effects
------------
``fetch()`` writes under ``data/raw/rba_d/``:

- ``<YYYY-MM-DD>__d1.csv`` and ``<YYYY-MM-DD>__d2.csv`` — verbatim CSV
  bytes per table.
- ``_metadata.json`` — provenance manifest with one entry per table:
  URL, SHA-256, download timestamp (UTC), byte count, observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import RAW_DATA_DIR
from rba.data.rba_d_release_calendar import build_rba_d_release_calendar

SOURCE_NAME = "rba_d"

# Observations earlier than this are not validated against the release
# calendar: the inflation-targeting era starts 1993, and that is also the
# earliest reference month for which an RBA release-page archive exists.
# Pre-1993 rows retain NaT publication_date, matching abs_labour_force.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

_TABLE_URLS: dict[str, str] = {
    "d1": "https://www.rba.gov.au/statistics/tables/csv/d1-data.csv",
    "d2": "https://www.rba.gov.au/statistics/tables/csv/d2-data.csv",
}


@dataclass(frozen=True)
class RbaDSeries:
    """A single RBA D1/D2 series, parameterised by table and RBA series ID."""

    series_id: str
    table: str
    rba_series_id: str


SERIES: tuple[RbaDSeries, ...] = (
    # D2 levels ($ billion, seasonally adjusted).
    # The "_legacy" pair stops 2019-06 (methodology change); the
    # "_inc_fin" pair starts 2019-07.
    RbaDSeries(
        series_id="credit_total_legacy_sa",
        table="d2",
        rba_series_id="DLCACS",
    ),
    RbaDSeries(
        series_id="credit_total_inc_fin_sa",
        table="d2",
        rba_series_id="DLCACSFS",
    ),
    RbaDSeries(
        series_id="credit_business_legacy_sa",
        table="d2",
        rba_series_id="DLCACBS",
    ),
    RbaDSeries(
        series_id="credit_business_inc_fin_sa",
        table="d2",
        rba_series_id="DLCACSFBS",
    ),
    RbaDSeries(
        series_id="credit_owner_occupier_housing_sa",
        table="d2",
        rba_series_id="DLCACOHS",
    ),
    RbaDSeries(
        series_id="credit_investor_housing_sa",
        table="d2",
        rba_series_id="DLCACIHS",
    ),
    RbaDSeries(
        series_id="credit_other_personal_sa",
        table="d2",
        rba_series_id="DLCACOPS",
    ),
    # D1 growth rates (%, seasonally adjusted). Total-credit YoY also has
    # the 2019 series-break (legacy stops 2019-06; new series starts
    # 2020-07 since it needs 12 months of underlying levels). Broad money
    # is the one series here that D2 cannot provide via level → growth,
    # so it's mandatory.
    RbaDSeries(
        series_id="credit_total_yoy_growth_legacy_sa",
        table="d1",
        rba_series_id="DGFAC12",
    ),
    RbaDSeries(
        series_id="credit_total_inc_fin_yoy_growth_sa",
        table="d1",
        rba_series_id="DGFACNW12",
    ),
    RbaDSeries(
        series_id="broad_money_yoy_growth_sa",
        table="d1",
        rba_series_id="DGFABM12",
    ),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull D1 + D2, extract headline series, attach publication dates.

    Parameters
    ----------
    force_download
        If True (default), refresh both tables from the live RBA endpoints.
        If False, reuse the most recent dated snapshot per table and only
        download when no snapshot is present.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) with one row per (series_id, observation_date).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    table_bytes: dict[str, bytes] = {}
    manifest: list[dict[str, object]] = []
    for table, url in _TABLE_URLS.items():
        snapshot_path, entry, raw_bytes = _resolve_snapshot(
            dest_dir, table=table, url=url, force_download=force_download
        )
        table_bytes[table] = raw_bytes
        manifest.append(entry)

    frames: list[pd.DataFrame] = []
    obs_counts: dict[str, int] = {"d1": 0, "d2": 0}
    for series in SERIES:
        df = _parse(table_bytes[series.table], spec=series)
        obs_counts[series.table] += len(df)
        frames.append(df)

    # Annotate the manifest with the total observations contributed by each
    # source table (summed across the logical series we extract from it).
    for entry in manifest:
        entry["observations"] = obs_counts[str(entry["table"])]

    _write_manifest(dest_dir, manifest)

    out = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(out)
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge algorithmic D1/D2 release dates onto each observation.

    Defensively drops any pre-existing ``publication_date`` column before
    merging so a stale value cannot leak through. Raises ``ValueError`` if
    any observation on or after ``_CALENDAR_VALIDATION_FLOOR`` is unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_rba_d_release_calendar(start_year=1993)
    merged = df.merge(
        calendar_df.rename(columns={"reference_month_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} RBA D1/D2 observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    table: str,
    url: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes)."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{table}.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing {} snapshot at {}", table.upper(), path)
        entry: dict[str, object] = {
            "table": table,
            "url": url,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir, table=table, url=url)


def _download(
    dest_dir: Path, *, table: str, url: str
) -> tuple[Path, dict[str, object], bytes]:
    """Download one table CSV, save dated snapshot, return path + manifest + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{table}.csv"
    logger.info("Downloading {} -> {}", url, snapshot_path)

    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": table,
        "url": url,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry, raw_bytes


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


def _parse(csv_bytes: bytes, *, spec: RbaDSeries) -> pd.DataFrame:
    """Extract one series from a D1 or D2 CSV by RBA series ID.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes downloaded from a ``/statistics/tables/csv/d{1,2}-data.csv``
        endpoint. Expected layout described in the module docstring.
    spec
        ``RbaDSeries`` identifying which series to extract.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-month),
        ``series_id`` (object, the logical name from ``spec.series_id``),
        ``value`` (float64). One row per non-null monthly observation.
        Publication dates are attached later by ``_attach_publication_dates``
        via merge against the release calendar.

    Shapes
    ------
    Returns: (n_months, 3).
    """
    # header=1 makes the "Title" row the header. The Series ID row then
    # sits in the body under df["Title"] == "Series ID" — that row maps
    # each column display name to its RBA series ID.
    raw_df = pd.read_csv(
        io.BytesIO(csv_bytes), header=1, low_memory=False, skipinitialspace=False
    )

    sid_row = raw_df[raw_df["Title"] == "Series ID"]
    if sid_row.empty:
        raise ValueError(
            f"No 'Series ID' metadata row found while extracting "
            f"{spec.rba_series_id!r}; CSV layout may have changed."
        )
    sid_mapping = sid_row.iloc[0].to_dict()

    target_col: str | None = None
    for col, rba_id in sid_mapping.items():
        if col == "Title":
            continue
        if rba_id == spec.rba_series_id:
            target_col = str(col)
            break
    if target_col is None:
        raise ValueError(
            f"RBA series ID {spec.rba_series_id!r} not found among "
            f"columns in {spec.table.upper()} CSV; series may have been "
            f"renamed or removed."
        )

    parsed_dates = pd.to_datetime(
        raw_df["Title"], format="%d/%m/%Y", errors="coerce"
    )
    data_mask = parsed_dates.notna()

    df = pd.DataFrame(
        {
            "observation_date": _to_month_end(parsed_dates[data_mask]),
            "series_id": spec.series_id,
            "value": pd.to_numeric(raw_df.loc[data_mask, target_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Parsed zero observations for {spec.series_id!r} "
            f"(RBA {spec.rba_series_id!r}); CSV layout may have changed "
            f"or the column is empty in the current vintage."
        )

    return df[["observation_date", "series_id", "value"]]


def _to_month_end(dates: pd.Series) -> pd.Series:
    """Snap each timestamp to the last calendar day of its month.

    The CSV publishes month-end dates already, but using
    ``MonthEnd().rollforward`` (idempotent on already-month-end dates) makes
    the contract explicit and resilient to any future format change.
    Normalised to nanosecond resolution to match other source modules.
    """
    offset = pd.offsets.MonthEnd()
    return pd.to_datetime(dates.map(offset.rollforward)).dt.as_unit("ns")
