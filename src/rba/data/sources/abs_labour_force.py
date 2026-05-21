"""ABS Labour Force — monthly, Australia (national, seasonally adjusted).

Pulls the three headline RBA-relevant labour-market series from the ABS Data
API (SDMX-JSON). They live across two dataflows:

==================================  ============  ============================  =============
Series                              Dataflow      Datakey                       Coverage from
----------------------------------  ------------  ----------------------------  -------------
Unemployment rate (SA, %)           ``LF``        ``M13.3.1599.20.AUS.M``       1978-02
Underemployment rate (SA, %)        ``LF_UNDER``  ``M23.3.1599.20.AUS.M``       1978-02
Participation rate (SA, %)          ``LF``        ``M12.3.1599.20.AUS.M``       1978-02
==================================  ============  ============================  =============

Datakey order is ``MEASURE.SEX.AGE.TSEST.REGION.FREQ`` for ``LF`` and
``PARM_ITEM.SEX.AGE.TSEST.REGION.FREQ`` for ``LF_UNDER``. The first dimension
plays the same role in both; the parser reads observation values by position
and is dimension-name-agnostic.

Key dimension picks:

- ``SEX=3``      — Persons (both sexes combined).
- ``AGE=1599``   — Total (all ages 15+).
- ``TSEST=20``   — Seasonally adjusted (the headline numbers RBA quotes).
- ``REGION=AUS`` — Australia (national).
- ``FREQ=M``     — Monthly.

All three series report in percent (``UNIT_MEASURE=PCT``).

Publication-date convention
---------------------------
LFS publication dates are computed algorithmically via
``rba.data.lfs_release_calendar`` using two era-aware rules plus a December
exception:

- 1993 – Dec 2015 reference months: 2nd Thursday of the following month.
- Jan 2016 – present reference months: 3rd Thursday of the following month.
- Any December reference month: 4th Thursday of the following January
  (permanent exception in both eras, longer collection window).

Public-holiday / ABS rescheduling exceptions go in the ``_OVERRIDES`` dict in
``rba.data.lfs_release_calendar`` — not here. The flat day-offset heuristic
is **not** used for LFS.

This module merges the algorithmic calendar onto every parsed observation in
``_attach_publication_dates``. Observations earlier than 1993 (the
verification window for the era rules) are allowed to retain NaN
``publication_date``; any unmatched row on/after 1993 raises a ``ValueError``.

Vintage policy
--------------
Values returned by ``fetch()`` are the **current ABS vintage** at the time of
download — *not* the original first-release value. LFS seasonal adjustment
is recomputed every release and historical observations are revised
materially (especially for sub-aggregates like underemployment); we do not
reconstruct prior vintages. So for any month that has been revised since
first publication, the ``value`` in the returned frame is not exactly the
number the RBA board saw on ``publication_date``. This is a known deviation
from strict real-time correctness, acknowledged in the project README and
discussed under Invariant #1 in ``CONTEXT.md``. The revision bias is larger
here than for headline CPI.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference month.
- ``publication_date`` (datetime64[ns]) — algorithmic ABS release date from
  ``rba.data.lfs_release_calendar.build_lfs_release_calendar``. ``NaT`` for
  pre-1993 observations (outside the verified era-rule window).
- ``series_id`` (object) — one of ``unemployment_rate_sa``,
  ``underemployment_rate_sa``, ``participation_rate_sa``.
- ``value`` (float64) — the percent value for that month.

Side effects
------------
``fetch()`` writes under ``data/raw/abs_labour_force/``:

- ``<YYYY-MM-DD>__<series_id>.json`` — verbatim SDMX-JSON bytes per series.
- ``_metadata.json`` — provenance manifest with one entry per series:
  URL, SHA-256, download timestamp (UTC), byte count, observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import RAW_DATA_DIR
from rba.data.lfs_release_calendar import build_lfs_release_calendar

SOURCE_NAME = "abs_labour_force"
API_BASE_URL = "https://data.api.abs.gov.au/rest/data"

# Observations earlier than this date are not validated against the release
# calendar: the era rules in lfs_release_calendar are only user-verified from
# 1993 onward. Pre-1993 rows retain NaT publication_date for downstream code
# to filter (the inflation-targeting era starts 1993 anyway).
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class LabourForceSeries:
    """A single labour-force series, parameterised by dataflow + datakey."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "1978-01") -> str:
        return (
            f"{API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


SERIES: tuple[LabourForceSeries, ...] = (
    LabourForceSeries(
        series_id="unemployment_rate_sa",
        dataflow="LF",
        datakey="M13.3.1599.20.AUS.M",
    ),
    LabourForceSeries(
        series_id="underemployment_rate_sa",
        dataflow="LF_UNDER",
        datakey="M23.3.1599.20.AUS.M",
    ),
    LabourForceSeries(
        series_id="participation_rate_sa",
        dataflow="LF",
        datakey="M12.3.1599.20.AUS.M",
    ),
)


def fetch(*, force_download: bool = True, start_period: str = "1978-01") -> pd.DataFrame:
    """Pull all three monthly labour-force series, persist raw, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live API. If False,
        reuse the most recent dated snapshot per series and only download when
        no snapshot is present.
    start_period
        SDMX period string for ``startPeriod`` (e.g. ``"1993-01"``). Default
        ``"1978-01"`` pulls the full ABS LFS history.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) where ``n_obs`` ≈ 1734 for ``start_period="1978-01"``
    (≈ 578 obs × 3 series as of 2026-05).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []
    for series in SERIES:
        snapshot_path, entry = _resolve_snapshot(
            dest_dir,
            series=series,
            start_period=start_period,
            force_download=force_download,
        )
        df = _parse(snapshot_path.read_bytes(), series_id=series.series_id)
        entry["observations"] = len(df)
        manifest.append(entry)
        frames.append(df)

    _write_manifest(dest_dir, manifest)

    out = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(out)
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge algorithmic LFS release dates onto each observation.

    Defensively drops any pre-existing ``publication_date`` column before
    merging so a stale value cannot leak through. Raises ``ValueError`` if any
    observation on or after ``_CALENDAR_VALIDATION_FLOOR`` is unmatched —
    that signals a calendar / data mismatch (e.g. observation date not landing
    on a month-end the calendar emitted).
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_lfs_release_calendar(start_year=1993)
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
            f"{len(unmatched)} LFS observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    series: LabourForceSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    """Return (snapshot path, manifest entry for that snapshot)."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing LFS snapshot at {}", path)
        entry: dict[str, object] = {
            "series_id": series.series_id,
            "dataflow": series.dataflow,
            "datakey": series.datakey,
            "url": series.url(start_period=start_period),
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry
    return _download(dest_dir, series=series, start_period=start_period)


def _download(
    dest_dir: Path,
    *,
    series: LabourForceSeries,
    start_period: str,
) -> tuple[Path, dict[str, object]]:
    """Download one series, save dated snapshot, return path + manifest entry."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{series.series_id}.json"
    url = series.url(start_period=start_period)
    logger.info("Downloading {} -> {}", url, snapshot_path)

    req = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.sdmx.data+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310 — public ABS URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "series_id": series.series_id,
        "dataflow": series.dataflow,
        "datakey": series.datakey,
        "url": url,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


def _parse(json_bytes: bytes, *, series_id: str) -> pd.DataFrame:
    """Parse one SDMX-JSON payload into a long-format frame.

    Parameters
    ----------
    json_bytes
        Raw SDMX-JSON response from the ABS Data API queried with
        ``dimensionAtObservation=AllDimensions``. Expected structure:
        ``data.dataSets[0].observations`` keyed by colon-delimited dimension
        indices, with ``data.structures[0].dimensions.observation`` providing
        the index→code mapping for each dimension position (including
        ``TIME_PERIOD``).
    series_id
        Logical name to stamp on every output row (e.g.
        ``"unemployment_rate_sa"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-month),
        ``series_id`` (object), ``value`` (float64). One row per month.
        Publication dates are attached later by ``_attach_publication_dates``
        via merge against the algorithmic release calendar.

    Shapes
    ------
    Returns: (n_months, 3).
    """
    payload = json.loads(json_bytes)
    structure = payload["data"]["structures"][0]
    obs_dims = structure["dimensions"]["observation"]

    time_index = next(
        i for i, d in enumerate(obs_dims) if d["id"] == "TIME_PERIOD"
    )
    time_codes = [v["id"] for v in obs_dims[time_index]["values"]]

    observations = payload["data"]["dataSets"][0]["observations"]
    rows: list[dict[str, object]] = []
    for key, values in observations.items():
        # key format: "0:0:0:0:0:0:7" for 6 series-defining dims + TIME_PERIOD.
        position = int(key.split(":")[time_index])
        period_str = time_codes[position]
        raw_value = values[0]
        if raw_value is None:
            continue
        rows.append(
            {
                "observation_date": _month_end(period_str),
                "series_id": series_id,
                "value": float(raw_value),
            }
        )

    if not rows:
        raise ValueError(
            f"Parsed zero observations for {series_id!r}; SDMX-JSON layout "
            "may have changed or the dataflow returned no data."
        )

    return pd.DataFrame(rows)[["observation_date", "series_id", "value"]]


def _month_end(period: str) -> pd.Timestamp:
    """Convert an ABS monthly period string (e.g. ``"1978-02"``) to the
    last calendar day of that month (``1978-02-28``).

    Normalised to nanosecond resolution to keep ``datetime64[ns]`` consistent
    across all sources (see ``rba.data.sources.abs_cpi._quarter_end``).
    """
    return pd.Period(period, freq="M").end_time.normalize().as_unit("ns")
