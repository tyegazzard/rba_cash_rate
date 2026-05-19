"""ABS Consumer Price Index — quarterly, Australia (8-capitals weighted avg).

Pulls headline CPI, trimmed mean, and weighted median **index numbers** from
the ABS Data API (SDMX-JSON). The three series live across two dataflows:

============================  ============  ===========================  =============
Series                        Dataflow      Datakey                      Coverage from
----------------------------  ------------  ---------------------------  -------------
Headline CPI (orig)           ``CPI``       ``1.10001.10.50.Q``          1948-Q3
Trimmed Mean (seas adj)       ``CPI_Q``     ``1.999902.20.50.Q``         1982-Q1
Weighted Median (seas adj)    ``CPI_Q``     ``1.999903.20.50.Q``         1982-Q1
============================  ============  ===========================  =============

Datakey order is ``MEASURE.INDEX.TSEST.REGION.FREQ`` for both dataflows.
``MEASURE=1`` is "index numbers" (the raw level); we deliberately do *not* pull
the pre-computed ``MEASURE=2`` (% Δ vs prior period) or ``MEASURE=3``
(% Δ vs same period last year), because the feature layer is the canonical
place for derived series — pulling raw avoids rounding mismatches and keeps
all the derivation choices (lag length, smoothing) in one place.

Headline is published "original" (un-seasonally-adjusted); the two analytical
core measures are only published as seasonally-adjusted series by the ABS.

Publication-date convention
---------------------------
The ABS Data API does not expose per-release publication dates. Per the ABS
release calendar, quarterly CPI is published on the **last Wednesday of the
month following** the reference quarter. ``rba.data.cpi_release_calendar``
computes this rule algorithmically and is the source of truth for CPI
publication dates here; this module merges its output onto every parsed
observation. Known public-holiday exceptions go in that module's ``_OVERRIDES``
dict — not here.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference quarter.
- ``publication_date`` (datetime64[ns]) — algorithmic ABS release date from
  ``rba.data.cpi_release_calendar.build_cpi_release_calendar``.
- ``series_id`` (object) — one of ``headline_cpi_index``,
  ``trimmed_mean_index``, ``weighted_median_index``.
- ``value`` (float64) — the index number for that quarter.

Side effects
------------
``fetch()`` writes under ``data/raw/abs_cpi/``:

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
from rba.data.cpi_release_calendar import build_cpi_release_calendar

SOURCE_NAME = "abs_cpi"
API_BASE_URL = "https://data.api.abs.gov.au/rest/data"

# Observations earlier than this date are not validated against the release
# calendar (the calendar defaults to start_year=1993 and the inflation-targeting
# era begins here). Pre-1993 rows retain NaN publication_date for downstream
# code to filter.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class CpiSeries:
    """A single CPI series we pull, parameterised by dataflow + datakey."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "1948-Q1") -> str:
        return (
            f"{API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


SERIES: tuple[CpiSeries, ...] = (
    CpiSeries(
        series_id="headline_cpi_index",
        dataflow="CPI",
        datakey="1.10001.10.50.Q",
    ),
    CpiSeries(
        series_id="trimmed_mean_index",
        dataflow="CPI_Q",
        datakey="1.999902.20.50.Q",
    ),
    CpiSeries(
        series_id="weighted_median_index",
        dataflow="CPI_Q",
        datakey="1.999903.20.50.Q",
    ),
)


def fetch(*, force_download: bool = True, start_period: str = "1948-Q1") -> pd.DataFrame:
    """Pull all three quarterly CPI series, persist raw, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live API. If False,
        reuse the most recent dated snapshot per series and only download when
        no snapshot is present.
    start_period
        SDMX period string for ``startPeriod`` (e.g. ``"1993-Q1"``). Default
        ``"1948-Q1"`` pulls the full ABS history; downstream code restricts
        to the inflation-targeting era.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) where ``n_obs`` ≈ 660 for ``start_period="1948-Q1"``
    (≈ 310 headline + 176 trimmed + 176 weighted-median observations).
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
    """Merge algorithmic CPI release dates onto each observation.

    Defensively drops any pre-existing ``publication_date`` column before
    merging so a stale value (e.g. from cached upstream code) cannot leak
    through. Raises ``ValueError`` if any observation on or after
    ``_CALENDAR_VALIDATION_FLOOR`` is unmatched — that signals a calendar /
    data mismatch (e.g. quarter-end falling on a day the calendar didn't emit).
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_cpi_release_calendar(start_year=1948)
    merged = df.merge(
        calendar_df.rename(columns={"reference_quarter_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} CPI observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    series: CpiSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    """Return (snapshot path, manifest entry for that snapshot)."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing CPI snapshot at {}", path)
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
    series: CpiSeries,
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
        Logical name to stamp on every output row (e.g. ``"headline_cpi_index"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-quarter),
        ``series_id`` (object), ``value`` (float64). One row per quarter.
        Publication dates are attached later by ``_attach_publication_dates``
        via merge against the algorithmic release calendar.

    Shapes
    ------
    Returns: (n_quarters, 3).
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
        # key format: "0:0:0:0:0:7" — one position per observation dimension.
        position = int(key.split(":")[time_index])
        period_str = time_codes[position]
        raw_value = values[0]
        if raw_value is None:
            continue
        rows.append(
            {
                "observation_date": _quarter_end(period_str),
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


def _quarter_end(period: str) -> pd.Timestamp:
    """Convert an ABS quarterly period string (e.g. ``"1993-Q1"``) to the
    last calendar day of that quarter (``1993-03-31``).

    Normalised to nanosecond resolution so every source in this project
    produces ``datetime64[ns]`` and the point-in-time join in
    ``rba.data.align`` does not need to coerce units.
    """
    return pd.Period(period, freq="Q").end_time.normalize().as_unit("ns")
