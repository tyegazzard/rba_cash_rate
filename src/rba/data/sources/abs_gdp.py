"""ABS National Accounts (Cat. 5206.0) — quarterly Australian GDP series.

Pulls headline real GDP, nominal GDP and real GDP per capita **levels** from
the ABS Data API (SDMX-JSON). All three series live in the ``ANA_AGG``
dataflow (Australian National Accounts: Key Aggregates):

================================================  ===========================  =============
Series                                            Datakey                      Coverage from
------------------------------------------------  ---------------------------  -------------
Real GDP (chain volume measure, SA)               ``M1.GPM.20.AUS.Q``          1959-Q3
Nominal GDP (current prices, SA)                  ``M3.GPM.20.AUS.Q``          1959-Q3
Real GDP per capita (chain volume measure, SA)    ``M1.GPM_PCA.20.AUS.Q``      1973-Q3
================================================  ===========================  =============

Datakey order is ``MEASURE.DATA_ITEM.TSEST.REGION.FREQ``. Picks:

- ``MEASURE=M1`` — Chain volume measures (real, the headline RBA tracks);
  ``MEASURE=M3`` — Current prices (nominal).
- ``DATA_ITEM=GPM`` — Gross domestic product; ``GPM_PCA`` — GDP per capita.
  We deliberately pull **levels** (the index/dollar series) and let the
  feature layer derive growth rates, mirroring the convention used by
  ``abs_cpi`` and ``abs_wpi``. The pre-computed percentage-change measures
  (``M2`` / ``M4`` / ``M6``) are available but introduce rounding
  mismatches against any custom YoY/QoQ derivation downstream.
- ``TSEST=20`` — Seasonally adjusted (the headline series RBA quotes).
- ``REGION=AUS`` — Australia (national).
- ``FREQ=Q`` — Quarterly.

Publication-date convention
---------------------------
The post-2003 ABS GDP release rule "first Wednesday of the third month after
quarter-end" holds cleanly for recent history (verified against 20 release
pages 2019-Q2 → 2025-Q4) but **does not** extend back to 1993 — the
pre-2003 era has irregular weekday slots (Tue/Fri observed), 2-month rather
than 3-month lags, and 2nd/3rd-Wednesday releases. Publication dates are
therefore **scraped from the ABS release pages** by
``rba.data.gdp_release_calendar``, which materialises a CSV at
``data/external/gdp_release_dates.csv`` with one row per (reference quarter,
original release date) spanning 1993-Q1 onward (the calendar uses two URL
patterns: legacy ``/ausstats/`` for 1993-Q1 → 2019-Q1 and modern
``/statistics/`` for 2019-Q2 onward).

Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (1993-01-01) MUST match
the scraped calendar; an unmatched in-window row raises ``ValueError`` —
that signals a calendar / data mismatch (e.g. the ABS adding a quarter the
scraper has not refreshed for). Pre-1993 observations (back to 1959-Q3)
retain ``NaT`` ``publication_date``; downstream code filters to the
inflation-targeting era anyway and those rows are not in scope for the
model.

Vintage policy
--------------
Values returned by ``fetch()`` are the **current ABS vintage** at the time
of download — *not* the original first-release value. GDP is among the
most heavily revised major series (chain volume re-referencing, seasonal
re-estimation, and methodology updates routinely shift historical
observations); we do not reconstruct prior vintages. For any quarter that
has been revised since first publication, the ``value`` in the returned
frame is not exactly the number the RBA board saw on ``publication_date``.
Acknowledged as a deviation from strict real-time correctness in the
project README and discussed under Invariant #1 in ``CONTEXT.md``. The
revision bias is materially larger here than for headline CPI.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference quarter.
- ``publication_date`` (datetime64[ns]) — scraped original release date
  from ``rba.data.gdp_release_calendar`` for 1993-Q1 onward; ``NaT`` for
  earlier quarters.
- ``series_id`` (object) — one of ``gdp_real_chain_volume_sa``,
  ``gdp_nominal_current_prices_sa``,
  ``gdp_per_capita_real_chain_volume_sa``.
- ``value`` (float64) — the quarterly level (chain-volume index basis for
  real series, dollars for nominal).

Side effects
------------
``fetch()`` writes under ``data/raw/abs_gdp/``:

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
from rba.data.gdp_release_calendar import build_gdp_release_calendar

SOURCE_NAME = "abs_gdp"
API_BASE_URL = "https://data.api.abs.gov.au/rest/data"

# Floor for "must match scraped calendar". The scraped calendar reaches
# back to 1993-Q1 (start of inflation targeting), so any GDP observation
# on/after this date must resolve to a calendar row. Earlier observations
# (back to 1959-Q3 from the API) retain NaT publication_date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class GdpSeries:
    """A single GDP series, parameterised by dataflow + datakey."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "1959-Q3") -> str:
        return (
            f"{API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


SERIES: tuple[GdpSeries, ...] = (
    GdpSeries(
        series_id="gdp_real_chain_volume_sa",
        dataflow="ANA_AGG",
        datakey="M1.GPM.20.AUS.Q",
    ),
    GdpSeries(
        series_id="gdp_nominal_current_prices_sa",
        dataflow="ANA_AGG",
        datakey="M3.GPM.20.AUS.Q",
    ),
    GdpSeries(
        series_id="gdp_per_capita_real_chain_volume_sa",
        dataflow="ANA_AGG",
        datakey="M1.GPM_PCA.20.AUS.Q",
    ),
)


def fetch(*, force_download: bool = True, start_period: str = "1959-Q3") -> pd.DataFrame:
    """Pull all three quarterly GDP series, persist raw, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live API. If False,
        reuse the most recent dated snapshot per series and only download when
        no snapshot is present.
    start_period
        SDMX period string for ``startPeriod`` (e.g. ``"1993-Q1"``). Default
        ``"1959-Q3"`` covers the full ABS history; downstream code restricts
        to the inflation-targeting era.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) where ``n_obs`` ≈ 742 for ``start_period="1959-Q3"``
    (266 + 266 + 210 obs across the three series as of 2026-Q1).
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
    """Attach publication dates to every observation.

    Merges in the scraped original release date from
    ``rba.data.gdp_release_calendar`` for every quarter the calendar covers
    (1993-Q1 onward). Pre-1993 observations retain ``NaT`` — the
    inflation-targeting-era cut applied by downstream model code drops
    them anyway.

    Defensively drops any pre-existing ``publication_date`` column. Raises
    ``ValueError`` if any in-window observation (on/after
    ``_CALENDAR_VALIDATION_FLOOR``) is unmatched in the calendar — that
    signals a calendar / data mismatch.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_gdp_release_calendar()
    merged = df.merge(
        calendar_df.rename(columns={"reference_quarter_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched_in_window = merged[in_window & merged["publication_date"].isna()]
    if not unmatched_in_window.empty:
        sample = unmatched_in_window[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched_in_window)} GDP observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the scraped release calendar. Re-run "
            f"`python -m rba.data.gdp_release_calendar` to refresh. "
            f"Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    series: GdpSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    """Return (snapshot path, manifest entry for that snapshot)."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing GDP snapshot at {}", path)
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
    series: GdpSeries,
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
        ``"gdp_real_chain_volume_sa"``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-quarter),
        ``series_id`` (object), ``value`` (float64). One row per quarter.
        Publication dates are attached later by ``_attach_publication_dates``.

    Shapes
    ------
    Returns: (n_quarters, 3).
    """
    payload = json.loads(json_bytes)
    structure = payload["data"]["structures"][0]
    obs_dims = structure["dimensions"]["observation"]

    time_index = next(i for i, d in enumerate(obs_dims) if d["id"] == "TIME_PERIOD")
    time_codes = [v["id"] for v in obs_dims[time_index]["values"]]

    observations = payload["data"]["dataSets"][0]["observations"]
    rows: list[dict[str, object]] = []
    for key, values in observations.items():
        # key format: "0:0:0:0:0:N" — 5 series dims + TIME_PERIOD position.
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
