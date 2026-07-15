"""ABS Wage Price Index — quarterly, Australia (all industries).

Pulls headline and sector-split WPI **index numbers** from the ABS Data API
(SDMX-JSON). All three series live in the ``WPI`` dataflow (Cat. 6345.0):

================================================  ============================  =============
Series                                            Datakey                       Coverage from
------------------------------------------------  ----------------------------  -------------
Headline — total hourly excl. bonuses, all sec.   ``1.THRPEB.7.TOT.20.AUS.Q``   1997-Q3
Private sector — total hourly excl. bonuses, SA   ``1.THRPEB.1.TOT.20.AUS.Q``   1997-Q3
Public  sector — total hourly excl. bonuses, SA   ``1.THRPEB.2.TOT.20.AUS.Q``   1997-Q3
================================================  ============================  =============

Datakey order is ``MEASURE.INDEX.SECTOR.INDUSTRY.TSEST.REGION.FREQ``. Picks:

- ``MEASURE=1``  — Quarterly Index (raw level). We pull index numbers only and
  let the feature layer derive YoY/QoQ — same convention as ``abs_cpi``.
- ``INDEX=THRPEB`` — Total hourly rates of pay excluding bonuses (the
  headline measure RBA tracks; bonuses introduce volatility unrelated to
  underlying wage pressure).
- ``SECTOR``     — ``7`` = Private and Public combined (headline); ``1`` =
  Private only; ``2`` = Public only.
- ``INDUSTRY=TOT`` — All industries.
- ``TSEST=20``   — Seasonally adjusted (the headline series the RBA quotes).
- ``REGION=AUS`` — Australia (national).
- ``FREQ=Q``     — Quarterly.

Publication-date convention
---------------------------
The ABS WPI has no clean algorithmic release rule (mix of 2nd/3rd Wednesdays
with at least one Tuesday on record; observed lag 43-52 days). Publication
dates are therefore **scraped from the ABS release pages** by
``rba.data.wpi_release_calendar``, which materialises a CSV at
``data/external/wpi_release_dates.csv`` with one row per (reference quarter,
original release date).

The post-redesign ABS URL pattern resolves back to **2019-Q3**; earlier
quarterly pages live under the legacy ``/Ausstats/`` namespace and are not
scraped. To keep pre-2019-Q3 WPI history usable in the model, this source
applies a conservative **flat 60-day offset** to those rows (60d > the
worst-observed 52d lag in the scraped era, so we never falsely mark a value
as available before the ABS actually published it). The split is documented
on every row implicitly through the calendar merge.

Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (2019-Q3) MUST match the
scraped calendar; an unmatched in-window row raises ``ValueError`` — that
signals a calendar/data mismatch (e.g. ABS adding a quarter the scraper has
not refreshed for, or an observation_date drifting off a quarter-end).

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference quarter.
- ``publication_date`` (datetime64[ns]) — scraped original release date
  from ``rba.data.wpi_release_calendar`` for 2019-Q3 onward; ``observation_date
  + 60 days`` for earlier quarters. No NaT.
- ``series_id`` (object) — one of ``wpi_total_hourly_excl_bonuses_all_sectors_sa``,
  ``wpi_total_hourly_excl_bonuses_private_sa``,
  ``wpi_total_hourly_excl_bonuses_public_sa``.
- ``value`` (float64) — the quarterly index number.

Side effects
------------
``fetch()`` writes under ``data/raw/abs_wpi/``:

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
from rba.data.wpi_release_calendar import build_wpi_release_calendar

SOURCE_NAME = "abs_wpi"
API_BASE_URL = "https://data.api.abs.gov.au/rest/data"

# Floor for "must match scraped calendar" — the earliest quarter that resolves
# on the post-redesign ABS site. Older WPI observations get the flat-offset
# fallback below.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("2019-09-30")

# Flat conservative offset applied to observations earlier than the
# validation floor. 60 days > the worst-observed lag in the scraped era
# (52 days for Q4 -> late Feb releases), so the marker never falsely claims
# a value was available before its true release.
_PUBLICATION_OFFSET_DAYS = 60


@dataclass(frozen=True)
class WpiSeries:
    """A single WPI series, parameterised by dataflow + datakey."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "1997-Q1") -> str:
        return (
            f"{API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


SERIES: tuple[WpiSeries, ...] = (
    WpiSeries(
        series_id="wpi_total_hourly_excl_bonuses_all_sectors_sa",
        dataflow="WPI",
        datakey="1.THRPEB.7.TOT.20.AUS.Q",
    ),
    WpiSeries(
        series_id="wpi_total_hourly_excl_bonuses_private_sa",
        dataflow="WPI",
        datakey="1.THRPEB.1.TOT.20.AUS.Q",
    ),
    WpiSeries(
        series_id="wpi_total_hourly_excl_bonuses_public_sa",
        dataflow="WPI",
        datakey="1.THRPEB.2.TOT.20.AUS.Q",
    ),
)


def fetch(*, force_download: bool = True, start_period: str = "1997-Q1") -> pd.DataFrame:
    """Pull all three quarterly WPI series, persist raw, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live API. If False,
        reuse the most recent dated snapshot per series and only download when
        no snapshot is present.
    start_period
        SDMX period string for ``startPeriod`` (e.g. ``"1997-Q3"``). Default
        ``"1997-Q1"`` covers the full WPI history (the series actually begins
        1997-Q3; the API returns only what's available).

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) where ``n_obs`` ≈ 345 for ``start_period="1997-Q1"``
    (≈ 115 quarters × 3 series as of 2026-05).
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

    For observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (2019-Q3), merge
    in the scraped original release date from
    ``rba.data.wpi_release_calendar``. For earlier observations, apply the
    flat ``_PUBLICATION_OFFSET_DAYS`` fallback.

    Defensively drops any pre-existing ``publication_date`` column. Raises
    ``ValueError`` if any in-window observation is unmatched in the calendar
    — that signals a calendar / data mismatch.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_wpi_release_calendar()
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
            f"{len(unmatched_in_window)} WPI observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the scraped release calendar. Re-run "
            f"`python -m rba.data.wpi_release_calendar` to refresh. "
            f"Sample: {sample}"
        )

    fallback = (merged["observation_date"] + pd.Timedelta(days=_PUBLICATION_OFFSET_DAYS)).astype(
        "datetime64[ns]"
    )
    merged["publication_date"] = merged["publication_date"].fillna(fallback)

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    series: WpiSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    """Return (snapshot path, manifest entry for that snapshot)."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing WPI snapshot at {}", path)
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
    series: WpiSeries,
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
        ``"wpi_total_hourly_excl_bonuses_all_sectors_sa"``).

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
        # key format: "0:0:0:0:0:0:0:N" — 7 series dims + TIME_PERIOD position.
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
    """Convert an ABS quarterly period string (e.g. ``"1997-Q3"``) to the
    last calendar day of that quarter (``1997-09-30``).

    Normalised to nanosecond resolution so every source in this project
    produces ``datetime64[ns]`` and the point-in-time join in
    ``rba.data.align`` does not need to coerce units.
    """
    return pd.Period(period, freq="Q").end_time.normalize().as_unit("ns")
