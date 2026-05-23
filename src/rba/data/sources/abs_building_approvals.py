"""ABS Building Approvals (Cat. 8731.0) — national monthly headline series.

Pulls the RBA-relevant national Building Approvals series from **two**
endpoints. The ABS Data API publishes Original (NSA) values only via
SDMX-JSON; the seasonally-adjusted and trend series RBA quotes live in
RBA Table H3 ("Monthly Activity Indicators"), which the RBA derives
from the same ABS release. Both come from the same underlying ABS
release and therefore share a publication date.

NSA series — ABS Data API, dataflow ``BA_GCCSA``
------------------------------------------------
================================================  ==========================================================================
Series                                            Datakey (MEASURE.VALUE.SECTOR.WORK_TYPE.BUILDING_TYPE.TSEST.REGION.FREQ)
------------------------------------------------  --------------------------------------------------------------------------
``building_approvals_dwellings_total_nsa``        ``1.1.9.TOT.100.10.AUS.M``  (count of dwellings, total residential)
``building_approvals_dwellings_houses_nsa``       ``1.1.9.TOT.110.10.AUS.M``  (count of dwellings, houses only)
``building_approvals_dwellings_other_nsa``        ``1.1.9.TOT.150.10.AUS.M``  (count of dwellings, total other residential)
``building_approvals_value_residential_aud_thousand_nsa``      ``2.1.9.TOT.100.10.AUS.M``  ($'000 value, residential)
``building_approvals_value_non_residential_aud_thousand_nsa``  ``2.1.9.TOT.700.10.AUS.M``  ($'000 value, non-residential)
================================================  ==========================================================================

Dimension picks:

- ``MEASURE=1``       — Number of dwelling units approved.
- ``MEASURE=2``       — Value of building jobs ($'000).
- ``VALUE=1``         — Total (across all value bands).
- ``SECTOR=9``        — Total sectors (private + public).
- ``WORK_TYPE=TOT``   — Total Work (new + alterations).
- ``BUILDING_TYPE``   — 100 = Total Residential; 110 = Houses; 150 = Total
  Other Residential; 700 = Total Non-residential.
- ``TSEST=10``        — Original (the only ``TSEST`` value ``BA_GCCSA``
  publishes via the API; SA / Trend live in ABS Excel only).
- ``REGION=AUS``      — Australia (national).
- ``FREQ=M``          — Monthly.

SA + trend series — RBA Table H3, derived from the same ABS release
-------------------------------------------------------------------
==========================================  ==========  ==============  =====================
Series                                      Table       RBA series ID   Coverage from
------------------------------------------  ----------  --------------  ---------------------
``building_approvals_private_dwellings_sa``      H3     ``GISPSDA``      1965-01 (monthly)
``building_approvals_private_dwellings_trend``   H3     ``GISDWPRITR``   1965-01 (monthly)
==========================================  ==========  ==============  =====================

RBA H3 publishes monthly building-approvals indicators using the ABS
seasonal adjustment / trend results. The H3 series is **private sector
only**; the ABS series at the top is total sectors (private + public),
so the two are not directly comparable as a "SA version" of the NSA
total. The feature layer should treat them as complementary: NSA total
(broad coverage) + SA private (RBA's tracked headline).

Publication-date convention
---------------------------
The ABS BA monthly release has no clean algorithmic rule (sampled
release dates land on days 1-8 of the second following month, on any
weekday Mon-Thu), so publication dates are **scraped** from each
monthly Building Approvals page by
``rba.data.abs_ba_release_calendar``, materialised to
``data/external/abs_ba_release_dates.csv``. The modern URL space only
resolves back to the Dec 2019 reference month; observations with a
pre-floor reference month fall back to a flat conservative
``+ 40`` day offset (slightly longer than the worst observed in-window
lag of 38 days, with a buffer).

The RBA H3 SA + trend series share the ABS BA publication date because
the RBA simply republishes the SA/trend numbers ABS computed in the same
release. We attach publication dates from the same BA release calendar
for both NSA and SA / trend rows.

Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (1993-01-01) MUST
resolve to a publication date — either via the scraped calendar
(``observation_date >= _SCRAPE_FLOOR``) or via the flat-offset fallback
(``< _SCRAPE_FLOOR``). An unmatched in-window row raises ``ValueError``.

Vintage policy
--------------
Values are the **current ABS vintage** at the time of download — *not*
the original first-release values. ABS revises monthly building
approvals routinely (seasonal re-estimation, late builder data); we do
not reconstruct prior vintages. Acknowledged limitation under
Invariant #1 in ``CONTEXT.md``.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference month.
- ``publication_date`` (datetime64[ns]) — scraped ABS BA release date
  (or flat-offset fallback for pre-floor months).
- ``series_id`` (object) — logical name from the SERIES registry.
- ``value`` (float64) — count of dwellings ('000 for the H3 series and
  units for the ABS NSA series — see ``Units`` in the SERIES registry)
  or $-value as appropriate.

Side effects
------------
``fetch()`` writes under ``data/raw/abs_building_approvals/``:

- ``<YYYY-MM-DD>__<series_id>.json`` — SDMX-JSON bytes per ABS NSA series.
- ``<YYYY-MM-DD>__rba_h3.csv`` — verbatim CSV bytes for RBA H3 (one file
  re-used across both SA + trend series, since they share the same CSV).
- ``_metadata.json`` — provenance manifest with one entry per source
  artifact: URL, SHA-256, download timestamp (UTC), byte count,
  observation count.
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
from rba.data.abs_ba_release_calendar import build_ba_release_calendar

SOURCE_NAME = "abs_building_approvals"
ABS_API_BASE_URL = "https://data.api.abs.gov.au/rest/data"
RBA_H3_URL = "https://www.rba.gov.au/statistics/tables/csv/h3-data.csv"

# Inflation-targeting-era floor. Observations on/after this must resolve
# to a publication date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Boundary at which scraped publication dates start. Pre-floor months
# use the flat-offset fallback below; on/after this date the calendar
# must match.
_SCRAPE_FLOOR = pd.Timestamp("2019-12-31")

# Conservative flat offset for pre-scrape-floor observations. Observed
# in-window lags ran 32-38 days; +40 buffers comfortably without
# overstating delay.
_FLAT_OFFSET_DAYS = 40


@dataclass(frozen=True)
class AbsBaSeries:
    """A single Building Approvals series fetched from the ABS Data API."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "1983-07") -> str:
        return (
            f"{ABS_API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


@dataclass(frozen=True)
class RbaH3Series:
    """A single RBA H3 series identified by its RBA series ID."""

    series_id: str
    rba_series_id: str


# Datakey is MEASURE.VALUE.SECTOR.WORK_TYPE.BUILDING_TYPE.TSEST.REGION.FREQ.
ABS_NSA_SERIES: tuple[AbsBaSeries, ...] = (
    AbsBaSeries(
        series_id="building_approvals_dwellings_total_nsa",
        dataflow="BA_GCCSA",
        datakey="1.1.9.TOT.100.10.AUS.M",
    ),
    AbsBaSeries(
        series_id="building_approvals_dwellings_houses_nsa",
        dataflow="BA_GCCSA",
        datakey="1.1.9.TOT.110.10.AUS.M",
    ),
    AbsBaSeries(
        series_id="building_approvals_dwellings_other_nsa",
        dataflow="BA_GCCSA",
        datakey="1.1.9.TOT.150.10.AUS.M",
    ),
    AbsBaSeries(
        series_id="building_approvals_value_residential_aud_thousand_nsa",
        dataflow="BA_GCCSA",
        datakey="2.1.9.TOT.100.10.AUS.M",
    ),
    AbsBaSeries(
        series_id="building_approvals_value_non_residential_aud_thousand_nsa",
        dataflow="BA_GCCSA",
        datakey="2.1.9.TOT.700.10.AUS.M",
    ),
)

RBA_H3_SERIES: tuple[RbaH3Series, ...] = (
    RbaH3Series(
        series_id="building_approvals_private_dwellings_sa",
        rba_series_id="GISPSDA",
    ),
    RbaH3Series(
        series_id="building_approvals_private_dwellings_trend",
        rba_series_id="GISDWPRITR",
    ),
)


def fetch(*, force_download: bool = True, start_period: str = "1983-07") -> pd.DataFrame:
    """Pull all NSA + SA / trend BA series, persist raw, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh from the live endpoints. If False,
        reuse the most recent dated snapshot per artifact and only
        download when no snapshot is present.
    start_period
        SDMX period string for ABS ``startPeriod`` (e.g. ``"1993-01"``).
        Default pulls the full ABS BA history from 1983.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, object]] = []

    for abs_series in ABS_NSA_SERIES:
        snapshot_path, entry = _resolve_abs_snapshot(
            dest_dir,
            series=abs_series,
            start_period=start_period,
            force_download=force_download,
        )
        df = _parse_sdmx(snapshot_path.read_bytes(), series_id=abs_series.series_id)
        entry["observations"] = len(df)
        manifest.append(entry)
        frames.append(df)

    # RBA H3 — one CSV file shared across both H3 series we extract.
    h3_path, h3_entry, h3_bytes = _resolve_h3_snapshot(
        dest_dir, force_download=force_download
    )
    h3_obs_total = 0
    for h3_series in RBA_H3_SERIES:
        df = _parse_h3(h3_bytes, spec=h3_series)
        h3_obs_total += len(df)
        frames.append(df)
    h3_entry["observations"] = h3_obs_total
    manifest.append(h3_entry)

    _write_manifest(dest_dir, manifest)

    out = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(out)
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach BA publication dates to every observation.

    Scraped calendar covers ``observation_date >= _SCRAPE_FLOOR``.
    Earlier observations use a flat ``+ _FLAT_OFFSET_DAYS`` offset.
    Raises ``ValueError`` if any observation on/after
    ``_CALENDAR_VALIDATION_FLOOR`` remains unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_ba_release_calendar()[["reference_month_end", "publication_date"]]
    merged = df.merge(
        calendar_df.rename(columns={"reference_month_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    pre_floor = merged["publication_date"].isna() & (
        merged["observation_date"] < _SCRAPE_FLOOR
    )
    merged.loc[pre_floor, "publication_date"] = merged.loc[
        pre_floor, "observation_date"
    ] + pd.Timedelta(days=_FLAT_OFFSET_DAYS)

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} Building Approvals observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date. Re-run `python -m rba.data.abs_ba_release_calendar` if "
            f"the scraped calendar is stale. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_abs_snapshot(
    dest_dir: Path,
    *,
    series: AbsBaSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    """Return (snapshot path, manifest entry) for one ABS NSA series."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing ABS BA snapshot at {}", path)
        entry: dict[str, object] = {
            "source": "abs_api",
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
    return _download_abs(dest_dir, series=series, start_period=start_period)


def _download_abs(
    dest_dir: Path,
    *,
    series: AbsBaSeries,
    start_period: str,
) -> tuple[Path, dict[str, object]]:
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{series.series_id}.json"
    url = series.url(start_period=start_period)
    logger.info("Downloading {} -> {}", url, snapshot_path)

    req = urllib.request.Request(url, headers={"Accept": "application/vnd.sdmx.data+json"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310 — public ABS URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "source": "abs_api",
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


def _resolve_h3_snapshot(
    dest_dir: Path, *, force_download: bool
) -> tuple[Path, dict[str, object], bytes]:
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__rba_h3.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing RBA H3 snapshot at {}", path)
        entry: dict[str, object] = {
            "source": "rba_h3",
            "url": RBA_H3_URL,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__rba_h3.csv"
    logger.info("Downloading {} -> {}", RBA_H3_URL, snapshot_path)
    with urllib.request.urlopen(RBA_H3_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)
    entry = {
        "source": "rba_h3",
        "url": RBA_H3_URL,
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


def _parse_sdmx(json_bytes: bytes, *, series_id: str) -> pd.DataFrame:
    """Parse one SDMX-JSON payload into a long-format frame.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-month),
        ``series_id`` (object), ``value`` (float64). One row per month.

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


def _parse_h3(csv_bytes: bytes, *, spec: RbaH3Series) -> pd.DataFrame:
    """Extract one series from the RBA H3 CSV by RBA series ID.

    Identical CSV layout to D1/D2: metadata rows 0-9 (Title / Description /
    Frequency / Type / Units / blanks / Source / Publication date /
    Series ID), data rows from row 10+. The "Title" header at row index 1
    is used as the column header.
    """
    raw_df = pd.read_csv(
        io.BytesIO(csv_bytes), header=1, low_memory=False, skipinitialspace=False
    )

    sid_row = raw_df[raw_df["Title"] == "Series ID"]
    if sid_row.empty:
        raise ValueError(
            f"No 'Series ID' metadata row found while extracting "
            f"{spec.rba_series_id!r} from RBA H3; CSV layout may have changed."
        )
    sid_mapping = sid_row.iloc[0].to_dict()

    target_col: str | None = None
    for col, rba_id in sid_mapping.items():
        if col == "Title":
            continue
        if rba_id == spec.rba_series_id:
            target_col = col
            break
    if target_col is None:
        raise ValueError(
            f"RBA series ID {spec.rba_series_id!r} not found among columns "
            f"in H3 CSV; series may have been renamed or removed."
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
            f"(RBA {spec.rba_series_id!r}); H3 CSV layout may have changed "
            f"or the column is empty in the current vintage."
        )

    return df[["observation_date", "series_id", "value"]]


def _month_end(period: str) -> pd.Timestamp:
    """Convert an ABS monthly period string (e.g. ``"2024-03"``) to the
    last calendar day of that month.
    """
    return pd.Period(period, freq="M").end_time.normalize().as_unit("ns")


def _to_month_end(dates: pd.Series) -> pd.Series:
    """Snap each timestamp to the last calendar day of its month."""
    offset = pd.offsets.MonthEnd()
    return pd.to_datetime(dates.map(offset.rollforward)).dt.as_unit("ns")
