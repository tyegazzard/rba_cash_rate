"""ABS Total Value of Dwellings (Cat. 6432.0) — quarterly housing-market series.

Pulls headline housing-market measures from **two** ABS dataflows plus
the **discontinued** Residential Property Price Indexes (RPPI) used as a
splice baseline. The TVD release is the live successor to RPPI: ABS
discontinued RPPI after 2021-Q4 and TVD covers 2011-Q3 → present with a
different headline measure (mean / median levels rather than a
quality-adjusted price index).

National-level value and stock — dataflow ``RES_DWELL_ST``
----------------------------------------------------------
=========================================  ========================  =================================
Series                                     Datakey (MEASURE.REGION.FREQ)  Units
-----------------------------------------  ------------------------  ---------------------------------
``tvd_value_all_sectors_aud_million``      ``1.AUS.Q``               $ million (all sectors)
``tvd_dwelling_count_thousand``            ``4.AUS.Q``               '000 (residential dwellings)
``tvd_mean_price_aud_thousand``            ``5.AUS.Q``               $ thousand (mean dwelling price)
=========================================  ========================  =================================

Coverage: 2011-Q3 → present.

Per-capital-city medians and transfer counts — dataflow ``RES_DWELL``
---------------------------------------------------------------------
``RES_DWELL`` publishes by REGION = capital-city (``1GSYD`` ... ``8ACTE``)
and by rest-of-state. There is **no Australia-aggregate row** in this
dataflow (the AUS code in the codelist has no observations); the closest
nationally meaningful aggregation is the weighted-8-capital code
``REGION=100``, but it also returns no data for ``RES_DWELL``. We pull
all 8 capital cities separately and let the feature layer aggregate /
weight as needed.

Datakey order is ``MEASURE.REGION.FREQ``. For each capital city we
expose 4 series:

- ``housing_median_price_house_<city>_aud_thousand`` (MEASURE=3)
- ``housing_median_price_attached_<city>_aud_thousand`` (MEASURE=4)
- ``housing_transfer_count_house_<city>`` (MEASURE=1)
- ``housing_transfer_count_attached_<city>`` (MEASURE=2)

City suffixes match the capital-city ABS REGION codes:

============  ================  ===================
City suffix   REGION code       Greater capital area
------------  ----------------  -------------------
``sydney``    ``1GSYD``         Greater Sydney
``melbourne`` ``2GMEL``         Greater Melbourne
``brisbane``  ``3GBRI``         Greater Brisbane
``adelaide``  ``4GADE``         Greater Adelaide
``perth``     ``5GPER``         Greater Perth
``hobart``    ``6GHOB``         Greater Hobart
``darwin``    ``7GDAR``         Greater Darwin
``canberra``  ``8ACTE``         Australian Capital Territory
============  ================  ===================

⚠ **Methodological caveat — read before using.** Mean and median prices
in RES_DWELL_ST / RES_DWELL are **unstratified**: they reflect the
composition of sales in each quarter (apartment vs house mix, location
mix within capital), not a quality-adjusted price level. ABS warns
against interpreting movements as price changes. Composition shifts can
move the unstratified mean by 5-10% per quarter without any underlying
quality-adjusted price change. The discontinued RPPI hedonic index was
the proper price gauge; the spliced index below uses TVD mean growth
*as a proxy* after 2021-Q4 and inherits this limitation. Treat as a
noisier signal. The feature layer should not present the unstratified
mean as RPPI-quality.

Legacy RPPI baseline — dataflow ``RPPI`` (discontinued)
-------------------------------------------------------
The single RPPI series we pull is the weighted-8-capital all-residential
hedonic index:

==========================================  =====================================  =====================
Series                                      Datakey                                Coverage
------------------------------------------  -------------------------------------  ---------------------
``housing_price_index_rppi_legacy_8cap``    ``RPPI / 1.3.100.Q``                   2003-Q3 → 2021-Q4
==========================================  =====================================  =====================

Datakey order is ``MEASURE.PROPERTY_TYPE.REGION.FREQ``: MEASURE=1 (Index
numbers), PROPERTY_TYPE=3 (Residential property — all dwellings),
REGION=100 (Weighted average of eight capital cities). The series ends
at 2021-Q4 and does **not** update; we pull it once and rely on the
splice for any post-2021-Q4 housing-price signal.

Spliced index — ``housing_price_index_spliced`` (derived)
---------------------------------------------------------
A continuous quality-adjusted-then-mean-extrapolated series for the
2003-Q3 → present window. Formula:

- 2003-Q3 → 2021-Q4: ``spliced(t) = rppi(t)`` (raw RPPI index).
- 2022-Q1 → present: ``spliced(t) = rppi(2021-Q4) * tvd_mean(t) / tvd_mean(2021-Q4)``.

The pre-splice multiplier ``rppi(2021-Q4) / tvd_mean(2021-Q4)`` keeps
the level continuous across the boundary so YoY / QoQ growth derivations
match the underlying source on either side.

Splice validation: at construction we compare the QoQ growth rate
implied by RPPI at its last observation (2021-Q3 → 2021-Q4) against the
QoQ growth rate implied by TVD mean at the boundary (2021-Q4 → 2022-Q1).
``fetch()`` raises ``ValueError`` if ``|tvd_qoq - rppi_qoq| > 0.10``
(absolute percentage points), which would indicate a wrong base, a
unit mistake, or a TVD/RPPI methodology gap we do not currently model.
The check matches the user-specified guard in the design brief.

Publication-date convention
---------------------------
The TVD release has no clean algorithmic rule (Tuesday is consistent but
the *week* of M+3 alternates between 1st and 2nd Tuesday), so dates are
scraped by ``rba.data.abs_tvd_release_calendar`` and materialised to
``data/external/abs_tvd_release_dates.csv`` with one row per
(reference quarter, original release date) covering 2022-Q1 onward.

Pre-scrape-floor observations (TVD 2011-Q3 → 2021-Q4 and all RPPI rows)
fall back to a flat ``+ 76`` day offset from quarter-end. The worst
observed in-window TVD lag was 73 days (2024-Q4 ref → 2025-03-11
release, 70 days); 76 buffers safely. The flat-offset rows carry
``publication_date`` calculated as ``observation_date + 76 days``;
``_attach_publication_dates`` keeps these rows distinguishable by their
matched-vs-fallback origin only by inspection of the publication
calendar — the source-tagged provenance is in the materialised CSV, not
in the per-observation frame.

Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (1993-01-01) MUST
resolve to a publication date — either via the scraped calendar
(observation ≥ 2022-Q1) or via the flat-offset fallback (< 2022-Q1).
Unmatched in-window rows raise ``ValueError``. Pre-1993 observations
retain ``NaT`` (none are in scope here because RPPI starts 2003-Q3,
TVD starts 2011-Q3 — all in-window).

Vintage policy
--------------
Values are the **current ABS vintage** at the time of download — *not*
the original first-release value. TVD revises historical observations
as sales / valuation data are reconciled; RPPI was revised in its
publishing window too. We do not reconstruct prior vintages.
Acknowledged limitation under Invariant #1 in ``CONTEXT.md``.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference quarter.
- ``publication_date`` (datetime64[ns]) — scraped TVD release date for
  post-2022-Q1 observations; flat-offset fallback otherwise.
- ``series_id`` (object) — logical name (see SERIES + RPPI_SERIES +
  ``housing_price_index_spliced`` for the full set).
- ``value`` (float64) — the per-series unit (see Units columns above).

Side effects
------------
``fetch()`` writes under ``data/raw/abs_total_value_dwellings/``:

- ``<YYYY-MM-DD>__<series_id>.json`` — SDMX-JSON bytes per series.
- ``_metadata.json`` — provenance manifest with one entry per fetched
  series.
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
from rba.data.abs_tvd_release_calendar import build_tvd_release_calendar

SOURCE_NAME = "abs_total_value_dwellings"
ABS_API_BASE_URL = "https://data.api.abs.gov.au/rest/data"

# Inflation-targeting-era floor — any observation on/after this date
# must resolve to a publication date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Scrape floor used by abs_tvd_release_calendar. Pre-floor observations
# use the flat-offset fallback below.
_SCRAPE_FLOOR = pd.Timestamp("2022-03-31")

# Conservative flat offset for pre-scrape-floor observations. Observed
# in-window TVD lags ran 70-73 days; +76 buffers comfortably.
_FLAT_OFFSET_DAYS = 76

# Splice boundary — last RPPI quarter and first TVD quarter the spliced
# index uses TVD growth for.
_SPLICE_BOUNDARY = pd.Timestamp("2021-12-31")
_SPLICE_FIRST_TVD_QUARTER = pd.Timestamp("2022-03-31")

# Hard tolerance on the QoQ growth comparison at the splice boundary
# (absolute decimal — 0.10 = 10 percentage points).
_SPLICE_QOQ_TOLERANCE = 0.10

# Capital-city REGION codes used by RES_DWELL. Mapped to short logical
# suffixes for the per-city series names.
_CAPITAL_CITIES: tuple[tuple[str, str], ...] = (
    ("sydney", "1GSYD"),
    ("melbourne", "2GMEL"),
    ("brisbane", "3GBRI"),
    ("adelaide", "4GADE"),
    ("perth", "5GPER"),
    ("hobart", "6GHOB"),
    ("darwin", "7GDAR"),
    ("canberra", "8ACTE"),
)


@dataclass(frozen=True)
class AbsHousingSeries:
    """A single ABS housing series fetched from the ABS Data API."""

    series_id: str
    dataflow: str
    datakey: str

    def url(self, *, start_period: str = "2003-Q1") -> str:
        return (
            f"{ABS_API_BASE_URL}/{self.dataflow}/{self.datakey}"
            f"?startPeriod={start_period}"
            "&format=jsondata"
            "&dimensionAtObservation=AllDimensions"
        )


def _build_series_registry() -> tuple[AbsHousingSeries, ...]:
    """Build the full series registry: 3 national + 32 per-city + 1 RPPI legacy."""
    registry: list[AbsHousingSeries] = [
        AbsHousingSeries(
            series_id="tvd_value_all_sectors_aud_million",
            dataflow="RES_DWELL_ST",
            datakey="1.AUS.Q",
        ),
        AbsHousingSeries(
            series_id="tvd_dwelling_count_thousand",
            dataflow="RES_DWELL_ST",
            datakey="4.AUS.Q",
        ),
        AbsHousingSeries(
            series_id="tvd_mean_price_aud_thousand",
            dataflow="RES_DWELL_ST",
            datakey="5.AUS.Q",
        ),
    ]
    # 8 capitals × 4 measures = 32 RES_DWELL series.
    # MEASURE: 1 = house transfer count, 2 = attached transfer count,
    # 3 = median house price, 4 = median attached price.
    for city_suffix, region_code in _CAPITAL_CITIES:
        registry.append(
            AbsHousingSeries(
                series_id=f"housing_transfer_count_house_{city_suffix}",
                dataflow="RES_DWELL",
                datakey=f"1.{region_code}.Q",
            )
        )
        registry.append(
            AbsHousingSeries(
                series_id=f"housing_transfer_count_attached_{city_suffix}",
                dataflow="RES_DWELL",
                datakey=f"2.{region_code}.Q",
            )
        )
        registry.append(
            AbsHousingSeries(
                series_id=f"housing_median_price_house_{city_suffix}_aud_thousand",
                dataflow="RES_DWELL",
                datakey=f"3.{region_code}.Q",
            )
        )
        registry.append(
            AbsHousingSeries(
                series_id=f"housing_median_price_attached_{city_suffix}_aud_thousand",
                dataflow="RES_DWELL",
                datakey=f"4.{region_code}.Q",
            )
        )
    registry.append(
        AbsHousingSeries(
            series_id="housing_price_index_rppi_legacy_8cap",
            dataflow="RPPI",
            datakey="1.3.100.Q",
        )
    )
    return tuple(registry)


SERIES: tuple[AbsHousingSeries, ...] = _build_series_registry()

SPLICED_SERIES_ID = "housing_price_index_spliced"


def fetch(*, force_download: bool = True, start_period: str = "2003-Q1") -> pd.DataFrame:
    """Pull all TVD + RPPI series, build the spliced index, return long frame.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live API. If False,
        reuse the most recent dated snapshot per series.
    start_period
        SDMX period string for ``startPeriod``. Default ``"2003-Q1"`` covers
        the full RPPI / TVD inflation-targeting-relevant history.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4).

    Raises
    ------
    ValueError
        If the splice-boundary QoQ growth-rate gap exceeds
        ``_SPLICE_QOQ_TOLERANCE`` (likely indicates a wrong base or
        unit mismatch), or if any in-window observation cannot be
        matched to a publication date.
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
    spliced = _build_spliced_index(out)
    out = pd.concat([out, spliced], ignore_index=True)
    out = _attach_publication_dates(out)
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _build_spliced_index(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the spliced housing-price index from RPPI + TVD mean.

    See module docstring for the formula. Raises ``ValueError`` if the
    splice-boundary QoQ growth rates differ by more than
    ``_SPLICE_QOQ_TOLERANCE``.
    """
    rppi = (
        df[df["series_id"] == "housing_price_index_rppi_legacy_8cap"]
        .set_index("observation_date")["value"]
        .sort_index()
    )
    tvd_mean = (
        df[df["series_id"] == "tvd_mean_price_aud_thousand"]
        .set_index("observation_date")["value"]
        .sort_index()
    )

    if _SPLICE_BOUNDARY not in rppi.index or _SPLICE_BOUNDARY not in tvd_mean.index:
        raise ValueError(
            f"Cannot build spliced index: 2021-Q4 missing from RPPI "
            f"(present={_SPLICE_BOUNDARY in rppi.index}) or TVD mean "
            f"(present={_SPLICE_BOUNDARY in tvd_mean.index})."
        )

    prev_rppi_quarter = pd.Timestamp("2021-09-30")
    if prev_rppi_quarter not in rppi.index:
        raise ValueError(
            "Cannot validate splice: RPPI value for 2021-Q3 missing."
        )
    if _SPLICE_FIRST_TVD_QUARTER not in tvd_mean.index:
        raise ValueError(
            "Cannot validate splice: TVD mean value for 2022-Q1 missing."
        )

    rppi_qoq = rppi[_SPLICE_BOUNDARY] / rppi[prev_rppi_quarter] - 1.0
    tvd_qoq = tvd_mean[_SPLICE_FIRST_TVD_QUARTER] / tvd_mean[_SPLICE_BOUNDARY] - 1.0
    gap = abs(tvd_qoq - rppi_qoq)
    if gap > _SPLICE_QOQ_TOLERANCE:
        raise ValueError(
            f"Splice-boundary QoQ growth gap = {gap:.4f} exceeds tolerance "
            f"{_SPLICE_QOQ_TOLERANCE}. RPPI QoQ ({prev_rppi_quarter.date()} -> "
            f"{_SPLICE_BOUNDARY.date()}) = {rppi_qoq:+.4f}; TVD-mean QoQ "
            f"({_SPLICE_BOUNDARY.date()} -> {_SPLICE_FIRST_TVD_QUARTER.date()}) "
            f"= {tvd_qoq:+.4f}. Indicates a wrong base, unit mismatch, or "
            f"unmodelled methodology gap."
        )
    logger.info(
        "Spliced index QoQ check: RPPI={:.4f}, TVD={:.4f}, gap={:.4f} (tol {:.2f})",
        rppi_qoq,
        tvd_qoq,
        gap,
        _SPLICE_QOQ_TOLERANCE,
    )

    multiplier = rppi[_SPLICE_BOUNDARY] / tvd_mean[_SPLICE_BOUNDARY]
    tvd_extension = tvd_mean[tvd_mean.index > _SPLICE_BOUNDARY] * multiplier
    rppi_segment = rppi[rppi.index <= _SPLICE_BOUNDARY]
    spliced = pd.concat([rppi_segment, tvd_extension]).sort_index()

    return pd.DataFrame(
        {
            "observation_date": spliced.index,
            "series_id": SPLICED_SERIES_ID,
            "value": spliced.values,
        }
    )


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach TVD publication dates to every observation.

    Scraped calendar covers ``observation_date >= _SCRAPE_FLOOR``;
    earlier observations use a flat ``+ _FLAT_OFFSET_DAYS`` offset.
    Raises ``ValueError`` if any observation on/after
    ``_CALENDAR_VALIDATION_FLOOR`` remains unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_tvd_release_calendar()[
        ["reference_quarter_end", "publication_date"]
    ]
    merged = df.merge(
        calendar_df.rename(columns={"reference_quarter_end": "observation_date"}),
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
            f"{len(unmatched)} housing observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date. Re-run `python -m rba.data.abs_tvd_release_calendar` if "
            f"the scraped calendar is stale. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    series: AbsHousingSeries,
    start_period: str,
    force_download: bool,
) -> tuple[Path, dict[str, object]]:
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{series.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        logger.info("Reusing existing housing snapshot at {}", path)
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
    series: AbsHousingSeries,
    start_period: str,
) -> tuple[Path, dict[str, object]]:
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{series.series_id}.json"
    url = series.url(start_period=start_period)
    logger.info("Downloading {} -> {}", url, snapshot_path)

    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.sdmx.data+json"}
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

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-quarter),
        ``series_id`` (object), ``value`` (float64).

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
    """Convert an ABS quarterly period string (e.g. ``"2021-Q4"``) to the
    last calendar day of that quarter.
    """
    return pd.Period(period, freq="Q").end_time.normalize().as_unit("ns")
