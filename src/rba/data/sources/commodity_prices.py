"""Commodity prices — iron ore, copper, Brent crude, WTI crude.

Four global commodity series the RBA cites alongside its own I2
Commodity Prices Index in every Statement on Monetary Policy. Iron
ore in particular is Australia's largest single export by value
(>20% of goods exports), so the spot price drives terms-of-trade,
mining-sector capex, and the AUD via the export-revenue channel.
Copper proxies global industrial demand (especially China); Brent /
WTI carry world oil supply / demand. All four pull from the same
FRED unauthenticated CSV endpoint as
:mod:`rba.data.sources.fred_global_signals` — no new infrastructure
needed.

==============================  ================  =====================================
Logical series ID               FRED series ID    Coverage (current vintage)
------------------------------  ----------------  -------------------------------------
``iron_ore_spot`` (monthly)     ``PIORECRUSDM``   1992-01 → present (IMF, USD/dry MT)
``copper_spot``   (monthly)     ``PCOPPUSDM``     1992-01 → present (IMF, USD/MT, LME-A)
``brent_crude``   (daily)       ``DCOILBRENTEU``  1987-05 → present (EIA, USD/bbl)
``wti_crude``     (daily)       ``DCOILWTICO``    1986-01 → present (EIA, USD/bbl)
==============================  ================  =====================================

Why this matters
----------------
Iron ore prices are *the* terms-of-trade driver for Australia — a
$10/MT iron-ore move shows up in AUD/USD, mining-investment
expectations, and federal-budget revenue forecasts. The RBA tracks
the spot price (not just its own I2 index, which is a basket
aggregate) because moves in the largest export commodity are visible
in monetary-policy outcomes faster than the I2 aggregation reflects
them. Copper is the canonical "Dr Copper" cyclical-demand proxy
(particularly China-construction-and-grid exposure). Brent is the
global oil benchmark AU export crudes (Cossack, Vincent) price off;
the Brent–WTI spread captures world-vs-US supply differentials and
is a useful spread feature for energy-cycle features.

Source choice rationale
-----------------------
- **Iron ore (monthly)**: IMF Primary Commodity Prices on FRED. The
  market-watched series (Singapore SGX 62% Fe futures) is daily but
  licensed; the IMF China-import spot proxy is monthly only.
  Acceptable trade-off because terms-of-trade features evaluate at
  monthly granularity anyway. The IMF series is the canonical "free"
  iron-ore reference (cited by the RBA, AOFM, and Treasury).
- **Copper (monthly)**: IMF Primary Commodity Prices on FRED. LME-A
  cathode price daily exists but requires a licensed LME data feed;
  COMEX HG=F (CME futures) is a different basis (HG is high-grade,
  US-warehouse-delivered) and would silently change the conceptual
  series. The IMF monthly average is the safe free-data choice.
- **Brent + WTI (daily)**: EIA spot prices. Free, daily, redistributed
  through FRED, full history back to 1986/1987. No licensing concern.

Source format
-------------
FRED publishes simple 2-column CSVs identical to the layout the
:mod:`rba.data.sources.fred_global_signals` parser handles::

    observation_date,<SERIES_ID>
    1992-01-01,14.31000000000000
    1992-02-01,14.31000000000000
    ...

- Header: ``observation_date,<SERIES_ID>``.
- Date format: ``YYYY-MM-DD``.
- Missing values: **empty string** (Brent and WTI emit blanks on US
  federal holidays + a handful of CBOE-style oil-market closures;
  monthly IMF series carry no blanks because the upstream is a
  monthly average, not a point sample).
- Encoding: UTF-8. No metadata rows, no User-Agent challenge.

Schema convention — observation_date anchor
-------------------------------------------
Same as :mod:`rba.data.sources.fred_global_signals`. Daily-series
``observation_date`` is the trade date, unchanged. Monthly-series
``observation_date`` is **converted from FRED-native month-start to
month-end** during parsing so it matches the project-wide period-end
convention.

Publication-date convention
---------------------------
Per-observation publication dates come from the shared
:mod:`rba.data.fred_release_calendar`. Two rules across the four
series:

- ``brent_crude`` / ``wti_crude``: ``us_daily_t_plus_1`` — next US
  business day after the trade date (EIA publishes the prior session's
  closes the next US business morning).
- ``iron_ore_spot`` / ``copper_spot``: ``us_monthly_flat`` — flat
  ``+ 20 calendar days`` from end-of-reference-month. IMF Primary
  Commodity Prices typically publish in the first week of the
  following month (~5-day lag), so +20 is conservatively wide; see the
  release-calendar module docstring for the shared rationale (the same
  rule serves BLS CPI).

The validation floor is ``1993-01-01`` (inflation-targeting era start)
to match every other source module. Pre-floor observations still
carry algorithmically-computed publication dates — the floor is a
*validation* guard, not a *computation* floor.

Vintage policy
--------------
Stored values are the **current FRED vintage** at the time of
download — *not* the original first-release value. The IMF restates
historical commodity series ~annually on benchmark revisions (weight
updates, new contract-month reference); the EIA restates Brent / WTI
spot on rare data-correction events. Acknowledged limitation; see
CONTEXT.md Invariant #1 vintage policy. The upgrade path to scheduled
vintage accumulation is shared with every other FRED-fed source.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (daily series)
  or reference month-end (monthly series).
- ``publication_date`` (datetime64[ns]) — algorithmic FRED release
  date from :mod:`rba.data.fred_release_calendar`.
- ``series_id`` (object) — one of the 4 logical IDs listed in
  ``SERIES``.
- ``value`` (float64) — value in upstream units. Iron ore in USD per
  dry metric ton (62% Fe, China import spot); copper in USD per
  metric ton (LME-A cathode); Brent / WTI in USD per barrel. No unit
  conversion at the ingest layer.

Running this module as ``__main__`` materialises two wide-format CSVs
to ``data/external/``, mirroring the fred_global_signals split:

- ``commodity_prices_daily.csv`` keyed by ``trade_date`` with one
  column per daily series plus ``release_date``.
- ``commodity_prices_monthly.csv`` keyed by ``reference_month_end``
  with one column per monthly series plus ``release_date``.

Two files rather than one because a single mixed-frequency wide pivot
would have to forward-fill the monthly values onto every daily row
or leave them mostly NaN — neither is a faithful representation of
the cached vintage and the feature layer is the canonical place to
align across frequencies.

Side effects
------------
``fetch()`` writes under ``data/raw/commodity_prices/``:

- ``<YYYY-MM-DD>__<series_id>.csv`` — verbatim CSV bytes per series.
- ``_metadata.json`` — provenance manifest with one entry per series
  (URL, SHA-256, download timestamp UTC, byte count, observation
  count).
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

from rba.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR
from rba.data.fred_release_calendar import attach_fred_publication_dates

SOURCE_NAME = "commodity_prices"
CSV_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_series_id}"

_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class CommoditySeries:
    """One FRED commodity series and its project-side metadata.

    ``frequency`` drives observation-date anchoring (month-end
    conversion for monthly, pass-through for daily) and the wide-CSV
    routing (daily series go to the daily wide file, monthly to the
    monthly wide file).
    """

    series_id: str
    fred_series_id: str
    frequency: str  # "daily" or "monthly"


SERIES: tuple[CommoditySeries, ...] = (
    CommoditySeries(series_id="iron_ore_spot", fred_series_id="PIORECRUSDM", frequency="monthly"),
    CommoditySeries(series_id="copper_spot", fred_series_id="PCOPPUSDM", frequency="monthly"),
    CommoditySeries(series_id="brent_crude", fred_series_id="DCOILBRENTEU", frequency="daily"),
    CommoditySeries(series_id="wti_crude", fred_series_id="DCOILWTICO", frequency="daily"),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull every commodity series, parse, attach publication dates.

    Each series is downloaded as an independent CSV (FRED has no
    multi-series bundle endpoint without an API key). Files are cached
    under ``data/raw/commodity_prices/`` with a dated snapshot
    filename and a SHA-256 provenance entry in ``_metadata.json``.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from the live FRED
        endpoint. If False, reuse the most recent dated snapshot per
        series and only download when no snapshot is present.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by
        ``(series_id, observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) with one row per (series_id, observation_date).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    parsed_frames: list[pd.DataFrame] = []

    for spec in SERIES:
        snapshot_path, entry, raw_bytes = _resolve_snapshot(
            dest_dir, spec=spec, force_download=force_download
        )
        df = _parse_csv(raw_bytes, spec=spec)
        entry["observations"] = len(df)
        manifest.append(entry)
        parsed_frames.append(df)

    _write_manifest(dest_dir, manifest)

    combined = pd.concat(parsed_frames, ignore_index=True)
    combined = _attach_publication_dates(combined)
    return combined.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp algorithmic publication dates via the FRED release calendar.

    Delegates to :func:`rba.data.fred_release_calendar.attach_fred_publication_dates`,
    which dispatches per-series to the correct rule (daily T+1 US
    BDay for Brent / WTI, monthly + 20 days for iron ore / copper).
    Raises ``ValueError`` if any observation on/after
    ``_CALENDAR_VALIDATION_FLOOR`` is left without a publication date —
    which would only happen if the calendar's ``_RULES`` mapping drifts
    out of sync with this module's ``SERIES`` registry.
    """
    if df.empty:
        out = df.drop(columns=["publication_date"], errors="ignore").copy()
        out["publication_date"] = pd.Series(dtype="datetime64[ns]")
        return out[["observation_date", "publication_date", "series_id", "value"]]

    stamped = attach_fred_publication_dates(df)

    in_window = stamped["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = stamped[in_window & stamped["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} commodity observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return stamped[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    spec: CommoditySeries,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one series."""
    url = CSV_URL_TEMPLATE.format(fred_series_id=spec.fred_series_id)
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{spec.series_id}.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing commodity snapshot at {}", path)
        entry: dict[str, object] = {
            "series_id": spec.series_id,
            "fred_series_id": spec.fred_series_id,
            "url": url,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir, spec=spec, url=url)


def _download(
    dest_dir: Path, *, spec: CommoditySeries, url: str
) -> tuple[Path, dict[str, object], bytes]:
    """Download one series CSV, save a dated snapshot, return path + entry + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{spec.series_id}.csv"
    logger.info("Downloading {} -> {}", url, snapshot_path)

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — public FRED URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "series_id": spec.series_id,
        "fred_series_id": spec.fred_series_id,
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


def _parse_csv(csv_bytes: bytes, *, spec: CommoditySeries) -> pd.DataFrame:
    """Parse one FRED CSV and emit long-format rows.

    Drops blank-value rows (US holiday non-trading days for daily
    Brent / WTI; the monthly IMF series typically have no blanks but
    the same drop is applied defensively in case FRED revisions
    introduce a trailing blank). For monthly series, converts the
    FRED-native month-start anchor to month-end so observation_date
    matches the project-wide convention.

    Parameters
    ----------
    csv_bytes
        Raw bytes from ``fredgraph.csv?id=<spec.fred_series_id>``.
    spec
        ``CommoditySeries`` identifying the target series and its
        frequency.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns]),
        ``series_id`` (object), ``value`` (float64).

    Shapes
    ------
    Returns: (n_non_null_obs, 3).
    """
    raw_df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

    if "observation_date" not in raw_df.columns:
        raise ValueError(
            f"FRED CSV for {spec.fred_series_id!r} missing expected "
            "'observation_date' column; layout may have changed. Got "
            f"columns: {list(raw_df.columns)}"
        )
    if spec.fred_series_id not in raw_df.columns:
        raise ValueError(
            f"FRED CSV for {spec.fred_series_id!r} missing the value "
            f"column {spec.fred_series_id!r}; layout may have changed. "
            f"Got columns: {list(raw_df.columns)}"
        )

    obs_dates = pd.to_datetime(raw_df["observation_date"], errors="coerce")
    values = pd.to_numeric(raw_df[spec.fred_series_id], errors="coerce")

    keep = obs_dates.notna() & values.notna()

    obs_series = obs_dates[keep].dt.as_unit("ns").reset_index(drop=True)
    val_series = values[keep].astype("float64").reset_index(drop=True)

    if spec.frequency == "monthly":
        obs_series = (obs_series + pd.offsets.MonthEnd(0)).dt.as_unit("ns")
    elif spec.frequency != "daily":
        raise ValueError(
            f"Unknown frequency {spec.frequency!r} for series_id "
            f"{spec.series_id!r}; expected 'daily' or 'monthly'."
        )

    df = pd.DataFrame(
        {
            "observation_date": obs_series,
            "series_id": spec.series_id,
            "value": val_series,
        }
    )

    if df.empty:
        raise ValueError(
            f"Parsed zero observations for {spec.series_id!r} "
            f"(FRED {spec.fred_series_id!r}); CSV layout may have changed "
            "or the series may be empty."
        )

    return df[["observation_date", "series_id", "value"]]


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation — split daily / monthly (called from __main__)
# ---------------------------------------------------------------------------


_DAILY_SERIES_IDS: tuple[str, ...] = tuple(s.series_id for s in SERIES if s.frequency == "daily")
_MONTHLY_SERIES_IDS: tuple[str, ...] = tuple(
    s.series_id for s in SERIES if s.frequency == "monthly"
)


def _to_wide_daily(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the daily slice of ``fetch()`` output to a wide CSV schema.

    Filters to daily-frequency series and pivots on ``observation_date``.
    Brent (DCOILBRENTEU) starts 1987-05 and WTI (DCOILWTICO) starts
    1986-01, so the union daily index runs from 1986-01 with Brent NaN
    until 1987-05 — expected, not an alignment bug.

    Parameters
    ----------
    long_df
        Output of :func:`fetch`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``trade_date`` (datetime64[ns]), one float column per
        daily series in ``SERIES``, and ``release_date`` (datetime64[ns]).

    Shapes
    ------
    Returns: (n_trade_days, 2 + n_daily_series).
    """
    daily = long_df[long_df["series_id"].isin(_DAILY_SERIES_IDS)]
    return _pivot_wide(daily, column_order=_DAILY_SERIES_IDS, date_label="trade_date")


def _to_wide_monthly(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the monthly slice of ``fetch()`` output to a wide CSV schema.

    Parameters
    ----------
    long_df
        Output of :func:`fetch`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``reference_month_end`` (datetime64[ns]), one float
        column per monthly series in ``SERIES``, and ``release_date``
        (datetime64[ns]).

    Shapes
    ------
    Returns: (n_reference_months, 2 + n_monthly_series).
    """
    monthly = long_df[long_df["series_id"].isin(_MONTHLY_SERIES_IDS)]
    return _pivot_wide(monthly, column_order=_MONTHLY_SERIES_IDS, date_label="reference_month_end")


def _pivot_wide(
    long_df: pd.DataFrame,
    *,
    column_order: tuple[str, ...],
    date_label: str,
) -> pd.DataFrame:
    """Shared pivot helper for both daily and monthly wide outputs."""
    if long_df.empty:
        empty_cols = [date_label, *column_order, "release_date"]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in empty_cols}).astype(
            {
                date_label: "datetime64[ns]",
                "release_date": "datetime64[ns]",
            }
        )

    wide = long_df.pivot_table(
        index="observation_date",
        columns="series_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    pub_dates = long_df.groupby("observation_date")["publication_date"].first().reset_index()
    wide = wide.merge(pub_dates, on="observation_date", how="left")
    wide = wide.rename(
        columns={
            "observation_date": date_label,
            "publication_date": "release_date",
        }
    )

    final_cols = [date_label] + [c for c in column_order if c in wide.columns] + ["release_date"]
    return wide[final_cols].sort_values(date_label).reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()

    daily_wide = _to_wide_daily(long_df)
    daily_dest = EXTERNAL_DATA_DIR / "commodity_prices_daily.csv"
    daily_dest.parent.mkdir(parents=True, exist_ok=True)
    daily_wide.to_csv(daily_dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote commodity daily prices ({} trade dates, {} series) to {}",
        len(daily_wide),
        len(_DAILY_SERIES_IDS),
        daily_dest,
    )

    monthly_wide = _to_wide_monthly(long_df)
    monthly_dest = EXTERNAL_DATA_DIR / "commodity_prices_monthly.csv"
    monthly_wide.to_csv(monthly_dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote commodity monthly prices ({} reference months, {} series) to {}",
        len(monthly_wide),
        len(_MONTHLY_SERIES_IDS),
        monthly_dest,
    )
