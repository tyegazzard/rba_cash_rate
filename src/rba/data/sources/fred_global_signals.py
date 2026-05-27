"""FRED global signals — US CPI, Fed funds, US 10y, Fed broad USD, VIX.

Pulls six US / global macro series from the Federal Reserve Bank of St.
Louis FRED database via its unauthenticated CSV endpoint
(``fredgraph.csv?id=<SERIES_ID>``). This is the project's first
non-Australian macro source and the first to mix monthly and daily
frequencies in a single module.

================================  ================  ====================================
Logical series ID                 FRED series ID    Coverage (current vintage)
--------------------------------  ----------------  ------------------------------------
``us_headline_cpi`` (monthly SA)  ``CPIAUCSL``      1947-01 → present (BLS Cat. CUUR)
``us_core_cpi``     (monthly SA)  ``CPILFESL``      1957-01 → present (BLS, ex food/energy)
``us_fed_funds``    (daily)       ``DFF``           1954-07 → present (NY Fed effective FFR)
``us_10y_treasury`` (daily)       ``DGS10``         1962-01 → present (Treasury CMT 10y)
``us_dxy_broad``    (daily)       ``DTWEXBGS``      2006-01 → present (Fed Nominal Broad USD)
``us_vix``          (daily)       ``VIXCLS``        1990-01 → present (CBOE close)
================================  ================  ====================================

Why this matters
----------------
US CPI, Fed funds, and the US 10y are the three highest-citation
non-Australian series the RBA references in every Statement on
Monetary Policy. Fed-funds path and US-AU 10y spreads are core
inputs to the global-rate-cycle and US-spillover features; the
broad USD index proxies global USD strength (which mechanically
shifts the AUD-side of the TWI); and VIX is the canonical risk-off
indicator that lines up against AUD-USD widening and AGB-yield
flattening in turbulence episodes. None of these are derivable from
Australian data alone.

DXY substitution caveat
-----------------------
The licensed ICE Dollar Index (the actual "DXY" — a 6-currency basket
weighted to EUR/JPY/GBP/CAD/SEK/CHF) is **not** distributed via FRED.
The Fed's free analog **DTWEXBGS** ("Nominal Broad U.S. Dollar Index")
is shipped instead — a broader basket of ~26 currencies including
emerging markets, daily, 2006-01 onwards. Behaviour at quarterly /
monthly granularity is similar (correlation ≥ 0.95 historically) but
not identical, particularly during EM-stress episodes where the broad
index moves more than DXY. The substitution is intentional and
documented; if licensed DXY is wanted later, a separate paid source
module can be added. See CONTEXT.md "Excluded series" for the same
licensing logic applied elsewhere (CoreLogic dwelling prices).

Source format
-------------
FRED publishes simple 2-column CSVs::

    observation_date,<SERIES_ID>
    1947-01-01,21.480
    1947-02-01,21.620
    ...

- Header: ``observation_date,<SERIES_ID>`` — column 0 is literally
  named ``observation_date``, conveniently matching the project schema.
- Date format: ``YYYY-MM-DD``. Daily series carry every Mon-Fri (with
  blank cells on US federal holidays); monthly series carry every
  reference month at the **first day** of that month.
- Missing values: **empty string**. The FRED endpoint emits blanks on
  non-trading days for daily series (US federal holidays plus Good
  Friday for VIXCLS following CBOE closures) and for the
  currently-undefined trailing month of monthly series. ``pd.read_csv``
  decodes these as ``NaN`` and the parser drops them.
- Encoding: UTF-8. No metadata rows, no cp1252 quirks, no User-Agent
  challenge — much simpler than the RBA tables.

Schema convention — observation_date anchor
-------------------------------------------
FRED stores monthly observations at the **first day** of the
reference month (BLS native convention — CPI for Oct 2024 is keyed
2024-10-01). This module **converts to month-end** during parsing so
that ``observation_date`` matches the project-wide convention (period
end for monthly series — same anchor used by :mod:`rba.data.sources.abs_cpi`,
:mod:`rba.data.sources.abs_labour_force`,
:mod:`rba.data.sources.westpac_mi_consumer_sentiment`, etc.). Daily-series
``observation_date`` is the trade date, unchanged. The wide-format
monthly CSV is keyed by ``reference_month_end`` to make the convention
explicit on the cached artifact.

Publication-date convention
---------------------------
Per-observation publication dates come from
:mod:`rba.data.fred_release_calendar`. Three rules across the six
series:

- ``us_fed_funds`` / ``us_10y_treasury`` / ``us_dxy_broad``:
  ``+ 1 US business day`` (next Mon-Fri excluding US federal holidays).
- ``us_vix``: same day (CBOE publishes the VIX close 4:15pm ET).
- ``us_headline_cpi`` / ``us_core_cpi``: flat ``+ 20 calendar days``
  from end-of-reference-month (conservative — BLS typically releases
  CPI 10–15 days after period end; see the release-calendar module
  docstring for the rationale and upgrade path to ALFRED-derived
  first-release dates).

The validation floor is ``1993-01-01`` (inflation-targeting era start)
to match every other source module. Pre-floor observations still
carry algorithmically-computed publication dates — the floor is a
*validation* guard, not a *computation* floor. Pre-1993 rows are
emitted so they're available for any extended-history sanity-check
work, but the model's walk-forward CV consumes them from 1993 onwards
only.

Vintage policy
--------------
Stored values are the **current FRED vintage** at the time of
download — *not* the original first-release value. The BLS revises
seasonally-adjusted series on the annual seasonal-adjustment update
(typically released with the January CPI release each year); the Fed
restates effective FFR and Treasury yields on rare data-correction
events. Acknowledged limitation; see CONTEXT.md Invariant #1 vintage
policy. Upgrading to first-release vintages is tracked under the
scheduled-vintage-accumulation upgrade path in the same section.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (daily series)
  or reference month-end (monthly series).
- ``publication_date`` (datetime64[ns]) — algorithmic FRED release
  date from :mod:`rba.data.fred_release_calendar`.
- ``series_id`` (object) — one of the 6 logical IDs listed in
  ``SERIES`` (``us_headline_cpi`` / ``us_core_cpi`` / ``us_fed_funds``
  / ``us_10y_treasury`` / ``us_dxy_broad`` / ``us_vix``).
- ``value`` (float64) — value in upstream units. CPI series in index
  points (1982-84 = 100 for CPIAUCSL; ditto CPILFESL); DFF and DGS10 in
  per cent per annum; DTWEXBGS in index points (Jan 2006 = 100);
  VIXCLS in annualised volatility points. No unit conversion at the
  ingest layer.

Running this module as ``__main__`` materialises two wide-format CSVs
to ``data/external/``:

- ``fred_global_signals_daily.csv`` keyed by ``trade_date`` with one
  column per daily series plus ``release_date``.
- ``fred_global_signals_monthly.csv`` keyed by ``reference_month_end``
  with one column per monthly series plus ``release_date``.

Two files rather than one because a single mixed-frequency wide pivot
would have to forward-fill the monthly values onto every daily row
or leave them mostly NaN — neither is a faithful representation of
the cached vintage and the feature layer is the canonical place to
align across frequencies.

Side effects
------------
``fetch()`` writes under ``data/raw/fred_global_signals/``:

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

SOURCE_NAME = "fred_global_signals"
CSV_URL_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_series_id}"

_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class FredSeries:
    """One FRED series and its project-side metadata.

    ``frequency`` drives observation-date anchoring (month-end
    conversion for monthly, pass-through for daily) and the wide-CSV
    routing (daily series go to the daily wide file, monthly to the
    monthly wide file).
    """

    series_id: str
    fred_series_id: str
    frequency: str  # "daily" or "monthly"


SERIES: tuple[FredSeries, ...] = (
    FredSeries(series_id="us_headline_cpi", fred_series_id="CPIAUCSL", frequency="monthly"),
    FredSeries(series_id="us_core_cpi", fred_series_id="CPILFESL", frequency="monthly"),
    FredSeries(series_id="us_fed_funds", fred_series_id="DFF", frequency="daily"),
    FredSeries(series_id="us_10y_treasury", fred_series_id="DGS10", frequency="daily"),
    FredSeries(series_id="us_dxy_broad", fred_series_id="DTWEXBGS", frequency="daily"),
    FredSeries(series_id="us_vix", fred_series_id="VIXCLS", frequency="daily"),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull every FRED series, parse, attach publication dates.

    Each series is downloaded as an independent CSV (FRED has no
    multi-series bundle endpoint without an API key). Files are cached
    under ``data/raw/fred_global_signals/`` with a dated snapshot
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
    BDay, VIX same-day, or monthly CPI + 20 days). Raises
    ``ValueError`` if any observation on/after
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
            f"{len(unmatched)} FRED observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return stamped[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    spec: FredSeries,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one series."""
    url = CSV_URL_TEMPLATE.format(fred_series_id=spec.fred_series_id)
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{spec.series_id}.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing FRED snapshot at {}", path)
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
    dest_dir: Path, *, spec: FredSeries, url: str
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


def _parse_csv(csv_bytes: bytes, *, spec: FredSeries) -> pd.DataFrame:
    """Parse one FRED CSV and emit long-format rows.

    Drops blank-value rows (US holiday non-trading days for daily
    series; trailing reference months that haven't been published yet
    for monthly series). For monthly series, converts the FRED-native
    month-start anchor to month-end so observation_date matches the
    project-wide convention.

    Parameters
    ----------
    csv_bytes
        Raw bytes from ``fredgraph.csv?id=<spec.fred_series_id>``.
    spec
        ``FredSeries`` identifying the target series and its frequency.

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
_MONTHLY_SERIES_IDS: tuple[str, ...] = tuple(s.series_id for s in SERIES if s.frequency == "monthly")


def _to_wide_daily(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the daily slice of ``fetch()`` output to a wide CSV schema.

    Filters to daily-frequency series and pivots on ``observation_date``.
    Each daily series has independent trading-day coverage (DTWEXBGS
    starts 2006-01, DFF/DGS10 go back to mid-20th century, VIXCLS from
    1990), so pre-coverage rows are NaN for the late-starting series —
    expected, not an alignment bug.

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
    return _pivot_wide(
        monthly, column_order=_MONTHLY_SERIES_IDS, date_label="reference_month_end"
    )


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

    final_cols = (
        [date_label] + [c for c in column_order if c in wide.columns] + ["release_date"]
    )
    return wide[final_cols].sort_values(date_label).reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()

    daily_wide = _to_wide_daily(long_df)
    daily_dest = EXTERNAL_DATA_DIR / "fred_global_signals_daily.csv"
    daily_dest.parent.mkdir(parents=True, exist_ok=True)
    daily_wide.to_csv(daily_dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote FRED daily signals ({} trade dates, {} series) to {}",
        len(daily_wide),
        len(_DAILY_SERIES_IDS),
        daily_dest,
    )

    monthly_wide = _to_wide_monthly(long_df)
    monthly_dest = EXTERNAL_DATA_DIR / "fred_global_signals_monthly.csv"
    monthly_wide.to_csv(monthly_dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote FRED monthly signals ({} reference months, {} series) to {}",
        len(monthly_wide),
        len(_MONTHLY_SERIES_IDS),
        monthly_dest,
    )
