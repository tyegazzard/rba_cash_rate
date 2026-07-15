"""S&P/ASX 200 daily equity index and sector sub-indices.

Pulls daily end-of-day close prices for the S&P/ASX 200 headline index
plus four sector sub-indices the RBA cites alongside FX in the financial-
conditions block of the Statement on Monetary Policy:

================  ===============  =======================================
Series            Yahoo ticker     Coverage (Sydney trade date)
----------------  ---------------  ---------------------------------------
S&P/ASX 200       ``^AXJO``        1992-11-23 → present
Financials        ``^AXFJ``        2013-03-06 → present
Materials         ``^AXMJ``        2013-03-06 → present
Resources         ``^AXJR``        2013-03-08 → present
Energy            ``^AXEJ``        2013-03-06 → present
================  ===============  =======================================

The S&P/ASX 200 (XJO) officially launched on 2000-04-03 — pre-2000
values are S&P's back-extension to the same 200-stock methodology and
the index trades under the XJO ticker on the ASX. Yahoo carries the
back-extended series from 1992-11-23 (matching the start of the All
Ordinaries 4th-edition vintage). This is *not* spliced from the All
Ords; the All Ords (~500 stocks, broad market) is a different concept
and is intentionally not included as a pre-2013 backfill for the
Financials sub-index either — see the "Why no pre-2013 splice" note
below.

The four sector sub-indices all start 2013-03-06/08. Pre-2013 history
for the S&P sector cuts is not freely available; the ASX/S&P
methodology was reorganised in March 2013 (alignment with the GICS
classification refresh) so any pre-2013 sector index would be a
different cohort.

Why this matters
----------------
Equity prices are a forward-looking financial-conditions signal the
RBA references throughout the SoMP. The XJO captures broad market
risk-on/risk-off in Australia (correlates with credit spreads, term
premium, AUD); the Financials sub-index narrows that to the banks
(direct mortgage-book and credit-risk exposure, particularly relevant
during cash-rate cycles since the Big Four pass-through is a central
SoMP discussion point). Materials / Resources / Energy capture the
commodity-trade channel that is a key RBA theme for Australia's
terms-of-trade story.

Why Yahoo v8 chart JSON, not direct ASX endpoints?
--------------------------------------------------
ASX does not publish a free historical OHLCV feed. Probed alternatives
and outcomes (2026-05-25):

- ``asx.com.au/asx/1/share/XJO`` — returns a **single snapshot** only,
  no daily history.
- ``stooq.com/q/d/l/?s=^xjo&i=d`` — historically a free CSV, but as of
  2026 the endpoint is **captcha-gated and requires a per-user
  API key**. Adds a secret to manage; not viable for a checked-in
  pipeline.
- ``yfinance`` Python package — wraps the same Yahoo endpoint we hit
  here, but its newer versions use ``curl_cffi`` which bypasses the
  Windows certificate store, making it unusable on this machine
  without extra TLS workarounds. Adds heavy transitive dependencies
  for a single JSON endpoint.
- Yahoo Finance v8 chart JSON endpoint —
  ``query1.finance.yahoo.com/v8/finance/chart/<ticker>`` — public, no
  auth, returns one ~700 KB JSON per ticker for the full history.
  Reachable via plain ``urllib.request`` with the system SSL context
  (which uses the Windows cert store on this machine). One HTTP
  request per series; no third-party dependency.

Same trade-off shape as :mod:`rba.data.sources.asx_ib_futures` (which
also uses an unofficial-but-stable upstream because no free direct-ASX
historical feed exists for that series either).

Upstream schema
---------------
The Yahoo v8 chart endpoint returns JSON of the form::

    {
      "chart": {
        "result": [
          {
            "meta": {
              "symbol": "^AXJO",
              "timezone": "AEST",
              "gmtoffset": 36000,
              "firstTradeDate": <unix_seconds>,
              ...
            },
            "timestamp": [<unix_seconds>, ...],
            "indicators": {
              "quote": [
                {
                  "open": [...], "high": [...], "low": [...],
                  "close": [...], "volume": [...]
                }
              ]
            }
          }
        ],
        "error": null
      }
    }

Timestamps are **10:00 market-local** (Australia/Sydney) Unix seconds
per daily bar — *not* midnight UTC. During AEDT (October–April) the
10:00 Sydney bar timestamp converts to a *previous* UTC calendar date,
so interpreting the timestamps via UTC ``.date()`` is off by one day
for roughly half the year. Trade dates must therefore be derived via
``tz_convert('Australia/Sydney').floor('D').tz_localize(None)`` to
recover the wall-clock trade date. Verified against ASX market
calendars: 1992-11-23 (Mon) is the correct first trade date for ^AXJO
(22-Nov-1992 was a Sunday), and 2025-02-14 (Fri, AEDT) parses to the
correct trade date with this transformation.

Yahoo also emits ``null`` close values for ~100 rows per ticker — ASX
public-holiday timestamps where the index didn't trade. These are
dropped at parse time.

Why no pre-2013 splice for Financials
-------------------------------------
The All Ordinaries (^AORD) covers 1984-08-03 → present (10,700+ trade
days) and is the broad-market predecessor to the XJO. It could in
principle be substituted into the financials column for pre-2013
rows, but the All Ords is a ~500-stock broad-market index spanning
*every* sector, not a financials cut. Splicing it onto the Financials
sub-index would change the concept mid-series and introduce a level
break that is not faithful to either source. Same trade-off pattern
the project applied to the RPPI/TVD splice (which we *did* do because
both measure dwelling values via a comparable concept) and to the
F11.1 historical XLS splice (same indicative-rate methodology). Here
the methodologies are incompatible, so we accept the shorter coverage
and document the start date.

Publication-date convention
---------------------------
The ASX cash market closes at 16:00 AEST/AEDT after the closing-single-
price auction (CSPA). End-of-day index values are published to the
ASX/S&P index feed within ~15 minutes of close, and are publicly
available on Yahoo and every other index vendor by ~16:30 market-
local. The publication-date rule is therefore trivial::

    publication_date = observation_date

— same convention as :mod:`rba.data.sources.asx_ib_futures` (~6pm
AEST same-day) and :mod:`rba.data.sources.aud_exchange_rates` (4pm
AEST same-day). RBA cash-rate decisions are announced at 14:30 AEST,
so the trade-day-T close is not available at the time of any T-day
meeting but is fully available to any participant by midnight T
market-local. The alignment layer's strict ``publication_date <
meeting_date`` join enforces the conservative "strictly before" rule,
so no calendar module is needed for this source.

Vintage policy
--------------
Values are the **current vintage** at the time of download. S&P
occasionally restates historical index values (corporate-action
adjustments, methodology refinements). Acknowledged limitation; see
CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (ASX local).
- ``publication_date`` (datetime64[ns]) — same as ``observation_date``
  (see publication-date convention above).
- ``series_id`` (object) — one of the logical IDs in ``SERIES``
  (``xjo``, ``xfj``, ``xmj``, ``xjr``, ``xej``).
- ``value`` (float64) — closing index value (verbatim from Yahoo, no
  unit conversion). Index values are unitless points.

Running this module as ``__main__`` also materialises a wide-format
CSV at ``data/external/asx_200.csv`` keyed by ``trade_date`` with one
column per index plus the release date.

Side effects
------------
``fetch()`` writes under ``data/raw/asx_200/``:

- ``<YYYY-MM-DD>__<series_id>.json`` — verbatim JSON bytes per
  refresh, one file per series (five files: ``xjo`` → ``xej``).
- ``_metadata.json`` — provenance manifest: per-file URL, SHA-256,
  download timestamp UTC, byte count, observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

SOURCE_NAME = "asx_200"
YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")
_MARKET_TZ = "Australia/Sydney"


@dataclass(frozen=True)
class AsxIndexSeries:
    """One ASX index identified by its Yahoo ticker."""

    series_id: str
    yahoo_ticker: str
    name: str


SERIES: tuple[AsxIndexSeries, ...] = (
    AsxIndexSeries(series_id="xjo", yahoo_ticker="^AXJO", name="S&P/ASX 200"),
    AsxIndexSeries(series_id="xfj", yahoo_ticker="^AXFJ", name="S&P/ASX 200 Financials"),
    AsxIndexSeries(series_id="xmj", yahoo_ticker="^AXMJ", name="S&P/ASX 200 Materials"),
    AsxIndexSeries(series_id="xjr", yahoo_ticker="^AXJR", name="S&P/ASX 200 Resources"),
    AsxIndexSeries(series_id="xej", yahoo_ticker="^AXEJ", name="S&P/ASX 200 Energy"),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull every series in ``SERIES`` from the Yahoo v8 chart endpoint.

    One HTTPS request per ticker against
    ``query1.finance.yahoo.com/v8/finance/chart/<ticker>`` with
    ``period1=0&period2=9999999999&interval=1d`` for the full available
    history. Each response is cached as JSON under ``data/raw/asx_200/``
    with a dated snapshot filename and SHA-256 provenance entry.

    Parameters
    ----------
    force_download
        If True (default), refresh every series from Yahoo. If False,
        reuse the most recent dated snapshot per series and only
        download when none is present.

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
        df = _parse_chart_json(raw_bytes, spec=spec)
        entry["observations"] = int(len(df))
        manifest.append(entry)
        parsed_frames.append(df)

    _write_manifest(dest_dir, manifest)

    combined = pd.concat(parsed_frames, ignore_index=True)
    combined = _attach_publication_dates(combined)
    return combined.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp ``publication_date == observation_date`` per Invariant #1.

    ASX index EOD values are published by ~16:30 market-local after
    the cash-market close. Defensively drops any pre-existing
    ``publication_date`` column before stamping so a stale value
    cannot leak through.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    if df.empty:
        df = df.copy()
        df["publication_date"] = pd.Series(dtype="datetime64[ns]")
        return df[["observation_date", "publication_date", "series_id", "value"]]

    df = df.copy()
    df["publication_date"] = df["observation_date"]

    in_window = df["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = df[in_window & df["publication_date"].isna()]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} ASX 200 observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication date."
        )

    return df[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    spec: AsxIndexSeries,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one ticker."""
    pattern = f"[0-9][0-9][0-9][0-9]-??-??__{spec.series_id}.json"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing ASX 200 snapshot at {}", path)
        entry: dict[str, object] = {
            "series_id": spec.series_id,
            "yahoo_ticker": spec.yahoo_ticker,
            "url": _build_url(spec.yahoo_ticker),
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir, spec=spec)


def _download(dest_dir: Path, *, spec: AsxIndexSeries) -> tuple[Path, dict[str, object], bytes]:
    """Download one ticker's chart JSON, save dated snapshot, return entry."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__{spec.series_id}.json"
    url = _build_url(spec.yahoo_ticker)
    logger.info("Downloading {} -> {}", url, snapshot_path)

    request = urllib.request.Request(url, headers={"User-Agent": "rba-cash-rate/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 — public Yahoo URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "series_id": spec.series_id,
        "yahoo_ticker": spec.yahoo_ticker,
        "url": url,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry, raw_bytes


def _build_url(yahoo_ticker: str) -> str:
    """Build the Yahoo v8 chart URL for one ticker (full daily history)."""
    encoded = urllib.parse.quote(yahoo_ticker, safe="")
    return f"{YAHOO_BASE_URL}{encoded}?period1=0&period2=9999999999&interval=1d"


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


def _parse_chart_json(
    raw_bytes: bytes,
    *,
    spec: AsxIndexSeries,
) -> pd.DataFrame:
    """Extract one series from a Yahoo v8 chart JSON payload.

    Parameters
    ----------
    raw_bytes
        Raw response bytes from ``query1.finance.yahoo.com/v8/finance/
        chart/<ticker>``.
    spec
        ``AsxIndexSeries`` identifying the target series — used only for
        the ``series_id`` stamp and for diagnostic messages (the upstream
        response carries a single series).

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], Sydney trade
        date — see "Upstream schema" note in the module docstring),
        ``series_id`` (object), ``value`` (float64, closing index
        value). Rows with null close are dropped (ASX holidays).

    Shapes
    ------
    Returns: (n_trade_days, 3).
    """
    payload = json.loads(raw_bytes)
    chart = payload.get("chart", {})
    err = chart.get("error")
    if err is not None:
        raise ValueError(f"Yahoo chart endpoint returned error for {spec.yahoo_ticker!r}: {err}")
    results = chart.get("result")
    if not results:
        raise ValueError(
            f"Yahoo chart endpoint returned no results for {spec.yahoo_ticker!r}; "
            "ticker may have been delisted or renamed."
        )
    result = results[0]
    timestamps = result.get("timestamp", [])
    if not timestamps:
        raise ValueError(
            f"Yahoo chart endpoint returned zero timestamps for {spec.yahoo_ticker!r}; "
            "response shape may have changed."
        )

    quotes = result.get("indicators", {}).get("quote", [])
    if not quotes or "close" not in quotes[0]:
        raise ValueError(
            f"Yahoo chart endpoint missing close array for {spec.yahoo_ticker!r}; "
            "response shape may have changed."
        )
    close = quotes[0]["close"]
    if len(close) != len(timestamps):
        raise ValueError(
            f"Length mismatch for {spec.yahoo_ticker!r}: "
            f"{len(timestamps)} timestamps vs {len(close)} close values."
        )

    ts_utc = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True)
    trade_dates = ts_utc.dt.tz_convert(_MARKET_TZ).dt.floor("D").dt.tz_localize(None)

    df = pd.DataFrame(
        {
            "observation_date": trade_dates.astype("datetime64[ns]"),
            "series_id": spec.series_id,
            "value": pd.to_numeric(pd.Series(close), errors="coerce"),
        }
    )
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df[["observation_date", "series_id", "value"]]


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation (called from __main__)
# ---------------------------------------------------------------------------


_WIDE_COLUMN_ORDER: tuple[str, ...] = tuple(s.series_id for s in SERIES)


def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ``fetch()`` output to the wide CSV schema.

    Parameters
    ----------
    long_df
        Output of :func:`fetch`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``trade_date`` (datetime64[ns]), one float column per
        series in ``SERIES`` (only those present in the input), and
        ``release_date`` (datetime64[ns]) — equal to ``trade_date``
        under the same-day publication rule.

    Shapes
    ------
    Returns: (n_trade_days, 2 + n_series).
    """
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
            "observation_date": "trade_date",
            "publication_date": "release_date",
        }
    )

    column_order = (
        ["trade_date"] + [c for c in _WIDE_COLUMN_ORDER if c in wide.columns] + ["release_date"]
    )
    return wide[column_order].sort_values("trade_date").reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "asx_200.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    series_cols = [c for c in wide_df.columns if c in _WIDE_COLUMN_ORDER]
    logger.info(
        "Wrote ASX 200 ({} trade dates, {} series) to {}",
        len(wide_df),
        len(series_cols),
        dest,
    )
