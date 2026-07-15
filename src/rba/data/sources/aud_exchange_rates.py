"""RBA F11.1 — AUD exchange rates (daily).

Pulls daily AUD exchange rates against major counterparties plus the
Australian Dollar Trade-Weighted Index from RBA Statistical Table F11.1
"Exchange Rates — Daily".

================  ==============  ===========================================
Series            RBA ID          Coverage (spliced)
----------------  --------------  -------------------------------------------
AUD/USD           ``FXRUSD``      1991-01-02 → present
AUD TWI           ``FXRTWI``      1991-01-02 → present
AUD/JPY           ``FXRJY``       1991-01-02 → present
AUD/EUR           ``FXREUR``      1999-01-04 → present (euro inception)
AUD/GBP           ``FXRUKPS``     1991-01-02 → present
AUD/CNY           ``FXRCR``       1991-01-02 → present
AUD/NZD           ``FXRNZD``      1991-01-02 → present
================  ==============  ===========================================

The SoMP-core counterparty set (USD / TWI / JPY / EUR / GBP / CNY / NZD)
covers the seven currencies the RBA cites most often in the Statement on
Monetary Policy and in Governor speeches. F11.1 also publishes ~16 other
AUD bilaterals (KRW, SGD, INR, THB, TWD, MYR, IDR, VND, HKD, CAD, CHF,
PHP, PGK) plus the IMF Special Drawing Right (``FXRSDR``). Excluded from
v1:

- **AED and ZAR** — frozen in the live CSV at ``Publication
  date=01-Oct-2024``; the RBA no longer refreshes them. Adding a frozen
  series would mean every prediction after Oct 2024 carries the same
  stale FX value (same failure mode that dropped ABS Retail Trade — see
  CONTEXT.md "Excluded series"). If a future revision adds them back,
  re-include in ``SERIES`` and re-verify the publication-date header.
- **SDR** (``FXRSDR``) — IMF-sourced index, not a market exchange rate.
- **The other 13 AUD bilaterals** — out of scope for v1; trivial extension
  (add a tuple to ``SERIES``, re-verify coverage start).

Why this matters
----------------
AUD/USD and the AUD TWI are core macro inputs the RBA cites in every
Statement on Monetary Policy. A depreciating AUD lifts imported-goods
inflation and supports a tightening bias; an appreciating AUD pulls in
the opposite direction. The cross-rate against China (the largest
trading partner) and the broader SoMP basket complement the TWI for
trade-channel features.

Source format
-------------
F11.1 publishes at
``https://www.rba.gov.au/statistics/tables/csv/f11.1-data.csv`` using
the standard RBA statistical-table layout. Encoding is Windows-1252
(cp1252) — the file carries a UTF-8 BOM at byte 0 but the body uses
cp1252 (the ``F11.1  EXCHANGE RATES`` title row is ASCII-safe; later
quirks like the dash in source-note text use cp1252 bytes). Date format
is ``DD-MMM-YYYY`` (e.g. ``20-May-2026``), matching F2.

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (``F11.1  EXCHANGE RATES``)
1           ``Title,A$1=USD,Trade-weighted Index ...,A$1=CNY,...``
2           ``Description,<descriptions>``
3           ``Frequency,Daily,Daily,...``
4           ``Type,Indicative,Indicative,...``
5           ``Units,USD,Index,CNY,...``
6-7         blanks
8           ``Source,WM/Reuters,RBA,RBA,...``
9           ``Publication date,<refresh date, all cols>``
10          ``Series ID,FXRUSD,FXRTWI,FXRCR,...``
11+         ``DD-MMM-YYYY,<observation values per series>``
==========  =====================================================

The current vintage F11.1 CSV starts **2023-01-03** only — to extend
coverage to the project's 1993 floor, the long-history daily archive
under ``/statistics/tables/xls-hist/`` is spliced on. Eight XLS files
(``1991-1994.xls`` → ``2018-2022.xls``) cover the pre-2023 era. Each
XLS file uses the same Title / Description / Frequency / Type / Units /
blanks / Source / Publication date / Series ID metadata layout as the
live CSV, with the date column read as ``datetime64`` rather than
DD-MMM-YYYY strings (xlrd parses Excel serial dates natively). The
historical archive uses the same RBA-sourced indicative-rate
methodology as the current vintage — there is one documented source
change (from 1 July 2008 the AUD/USD column is the WM/Reuters 4.00pm
Sydney fix; prior to that it was the RBA's own observation of mid-points
of buying and selling rates around 4.00pm) but the level shift across
that boundary is negligible. Other currencies and the TWI remain on the
same RBA methodology throughout.

Per-currency coverage starts differ only for EUR: it begins 1999-01-04
(the euro replaced 11 EMU national currencies in Jan 1999); pre-1999
cells are empty in the historical XLS files and dropped during parsing.
USD / TWI / JPY / GBP / CNY / NZD all start 1991-01-02 (the earliest
XLS file's first row).

Publication-date convention
---------------------------
F11.1 publishes the daily indicative rates at 4.00pm AEST after the
Sydney market close. Verified against 5 Wayback Machine snapshots of
``f11.1-data.csv`` spanning 2023-03 → 2025-11: the ``Publication date``
header equals the latest data row's date exactly (zero business-day
lag), every time. The publication-date rule is therefore::

    publication_date = observation_date

— same convention as :mod:`rba.data.sources.asx_ib_futures`. Because
F11.1 rates are published end-of-day (4pm AEST) and RBA cash-rate
decisions are announced at 14:30 AEST, the trade-day-T closing rate is
not available at the time of any T-day meeting, but is fully available
to any participant by midnight T AEST. The alignment layer's strict
``publication_date < meeting_date`` join enforces the conservative
"strictly before" rule, so no calendar module is needed for this
source (unlike F2, which publishes the next morning and uses the
algorithmic ``rba.data.agb_yields_release_calendar``).

Verified samples
~~~~~~~~~~~~~~~~

==========================  =====================  ==========================  ==========
Wayback snapshot timestamp  Last observation date  ``Publication date`` header  BDay lag
--------------------------  ---------------------  --------------------------  ----------
2023-03-21 16:05 UTC         Tue 21-Mar-2023        Tue 21-Mar-2023             +0
2023-06-08 02:12 UTC         Wed 07-Jun-2023        Wed 07-Jun-2023             +0
2025-05-02 17:23 UTC         Fri 02-May-2025        Fri 02-May-2025             +0
2025-06-27 04:39 UTC         Thu 26-Jun-2025        Thu 26-Jun-2025             +0
2025-11-02 02:37 UTC         Fri 31-Oct-2025        Fri 31-Oct-2025             +0
==========================  =====================  ==========================  ==========

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release rate. The RBA occasionally restates F11
historical rates when input quotes are corrected. Acknowledged
limitation; see CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (4.00pm Sydney fix).
- ``publication_date`` (datetime64[ns]) — same as ``observation_date``
  (see publication-date convention above).
- ``series_id`` (object) — one of the logical IDs listed in ``SERIES``
  (``aud_usd``, ``aud_twi``, ``aud_jpy``, ``aud_eur``, ``aud_gbp``,
  ``aud_cny``, ``aud_nzd``).
- ``value`` (float64) — observed rate / index in the units listed in the
  table above (verbatim from RBA, no unit conversion).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/aud_exchange_rates.csv`` keyed by ``trade_date`` with
one column per currency plus the release date.

Side effects
------------
``fetch()`` writes under ``data/raw/aud_exchange_rates/``:

- ``<YYYY-MM-DD>__f11.1.csv`` — verbatim CSV bytes per refresh.
- ``<YYYY-MM-DD>__<range>.xls`` — verbatim XLS bytes for each historical
  archive (eight files: ``1991-1994`` → ``2018-2022``).
- ``_metadata.json`` — provenance manifest with one entry per file
  (URL, SHA-256, download timestamp UTC, byte count, observation count).
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

SOURCE_NAME = "aud_exchange_rates"
CSV_URL = "https://www.rba.gov.au/statistics/tables/csv/f11.1-data.csv"

HIST_XLS_URLS: tuple[str, ...] = (
    "https://www.rba.gov.au/statistics/tables/xls-hist/1991-1994.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/1995-1998.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/1999-2002.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/2003-2006.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/2007-2009.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/2010-2013.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/2014-2017.xls",
    "https://www.rba.gov.au/statistics/tables/xls-hist/2018-2022.xls",
)

_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class AudExchangeRateSeries:
    """One AUD bilateral / index identified by its RBA F11.1 series ID."""

    series_id: str
    rba_series_id: str
    counterparty: str


SERIES: tuple[AudExchangeRateSeries, ...] = (
    AudExchangeRateSeries(series_id="aud_usd", rba_series_id="FXRUSD", counterparty="USD"),
    AudExchangeRateSeries(series_id="aud_twi", rba_series_id="FXRTWI", counterparty="TWI"),
    AudExchangeRateSeries(series_id="aud_jpy", rba_series_id="FXRJY", counterparty="JPY"),
    AudExchangeRateSeries(series_id="aud_eur", rba_series_id="FXREUR", counterparty="EUR"),
    AudExchangeRateSeries(series_id="aud_gbp", rba_series_id="FXRUKPS", counterparty="GBP"),
    AudExchangeRateSeries(series_id="aud_cny", rba_series_id="FXRCR", counterparty="CNY"),
    AudExchangeRateSeries(series_id="aud_nzd", rba_series_id="FXRNZD", counterparty="NZD"),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull F11.1 (live CSV + historical XLS splice), extract every series.

    The eight dated XLS files under ``/statistics/tables/xls-hist/`` are
    concatenated with the live CSV and de-duplicated on
    ``(series_id, observation_date)``. Overlap at the splice boundary is
    resolved by keeping the live-CSV vintage (i.e. the most-recently-
    published RBA value wins). Each downloaded file is cached under
    ``data/raw/aud_exchange_rates/`` with a dated snapshot filename and a
    SHA-256 provenance entry in ``_metadata.json``.

    Parameters
    ----------
    force_download
        If True (default), refresh every file from the live RBA endpoints.
        If False, reuse the most recent dated snapshot per file and only
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

    for url in HIST_XLS_URLS:
        snapshot_path, entry, raw_bytes = _resolve_snapshot(
            dest_dir, url=url, force_download=force_download
        )
        file_obs = 0
        for spec in SERIES:
            df = _parse_xls(raw_bytes, spec=spec, missing_ok=True)
            if not df.empty:
                parsed_frames.append(df)
                file_obs += len(df)
        entry["observations"] = file_obs
        manifest.append(entry)

    snapshot_path, entry, raw_bytes = _resolve_snapshot(
        dest_dir, url=CSV_URL, force_download=force_download
    )
    file_obs = 0
    for spec in SERIES:
        df = _parse_csv(raw_bytes, spec=spec, missing_ok=False)
        parsed_frames.append(df)
        file_obs += len(df)
    entry["observations"] = file_obs
    manifest.append(entry)

    _write_manifest(dest_dir, manifest)

    combined = pd.concat(parsed_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["series_id", "observation_date"], keep="last")
    combined = _attach_publication_dates(combined)
    return combined.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Stamp ``publication_date == observation_date`` per Invariant #1.

    F11.1 publishes the daily indicative rates at 4.00pm AEST same-day
    (verified via Wayback samples — see module docstring). Defensively
    drops any pre-existing ``publication_date`` column before stamping so
    a stale value cannot leak through.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    if df.empty:
        df = df.copy()
        df["publication_date"] = pd.Series(dtype="datetime64[ns]")
        return df[["observation_date", "publication_date", "series_id", "value"]]

    df = df.copy()
    df["publication_date"] = df["observation_date"]
    return df[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    url: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one source file.

    Per-URL filename suffix is derived from the URL basename:
    ``f11.1-data.csv`` → ``__f11.1.csv``; ``2018-2022.xls`` →
    ``__2018-2022.xls``.
    """
    stem, suffix = _snapshot_suffix(url)
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{stem}{suffix}"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing F11.1 snapshot at {}", path)
        entry: dict[str, object] = {
            "table": stem,
            "url": url,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir, url=url)


def _download(dest_dir: Path, *, url: str) -> tuple[Path, dict[str, object], bytes]:
    """Download one source file, save dated snapshot, return path + entry + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    stem, suffix = _snapshot_suffix(url)
    snapshot_path = dest_dir / f"{today}__{stem}{suffix}"
    logger.info("Downloading {} -> {}", url, snapshot_path)

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": stem,
        "url": url,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry, raw_bytes


def _snapshot_suffix(url: str) -> tuple[str, str]:
    """Return (stem, extension) for the dated-snapshot filename of one URL."""
    basename = url.rsplit("/", 1)[-1]
    if basename.endswith("-data.csv"):
        return basename.removesuffix("-data.csv"), ".csv"
    if basename.endswith(".xls"):
        return basename.removesuffix(".xls"), ".xls"
    raise ValueError(f"Unexpected URL basename in F11.1 source list: {basename!r}")


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


def _parse_csv(
    csv_bytes: bytes,
    *,
    spec: AudExchangeRateSeries,
    missing_ok: bool = False,
) -> pd.DataFrame:
    """Extract one series from the live F11.1 CSV.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes from ``/statistics/tables/csv/f11.1-data.csv``.
        cp1252-encoded; ``DD-MMM-YYYY`` date format.
    spec
        ``AudExchangeRateSeries`` identifying the target series.
    missing_ok
        If True, return an empty frame when the spec's column is absent
        (used during historical splicing when a currency does not exist
        in older XLS files). If False (default for live CSV), raise.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns]),
        ``series_id`` (object), ``value`` (float64).

    Shapes
    ------
    Returns: (n_trade_days, 3).
    """
    raw_df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=1,
        low_memory=False,
        skipinitialspace=False,
        encoding="cp1252",
    )
    return _extract_series(raw_df, spec=spec, missing_ok=missing_ok)


def _parse_xls(
    xls_bytes: bytes,
    *,
    spec: AudExchangeRateSeries,
    missing_ok: bool = True,
) -> pd.DataFrame:
    """Extract one series from one historical F11.1 XLS archive.

    Parameters
    ----------
    xls_bytes
        Raw XLS bytes from ``/statistics/tables/xls-hist/<range>.xls``.
        Read via ``xlrd``; dates arrive as ``datetime`` objects (not
        DD-MMM-YYYY strings as in the live CSV).
    spec
        ``AudExchangeRateSeries`` identifying the target series.
    missing_ok
        If True (default for historical files), return an empty frame
        when the spec's column is absent in this file. If False, raise.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns]),
        ``series_id`` (object), ``value`` (float64).

    Shapes
    ------
    Returns: (n_trade_days_in_archive, 3).
    """
    raw_df = pd.read_excel(
        io.BytesIO(xls_bytes),
        sheet_name="Data",
        header=1,
        engine="xlrd",
    )
    return _extract_series(raw_df, spec=spec, missing_ok=missing_ok)


def _extract_series(
    raw_df: pd.DataFrame,
    *,
    spec: AudExchangeRateSeries,
    missing_ok: bool,
) -> pd.DataFrame:
    """Find the target column via the Series ID metadata row and emit rows.

    Shared by ``_parse_csv`` and ``_parse_xls``: both produce a DataFrame
    with ``Title`` as the leftmost column (carrying metadata-row labels
    and date values), and per-counterparty columns to the right. Date
    values arrive as DD-MMM-YYYY strings (CSV) or ``Timestamp`` objects
    (XLS); ``pd.to_datetime`` handles both via a two-phase parse.
    """
    sid_row = raw_df[raw_df["Title"] == "Series ID"]
    if sid_row.empty:
        raise ValueError(
            "No 'Series ID' metadata row found while extracting "
            f"{spec.rba_series_id!r}; F11.1 layout may have changed."
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
        if missing_ok:
            return pd.DataFrame(
                {
                    "observation_date": pd.Series(dtype="datetime64[ns]"),
                    "series_id": pd.Series(dtype="object"),
                    "value": pd.Series(dtype="float64"),
                }
            )
        raise ValueError(
            f"RBA series ID {spec.rba_series_id!r} not found among columns "
            "in F11.1 input; series may have been renamed or removed."
        )

    parsed_dates = pd.to_datetime(raw_df["Title"], format="%d-%b-%Y", errors="coerce")
    if parsed_dates.notna().sum() == 0:
        parsed_dates = pd.to_datetime(raw_df["Title"], errors="coerce")
    data_mask = parsed_dates.notna()

    df = pd.DataFrame(
        {
            "observation_date": parsed_dates[data_mask].dt.as_unit("ns").to_numpy(),
            "series_id": spec.series_id,
            "value": pd.to_numeric(raw_df.loc[data_mask, target_col], errors="coerce"),
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
        currency in ``SERIES`` (only those present in the input), and
        ``release_date`` (datetime64[ns]) — equal to ``trade_date``
        under the same-day publication rule.

    Shapes
    ------
    Returns: (n_trade_days, 2 + n_currencies).
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
    dest = EXTERNAL_DATA_DIR / "aud_exchange_rates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote AUD exchange rates ({} trade dates, {} series) to {}",
        len(wide_df),
        len([c for c in wide_df.columns if c.startswith("aud_")]),
        dest,
    )
