"""RBA F1 — Bank Bill Swap Rate (BBSW) daily fixings.

Pulls daily closing BBSW fixings for the three tenors carried by RBA
Statistical Table F1 "Interest Rates and Yields — Money Market":

================  ==============  ===========================================
Series            RBA ID          Coverage (spliced: xls-hist + live CSV)
----------------  --------------  -------------------------------------------
BBSW 1-month      ``FIRMMBAB30D``   1995-01-03 → present
BBSW 3-month      ``FIRMMBAB90D``   1976-04-07 → present
BBSW 6-month      ``FIRMMBAB180D``  1995-01-03 → present
================  ==============  ===========================================

BBSW = Bank Bill Swap Rate, the AUD short-term benchmark rate (analogue
of the former LIBOR / EURIBOR — IBOR-family). The fixing is computed
from bank-accepted bill / negotiable certificate of deposit quotes
across the prime bank panel and published end of trading day.

F1 also carries the daily cash rate, OIS (1m/3m/6m) and Treasury Notes
(1m/3m/6m) under the same metadata layout. They are deliberately out of
scope here — daily cash-rate semantics differ from the event-driven F11
decision series we already ingest, and OIS / Treasury Notes can ship as
a separate ``rba_f1.py`` module later without touching this one.

Why this matters
----------------
The RBA cites BBSW (especially 3M) alongside the cash rate in every
SoMP money-market discussion, and the spread ``BBSW − cash`` is the
canonical bank-funding-stress signal (the wide blowouts in March 2020
and again across the 2022–23 hiking cycle were the most-cited
contemporaneous indicator of money-market dislocation). The 3m-1m and
6m-3m slopes proxy the front-end rate-expectations path.

Source format
-------------
F1 publishes at
``https://www.rba.gov.au/statistics/tables/csv/f1-data.csv`` using the
standard RBA statistical-table layout — Windows-1252 (cp1252) encoding,
``DD-MMM-YYYY`` date format, an en-dash in the title row (byte ``0x96``)
that breaks UTF-8 decoding.

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (``F1 INTEREST RATES AND YIELDS – MONEY MARKET``)
1           ``Title,Cash Rate Target,...,EOD 1-month BABs/NCDs,...``
2           ``Description,<descriptions>``
3           ``Frequency,Daily,...``
4           ``Type,Original,...``
5           ``Units,Per cent,...``
6-7         blanks
8           ``Source,RBA,...,ASX,ASX,ASX,...`` (was AFMA pre-2017)
9           ``Publication date,<refresh date, all cols>``
10          ``Series ID,FIRMMCRTD,...,FIRMMBAB30D,FIRMMBAB90D,FIRMMBAB180D,...``
11+         ``DD-MMM-YYYY,<observation values per series>``
==========  =====================================================

The current vintage F1 CSV starts **2011-01-04** only — to extend
coverage to the project's 1993 floor, the long-history daily archive
``f01dhist.xls`` (1976-04-07 → 2010-12-31, ~8.8k rows) under
``/statistics/tables/xls-hist/`` is spliced on. The XLS file uses the
same Title / Description / Frequency / Type / Units / blanks / Source /
Publication date / Series ID metadata layout as the live CSV, with the
date column read as ``datetime64`` rather than DD-MMM-YYYY strings
(xlrd parses Excel serial dates natively). Overlap at the splice
boundary is resolved by ``drop_duplicates(keep="last")`` so the
live-CSV vintage wins.

Per-tenor coverage starts differ: 3m BAB begins 1976-04-07 (the
earliest row in the historical archive); 1m and 6m both begin
1995-01-03 (pre-1995 cells are empty in the historical XLS and dropped
during parsing). Every series is fully populated through the splice
boundary at 2010-12-31 / 2011-01-04 with no gap.

Methodology / administrator vintage shifts (documented, not adjusted)
---------------------------------------------------------------------
The BBSW series has two documented vintage shifts that are reflected
in the F1 data as-published:

1. **Administrator change (Jan 2017)**: AFMA → ASX. The ``Source``
   metadata row of ``f01dhist.xls`` reports ``AFMA`` for the BBSW
   columns; the live CSV reports ``ASX``. The level series is
   continuous across the handover.
2. **Methodology change (May 2018)**: ASX moved BBSW from a panel-mean
   ("National Best Bid and Offer") fixing to a volume-weighted-average-
   price (VWAP) calculated from actual transacted trades during the
   morning rate-set window. This is a documented level break around
   2018-05-21 that we do **not** splice-adjust — the project policy is
   to retain RBA-published vintage data as-is (see CONTEXT.md
   "Vintage policy"). Downstream features (BBSW − cash spreads,
   1m/3m/6m slopes) absorb the break naturally; any model that needs
   the break flagged should add a regime dummy at the methodology
   date rather than restating the level.

Publication-date convention
---------------------------
F1 publishes the prior trading session's closing fixings the next
morning (around 11:30am AEST). Verified against 10 Wayback Machine
snapshots of ``f1-data.csv`` spanning 2018-01 → 2023-06: the
``Publication date`` header sits exactly 1 Australian business day
after the latest populated BBSW data row's date, every time. The
publication-date rule is therefore::

    publication_date = observation_date + 1 AU business day

— **identical** to the F2 / AGB yields convention. Rather than
duplicate the algorithmic next-business-day calendar, this source
reuses :mod:`rba.data.agb_yields_release_calendar` directly. The
``_OVERRIDES`` dict in that module governs both F1 and F2
publication dates; if the two ever diverge (no known case to date)
the calendar can be forked.

Verified samples
~~~~~~~~~~~~~~~~

==========================  =====================  ==========================  ==========
Wayback snapshot timestamp  Last BBSW-3m obs       ``Publication date`` header  BDay lag
--------------------------  ---------------------  --------------------------  ----------
2018-01-21 23:10 UTC         Thu 18-Jan-2018        Fri 19-Jan-2018             +1
2018-03-25 01:32 UTC         Thu 22-Mar-2018        Fri 23-Mar-2018             +1
2018-04-25 21:38 UTC         Mon 23-Apr-2018        Tue 24-Apr-2018             +1
2018-05-25 19:49 UTC         Thu 24-May-2018        Fri 25-May-2018             +1
2018-08-08 11:46 UTC         Tue 07-Aug-2018        Wed 08-Aug-2018             +1
2019-03-18 15:21 UTC         Fri 15-Mar-2019        Mon 18-Mar-2019             +1
2020-06-22 00:25 UTC         Fri 19-Jun-2020        Mon 22-Jun-2020             +1
2021-03-16 16:10 UTC         Mon 15-Mar-2021        Tue 16-Mar-2021             +1
2023-03-21 16:03 UTC         Mon 20-Mar-2023        Tue 21-Mar-2023             +1
2023-06-08 02:12 UTC         Wed 07-Jun-2023        Thu 08-Jun-2023             +1
==========================  =====================  ==========================  ==========

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release fixing. The RBA occasionally restates F1
historical rates when input quotes are corrected. Acknowledged
limitation; see CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (EOD fixing).
- ``publication_date`` (datetime64[ns]) — algorithmic F1 release date
  from ``rba.data.agb_yields_release_calendar`` (= trade date + 1 AU
  business day).
- ``series_id`` (object) — one of the 3 logical IDs listed in
  ``SERIES`` (``bbsw_1m``, ``bbsw_3m``, ``bbsw_6m``).
- ``value`` (float64) — fixing in per cent per annum (verbatim from
  F1, no unit conversion).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/bbsw_rates.csv`` keyed by ``trade_date`` with one
column per tenor plus the release date.

Side effects
------------
``fetch()`` writes under ``data/raw/bbsw_rates/``:

- ``<YYYY-MM-DD>__f1.csv`` — verbatim CSV bytes per refresh.
- ``<YYYY-MM-DD>__f01dhist.xls`` — verbatim XLS bytes for the
  historical archive.
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
from rba.data.agb_yields_release_calendar import build_agb_yields_release_calendar

SOURCE_NAME = "bbsw_rates"
CSV_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
HIST_XLS_URL = "https://www.rba.gov.au/statistics/tables/xls-hist/f01dhist.xls"

_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Coverage start of the spliced (xls-hist + live CSV) F1 BBSW series.
# 3m starts 1976-04-07 in xls-hist; 1m/6m start 1995-01-03. Used only
# to size the release calendar window in _attach_publication_dates.
_BBSW_COVERAGE_START = pd.Timestamp("1976-04-07")


@dataclass(frozen=True)
class BbswSeries:
    """One BBSW tenor identified by its RBA F1 series ID."""

    series_id: str
    rba_series_id: str
    tenor_months: int


SERIES: tuple[BbswSeries, ...] = (
    BbswSeries(series_id="bbsw_1m", rba_series_id="FIRMMBAB30D", tenor_months=1),
    BbswSeries(series_id="bbsw_3m", rba_series_id="FIRMMBAB90D", tenor_months=3),
    BbswSeries(series_id="bbsw_6m", rba_series_id="FIRMMBAB180D", tenor_months=6),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull F1 (live CSV + f01dhist.xls splice), extract every BBSW tenor.

    The historical XLS archive is concatenated with the live CSV and
    de-duplicated on ``(series_id, observation_date)``. Overlap at the
    splice boundary is resolved by keeping the live-CSV vintage (i.e.
    the most-recently-published RBA value wins). Each downloaded file
    is cached under ``data/raw/bbsw_rates/`` with a dated snapshot
    filename and a SHA-256 provenance entry in ``_metadata.json``.

    Parameters
    ----------
    force_download
        If True (default), refresh every file from the live RBA
        endpoints. If False, reuse the most recent dated snapshot per
        file and only download when none is present.

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

    snapshot_path, entry, raw_bytes = _resolve_snapshot(
        dest_dir, url=HIST_XLS_URL, force_download=force_download
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
    combined = combined.drop_duplicates(
        subset=["series_id", "observation_date"], keep="last"
    )
    combined = _attach_publication_dates(combined)
    return combined.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge algorithmic F1 release dates onto each observation.

    Reuses :func:`rba.data.agb_yields_release_calendar.build_agb_yields_release_calendar`
    directly — the F1 publication-date rule (verified via 10 Wayback
    samples) is identical to the F2 rule it implements. Defensively
    drops any pre-existing ``publication_date`` column before merging
    so a stale value cannot leak through. Raises ``ValueError`` if any
    observation on or after ``_CALENDAR_VALIDATION_FLOOR`` is unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    if df.empty:
        df = df.copy()
        df["publication_date"] = pd.Series(dtype="datetime64[ns]")
        return df[["observation_date", "publication_date", "series_id", "value"]]

    obs_min = df["observation_date"].min()
    obs_max = df["observation_date"].max()
    calendar_start = min(_BBSW_COVERAGE_START, pd.Timestamp(obs_min))
    calendar_end = pd.Timestamp(obs_max) + pd.Timedelta(days=14)
    calendar_df = build_agb_yields_release_calendar(
        start_date=calendar_start.date(),
        end_date=calendar_end.date(),
    )

    merged = df.merge(calendar_df, on="observation_date", how="left")

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} BBSW observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    url: str,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for one source file."""
    stem, suffix = _snapshot_suffix(url)
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__{stem}{suffix}"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing F1 snapshot at {}", path)
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
    raise ValueError(f"Unexpected URL basename in F1 source list: {basename!r}")


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


def _parse_csv(
    csv_bytes: bytes,
    *,
    spec: BbswSeries,
    missing_ok: bool = False,
) -> pd.DataFrame:
    """Extract one BBSW tenor from the live F1 CSV.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes from ``/statistics/tables/csv/f1-data.csv``.
        cp1252-encoded; ``DD-MMM-YYYY`` date format.
    spec
        ``BbswSeries`` identifying the target tenor.
    missing_ok
        If True, return an empty frame when the spec's column is absent
        (defensive). If False (default for live CSV), raise.

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
    spec: BbswSeries,
    missing_ok: bool = True,
) -> pd.DataFrame:
    """Extract one BBSW tenor from the historical F1 XLS archive.

    Parameters
    ----------
    xls_bytes
        Raw XLS bytes from ``/statistics/tables/xls-hist/f01dhist.xls``.
        Read via ``xlrd``; dates arrive as ``datetime`` objects (not
        DD-MMM-YYYY strings as in the live CSV).
    spec
        ``BbswSeries`` identifying the target tenor.
    missing_ok
        If True (default for the historical file), return an empty
        frame when the spec's column is absent. If False, raise.

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
    spec: BbswSeries,
    missing_ok: bool,
) -> pd.DataFrame:
    """Find the target column via the Series ID metadata row and emit rows.

    Shared by ``_parse_csv`` and ``_parse_xls``: both produce a DataFrame
    with ``Title`` as the leftmost column (carrying metadata-row labels
    and date values), and per-series columns to the right. Date values
    arrive as DD-MMM-YYYY strings (CSV) or ``Timestamp`` objects (XLS);
    ``pd.to_datetime`` handles both via a two-phase parse.
    """
    sid_row = raw_df[raw_df["Title"] == "Series ID"]
    if sid_row.empty:
        raise ValueError(
            "No 'Series ID' metadata row found while extracting "
            f"{spec.rba_series_id!r}; F1 layout may have changed."
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
            "in F1 input; series may have been renamed or removed."
        )

    parsed_dates = pd.to_datetime(
        raw_df["Title"], format="%d-%b-%Y", errors="coerce"
    )
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
        tenor in ``SERIES`` (only those present in the input), and
        ``release_date`` (datetime64[ns]) from the release calendar.

    Shapes
    ------
    Returns: (n_trade_days, 2 + n_tenors).
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
        ["trade_date"]
        + [c for c in _WIDE_COLUMN_ORDER if c in wide.columns]
        + ["release_date"]
    )
    return wide[column_order].sort_values("trade_date").reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "bbsw_rates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote BBSW rates ({} trade dates, {} tenors) to {}",
        len(wide_df),
        len([c for c in wide_df.columns if c.startswith("bbsw_")]),
        dest,
    )
