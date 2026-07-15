"""RBA F2 — Australian Government Bond yields (ACGB).

Pulls daily closing yields for benchmark Australian Government Bond
maturities from RBA Statistical Table F2 "Capital Market Yields —
Government Bonds".

================  ============  ==========================================
Series            RBA ID        Coverage (current vintage)
----------------  ------------  ------------------------------------------
ACGB 2-year       ``FCMYGBAG2D``  2013-09-02 → present
ACGB 3-year       ``FCMYGBAG3D``  2013-09-02 → present
ACGB 5-year       ``FCMYGBAG5D``  2013-09-02 → present
ACGB 10-year      ``FCMYGBAG10D`` 2013-05-20 → present
================  ============  ==========================================

F2 also publishes an inflation-linked 10-year yield (``FCMYGBAGID``) and
prior to 2018-ish carried NSW Treasury Corporation 3y/5y/10y series.
Neither is included in v1: the indexed series is sparser (starts
2014-11) and only relevant once a breakeven-inflation feature is wanted;
the NSW TCorp series are sub-sovereign and out of scope. Both are
trivial extensions (add an entry to ``SERIES`` and a verification row to
the tests) if a future feature requests them.

Why this matters
----------------
ACGB yields are core macro inputs the RBA cites in every Statement on
Monetary Policy. The 10y-2y slope is a regime / recession proxy; the 3y
is the most direct market read on the cash-rate path 1-3 years out (and
the most-liquid ACGB futures contract); the spread vs US 10y captures
global rate-cycle divergence. These features feed both Taylor-rule
baselines and the surprise-vs-market stretch target.

Source format
-------------
F2 publishes at
``https://www.rba.gov.au/statistics/tables/csv/f2-data.csv`` using a
variant of the standard RBA statistical-table layout. Two differences
from I2 / D / E2 worth noting:

- **Encoding** is Windows-1252 (cp1252), not UTF-8. The table title
  contains an en-dash (byte ``0x96``) that breaks UTF-8 decoding.
- **Date format** is ``DD-MMM-YYYY`` (e.g. ``20-May-2013``), not
  ``DD/MM/YYYY`` as in I2 / D / E2.

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (``F2 CAPITAL MARKET YIELDS – GOVERNMENT BONDS``)
1           ``Title,<column display names>``
2           ``Description,<descriptions>``
3           ``Frequency,Daily,Daily,Daily,Daily,...``
4           ``Type,Original,Original,...``
5           ``Units,Per cent per annum,...``
6-7         blanks
8           ``Source,RBA,RBA,...`` (was ``Yieldbroker`` pre-Finlay-Wende)
9           ``Publication date,<refresh date, all cols>``
10          ``Series ID,FCMYGBAG2D,FCMYGBAG3D,...``
11+         ``DD-MMM-YYYY,<observation values per series>``
==========  =====================================================

The ``Publication date`` row (9) is the **date of the latest refresh**
(per-table, not per-observation). Per-observation publication dates
come from :mod:`rba.data.agb_yields_release_calendar` instead.

Publication-date convention
---------------------------
F2 publishes the prior trading session's closing yields the next
morning (around 11:30am AEST). Verified against 5 Wayback Machine
snapshots of ``f2-data.csv`` spanning 2018-01 → 2018-08: the
``Publication date`` header sits exactly 1 Australian business day
after the latest data row's date, every time. The publication-date
rule is therefore::

    publication_date = observation_date + 1 AU business day

implemented in :mod:`rba.data.agb_yields_release_calendar` (algorithmic
+1 BDay with the same AU/NSW public-holiday set used by
``rba_i2_release_calendar``).

Note that the live CSV occasionally lags by an extra business day at
the head — e.g. a refresh on Fri carrying data through Wed — but every
populated cell, once it appears, sits exactly 1 BDay before the header
date.

The validation floor is ``1993-01-01`` (inflation-targeting era start)
to match every other source module, but in practice every F2
observation we ingest sits well after the floor (current-vintage F2
begins 2013-05-20). There is no NaT-publication-date region for this
source.

Splice / long-history
---------------------
A long-history archive ``f02dhist.xls`` exists at
``https://www.rba.gov.au/statistics/tables/xls-hist/f02dhist.xls``
covering 1995 → 2013-05-17 daily, but is **not** spliced here. Two
reasons (documented for future contributors):

1. The pre-2014 yields use a different RBA yield-curve methodology
   (the bank adopted the Finlay-Wende zero-coupon construction in 2014;
   pre-2014 vintages come from a different model). Splicing would mix
   methodologies and require a documented break.
2. Even with a splice, 1993-1994 remains uncovered. The model's
   walk-forward CV already has 13+ years of post-2013 F2 history,
   which covers ≥1 hike and ≥1 cut cycle as required by the project
   spec.

Adding the splice later is a backwards-compatible extension (one extra
fetch path, plus an ``observation_date < 2013-05-20`` branch in
``_attach_publication_dates`` to use the same +1 BDay rule).

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release yield. The RBA occasionally restates F2
historical yields when input data is corrected. Acknowledged limitation;
see CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date (closing yield).
- ``publication_date`` (datetime64[ns]) — algorithmic F2 release date
  from ``rba.data.agb_yields_release_calendar`` (= trade date + 1 AU
  business day).
- ``series_id`` (object) — one of the 4 logical series IDs listed in
  ``SERIES`` below (``agb_yield_2y``, ``agb_yield_3y``, ``agb_yield_5y``,
  ``agb_yield_10y``).
- ``value`` (float64) — closing yield in per cent per annum (verbatim
  from F2, no unit conversion).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/agb_yields.csv`` keyed by ``trade_date`` with one
column per maturity plus the release date.

Side effects
------------
``fetch()`` writes under ``data/raw/agb_yields/``:

- ``<YYYY-MM-DD>__f2.csv`` — verbatim CSV bytes per refresh.
- ``_metadata.json`` — provenance manifest: URL, SHA-256, download
  timestamp (UTC), byte count, observation count.
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

SOURCE_NAME = "agb_yields"
TABLE_URL = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"

# Observations earlier than this are not validated against the release
# calendar — matches every other source module. Current-vintage F2
# starts 2013-05-20, so in practice every row is post-floor.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Coverage start of the current-vintage F2 table. Used to size the
# release calendar; if fetch() ever yields a row earlier than this the
# upstream has materially changed (or a long-history splice was added)
# and the calendar window should be widened.
_F2_COVERAGE_START = pd.Timestamp("2013-05-20")


@dataclass(frozen=True)
class AgbYieldSeries:
    """One ACGB benchmark-maturity yield identified by its RBA series ID."""

    series_id: str
    rba_series_id: str
    maturity_years: int


SERIES: tuple[AgbYieldSeries, ...] = (
    AgbYieldSeries(series_id="agb_yield_2y", rba_series_id="FCMYGBAG2D", maturity_years=2),
    AgbYieldSeries(series_id="agb_yield_3y", rba_series_id="FCMYGBAG3D", maturity_years=3),
    AgbYieldSeries(series_id="agb_yield_5y", rba_series_id="FCMYGBAG5D", maturity_years=5),
    AgbYieldSeries(series_id="agb_yield_10y", rba_series_id="FCMYGBAG10D", maturity_years=10),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull F2, extract every series, attach publication dates.

    Parameters
    ----------
    force_download
        If True (default), refresh from the live RBA endpoint. If False,
        reuse the most recent dated snapshot and only download when none
        is present.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(series_id,
        observation_date)``.

    Shapes
    ------
    Returns: (n_obs, 4) with one row per (series_id, observation_date).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path, entry, raw_bytes = _resolve_snapshot(dest_dir, force_download=force_download)

    frames: list[pd.DataFrame] = []
    obs_count = 0
    for spec in SERIES:
        df = _parse(raw_bytes, spec=spec)
        obs_count += len(df)
        frames.append(df)
    entry["observations"] = obs_count

    _write_manifest(dest_dir, [entry])

    out = pd.concat(frames, ignore_index=True)
    out = _attach_publication_dates(out)
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge algorithmic F2 release dates onto each observation.

    Defensively drops any pre-existing ``publication_date`` column before
    merging so a stale value cannot leak through. Raises ``ValueError``
    if any observation on or after ``_CALENDAR_VALIDATION_FLOOR`` is
    unmatched (which would mean the calendar window is too narrow — the
    calendar is built lazily here with a window spanning the data).
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    if df.empty:
        df = df.copy()
        df["publication_date"] = pd.Series(dtype="datetime64[ns]")
        return df[["observation_date", "publication_date", "series_id", "value"]]

    # Size the calendar to cover the observation window with a small
    # forward buffer. The calendar's default start_date is the current
    # F2 coverage start; widen if we ever splice older history.
    obs_min = df["observation_date"].min()
    obs_max = df["observation_date"].max()
    calendar_start = min(_F2_COVERAGE_START, pd.Timestamp(obs_min))
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
            f"{len(unmatched)} AGB yield observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for the F2 CSV."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__f2.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing F2 snapshot at {}", path)
        entry: dict[str, object] = {
            "table": "f2",
            "url": TABLE_URL,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir)


def _download(dest_dir: Path) -> tuple[Path, dict[str, object], bytes]:
    """Download the F2 CSV, save dated snapshot, return path + manifest + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__f2.csv"
    logger.info("Downloading {} -> {}", TABLE_URL, snapshot_path)

    with urllib.request.urlopen(TABLE_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": "f2",
        "url": TABLE_URL,
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


def _parse(csv_bytes: bytes, *, spec: AgbYieldSeries) -> pd.DataFrame:
    """Extract one F2 series by RBA series ID.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes from the ``/statistics/tables/csv/f2-data.csv``
        endpoint. Expected layout described in the module docstring.
    spec
        ``AgbYieldSeries`` identifying which series to extract.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], trade date),
        ``series_id`` (object, logical name from ``spec.series_id``),
        ``value`` (float64). One row per non-null daily observation.

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

    sid_row = raw_df[raw_df["Title"] == "Series ID"]
    if sid_row.empty:
        raise ValueError(
            "No 'Series ID' metadata row found while extracting "
            f"{spec.rba_series_id!r}; CSV layout may have changed."
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
        raise ValueError(
            f"RBA series ID {spec.rba_series_id!r} not found among columns "
            "in F2 CSV; series may have been renamed or removed."
        )

    parsed_dates = pd.to_datetime(raw_df["Title"], format="%d-%b-%Y", errors="coerce")
    data_mask = parsed_dates.notna()

    df = pd.DataFrame(
        {
            "observation_date": parsed_dates[data_mask].dt.as_unit("ns").to_numpy(),
            "series_id": spec.series_id,
            "value": pd.to_numeric(raw_df.loc[data_mask, target_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Parsed zero observations for {spec.series_id!r} "
            f"(RBA {spec.rba_series_id!r}); CSV layout may have changed "
            "or the column is empty in the current vintage."
        )

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
        maturity in ``SERIES`` (only those present in the input), and
        ``release_date`` (datetime64[ns]) from the release calendar.

    Shapes
    ------
    Returns: (n_trade_days, 2 + n_maturities).
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
    dest = EXTERNAL_DATA_DIR / "agb_yields.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote ACGB yields ({} trade dates, {} maturities) to {}",
        len(wide_df),
        len([c for c in wide_df.columns if c.startswith("agb_yield_")]),
        dest,
    )
