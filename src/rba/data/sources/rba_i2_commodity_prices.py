"""RBA I2 — Index of Commodity Prices.

Pulls the full set of commodity-price index series from RBA Table I2.
All series are monthly, in index points with a common base of
``2024/25 = 100``, and cover Australia's basket of major export
commodities valued in three currency / weighting bases (A$, SDR, US$):

================================================  ============  ============================
Series                                            RBA ID        Coverage
------------------------------------------------  ------------  ----------------------------
Index – All items; A$                             ``GRCPAIAD``  1982-07 → present
Index – All items; SDR                            ``GRCPAISDR`` 1982-07 → present
Index – All items; US$                            ``GRCPAIUSD`` 1982-07 → present
Index – Rural; A$                                 ``GRCPRCAD``  1982-07 → present
Index – Rural; SDR                                ``GRCPRCSDR`` 1982-07 → present
Index – Rural; US$                                ``GRCPRCUSD`` 1982-07 → present
Index – Non-rural; A$                             ``GRCPNRAD``  1982-07 → present
Index – Non-rural; SDR                            ``GRCPNRSDR`` 1982-07 → present
Index – Non-rural; US$                            ``GRCPNRUSD`` 1982-07 → present
Index – Base metals; A$                           ``GRCPBMAD``  1982-07 → present
Index – Base metals; SDR                          ``GRCPBMSDR`` 1982-07 → present
Index – Base metals; US$                          ``GRCPBMUSD`` 1982-07 → present
Index – Bulk commodities (export prices); A$      ``GRCPBCAD``  1982-07 → present
Index – Bulk commodities (export prices); SDR     ``GRCPBCSDR`` 1982-07 → present
Index – Bulk commodities (export prices); US$     ``GRCPBCUSD`` 1982-07 → present
Index – All items (with bulk spot); A$            ``GRCPAISAD`` 2009-01 → present
Index – All items (with bulk spot); SDR           ``GRCPAISSDR``2009-01 → present
Index – All items (with bulk spot); US$           ``GRCPAISUSD``2009-01 → present
Index – Bulk commodities (spot prices); A$        ``GRCPBCSAD`` 2009-01 → present
Index – Bulk commodities (spot prices); SDR       ``GRCPBCSSDR``2009-01 → present
Index – Bulk commodities (spot prices); US$       ``GRCPBCSUSD``2009-01 → present
================================================  ============  ============================

Why this matters
----------------
Commodity prices are an explicit RBA terms-of-trade driver cited
throughout the Statement on Monetary Policy. A rise in the AUD-weighted
index lifts national income and tradeables inflation; divergence
between the export-price and spot variants is itself a forward signal
(spot leads contract-renegotiation timing). Including all three currency
bases lets the feature layer separate genuine commodity-price moves
(USD / SDR variants) from AUD valuation effects.

Source format
-------------
I2 publishes at
``https://www.rba.gov.au/statistics/tables/csv/i2-data.csv`` using the
standard RBA statistical-table layout (identical to D1/D2/E2/H3):

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (``I2 COMMODITY PRICES``)
1           ``Title,<column display names>``
2           ``Description,<descriptions>``
3-9         Frequency / Type / Units / blanks / Source
10          ``Publication date,<refresh date, all cols>``
11          ``Series ID,<RBA series codes>``
12+         ``DD/MM/YYYY,<observation values per series>``
==========  =====================================================

The ``Publication date`` row (10) is the **date of the latest refresh**
(per-table, not per-observation). Per-observation publication dates
come from ``rba.data.rba_i2_release_calendar`` instead.

Publication-date convention
---------------------------
I2 publishes to a fixed schedule: "first business day of each month",
covering data for the *preceding* month. The algorithmic rule "first
weekday on/after the 1st of the following month that is not an AU/NSW
public holiday" matches every sampled Wayback ``Publication date``
header from 2015-04 through 2023-06 (14 snapshots, including a
verified Easter shift to Tue Apr 3 2018). See
``rba.data.rba_i2_release_calendar`` for the full verification table
and override mechanism.

The validation floor is 1993-01-31 (inflation-targeting era start).
Observations on/after the floor MUST match the calendar and raise on
miss; pre-1993 observations retain ``NaT`` publication dates (matches
``rba_d`` and ``abs_labour_force``).

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release value. The RBA revises I2 values backward
as upstream export-price data is finalised (the index is computed by
the RBA using a Laspeyres-weighted basket whose weights are rebased
every 5 years; weights change → historical levels rebase). Acknowledged
limitation; see CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference month.
- ``publication_date`` (datetime64[ns]) — algorithmic I2 release date
  from ``rba.data.rba_i2_release_calendar``. ``NaT`` for pre-1993
  observations.
- ``series_id`` (object) — one of the 21 logical series IDs listed in
  ``SERIES`` below.
- ``value`` (float64) — index points (base ``2024/25 = 100``).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/rba_i2_commodity_prices.csv`` keyed by
``reference_month_end`` with one column per logical series plus
``release_date`` and ``reference_label``.

Side effects
------------
``fetch()`` writes under ``data/raw/rba_i2_commodity_prices/``:

- ``<YYYY-MM-DD>__i2.csv`` — verbatim CSV bytes per refresh.
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
from rba.data.rba_i2_release_calendar import build_rba_i2_release_calendar

SOURCE_NAME = "rba_i2_commodity_prices"
TABLE_URL = "https://www.rba.gov.au/statistics/tables/csv/i2-data.csv"

# Observations earlier than this are not validated against the release
# calendar: the inflation-targeting era starts 1993, matching every
# other source module. Pre-1993 rows retain NaT publication_date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class RbaI2Series:
    """One I2 commodity-price index identified by its RBA series ID."""

    series_id: str
    rba_series_id: str


SERIES: tuple[RbaI2Series, ...] = (
    # Headline All-items index in all three currency / weighting bases.
    RbaI2Series(series_id="commodity_prices_all_aud", rba_series_id="GRCPAIAD"),
    RbaI2Series(series_id="commodity_prices_all_sdr", rba_series_id="GRCPAISDR"),
    RbaI2Series(series_id="commodity_prices_all_usd", rba_series_id="GRCPAIUSD"),
    # Rural sub-index.
    RbaI2Series(series_id="commodity_prices_rural_aud", rba_series_id="GRCPRCAD"),
    RbaI2Series(series_id="commodity_prices_rural_sdr", rba_series_id="GRCPRCSDR"),
    RbaI2Series(series_id="commodity_prices_rural_usd", rba_series_id="GRCPRCUSD"),
    # Non-rural sub-index (Base metals + Bulk + other non-rural).
    RbaI2Series(series_id="commodity_prices_non_rural_aud", rba_series_id="GRCPNRAD"),
    RbaI2Series(series_id="commodity_prices_non_rural_sdr", rba_series_id="GRCPNRSDR"),
    RbaI2Series(series_id="commodity_prices_non_rural_usd", rba_series_id="GRCPNRUSD"),
    # Base metals component of non-rural.
    RbaI2Series(series_id="commodity_prices_base_metals_aud", rba_series_id="GRCPBMAD"),
    RbaI2Series(series_id="commodity_prices_base_metals_sdr", rba_series_id="GRCPBMSDR"),
    RbaI2Series(series_id="commodity_prices_base_metals_usd", rba_series_id="GRCPBMUSD"),
    # Bulk commodities (computed from export-price movements).
    RbaI2Series(series_id="commodity_prices_bulk_export_aud", rba_series_id="GRCPBCAD"),
    RbaI2Series(series_id="commodity_prices_bulk_export_sdr", rba_series_id="GRCPBCSDR"),
    RbaI2Series(series_id="commodity_prices_bulk_export_usd", rba_series_id="GRCPBCUSD"),
    # All-items variant that splices in bulk-commodities *spot* prices (starts 2009-01).
    RbaI2Series(series_id="commodity_prices_all_with_spot_aud", rba_series_id="GRCPAISAD"),
    RbaI2Series(series_id="commodity_prices_all_with_spot_sdr", rba_series_id="GRCPAISSDR"),
    RbaI2Series(series_id="commodity_prices_all_with_spot_usd", rba_series_id="GRCPAISUSD"),
    # Bulk-commodities spot sub-index (starts 2009-01).
    RbaI2Series(series_id="commodity_prices_bulk_spot_aud", rba_series_id="GRCPBCSAD"),
    RbaI2Series(series_id="commodity_prices_bulk_spot_sdr", rba_series_id="GRCPBCSSDR"),
    RbaI2Series(series_id="commodity_prices_bulk_spot_usd", rba_series_id="GRCPBCSUSD"),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull I2, extract every series, attach publication dates.

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
    """Merge algorithmic I2 release dates onto each observation.

    Defensively drops any pre-existing ``publication_date`` column before
    merging so a stale value cannot leak through. Raises ``ValueError`` if
    any observation on or after ``_CALENDAR_VALIDATION_FLOOR`` is unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    # Calendar starts at the validation floor; pre-1993 observations
    # naturally fall through to NaT publication_date via the left merge.
    calendar_df = build_rba_i2_release_calendar(start_year=1993)
    merged = df.merge(
        calendar_df.rename(columns={"reference_month_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} RBA I2 observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date from the release calendar. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for the I2 CSV."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__i2.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing I2 snapshot at {}", path)
        entry: dict[str, object] = {
            "table": "i2",
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
    """Download the I2 CSV, save dated snapshot, return path + manifest + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__i2.csv"
    logger.info("Downloading {} -> {}", TABLE_URL, snapshot_path)

    with urllib.request.urlopen(TABLE_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": "i2",
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


def _parse(csv_bytes: bytes, *, spec: RbaI2Series) -> pd.DataFrame:
    """Extract one I2 series by RBA series ID.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes from the ``/statistics/tables/csv/i2-data.csv``
        endpoint. Expected layout described in the module docstring.
    spec
        ``RbaI2Series`` identifying which series to extract.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-month),
        ``series_id`` (object, logical name from ``spec.series_id``),
        ``value`` (float64). One row per non-null monthly observation.

    Shapes
    ------
    Returns: (n_months, 3).
    """
    raw_df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=1,
        low_memory=False,
        skipinitialspace=False,
        encoding="utf-8",
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
            "in I2 CSV; series may have been renamed or removed."
        )

    parsed_dates = pd.to_datetime(raw_df["Title"], format="%d/%m/%Y", errors="coerce")
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
            f"(RBA {spec.rba_series_id!r}); CSV layout may have changed "
            "or the column is empty in the current vintage."
        )

    return df[["observation_date", "series_id", "value"]]


def _to_month_end(dates: pd.Series) -> pd.Series:
    """Snap each timestamp to the last calendar day of its month.

    I2 publishes month-end dates already, but applying ``MonthEnd``
    (idempotent on already-month-end inputs) makes the contract
    explicit and resilient to any future format change. Normalised to
    nanosecond resolution to match other source modules.
    """
    offset = pd.offsets.MonthEnd()
    return pd.to_datetime(dates.map(offset.rollforward)).dt.as_unit("ns")


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation (called from __main__)
# ---------------------------------------------------------------------------


_WIDE_COLUMN_ORDER: tuple[str, ...] = tuple(s.series_id for s in SERIES)


def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ``fetch()`` output to the wide CSV schema."""
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
            "observation_date": "reference_month_end",
            "publication_date": "release_date",
        }
    )

    wide["reference_label"] = wide["reference_month_end"].dt.strftime("%b %Y")

    column_order = (
        ["reference_month_end", "reference_label"]
        + [c for c in _WIDE_COLUMN_ORDER if c in wide.columns]
        + ["release_date"]
    )
    return wide[column_order].sort_values("reference_month_end").reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "rba_i2_commodity_prices.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info("Wrote RBA I2 commodity prices ({} months) to {}", len(wide_df), dest)
