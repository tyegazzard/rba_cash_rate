"""Westpac-Melbourne Institute Consumer Sentiment Index — RBA H3 vintage.

Pulls the headline Westpac-MI Consumer Sentiment Index from **RBA Table
H3 (Monthly Activity Indicators)**, which republishes the Westpac-MI
series under an RBA series ID:

==========================================  ==========  ==============  =====================
Series                                      Table       RBA series ID   Coverage from
------------------------------------------  ----------  --------------  ---------------------
``consumer_sentiment``                          H3      ``GICWMICS``    2010-01 (monthly)
==========================================  ==========  ==============  =====================

The Westpac-MI survey itself goes back to 1974-09 but **H3 carries
only the most recent ~16 years** of the series (verified: 2010-01-31
→ present, 197 observations as of 2026-05). Deeper history would
require a Melbourne Institute subscription or a separate scrape of
the Westpac IQ archive; both are out of scope for this v1 module.

Why H3 rather than Westpac IQ
-----------------------------
Westpac-MI is a commercial publication. The Melbourne Institute makes
full historical data available only via a paid subscription, and the
Westpac IQ release pages cover roughly 2018+ only. The RBA H3 table is
the deepest free, machine-readable source of the Westpac-MI headline
index — same single number as Westpac's headline, full history back to
the survey's 1974 origin, refreshed monthly with the survey release.
Sub-component indices (family finances, time to buy, economic
conditions 12m/5y) are *not* in H3 and are out of scope for this v1
module; if a future iteration needs them, scrape Westpac IQ directly
and add them as additional columns.

Why the value is the same as Westpac's headline
-----------------------------------------------
The RBA H3 column is labelled "Westpac-Melbourne Institute consumer
sentiment index" with units "Index", type "Seasonally adjusted" — i.e.
the headline index level (100 = neutral, > 100 = optimists outnumber
pessimists). The RBA is not transforming the value; it republishes the
Westpac-MI seasonally-adjusted index unchanged. Cross-check: H3's May
2026 value matches the value reported on the Westpac IQ May 2026
release page.

Publication-date convention
---------------------------
Westpac-MI publishes mid-reference-month (the survey is conducted in
the first week and released ~2 weeks later). Publication dates come
from the **algorithmic** release calendar
``rba.data.westpac_mi_release_calendar`` (2nd Wednesday of the
reference month + ``_OVERRIDES`` for verified exceptions) — the same
pattern as ``lfs_release_calendar`` / ``cpi_release_calendar``. We do
not scrape per-observation: NAB / Westpac archive pages cover only
recent years, so a scraped calendar would be no better than the
algorithm for the inflation-targeting training window.

Unusual: publication date PRECEDES observation date for this series.
The survey covers the reference month but is released *during* that
month, so for a reference month ending day 31 the 2nd Wednesday lands
~10-18 days *before* month-end. The model's leakage check
(``publication_date <= meeting_date``) still works correctly — it
treats publication_date as the date the value became publicly known.

Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (1993-01-01,
inflation-targeting era) MUST resolve to a publication date. Pre-1993
observations retain ``NaT`` publication dates (matches every other
source module in this repo).

Vintage policy
--------------
Values are the **current RBA H3 vintage** at the time of download —
*not* the original first-release values. The RBA refreshes H3 monthly
with the latest Westpac-MI data including any revisions Westpac-MI
publishes. Acknowledged limitation under Invariant #1 in
``CONTEXT.md``.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference month.
- ``publication_date`` (datetime64[ns]) — algorithmic Westpac-MI
  release date (or ``NaT`` for pre-1993 observations).
- ``series_id`` (object) — logical name ``"consumer_sentiment"``.
- ``value`` (float64) — Westpac-MI Consumer Sentiment Index, SA, index
  level (100 = neutral).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/westpac_mi_consumer_sentiment.csv`` with columns:

- ``reference_month_end`` (date)
- ``reference_label`` (str, e.g. ``"May 2026"``)
- ``consumer_sentiment`` (float, GICWMICS)
- ``release_date`` (date)

The wide CSV omits the ``source`` column described in the brief: the
algorithmic calendar produces a single provenance value
("algorithmic") for every row, so the column carries no information.
Matches the LFS / CPI wide-output convention.

Side effects
------------
``fetch()`` writes under ``data/raw/westpac_mi_consumer_sentiment/``:

- ``<YYYY-MM-DD>__rba_h3.csv`` — verbatim H3 CSV bytes per refresh.
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
from rba.data.westpac_mi_release_calendar import (
    build_westpac_mi_release_calendar,
)

SOURCE_NAME = "westpac_mi_consumer_sentiment"
RBA_H3_URL = "https://www.rba.gov.au/statistics/tables/csv/h3-data.csv"

# Inflation-targeting-era floor. Observations on/after this must resolve
# to a publication date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")


@dataclass(frozen=True)
class RbaH3Series:
    """An H3 column identified by its RBA series ID."""

    series_id: str
    rba_series_id: str


SERIES: tuple[RbaH3Series, ...] = (
    RbaH3Series(
        series_id="consumer_sentiment",
        rba_series_id="GICWMICS",
    ),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull H3, extract the Westpac-MI consumer sentiment series.

    Parameters
    ----------
    force_download
        If True (default), refresh from the live RBA endpoint. If False,
        reuse the most recent dated snapshot and only download when
        none is present.

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

    snapshot_path, entry, raw_bytes = _resolve_snapshot(
        dest_dir, force_download=force_download
    )

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
    """Attach algorithmic Westpac-MI publication dates to every
    in-window observation.

    Pre-1993 observations retain ``NaT``. Raises ``ValueError`` if any
    observation on/after ``_CALENDAR_VALIDATION_FLOOR`` remains
    unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    in_window_obs = df["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    if not in_window_obs.any():
        # All observations are pre-floor: no calendar match exists, so every
        # publication_date is NaT (a NaT never satisfies ``<= meeting_date``, so
        # these rows are conservatively excluded from any point-in-time join —
        # no leakage). Build the column as an explicit datetime64 NaT Series so
        # the dtype matches the merge path below.
        df = df.assign(
            publication_date=pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        )
        return df[["observation_date", "publication_date", "series_id", "value"]]

    min_year = int(df.loc[in_window_obs, "observation_date"].dt.year.min())
    max_year = int(df.loc[in_window_obs, "observation_date"].dt.year.max())
    cal = build_westpac_mi_release_calendar(start_year=min_year, end_year=max_year)

    merged = df.merge(
        cal.rename(columns={"reference_month_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} Westpac-MI consumer-sentiment observation(s) "
            f"on/after {_CALENDAR_VALIDATION_FLOOR.date()} are missing a "
            f"publication date. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for the H3 CSV."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__rba_h3.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing H3 snapshot at {}", path)
        entry: dict[str, object] = {
            "table": "h3",
            "url": RBA_H3_URL,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir)


def _download(dest_dir: Path) -> tuple[Path, dict[str, object], bytes]:
    """Download H3, save dated snapshot, return path + manifest + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__rba_h3.csv"
    logger.info("Downloading {} -> {}", RBA_H3_URL, snapshot_path)

    with urllib.request.urlopen(RBA_H3_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": "h3",
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


def _parse(csv_bytes: bytes, *, spec: RbaH3Series) -> pd.DataFrame:
    """Extract one H3 series by RBA series ID.

    The H3 CSV uses the standard RBA layout: metadata rows 0-9 (Title /
    Description / Frequency / Type / Units / blanks / Source /
    Publication date / Series ID), data rows from row 10+. The "Title"
    header at row index 1 supplies the column headers.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-month),
        ``series_id`` (object, from ``spec.series_id``), ``value``
        (float64). One row per non-null monthly observation.

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
            f"{spec.rba_series_id!r} from RBA H3; CSV layout may have changed."
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
            "in H3 CSV; series may have been renamed or removed."
        )

    parsed_dates = pd.to_datetime(
        raw_df["Title"], format="%d/%m/%Y", errors="coerce"
    )
    data_mask = parsed_dates.notna()

    df = pd.DataFrame(
        {
            "observation_date": _to_month_end(parsed_dates[data_mask]),
            "series_id": spec.series_id,
            "value": pd.to_numeric(
                raw_df.loc[data_mask, target_col], errors="coerce"
            ),
        }
    )
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Parsed zero observations for {spec.series_id!r} "
            f"(RBA {spec.rba_series_id!r}); H3 column may be empty in the "
            "current vintage."
        )

    return df[["observation_date", "series_id", "value"]]


def _to_month_end(dates: pd.Series) -> pd.Series:
    """Snap each timestamp to the last calendar day of its month."""
    offset = pd.offsets.MonthEnd()
    return pd.to_datetime(dates.map(offset.rollforward)).dt.as_unit("ns")


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation (called from __main__)
# ---------------------------------------------------------------------------


def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ``fetch()`` output to the wide CSV schema."""
    wide = long_df.pivot_table(
        index="observation_date",
        columns="series_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    pub_dates = (
        long_df.groupby("observation_date")["publication_date"].first().reset_index()
    )
    wide = wide.merge(pub_dates, on="observation_date", how="left")
    wide = wide.rename(
        columns={
            "observation_date": "reference_month_end",
            "publication_date": "release_date",
        }
    )
    wide["reference_label"] = wide["reference_month_end"].dt.strftime("%b %Y")

    column_order = [
        "reference_month_end",
        "reference_label",
        "consumer_sentiment",
        "release_date",
    ]
    return wide[column_order].sort_values("reference_month_end").reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "westpac_mi_consumer_sentiment.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote Westpac-MI consumer sentiment ({} months) to {}",
        len(wide_df),
        dest,
    )
