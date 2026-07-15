"""RBA E2 — Household Finances, Selected Ratios (debt-to-income).

Pulls three household-leverage ratios from RBA Table E2. All three are
RBA-published, pre-computed ratios (we do not derive them from E1
component levels) so values match the RBA's official numbers exactly:

================================================  =============  ============================
Series                                            RBA ID         Coverage
------------------------------------------------  -------------  ----------------------------
Household debt to income (%)                      ``BHFDDIT``    1988-06 → present
Housing debt to income (%)                        ``BHFDDIH``    1988-09 → present
Owner-occupier housing debt to income (%)         ``BHFDDIO``    1990-03 → present
================================================  =============  ============================

All three are quarterly, in per cent, with the *income* in the
denominator being annualised household disposable income (sourced from
ABS Cat 5206.0, seasonally adjusted, before deduction of interest
payments). See the ``Notes`` sheet of ``e02hist.xlsx`` for the full
methodology.

Why these three
---------------
Household debt-service and leverage measures are explicit RBA financial-
stability inputs cited across speeches, the Statement on Monetary Policy,
and the Financial Stability Review. The three ratios capture different
slices of household leverage:

- ``BHFDDIT`` (total household debt-to-income) — broadest leverage
  measure including housing + personal debt + unincorporated-enterprise
  debt.
- ``BHFDDIH`` (housing debt-to-income) — excludes personal/other debt;
  most directly comparable to housing-market signals.
- ``BHFDDIO`` (owner-occupier housing debt-to-income) — RBA-calculated
  series that *excludes* unincorporated-enterprise debt and is
  pre-spliced across the methodology breaks the RBA documents.

Why no interest-paid-to-income ratio
------------------------------------
The RBA removed the household interest-payments-to-income series from E2
in February 2023 (per the E2 notes sheet). The closest live series is
``LPHTICRI`` (interest charged on total housing loans relative to
household disposable income) in Table **E13**, but it covers only
2009-Q1 onward and is housing-interest-only — not total household
interest. E13 is out of scope for this module; if a future iteration
adds it, write it as a separate ``rba_e13_*`` source rather than
splicing.

Source format
-------------
The E2 CSV at
``https://www.rba.gov.au/statistics/tables/csv/e2-data.csv`` uses the
standard RBA statistical-table layout (identical to D1/D2/H3):

==========  =====================================================
Row index   Content
----------  -----------------------------------------------------
0           Table name (``E2 HOUSEHOLD FINANCES – SELECTED RATIOS``)
1           ``Title,<column display names>``
2           ``Description,<descriptions>``
3-9         Frequency / Type / Units / blanks / Source
10          ``Publication date,<refresh date, all cols>``
11          ``Series ID,<RBA series codes>``
12+         ``DD/MM/YYYY,<observation values per series>``
==========  =====================================================

The ``Publication date`` row (10) is the **date of the latest refresh**
(per-table, not per-observation). The release calendar in
``rba.data.rba_e_release_calendar`` provides per-observation
publication dates instead.

Note: the RBA also exposes ``e02hist.xlsx`` (multi-sheet Excel with a
``Notes`` tab carrying the full methodology) — we use the CSV because
it carries the same data, is faster to fetch, and matches the parsing
pattern of every other RBA source module in this repo. The XLSX notes
are summarised in this docstring instead.

Publication-date convention
---------------------------
E2 release dates are scraped from the Wayback Machine + the live RBA
CSV by ``rba.data.rba_e_release_calendar``, materialised to
``data/external/rba_e_release_dates.csv``. The harvest covers a partial
subset of quarters from **2014-Q4 onward** — pre-2014 observations and
gaps in the Wayback crawl fall back to a flat conservative
``+ 95`` day offset (slightly longer than the worst observed in-window
lag of ~92 days). Pre-fallback rows are tagged
``source = "inferred"`` in the wide-format CSV; scraped rows are tagged
``rba_page`` or ``archive_org``.

Validation floor
----------------
Observations on/after ``_CALENDAR_VALIDATION_FLOOR`` (1993-01-01,
inflation-targeting era) MUST resolve to a publication date — either
via the scraped calendar or via the flat-offset fallback. Pre-1993
observations retain ``NaT`` publication dates (matches ``rba_d`` and
``abs_labour_force``). An unmatched in-window row raises ``ValueError``.

Series-break handling
---------------------
The E2 notes sheet documents two RBA methodology events: (1) a 2019-Q3
recalibration of interest-payment calculation using EFS-based weighted
interest rates, and (2) the Feb 2023 removal of the interest-payments
series. **Neither affects the three debt-to-income ratios we pull
here** — for ``BHFDDIO`` the RBA explicitly notes "These data are
adjusted for the effects of breaks in the series" (i.e. the series is
internally pre-spliced). We therefore do not surface a
``series_break_indicator`` column; there are no in-scope breaks for the
feature layer to handle.

Vintage policy
--------------
Values are the **current RBA vintage** at the time of download — *not*
the original first-release value. The RBA revises ratios as upstream
ABS national-accounts data revises. Acknowledged limitation; see
CONTEXT.md Invariant #1 vintage policy.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — end of the reference quarter.
- ``publication_date`` (datetime64[ns]) — release date from the E
  release calendar (or flat-offset fallback for pre-2014 quarters).
- ``series_id`` (object) — one of the three logical names below.
- ``value`` (float64) — observation value (per cent).

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/rba_e2_household_ratios.csv`` with columns:

- ``reference_quarter_end`` (date)
- ``reference_label`` (str, e.g. ``"Dec 2024"``)
- ``household_debt_to_income`` (float, BHFDDIT)
- ``housing_debt_to_income`` (float, BHFDDIH)
- ``owner_occupier_housing_debt_to_income`` (float, BHFDDIO)
- ``release_date`` (date)
- ``source`` (str, one of ``rba_page``, ``archive_org``, ``inferred``)

Side effects
------------
``fetch()`` writes under ``data/raw/rba_e/``:

- ``<YYYY-MM-DD>__e2.csv`` — verbatim CSV bytes per refresh.
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
from rba.data.rba_e_release_calendar import build_rba_e_release_calendar

SOURCE_NAME = "rba_e"
TABLE_URL = "https://www.rba.gov.au/statistics/tables/csv/e2-data.csv"

# Observations earlier than this are not validated against the release
# calendar: the inflation-targeting era starts 1993, and that is the
# floor used by every other source module. Pre-1993 rows retain NaT
# publication_date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Boundary at which scraped publication dates *start* being available.
# Pre-floor observations (post-1993, pre-2014-Q4) all fall through to
# the flat-offset fallback. Some post-floor observations also fall
# through (gaps in the Wayback crawl) — those also use the offset.
_SCRAPE_FLOOR = pd.Timestamp("2014-12-31")

# Conservative flat offset for pre-scrape-floor + Wayback-gap
# observations. Observed lags ran 79-92 days; +95 buffers comfortably
# without overstating delay.
_FLAT_OFFSET_DAYS = 95


@dataclass(frozen=True)
class RbaE2Series:
    """One E2 ratio identified by its RBA series ID."""

    series_id: str
    rba_series_id: str


SERIES: tuple[RbaE2Series, ...] = (
    RbaE2Series(
        series_id="household_debt_to_income",
        rba_series_id="BHFDDIT",
    ),
    RbaE2Series(
        series_id="housing_debt_to_income",
        rba_series_id="BHFDDIH",
    ),
    RbaE2Series(
        series_id="owner_occupier_housing_debt_to_income",
        rba_series_id="BHFDDIO",
    ),
)


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull E2, extract the three debt-to-income series, attach pub dates.

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
    """Attach E2 publication dates to every observation.

    Scraped calendar covers a sparse subset of observations from
    ``_SCRAPE_FLOOR`` onward. All other in-window observations use a
    flat ``+ _FLAT_OFFSET_DAYS`` offset. Pre-1993 observations retain
    ``NaT`` (matches rba_d / abs_labour_force).

    Raises ``ValueError`` if any observation on/after
    ``_CALENDAR_VALIDATION_FLOOR`` remains unmatched.
    """
    df = df.drop(columns=["publication_date"], errors="ignore")
    calendar_df = build_rba_e_release_calendar()[["reference_quarter_end", "publication_date"]]
    merged = df.merge(
        calendar_df.rename(columns={"reference_quarter_end": "observation_date"}),
        on="observation_date",
        how="left",
    )

    in_window = merged["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    needs_fallback = in_window & merged["publication_date"].isna()
    merged.loc[needs_fallback, "publication_date"] = merged.loc[
        needs_fallback, "observation_date"
    ] + pd.Timedelta(days=_FLAT_OFFSET_DAYS)

    unmatched = merged[in_window & merged["publication_date"].isna()]
    if not unmatched.empty:
        sample = unmatched[["observation_date", "series_id"]].head().to_dict("records")
        raise ValueError(
            f"{len(unmatched)} RBA E2 observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication "
            f"date. Sample: {sample}"
        )

    return merged[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for the E2 CSV."""
    pattern = f"[0-9]{'[0-9]' * 3}-??-??__e2.csv"
    existing = sorted(dest_dir.glob(pattern))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing E2 snapshot at {}", path)
        entry: dict[str, object] = {
            "table": "e2",
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
    """Download the E2 CSV, save dated snapshot, return path + manifest + bytes."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__e2.csv"
    logger.info("Downloading {} -> {}", TABLE_URL, snapshot_path)

    with urllib.request.urlopen(TABLE_URL, timeout=60) as response:  # noqa: S310 — public RBA URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "table": "e2",
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


def _parse(csv_bytes: bytes, *, spec: RbaE2Series) -> pd.DataFrame:
    """Extract one E2 ratio by RBA series ID.

    Parameters
    ----------
    csv_bytes
        Raw CSV bytes from the ``/statistics/tables/csv/e2-data.csv``
        endpoint. Expected layout described in the module docstring.
    spec
        ``RbaE2Series`` identifying which series to extract.

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], end-of-quarter),
        ``series_id`` (object, logical name from ``spec.series_id``),
        ``value`` (float64). One row per non-null quarterly observation.

    Shapes
    ------
    Returns: (n_quarters, 3).
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
            "in E2 CSV; series may have been renamed or removed."
        )

    # E2 currently uses DD/MM/YYYY in the CSV. Historical Wayback vintages
    # used Mon-YYYY; we don't expect to encounter those at parse time for
    # the live source, but handle both for robustness.
    modern = pd.to_datetime(raw_df["Title"], format="%d/%m/%Y", errors="coerce")
    legacy = pd.to_datetime(raw_df["Title"], format="%b-%Y", errors="coerce")
    legacy = legacy + pd.offsets.MonthEnd(0)
    parsed_dates = modern.fillna(legacy)
    data_mask = parsed_dates.notna()

    df = pd.DataFrame(
        {
            "observation_date": _to_quarter_end(parsed_dates[data_mask]),
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


def _to_quarter_end(dates: pd.Series) -> pd.Series:
    """Snap each timestamp to the last calendar day of its quarter.

    E2 publishes quarter-end dates already, but applying ``QuarterEnd``
    (idempotent on already-quarter-end inputs) makes the contract
    explicit and resilient to any future format change. Normalised to
    nanosecond resolution to match other source modules.
    """
    offset = pd.offsets.QuarterEnd(startingMonth=12)
    return pd.to_datetime(dates.map(offset.rollforward)).dt.as_unit("ns")


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation (called from __main__)
# ---------------------------------------------------------------------------


def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ``fetch()`` output to the wide CSV schema.

    Joins the release calendar a second time to expose the ``source``
    column (which ``fetch()`` strips for parity with other source
    modules). Unmatched in-window rows take ``source = "inferred"``.
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
            "observation_date": "reference_quarter_end",
            "publication_date": "release_date",
        }
    )

    calendar_df = build_rba_e_release_calendar()[["reference_quarter_end", "source"]]
    wide = wide.merge(calendar_df, on="reference_quarter_end", how="left")
    # Rows with a release_date but no calendar entry came from the
    # flat-offset fallback in _attach_publication_dates -> source="inferred".
    # Rows with no release_date at all (pre-1993) keep source=NaN.
    has_release = wide["release_date"].notna()
    needs_inferred = has_release & wide["source"].isna()
    wide.loc[needs_inferred, "source"] = "inferred"
    wide["source"] = wide["source"].astype(object)

    wide["reference_label"] = wide["reference_quarter_end"].dt.strftime("%b %Y")

    column_order = [
        "reference_quarter_end",
        "reference_label",
        "household_debt_to_income",
        "housing_debt_to_income",
        "owner_occupier_housing_debt_to_income",
        "release_date",
        "source",
    ]
    return wide[column_order].sort_values("reference_quarter_end").reset_index(drop=True)


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "rba_e2_household_ratios.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info("Wrote RBA E2 household ratios ({} quarters) to {}", len(wide_df), dest)
