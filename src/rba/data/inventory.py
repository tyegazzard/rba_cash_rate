"""Data inventory & catalog — one generated source of truth for what data we have.

This module sits alongside :mod:`rba.data.refresh` in ``src/rba/data/``. Where
``refresh.py`` answers *"how do I re-pull every source"*, ``inventory.py``
answers *"what series exist, where their raw bytes live, and when each was last
observed"* — and materialises that catalog to three artifacts:

- ``DATA.md`` at the repo root — a checked-in, diffable Markdown table that
  renders on GitHub.
- ``data/external/inventory.parquet`` — the machine-readable counterpart.
- (optional, ``--xlsx``) ``data/external/inventory.xlsx`` — a spreadsheet copy
  for teammates who prefer Excel. A generated convenience, never the source of
  truth.

Design
------
The source list is heterogeneous and the inventory copes with it explicitly
rather than assuming uniformity (see CONTEXT.md "Adding a new data source"):

- **Reuse, don't redefine.** Every numeric source's ``SERIES`` tuple is imported
  from its module — never re-listed here. The catalog (:data:`SERIES_SOURCES` /
  :data:`DOCUMENT_SOURCES`) is the inventory's own per-source spec; the CI test
  (``tests/data/test_inventory.py``) cross-checks it against ``refresh.REGISTRY``
  for the wired subset so the two cannot silently diverge. As the remaining
  sources are wired into ``REGISTRY``, the catalog converges onto it.
- **Heterogeneous ``SERIES``.** The element dataclasses differ per module
  (``CpiSeries`` with ``dataflow``/``datakey``; ``AgbYieldSeries`` with
  ``rba_series_id``; ``FredSeries`` with ``fred_series_id``; ``AsxIndexSeries``
  with ``yahoo_ticker``; …). Each :class:`SeriesSource` carries small callables
  (``identifier`` / ``parse_one`` / ``resolve`` / ``source_url``) that adapt its
  own dataclass — no fragile duck-typed reflection.
- **Heterogeneous ``_metadata.json``.** ``rba_f11`` writes a single dict; the
  multi-series ABS / FRED / Yahoo sources write a per-``series_id`` list; the RBA
  CSV-table sources write a single per-``table`` entry whose ``observations`` is a
  sum across every series in that file. :func:`read_metadata` normalises all of
  these to ``list[dict]`` and each source's ``resolve`` maps a spec to its row.
- **No ``SERIES``.** ``rba_f11``, ``asx_ib_futures`` (dynamic per-contract), and
  the four text sources expose no series registry; they appear as one synthetic
  :class:`DocumentSource` row each, with provenance from their ``_metadata.json``
  and ``last_observation_date`` from their materialised artifact (or, for
  ``rba_f11``, its cached raw snapshot).

Provenance / Invariant #5
-------------------------
The inventory is **read-only over** ``data/raw/`` and **generate-only over** the
three artifacts. It never re-hashes or re-downloads: it reads the existing
``_metadata.json`` for URL / SHA-256 / ``downloaded_at_utc`` and parses the
cached raw snapshot (via each source's own parse helpers) only to read off the
last ``observation_date`` and per-series row count. It deliberately does **not**
call any source ``fetch()`` — even ``force_download=False`` rewrites
``_metadata.json`` (flipping ``reused``/nulling ``downloaded_at_utc``) and would
destroy the provenance the inventory reports.

CLI
---
Run as ``python -m rba.data.inventory``:

- (default)   regenerate ``DATA.md`` + ``data/external/inventory.parquet``
- ``--xlsx``  additionally emit ``data/external/inventory.xlsx``
- ``--check`` report any registered series with no matching ``_metadata.json``
              row and exit non-zero if any are missing (no files written)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger
import pandas as pd

from rba.config import EXTERNAL_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR
from rba.data.sources import (
    abs_building_approvals,
    abs_cpi,
    abs_gdp,
    abs_labour_force,
    abs_total_value_dwellings,
    abs_wpi,
    agb_yields,
    asx_200,
    asx_ib_futures,
    aud_exchange_rates,
    bbsw_rates,
    commodity_prices,
    fred_global_signals,
    nab_business_survey,
    rba_d,
    rba_e2_household_ratios,
    rba_f11,
    rba_i2_commodity_prices,
    rba_media_releases,
    rba_minutes,
    rba_somp,
    rba_speeches,
    westpac_mi_consumer_sentiment,
)

# -----------------------------------------------------------------------------
# Artifact locations + the catalog row schema.
# -----------------------------------------------------------------------------
DATA_MD_PATH = PROJ_ROOT / "DATA.md"
INVENTORY_PARQUET_PATH = EXTERNAL_DATA_DIR / "inventory.parquet"
INVENTORY_XLSX_PATH = EXTERNAL_DATA_DIR / "inventory.xlsx"

METADATA_FILENAME = "_metadata.json"

# Sentinel used in the ``dataflow/table`` column for document (no-SERIES) sources.
DOCUMENT_TABLE = "<documents>"

# Column order for DATA.md / inventory.parquet (the checklist-mandated schema).
COLUMNS: tuple[str, ...] = (
    "series_id",
    "source_module",
    "dataflow_or_table",
    "source_url",
    "raw_dir",
    "snapshot_filename",
    "observations",
    "last_observation_date",
    "release_calendar_module",
)


# -----------------------------------------------------------------------------
# Catalog dataclasses.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SeriesSource:
    """A source exposing a module-level ``SERIES``-style registry.

    The callables adapt this source's particular series dataclass and
    ``_metadata.json`` shape so the generic inventory core never has to branch
    on source identity.

    Attributes
    ----------
    name
        Source identifier == ``data/raw/<name>/`` directory.
    module
        The source module (provides the parse helpers reused read-only).
    calendar_module
        Bare name of the release-calendar module under ``src/rba/data/``
        (e.g. ``"cpi_release_calendar"``), or ``None`` if the source has none.
    series
        The heterogeneous series specs (imported from ``module``).
    identifier
        ``spec -> str`` for the ``dataflow/table`` column.
    parse_one
        ``(raw_bytes, spec) -> DataFrame`` — the source's own read-only parser
        for one series. Used only to read off ``last_observation_date`` + the
        per-series row count; never writes.
    resolve
        ``(spec, entries) -> entry | None`` mapping a spec to its
        ``_metadata.json`` row (per ``series_id``, per ``table``, …).
    """

    name: str
    module: ModuleType
    calendar_module: str | None
    series: tuple[Any, ...]
    identifier: Callable[[Any], str]
    parse_one: Callable[[bytes, Any], pd.DataFrame]
    resolve: Callable[[Any, list[dict[str, Any]]], dict[str, Any] | None]


@dataclass(frozen=True)
class DocumentSource:
    """A source with no ``SERIES`` registry — one synthetic catalog row.

    Covers ``rba_f11`` (the decision frame), ``asx_ib_futures`` (dynamic
    per-contract), and the four text scrapers. ``last_observation_date`` and the
    document count come from the materialised artifact when present, else (for
    ``rba_f11``) from re-parsing its single cached raw snapshot.

    Attributes
    ----------
    name
        Source identifier == ``data/raw/<name>/`` directory.
    module
        The source module.
    calendar_module
        Release-calendar module name, or ``None``.
    identifier
        Value for the ``dataflow/table`` column (a short descriptive label).
    external_artifact
        Filename under ``data/external/`` whose row count + ``date_column`` give
        the document count and ``last_observation_date``, or ``None``.
    date_column
        Column in ``external_artifact`` (or in the re-parsed raw frame when
        ``parse_raw`` is set) to take the max of for ``last_observation_date``.
    parse_raw
        Optional ``raw_bytes -> DataFrame`` used when there is no external
        artifact (``rba_f11``): re-parses the single cached snapshot read-only.
    """

    name: str
    module: ModuleType
    calendar_module: str | None
    identifier: str
    external_artifact: str | None
    date_column: str
    parse_raw: Callable[[bytes], pd.DataFrame] | None = None


# -----------------------------------------------------------------------------
# Resolution strategies (spec -> _metadata.json entry).
# -----------------------------------------------------------------------------
def _resolve_by_series_id(spec: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-``series_id`` metadata (ABS / FRED / Yahoo multi-series sources)."""
    return next((e for e in entries if e.get("series_id") == spec.series_id), None)


def _resolve_single(spec: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Single shared-file metadata (one RBA CSV table covering every series)."""
    return entries[0] if entries else None


def _resolve_by_table(spec: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-``table`` metadata where the spec names its table (``rba_d``: d1/d2)."""
    return next((e for e in entries if e.get("table") == spec.table), None)


def _resolve_rba_h3(spec: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The RBA-H3 leg of ``abs_building_approvals`` (its ``source == 'rba_h3'``)."""
    return next((e for e in entries if e.get("source") == "rba_h3"), None)


def _resolve_live_csv(spec: Any, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Splice sources (``aud``/``bbsw``): the live ``.csv`` carries the latest data."""
    return next(
        (e for e in entries if str(e.get("snapshot_filename", "")).endswith(".csv")),
        None,
    )


# -----------------------------------------------------------------------------
# ``dataflow/table`` identifier helpers.
# -----------------------------------------------------------------------------
def _abs_identifier(spec: Any) -> str:
    return f"{spec.dataflow}/{spec.datakey}"


def _fred_identifier(spec: Any) -> str:
    return str(spec.fred_series_id)


def _rba_identifier(table: str) -> Callable[[Any], str]:
    return lambda spec: f"{table}:{spec.rba_series_id}"


# -----------------------------------------------------------------------------
# parse_one adapters (read-only; reuse each source's own helpers).
# -----------------------------------------------------------------------------
def _abs_sdmx_parse(module: ModuleType) -> Callable[[bytes, Any], pd.DataFrame]:
    return lambda raw, spec: module._parse(raw, series_id=spec.series_id)


def _spec_parse(parse_fn: Callable[..., pd.DataFrame]) -> Callable[[bytes, Any], pd.DataFrame]:
    return lambda raw, spec: parse_fn(raw, spec=spec)


# -----------------------------------------------------------------------------
# The catalog — single source of truth for the inventory (cross-checked against
# refresh.REGISTRY by the CI test). Order here is the DATA.md row order.
# -----------------------------------------------------------------------------
SERIES_SOURCES: tuple[SeriesSource, ...] = (
    SeriesSource(
        name=abs_cpi.SOURCE_NAME,
        module=abs_cpi,
        calendar_module="cpi_release_calendar",
        series=abs_cpi.SERIES,
        identifier=_abs_identifier,
        parse_one=_abs_sdmx_parse(abs_cpi),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=abs_labour_force.SOURCE_NAME,
        module=abs_labour_force,
        calendar_module="lfs_release_calendar",
        series=abs_labour_force.SERIES,
        identifier=_abs_identifier,
        parse_one=_abs_sdmx_parse(abs_labour_force),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=abs_wpi.SOURCE_NAME,
        module=abs_wpi,
        calendar_module="wpi_release_calendar",
        series=abs_wpi.SERIES,
        identifier=_abs_identifier,
        parse_one=_abs_sdmx_parse(abs_wpi),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=abs_gdp.SOURCE_NAME,
        module=abs_gdp,
        calendar_module="gdp_release_calendar",
        series=abs_gdp.SERIES,
        identifier=_abs_identifier,
        parse_one=_abs_sdmx_parse(abs_gdp),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=abs_total_value_dwellings.SOURCE_NAME,
        module=abs_total_value_dwellings,
        calendar_module="abs_tvd_release_calendar",
        series=abs_total_value_dwellings.SERIES,
        identifier=_abs_identifier,
        parse_one=_abs_sdmx_parse(abs_total_value_dwellings),
        resolve=_resolve_by_series_id,
    ),
    # abs_building_approvals is mixed: an ABS-SDMX leg + an RBA-H3 leg.
    SeriesSource(
        name=abs_building_approvals.SOURCE_NAME,
        module=abs_building_approvals,
        calendar_module="abs_ba_release_calendar",
        series=abs_building_approvals.ABS_NSA_SERIES,
        identifier=_abs_identifier,
        parse_one=lambda raw, spec: abs_building_approvals._parse_sdmx(
            raw, series_id=spec.series_id
        ),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=abs_building_approvals.SOURCE_NAME,
        module=abs_building_approvals,
        calendar_module="abs_ba_release_calendar",
        series=abs_building_approvals.RBA_H3_SERIES,
        identifier=_rba_identifier("h3"),
        parse_one=_spec_parse(abs_building_approvals._parse_h3),
        resolve=_resolve_rba_h3,
    ),
    SeriesSource(
        name=rba_d.SOURCE_NAME,
        module=rba_d,
        calendar_module="rba_d_release_calendar",
        series=rba_d.SERIES,
        identifier=lambda spec: f"{spec.table}:{spec.rba_series_id}",
        parse_one=_spec_parse(rba_d._parse),
        resolve=_resolve_by_table,
    ),
    SeriesSource(
        name=rba_e2_household_ratios.SOURCE_NAME,
        module=rba_e2_household_ratios,
        calendar_module="rba_e_release_calendar",
        series=rba_e2_household_ratios.SERIES,
        identifier=_rba_identifier("e2"),
        parse_one=_spec_parse(rba_e2_household_ratios._parse),
        resolve=_resolve_single,
    ),
    SeriesSource(
        name=rba_i2_commodity_prices.SOURCE_NAME,
        module=rba_i2_commodity_prices,
        calendar_module="rba_i2_release_calendar",
        series=rba_i2_commodity_prices.SERIES,
        identifier=_rba_identifier("i2"),
        parse_one=_spec_parse(rba_i2_commodity_prices._parse),
        resolve=_resolve_single,
    ),
    SeriesSource(
        name=nab_business_survey.SOURCE_NAME,
        module=nab_business_survey,
        calendar_module="nab_release_calendar",
        series=nab_business_survey.SERIES,
        identifier=_rba_identifier("h3"),
        parse_one=_spec_parse(nab_business_survey._parse),
        resolve=_resolve_single,
    ),
    SeriesSource(
        name=westpac_mi_consumer_sentiment.SOURCE_NAME,
        module=westpac_mi_consumer_sentiment,
        calendar_module="westpac_mi_release_calendar",
        series=westpac_mi_consumer_sentiment.SERIES,
        identifier=_rba_identifier("h3"),
        parse_one=_spec_parse(westpac_mi_consumer_sentiment._parse),
        resolve=_resolve_single,
    ),
    SeriesSource(
        name=agb_yields.SOURCE_NAME,
        module=agb_yields,
        calendar_module="agb_yields_release_calendar",
        series=agb_yields.SERIES,
        identifier=_rba_identifier("f2"),
        parse_one=_spec_parse(agb_yields._parse),
        resolve=_resolve_single,
    ),
    SeriesSource(
        name=bbsw_rates.SOURCE_NAME,
        module=bbsw_rates,
        calendar_module="agb_yields_release_calendar",  # reused (identical +1 BDay rule)
        series=bbsw_rates.SERIES,
        identifier=_rba_identifier("f1"),
        parse_one=_spec_parse(bbsw_rates._parse_csv),
        resolve=_resolve_live_csv,
    ),
    SeriesSource(
        name=aud_exchange_rates.SOURCE_NAME,
        module=aud_exchange_rates,
        calendar_module=None,
        series=aud_exchange_rates.SERIES,
        identifier=_rba_identifier("f11.1"),
        parse_one=_spec_parse(aud_exchange_rates._parse_csv),
        resolve=_resolve_live_csv,
    ),
    SeriesSource(
        name=asx_200.SOURCE_NAME,
        module=asx_200,
        calendar_module=None,
        series=asx_200.SERIES,
        identifier=lambda spec: spec.yahoo_ticker,
        parse_one=_spec_parse(asx_200._parse_chart_json),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=fred_global_signals.SOURCE_NAME,
        module=fred_global_signals,
        calendar_module="fred_release_calendar",
        series=fred_global_signals.SERIES,
        identifier=_fred_identifier,
        parse_one=_spec_parse(fred_global_signals._parse_csv),
        resolve=_resolve_by_series_id,
    ),
    SeriesSource(
        name=commodity_prices.SOURCE_NAME,
        module=commodity_prices,
        calendar_module="fred_release_calendar",  # shared with fred_global_signals
        series=commodity_prices.SERIES,
        identifier=_fred_identifier,
        parse_one=_spec_parse(commodity_prices._parse_csv),
        resolve=_resolve_by_series_id,
    ),
)

DOCUMENT_SOURCES: tuple[DocumentSource, ...] = (
    DocumentSource(
        name=rba_f11.SOURCE_NAME,
        module=rba_f11,
        calendar_module=None,
        identifier="F1.1/decisions",
        external_artifact=None,
        date_column="observation_date",
        parse_raw=rba_f11._parse,
    ),
    DocumentSource(
        name=asx_ib_futures.SOURCE_NAME,
        module=asx_ib_futures,
        calendar_module=None,
        identifier="ASX/ib_futures",
        external_artifact="asx_ib_futures.csv",
        date_column="trade_date",
    ),
    DocumentSource(
        name=rba_media_releases.SOURCE_NAME,
        module=rba_media_releases,
        calendar_module=None,
        identifier="media-releases",
        external_artifact="rba_media_releases.parquet",
        date_column="publication_date",
    ),
    DocumentSource(
        name=rba_minutes.SOURCE_NAME,
        module=rba_minutes,
        calendar_module="rba_minutes_release_calendar",
        identifier="board-minutes",
        external_artifact="rba_minutes.parquet",
        date_column="publication_date",
    ),
    DocumentSource(
        name=rba_somp.SOURCE_NAME,
        module=rba_somp,
        calendar_module=None,
        identifier="smp/overview",
        external_artifact="rba_somp.parquet",
        date_column="publication_date",
    ),
    DocumentSource(
        name=rba_speeches.SOURCE_NAME,
        module=rba_speeches,
        calendar_module=None,
        identifier="speeches",
        external_artifact="rba_speeches.parquet",
        date_column="publication_date",
    ),
)


# -----------------------------------------------------------------------------
# Raw-metadata access (read-only).
# -----------------------------------------------------------------------------
def read_metadata(raw_root: Path, source_name: str) -> list[dict[str, Any]]:
    """Read ``data/raw/<source>/_metadata.json``, normalised to a list.

    ``rba_f11`` writes a single dict; every other source writes a list. Both are
    returned as ``list[dict]``. A missing file (source never refreshed) returns
    an empty list rather than raising.

    Parameters
    ----------
    raw_root
        The ``data/raw`` root (parameterised so tests can stub ``tmp_path``).
    source_name
        Source identifier == the ``data/raw`` subdirectory name.

    Returns
    -------
    list[dict]
        Provenance entries, possibly empty.
    """
    path = raw_root / source_name / METADATA_FILENAME
    if not path.exists():
        logger.warning("No {} for source {!r} (never refreshed?).", METADATA_FILENAME, source_name)
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def _read_snapshot(raw_root: Path, source_name: str, entry: dict[str, Any], cache: dict[Path, bytes]) -> bytes | None:
    """Read a cached raw snapshot's bytes, memoised per path within a run."""
    filename = entry.get("snapshot_filename")
    if not filename:
        return None
    path = raw_root / source_name / str(filename)
    if path in cache:
        return cache[path]
    if not path.exists():
        logger.warning("Snapshot {} missing for source {!r}.", filename, source_name)
        return None
    data = path.read_bytes()
    cache[path] = data
    return data


def _max_date(frame: pd.DataFrame, column: str) -> pd.Timestamp | None:
    """Max of ``column`` as a tz-naive Timestamp, or ``None`` if unavailable."""
    if column not in frame.columns or frame.empty:
        # Fall back to any single date-like column when the named one is absent.
        candidates = [c for c in frame.columns if "date" in c.lower()]
        if not candidates:
            return None
        column = candidates[0]
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max())


# -----------------------------------------------------------------------------
# Row builders.
# -----------------------------------------------------------------------------
def _series_rows(
    source: SeriesSource,
    raw_root: Path,
    cache: dict[Path, bytes],
) -> list[dict[str, Any]]:
    """Build one inventory row per series in ``source`` (read-only)."""
    entries = read_metadata(raw_root, source.name)
    raw_dir = _relpath(RAW_DATA_DIR / source.name)
    rows: list[dict[str, Any]] = []

    for spec in source.series:
        entry = source.resolve(spec, entries)
        observations: int | None = None
        last_obs: pd.Timestamp | None = None

        if entry is None:
            logger.warning(
                "Series {!r} ({}) has no matching {} row — refresh raw?",
                spec.series_id,
                source.name,
                METADATA_FILENAME,
            )
        else:
            raw_bytes = _read_snapshot(raw_root, source.name, entry, cache)
            if raw_bytes is not None:
                try:
                    parsed = source.parse_one(raw_bytes, spec)
                    observations = int(len(parsed))
                    last_obs = _max_date(parsed, "observation_date")
                except Exception as exc:  # noqa: BLE001 — best-effort, never fatal
                    logger.warning(
                        "Could not parse cached snapshot for {!r} ({}): {}",
                        spec.series_id,
                        source.name,
                        exc,
                    )

        rows.append(
            {
                "series_id": spec.series_id,
                "source_module": _module_name(source.module),
                "dataflow_or_table": source.identifier(spec),
                "source_url": (entry or {}).get("url", ""),
                "raw_dir": raw_dir,
                "snapshot_filename": (entry or {}).get("snapshot_filename", ""),
                "observations": observations,
                "last_observation_date": last_obs,
                "release_calendar_module": source.calendar_module or "",
            }
        )
    return rows


def _document_row(
    source: DocumentSource,
    raw_root: Path,
    external_root: Path,
    cache: dict[Path, bytes],
) -> dict[str, Any]:
    """Build the single synthetic inventory row for a no-SERIES source."""
    entries = read_metadata(raw_root, source.name)
    observations: int | None = None
    last_obs: pd.Timestamp | None = None

    if source.external_artifact is not None:
        artifact = external_root / source.external_artifact
        if artifact.exists():
            try:
                frame = (
                    pd.read_parquet(artifact)
                    if artifact.suffix == ".parquet"
                    else pd.read_csv(artifact)
                )
                observations = int(len(frame))
                last_obs = _max_date(frame, source.date_column)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read artifact {}: {}", artifact, exc)
        else:
            logger.warning("Artifact {} missing for source {!r}.", artifact, source.name)
    elif source.parse_raw is not None and entries:
        raw_bytes = _read_snapshot(raw_root, source.name, entries[0], cache)
        if raw_bytes is not None:
            try:
                frame = source.parse_raw(raw_bytes)
                observations = int(len(frame))
                last_obs = _max_date(frame, source.date_column)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not parse raw for {!r}: {}", source.name, exc)

    return {
        "series_id": source.name,
        "source_module": _module_name(source.module),
        "dataflow_or_table": source.identifier,
        "source_url": _doc_url(entries),
        "raw_dir": _relpath(RAW_DATA_DIR / source.name),
        "snapshot_filename": _doc_snapshot(entries),
        "observations": observations,
        "last_observation_date": last_obs,
        "release_calendar_module": source.calendar_module or "",
    }


def _doc_url(entries: list[dict[str, Any]]) -> str:
    """A representative provenance URL for a document source."""
    for entry in entries:
        url = entry.get("url")
        if url:
            return str(url)
    return ""


def _doc_snapshot(entries: list[dict[str, Any]]) -> str:
    """Snapshot label: the filename if a single snapshot, else a count."""
    if not entries:
        return ""
    if len(entries) == 1:
        return str(entries[0].get("snapshot_filename", ""))
    return f"<{len(entries)} snapshots>"


# -----------------------------------------------------------------------------
# Inventory assembly + the metadata-coverage guard.
# -----------------------------------------------------------------------------
def build_inventory(
    raw_root: Path = RAW_DATA_DIR,
    external_root: Path = EXTERNAL_DATA_DIR,
) -> pd.DataFrame:
    """Assemble the full inventory catalog as a DataFrame.

    Parameters
    ----------
    raw_root
        The ``data/raw`` root (parameterised for tests).
    external_root
        The ``data/external`` root (parameterised for tests).

    Returns
    -------
    pandas.DataFrame
        One row per series (numeric sources) plus one synthetic row per
        document source.

    Shapes
    ------
    Returns: (n_series + n_document_sources, 9) with :data:`COLUMNS` order.
    """
    cache: dict[Path, bytes] = {}
    rows: list[dict[str, Any]] = []
    for source in SERIES_SOURCES:
        rows.extend(_series_rows(source, raw_root, cache))
    for doc in DOCUMENT_SOURCES:
        rows.append(_document_row(doc, raw_root, external_root, cache))

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["last_observation_date"] = pd.to_datetime(
        frame["last_observation_date"], errors="coerce"
    ).astype("datetime64[ns]")
    frame["observations"] = frame["observations"].astype("Int64")
    return frame


def find_missing_metadata(raw_root: Path = RAW_DATA_DIR) -> list[tuple[str, str]]:
    """Registered series whose backing ``_metadata.json`` row is absent.

    This is the inventory's CI guard: it catches "added (or renamed) a series in
    code but forgot to refresh raw". Document sources are exempt (no ``SERIES``).

    Parameters
    ----------
    raw_root
        The ``data/raw`` root (parameterised for tests).

    Returns
    -------
    list[tuple[str, str]]
        ``(source_name, series_id)`` pairs with no matching metadata row.
    """
    missing: list[tuple[str, str]] = []
    for source in SERIES_SOURCES:
        entries = read_metadata(raw_root, source.name)
        for spec in source.series:
            if source.resolve(spec, entries) is None:
                missing.append((source.name, spec.series_id))
    return missing


# -----------------------------------------------------------------------------
# Rendering / materialisation.
# -----------------------------------------------------------------------------
def render_markdown(frame: pd.DataFrame) -> str:
    """Render the inventory as a GitHub-flavoured Markdown document.

    Parameters
    ----------
    frame
        The inventory frame from :func:`build_inventory`.

    Returns
    -------
    str
        The full ``DATA.md`` body (header + table), newline-terminated.
    """
    n_series = sum(len(s.series) for s in SERIES_SOURCES)
    header = (
        "# DATA.md — data inventory & catalog\n\n"
        "**Generated** — do not edit by hand. Regenerate with "
        "`python -m rba.data.inventory`.\n\n"
        f"{n_series} series across {len(_distinct_series_sources())} numeric sources "
        f"+ {len(DOCUMENT_SOURCES)} document/dynamic sources. "
        "Provenance (URL / snapshot / observations) is read from each source's "
        "`data/raw/<source>/_metadata.json`; `last_observation_date` is read from "
        "the latest cached raw snapshot (or the materialised artifact for document "
        "sources). See CONTEXT.md “Data inventory & catalog” for the contract.\n\n"
    )

    display = frame.copy()
    display["last_observation_date"] = (
        display["last_observation_date"].dt.strftime("%Y-%m-%d").fillna("")
    )
    display["observations"] = display["observations"].apply(
        lambda v: "" if pd.isna(v) else str(int(v))
    )
    headers = [
        "series_id",
        "source_module",
        "dataflow/table",
        "source_url",
        "raw_dir",
        "snapshot_filename",
        "observations",
        "last_observation_date",
        "release_calendar_module",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in display.iterrows():
        cells = [_md_cell(row[col]) for col in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return header + "\n".join(lines) + "\n"


def _md_cell(value: Any) -> str:
    """Escape a cell value for a Markdown table (pipes break columns)."""
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return text.replace("|", "\\|")


def write_data_md(frame: pd.DataFrame, path: Path = DATA_MD_PATH) -> None:
    """Materialise ``DATA.md``."""
    path.write_text(render_markdown(frame), encoding="utf-8")
    logger.success("Wrote {} ({} rows).", _relpath(path), len(frame))


def write_parquet(frame: pd.DataFrame, path: Path = INVENTORY_PARQUET_PATH) -> None:
    """Materialise ``data/external/inventory.parquet``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.success("Wrote {} ({} rows).", _relpath(path), len(frame))


def write_xlsx(frame: pd.DataFrame, path: Path = INVENTORY_XLSX_PATH) -> None:
    """Materialise the optional ``data/external/inventory.xlsx`` (openpyxl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False, engine="openpyxl")
    logger.success("Wrote {} ({} rows).", _relpath(path), len(frame))


# -----------------------------------------------------------------------------
# Small shared helpers.
# -----------------------------------------------------------------------------
def _module_name(module: ModuleType) -> str:
    """Short module name (``abs_cpi``) rather than the dotted path."""
    return module.__name__.rsplit(".", 1)[-1]


def _relpath(path: Path) -> str:
    """Repo-root-relative POSIX path for display, falling back to the name."""
    try:
        return path.resolve().relative_to(PROJ_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _distinct_series_sources() -> set[str]:
    """Distinct numeric source names (``abs_building_approvals`` appears twice)."""
    return {s.name for s in SERIES_SOURCES}


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.data.inventory`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.data.inventory",
        description="Regenerate the data inventory (DATA.md + inventory.parquet).",
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        help="Also emit data/external/inventory.xlsx (optional spreadsheet copy).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report series with no matching _metadata.json row and exit "
        "non-zero if any are missing (writes nothing).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.check:
        missing = find_missing_metadata()
        if missing:
            for source_name, series_id in missing:
                logger.error("MISSING metadata row: {} / {}", source_name, series_id)
            logger.error("{} registered series have no _metadata.json row.", len(missing))
            return 1
        logger.success("All registered series have a matching _metadata.json row.")
        return 0

    frame = build_inventory()
    write_data_md(frame)
    write_parquet(frame)
    if args.xlsx:
        write_xlsx(frame)

    missing = find_missing_metadata()
    if missing:
        logger.warning(
            "{} registered series have no _metadata.json row (run with --check for the list).",
            len(missing),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
