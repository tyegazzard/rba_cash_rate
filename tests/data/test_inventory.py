"""Tests for ``rba.data.inventory`` — the data inventory & catalog.

No network and no dependence on the real ``data/raw/`` contents: the metadata
walk and the coverage guard are exercised against synthetic ``_metadata.json``
files written under ``tmp_path``. The catalog itself (the ``SERIES`` tuples and
parse helpers) is imported from the real source modules — that is in-process and
offline.

The headline test is the CI guard: every registered ``SERIES`` entry must have a
matching ``_metadata.json`` row (the "added/renamed a series but forgot to
refresh raw" catch).
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import pytest

from rba.data import inventory, refresh
from rba.data.inventory import (
    COLUMNS,
    DOCUMENT_SOURCES,
    SERIES_SOURCES,
    build_inventory,
    find_missing_metadata,
    read_metadata,
    render_markdown,
)


# -----------------------------------------------------------------------------
# Helpers — write a synthetic raw tree that fully covers every registered series.
# -----------------------------------------------------------------------------
def _entry_for(source, spec) -> dict[str, object]:
    """A minimal ``_metadata.json`` entry that ``source.resolve`` will match.

    Mirrors the real heterogeneity: per-``series_id`` sources key on series_id;
    the RBA-H3 leg keys on ``source == 'rba_h3'``; ``rba_d`` keys on ``table``;
    splice sources resolve to the live ``.csv``; single-table sources take the
    sole entry.
    """
    base = {
        "url": f"https://example.test/{spec.series_id}",
        "snapshot_filename": f"2026-01-01__{spec.series_id}.csv",
        "sha256": "0" * 64,
        "downloaded_at_utc": "2026-01-01T00:00:00+00:00",
        "observations": 1,
    }
    if source.resolve is inventory._resolve_by_series_id:
        base["series_id"] = spec.series_id
    elif source.resolve is inventory._resolve_rba_h3:
        base["source"] = "rba_h3"
    elif source.resolve is inventory._resolve_by_table:
        base["table"] = spec.table
    # _resolve_single / _resolve_live_csv match on the .csv snapshot already set.
    return base


def _write_full_raw_tree(raw_root: Path) -> None:
    """Write one ``_metadata.json`` per source that covers all its series.

    Entries accumulate by source *name* — ``abs_building_approvals`` appears as
    two catalog records (an ABS-SDMX leg + an RBA-H3 leg) sharing one raw dir, so
    their rows must merge into a single file rather than overwrite each other.
    """
    by_name: dict[str, list[dict[str, object]]] = {}
    for source in SERIES_SOURCES:
        entries = by_name.setdefault(source.name, [])
        for spec in source.series:
            entry = _entry_for(source, spec)
            # Per-series-id / per-table / rba-h3 need a distinct row each; the
            # single-shared and live-csv strategies match any one ``.csv`` entry.
            if source.resolve in (
                inventory._resolve_by_series_id,
                inventory._resolve_by_table,
                inventory._resolve_rba_h3,
            ):
                entries.append(entry)
            elif not entries:
                entries.append(entry)
    for name, entries in by_name.items():
        _write_metadata(raw_root, name, entries)


def _write_metadata(raw_root: Path, name: str, payload: object) -> None:
    source_dir = raw_root / name
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / inventory.METADATA_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def loguru_messages():
    """Capture loguru messages (all levels) emitted within the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# -----------------------------------------------------------------------------
# The CI guard.
# -----------------------------------------------------------------------------
def test_full_tree_has_no_missing_metadata(tmp_path: Path) -> None:
    """A raw tree covering every series → the coverage guard is clean."""
    _write_full_raw_tree(tmp_path)
    assert find_missing_metadata(raw_root=tmp_path) == []


def test_dropped_series_is_flagged(tmp_path: Path) -> None:
    """Removing one series' backing row → exactly that series is flagged."""
    _write_full_raw_tree(tmp_path)

    # Pick a per-series-id source and drop one of its metadata rows.
    target = next(s for s in SERIES_SOURCES if s.resolve is inventory._resolve_by_series_id)
    dropped = target.series[0]
    path = tmp_path / target.name / inventory.METADATA_FILENAME
    entries = [e for e in json.loads(path.read_text()) if e.get("series_id") != dropped.series_id]
    path.write_text(json.dumps(entries), encoding="utf-8")

    missing = find_missing_metadata(raw_root=tmp_path)
    assert (target.name, dropped.series_id) in missing
    assert len(missing) == 1


def test_missing_source_dir_flags_all_its_series(tmp_path: Path) -> None:
    """A source with no ``_metadata.json`` at all → every series flagged."""
    _write_full_raw_tree(tmp_path)
    target = SERIES_SOURCES[0]
    (tmp_path / target.name / inventory.METADATA_FILENAME).unlink()

    missing = dict.fromkeys(find_missing_metadata(raw_root=tmp_path))
    for spec in target.series:
        assert (target.name, spec.series_id) in missing


# -----------------------------------------------------------------------------
# Metadata reading — both on-disk shapes.
# -----------------------------------------------------------------------------
def test_read_metadata_normalises_single_dict(tmp_path: Path) -> None:
    """The single-dict shape (rba_f11) is normalised to a one-element list."""
    _write_metadata(tmp_path, "rba_f11", {"url": "x", "snapshot_filename": "s.html"})
    entries = read_metadata(tmp_path, "rba_f11")
    assert isinstance(entries, list)
    assert entries[0]["snapshot_filename"] == "s.html"


def test_read_metadata_missing_returns_empty(tmp_path: Path, loguru_messages: list) -> None:
    """A never-refreshed source returns ``[]`` and warns rather than raising."""
    assert read_metadata(tmp_path, "nope") == []
    assert any("never refreshed" in str(m) for m in loguru_messages)


# -----------------------------------------------------------------------------
# Catalog integrity + REGISTRY cross-check.
# -----------------------------------------------------------------------------
def test_every_series_source_imports_a_real_series_tuple() -> None:
    """Each catalog source's ``series`` is the module's own registry (non-empty)."""
    for source in SERIES_SOURCES:
        assert len(source.series) > 0
        for spec in source.series:
            assert isinstance(spec.series_id, str) and spec.series_id


def test_catalog_consistent_with_refresh_registry() -> None:
    """Every source wired in refresh.REGISTRY appears in the inventory catalog.

    REGISTRY is the source-of-truth for "how to refresh"; the catalog is the
    source-of-truth for "what to inventory". For the wired subset they must
    agree, so the two cannot silently diverge as more sources are wired.
    """
    catalog_names = {s.name for s in SERIES_SOURCES} | {d.name for d in DOCUMENT_SOURCES}
    registry_names = {spec.name for spec in refresh.REGISTRY}
    assert registry_names <= catalog_names


def test_document_sources_are_disjoint_from_series_sources() -> None:
    series_names = {s.name for s in SERIES_SOURCES}
    doc_names = {d.name for d in DOCUMENT_SOURCES}
    assert series_names.isdisjoint(doc_names)


# -----------------------------------------------------------------------------
# build_inventory + rendering (offline, against the stub tree).
# -----------------------------------------------------------------------------
def test_build_inventory_shape_and_columns(tmp_path: Path) -> None:
    """One row per series + one per document source, in the mandated schema."""
    _write_full_raw_tree(tmp_path)
    frame = build_inventory(raw_root=tmp_path, external_root=tmp_path / "external")

    assert list(frame.columns) == list(COLUMNS)
    expected = sum(len(s.series) for s in SERIES_SOURCES) + len(DOCUMENT_SOURCES)
    assert len(frame) == expected
    assert str(frame["last_observation_date"].dtype) == "datetime64[ns]"
    assert str(frame["observations"].dtype) == "Int64"


def test_render_markdown_table_well_formed(tmp_path: Path) -> None:
    """The rendered table has a header, a divider, and one row per record."""
    _write_full_raw_tree(tmp_path)
    frame = build_inventory(raw_root=tmp_path, external_root=tmp_path / "external")
    md = render_markdown(frame)

    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert lines[0].startswith("| series_id |")
    assert set(lines[1].replace("|", "").strip()) == {"-"}
    assert len(lines) == len(frame) + 2  # header + divider + rows


def test_render_markdown_escapes_pipes() -> None:
    """A pipe in a cell is escaped so it cannot break the column layout."""
    frame = pd.DataFrame(
        [
            {
                "series_id": "a|b",
                "source_module": "m",
                "dataflow_or_table": "t",
                "source_url": "u",
                "raw_dir": "d",
                "snapshot_filename": "s",
                "observations": pd.NA,
                "last_observation_date": pd.NaT,
                "release_calendar_module": "",
            }
        ],
        columns=list(COLUMNS),
    )
    frame["observations"] = frame["observations"].astype("Int64")
    md = render_markdown(frame)
    assert "a\\|b" in md
