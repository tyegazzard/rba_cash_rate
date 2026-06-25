"""Tests for ``rba.data.refresh`` — the data-source refresh orchestrator.

No network is touched: the registered ``fetch`` callables are replaced with
synthetic stubs that return tiny frames (or raise), and the module-level
``REGISTRY`` is monkeypatched for the CLI / ``main`` paths. The tests assert
the registry contents, per-source error isolation, the succeeded / failed /
skipped summary + exit code, ``force_download`` threading, F11 reuse + the
dependency-skip rule, ``--list`` and ``--source`` subset selection, and the CLI
arg parsing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from rba.data import refresh
from rba.data.refresh import (
    REGISTRY,
    REGISTRY_BY_NAME,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCEEDED,
    RefreshSummary,
    SourceResult,
    SourceSpec,
    build_parser,
    format_registry,
    main,
    select_specs,
)
from rba.data.refresh import (
    refresh as run_refresh,
)

# -----------------------------------------------------------------------------
# Synthetic stub fetch callables (no network).
# -----------------------------------------------------------------------------


def _tiny_frame(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "value": range(n),
        }
    )


def _ok_fetch(rows: int = 2, *, record: list[dict] | None = None):
    """A fetch stub that records its kwargs and returns a tiny frame."""

    def fetch(**kwargs: object) -> pd.DataFrame:
        if record is not None:
            record.append(kwargs)
        return _tiny_frame(rows)

    return fetch


def _boom_fetch(exc: Exception | None = None):
    """A fetch stub that raises (network/parse failure simulation)."""

    def fetch(**kwargs: object) -> pd.DataFrame:
        raise exc or RuntimeError("simulated WAF 403")

    return fetch


@pytest.fixture
def loguru_messages():
    """Capture loguru messages (all levels) emitted within the test."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level=0)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


# -----------------------------------------------------------------------------
# Registry contents.
# -----------------------------------------------------------------------------


def test_registry_contains_expected_entries() -> None:
    names = [spec.name for spec in REGISTRY]
    assert names == ["rba_f11", "abs_cpi"]


def test_registry_f11_listed_first() -> None:
    # F11 must precede any needs_f11 source so its frame is available to reuse.
    assert REGISTRY[0].name == refresh.F11_SOURCE_NAME


def test_registry_wired_sources_do_not_materialise() -> None:
    # rba_f11 and abs_cpi write nothing to data/external/ — materialise is None.
    assert all(spec.materialise is None for spec in REGISTRY)
    assert all(not spec.needs_f11 for spec in REGISTRY)


def test_registry_by_name_matches_registry() -> None:
    assert set(REGISTRY_BY_NAME) == {spec.name for spec in REGISTRY}
    assert REGISTRY_BY_NAME["rba_f11"] is REGISTRY[0]


def test_registry_fetch_callables_are_the_source_fetches() -> None:
    from rba.data.sources import abs_cpi, rba_f11

    assert REGISTRY_BY_NAME["rba_f11"].fetch is rba_f11.fetch
    assert REGISTRY_BY_NAME["abs_cpi"].fetch is abs_cpi.fetch


# -----------------------------------------------------------------------------
# select_specs.
# -----------------------------------------------------------------------------


def test_select_specs_none_returns_all() -> None:
    assert select_specs(None) == list(REGISTRY)


def test_select_specs_empty_returns_all() -> None:
    assert select_specs([]) == list(REGISTRY)


def test_select_specs_subset() -> None:
    selected = select_specs(["abs_cpi"])
    assert [s.name for s in selected] == ["abs_cpi"]


def test_select_specs_preserves_registry_order_and_dedupes() -> None:
    # Requested out of order + duplicated; output is registry order, unique.
    selected = select_specs(["abs_cpi", "rba_f11", "abs_cpi"])
    assert [s.name for s in selected] == ["rba_f11", "abs_cpi"]


def test_select_specs_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        select_specs(["not_a_source"])


# -----------------------------------------------------------------------------
# refresh — error isolation + summary + exit code.
# -----------------------------------------------------------------------------


def test_refresh_all_success_exit_zero() -> None:
    specs = [
        SourceSpec(name="a", fetch=_ok_fetch(3)),
        SourceSpec(name="b", fetch=_ok_fetch(5)),
    ]
    summary = run_refresh(specs)
    assert summary.exit_code == 0
    assert [r.name for r in summary.succeeded] == ["a", "b"]
    assert summary.succeeded[0].rows == 3
    assert summary.succeeded[1].rows == 5


def test_refresh_isolates_failing_source() -> None:
    # 'b' raises but must not abort 'a' or 'c'.
    specs = [
        SourceSpec(name="a", fetch=_ok_fetch()),
        SourceSpec(name="b", fetch=_boom_fetch()),
        SourceSpec(name="c", fetch=_ok_fetch()),
    ]
    summary = run_refresh(specs)
    assert [r.name for r in summary.succeeded] == ["a", "c"]
    assert [r.name for r in summary.failed] == ["b"]
    assert summary.exit_code == 1
    assert "simulated WAF 403" in summary.failed[0].error


def test_refresh_logs_failure_with_traceback(loguru_messages) -> None:
    run_refresh([SourceSpec(name="b", fetch=_boom_fetch())])
    joined = "".join(loguru_messages)
    assert "failed" in joined
    # loguru's opt(exception=True) appends the traceback block.
    assert "Traceback" in joined


def test_refresh_summary_is_logged(loguru_messages) -> None:
    run_refresh(
        [
            SourceSpec(name="a", fetch=_ok_fetch()),
            SourceSpec(name="b", fetch=_boom_fetch()),
        ]
    )
    joined = "".join(loguru_messages)
    assert "1 succeeded, 1 failed, 0 skipped" in joined


def test_refresh_fail_fast_reraises() -> None:
    specs = [
        SourceSpec(name="a", fetch=_ok_fetch()),
        SourceSpec(name="b", fetch=_boom_fetch(ValueError("parse error"))),
        SourceSpec(name="c", fetch=_ok_fetch()),
    ]
    with pytest.raises(ValueError, match="parse error"):
        run_refresh(specs, fail_fast=True)


def test_refresh_materialise_called_with_frame() -> None:
    captured: list[pd.DataFrame] = []
    spec = SourceSpec(
        name="a",
        fetch=_ok_fetch(4),
        materialise=lambda df: captured.append(df),
    )
    run_refresh([spec])
    assert len(captured) == 1
    assert len(captured[0]) == 4


def test_refresh_materialise_failure_marks_source_failed() -> None:
    def boom_materialise(df: pd.DataFrame) -> None:
        raise OSError("disk full")

    summary = run_refresh(
        [SourceSpec(name="a", fetch=_ok_fetch(), materialise=boom_materialise)]
    )
    assert summary.exit_code == 1
    assert "disk full" in summary.failed[0].error


# -----------------------------------------------------------------------------
# force_download threading.
# -----------------------------------------------------------------------------


def test_refresh_threads_force_download_true_by_default() -> None:
    record: list[dict] = []
    run_refresh([SourceSpec(name="a", fetch=_ok_fetch(record=record))])
    assert record[0]["force_download"] is True


def test_refresh_threads_force_download_false() -> None:
    record: list[dict] = []
    run_refresh(
        [SourceSpec(name="a", fetch=_ok_fetch(record=record))],
        force_download=False,
    )
    assert record[0]["force_download"] is False


def test_refresh_passes_spec_kwargs() -> None:
    record: list[dict] = []
    spec = SourceSpec(
        name="a",
        fetch=_ok_fetch(record=record),
        kwargs={"start_period": "1993-Q1"},
    )
    run_refresh([spec])
    assert record[0]["start_period"] == "1993-Q1"
    assert record[0]["force_download"] is True


# -----------------------------------------------------------------------------
# F11 reuse + dependency-skip.
# -----------------------------------------------------------------------------


def test_refresh_threads_f11_frame_into_dependent() -> None:
    f11_frame = _tiny_frame(7)
    record: list[dict] = []
    specs = [
        SourceSpec(name="rba_f11", fetch=lambda **k: f11_frame),
        SourceSpec(name="dependent", fetch=_ok_fetch(record=record), needs_f11=True),
    ]
    run_refresh(specs)
    assert "f11_meetings" in record[0]
    assert record[0]["f11_meetings"] is f11_frame


def test_refresh_skips_dependent_when_f11_fails() -> None:
    specs = [
        SourceSpec(name="rba_f11", fetch=_boom_fetch()),
        SourceSpec(name="dependent", fetch=_ok_fetch(), needs_f11=True),
    ]
    summary = run_refresh(specs)
    assert [r.name for r in summary.failed] == ["rba_f11"]
    assert [r.name for r in summary.skipped] == ["dependent"]
    # A skipped dependency does not, by itself, change that the run had a failure.
    assert summary.exit_code == 1


def test_refresh_dependent_self_fetches_when_f11_absent() -> None:
    # needs_f11 source run without F11 in the set: no f11_meetings injected,
    # so the source falls back to its own cached F11 (default None).
    record: list[dict] = []
    run_refresh(
        [SourceSpec(name="dependent", fetch=_ok_fetch(record=record), needs_f11=True)]
    )
    assert "f11_meetings" not in record[0]


# -----------------------------------------------------------------------------
# RefreshSummary.
# -----------------------------------------------------------------------------


def test_summary_exit_code_zero_when_only_skipped() -> None:
    summary = RefreshSummary(
        results=[
            SourceResult(name="a", status=STATUS_SUCCEEDED, rows=1),
            SourceResult(name="b", status=STATUS_SKIPPED),
        ]
    )
    assert summary.exit_code == 0


def test_summary_partitions_by_status() -> None:
    summary = RefreshSummary(
        results=[
            SourceResult(name="a", status=STATUS_SUCCEEDED),
            SourceResult(name="b", status=STATUS_FAILED),
            SourceResult(name="c", status=STATUS_SKIPPED),
        ]
    )
    assert [r.name for r in summary.succeeded] == ["a"]
    assert [r.name for r in summary.failed] == ["b"]
    assert [r.name for r in summary.skipped] == ["c"]


# -----------------------------------------------------------------------------
# format_registry.
# -----------------------------------------------------------------------------


def test_format_registry_lists_every_source() -> None:
    rendered = format_registry()
    assert "rba_f11" in rendered
    assert "abs_cpi" in rendered
    # Neither wired source materialises an external artifact.
    assert "raw-only" in rendered


# -----------------------------------------------------------------------------
# CLI parsing + main.
# -----------------------------------------------------------------------------


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.sources is None
    assert args.list_registry is False
    assert args.no_download is False
    assert args.fail_fast is False


def test_parser_source_is_repeatable() -> None:
    args = build_parser().parse_args(["--source", "rba_f11", "--source", "abs_cpi"])
    assert args.sources == ["rba_f11", "abs_cpi"]


def test_parser_only_is_alias_for_source() -> None:
    args = build_parser().parse_args(["--only", "abs_cpi"])
    assert args.sources == ["abs_cpi"]


def test_parser_flags() -> None:
    args = build_parser().parse_args(["--no-download", "--fail-fast", "--list"])
    assert args.no_download is True
    assert args.fail_fast is True
    assert args.list_registry is True


def test_main_list_returns_zero_and_logs(loguru_messages) -> None:
    code = main(["--list"])
    assert code == 0
    assert any("Registered data sources" in m for m in loguru_messages)


def test_main_runs_selected_subset(monkeypatch) -> None:
    calls: list[str] = []

    def make(name: str):
        def fetch(**kwargs: object) -> pd.DataFrame:
            calls.append(name)
            return _tiny_frame()

        return fetch

    stub_registry = (
        SourceSpec(name="rba_f11", fetch=make("rba_f11")),
        SourceSpec(name="abs_cpi", fetch=make("abs_cpi")),
    )
    monkeypatch.setattr(refresh, "REGISTRY", stub_registry)
    monkeypatch.setattr(
        refresh, "REGISTRY_BY_NAME", {s.name: s for s in stub_registry}
    )

    code = main(["--source", "abs_cpi"])
    assert code == 0
    assert calls == ["abs_cpi"]  # rba_f11 not selected, not called


def test_main_threads_no_download(monkeypatch) -> None:
    record: list[dict] = []
    stub_registry = (SourceSpec(name="abs_cpi", fetch=_ok_fetch(record=record)),)
    monkeypatch.setattr(refresh, "REGISTRY", stub_registry)
    monkeypatch.setattr(
        refresh, "REGISTRY_BY_NAME", {s.name: s for s in stub_registry}
    )

    main(["--no-download"])
    assert record[0]["force_download"] is False


def test_main_nonzero_exit_on_failure(monkeypatch) -> None:
    stub_registry = (SourceSpec(name="abs_cpi", fetch=_boom_fetch()),)
    monkeypatch.setattr(refresh, "REGISTRY", stub_registry)
    monkeypatch.setattr(
        refresh, "REGISTRY_BY_NAME", {s.name: s for s in stub_registry}
    )
    assert main([]) == 1


def test_main_unknown_source_exits_two(monkeypatch) -> None:
    # argparse parser.error raises SystemExit(2).
    with pytest.raises(SystemExit) as excinfo:
        main(["--source", "nope"])
    assert excinfo.value.code == 2
