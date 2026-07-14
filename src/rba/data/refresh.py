"""Data-source refresh orchestrator — re-pull every registered source.

This is the first non-source, non-feature module in ``src/rba/data/``. It sits
on top of the ~23 source modules in ``src/rba/data/sources/`` and provides a
single CLI to re-pull them with per-source error isolation, structured logging,
and a succeeded / failed / skipped summary.

Design
------
The registry (:data:`REGISTRY`) is a tuple of :class:`SourceSpec` entries — the
single source of truth for "what sources exist and how to refresh them". It is
deliberately importable so the later ``inventory.py`` (see CHECKLIST §3 "Data
inventory & catalog") can reuse it rather than redefining the source list.

Each source's ``fetch()`` already writes the immutable raw snapshot plus
``data/raw/<source>/_metadata.json`` (provenance / SHA-256 — Invariant #5). The
orchestrator therefore does **not** re-implement hashing or raw writes; it only
*drives* ``fetch()`` and, optionally, a per-source ``materialise`` callable for
the ``data/external/`` wide artifact. Most numeric sources currently materialise
their wide CSV inside their own ``if __name__ == "__main__"`` block, and the two
sources wired here (``rba_f11``, ``abs_cpi``) materialise *nothing* to
``data/external/`` — so their ``materialise`` is ``None``. The hook exists so the
remaining ~21 sources can be wired with a single registry line each later,
without re-architecting this module.

F11 reuse
---------
The F11-dependent text sources (``rba_media_releases`` / ``rba_minutes`` /
``rba_somp``) accept an already-fetched F11 meetings frame via an
``f11_meetings`` keyword rather than each re-downloading it. When such a source
is registered with ``needs_f11=True`` and F11 is refreshed earlier in the same
run, the orchestrator threads F11's result into it. None of the two sources
wired today need this, but the mechanism is in place for the rollout.

CLI
---
Run as ``python -m rba.data.refresh``:

- (default)        refresh every registered source
- ``--source`` / ``--only NAME``  refresh only NAME (repeatable)
- ``--list``       print the registry and exit
- ``--no-download``  pass ``force_download=False`` so sources reuse cached raw
                     snapshots (the live crawls are large / slow)
- ``--fail-fast``  abort on the first failure instead of isolating it

Exit code is non-zero if any source failed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time

from loguru import logger
import pandas as pd

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

# Status labels for a single source's refresh outcome.
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# The source whose meeting frame the F11-dependent text sources reuse.
F11_SOURCE_NAME = rba_f11.SOURCE_NAME


@dataclass(frozen=True)
class SourceSpec:
    """How to refresh one data source.

    Attributes
    ----------
    name
        Stable source identifier; matches the source module's ``SOURCE_NAME``
        and the ``data/raw/<name>/`` directory.
    fetch
        The source's ``fetch`` callable. Always invoked with
        ``force_download=<bool>``; receives ``**kwargs`` and, when
        ``needs_f11`` is set and an F11 frame is available, ``f11_meetings=``.
    kwargs
        Extra keyword arguments to pass to ``fetch`` (e.g. ``start_period`` for
        the ABS sources). ``force_download`` / ``f11_meetings`` are injected by
        the orchestrator and must not be set here.
    materialise
        Optional callable that writes the ``data/external/`` artifact from the
        frame ``fetch`` returns. ``None`` for sources that don't materialise an
        external artifact (e.g. ``rba_f11``, ``abs_cpi``).
    needs_f11
        If True, ``fetch`` accepts an ``f11_meetings`` keyword and the
        orchestrator threads the F11 frame in when it is available this run.
    """

    name: str
    fetch: Callable[..., pd.DataFrame]
    kwargs: Mapping[str, object] = field(default_factory=dict)
    materialise: Callable[[pd.DataFrame], None] | None = None
    needs_f11: bool = False


# -----------------------------------------------------------------------------
# The registry — single source of truth (also imported by inventory.py later).
# rba_f11 is listed first so the F11 frame is available to any needs_f11 source
# refreshed after it in the same run. Ordering after F11: numeric sources
# (ABS macro → RBA aggregates → market/global) then text sources (three of
# which are F11-driven, one is archive-driven).
# -----------------------------------------------------------------------------
REGISTRY: tuple[SourceSpec, ...] = (
    # Target + meeting frame — must be first.
    SourceSpec(name=rba_f11.SOURCE_NAME, fetch=rba_f11.fetch),
    # ABS macro (CPI / labour / wages / GDP / housing).
    SourceSpec(name=abs_cpi.SOURCE_NAME, fetch=abs_cpi.fetch),
    SourceSpec(name=abs_labour_force.SOURCE_NAME, fetch=abs_labour_force.fetch),
    SourceSpec(name=abs_wpi.SOURCE_NAME, fetch=abs_wpi.fetch),
    SourceSpec(name=abs_gdp.SOURCE_NAME, fetch=abs_gdp.fetch),
    SourceSpec(name=abs_building_approvals.SOURCE_NAME, fetch=abs_building_approvals.fetch),
    SourceSpec(name=abs_total_value_dwellings.SOURCE_NAME, fetch=abs_total_value_dwellings.fetch),
    # RBA statistical aggregates + commodity price index.
    SourceSpec(name=rba_d.SOURCE_NAME, fetch=rba_d.fetch),
    SourceSpec(name=rba_e2_household_ratios.SOURCE_NAME, fetch=rba_e2_household_ratios.fetch),
    SourceSpec(name=rba_i2_commodity_prices.SOURCE_NAME, fetch=rba_i2_commodity_prices.fetch),
    # Sentiment surveys.
    SourceSpec(name=nab_business_survey.SOURCE_NAME, fetch=nab_business_survey.fetch),
    SourceSpec(
        name=westpac_mi_consumer_sentiment.SOURCE_NAME,
        fetch=westpac_mi_consumer_sentiment.fetch,
    ),
    # Market data (rates / FX / equities / futures).
    SourceSpec(name=agb_yields.SOURCE_NAME, fetch=agb_yields.fetch),
    SourceSpec(name=bbsw_rates.SOURCE_NAME, fetch=bbsw_rates.fetch),
    SourceSpec(name=aud_exchange_rates.SOURCE_NAME, fetch=aud_exchange_rates.fetch),
    SourceSpec(name=asx_200.SOURCE_NAME, fetch=asx_200.fetch),
    SourceSpec(name=asx_ib_futures.SOURCE_NAME, fetch=asx_ib_futures.fetch),
    # Global signals + commodities via FRED.
    SourceSpec(name=fred_global_signals.SOURCE_NAME, fetch=fred_global_signals.fetch),
    SourceSpec(name=commodity_prices.SOURCE_NAME, fetch=commodity_prices.fetch),
    # Text sources — three F11-driven, one archive-driven.
    SourceSpec(
        name=rba_media_releases.SOURCE_NAME, fetch=rba_media_releases.fetch, needs_f11=True
    ),
    SourceSpec(name=rba_minutes.SOURCE_NAME, fetch=rba_minutes.fetch, needs_f11=True),
    SourceSpec(name=rba_somp.SOURCE_NAME, fetch=rba_somp.fetch, needs_f11=True),
    SourceSpec(name=rba_speeches.SOURCE_NAME, fetch=rba_speeches.fetch),
)

REGISTRY_BY_NAME: dict[str, SourceSpec] = {spec.name: spec for spec in REGISTRY}


@dataclass(frozen=True)
class SourceResult:
    """Outcome of refreshing a single source."""

    name: str
    status: str
    rows: int | None = None
    error: str | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class RefreshSummary:
    """Aggregate outcome of a refresh run.

    ``exit_code`` is non-zero iff at least one source failed; a skipped source
    (a dependency was unavailable) does not by itself force a non-zero exit.
    """

    results: list[SourceResult]

    @property
    def succeeded(self) -> list[SourceResult]:
        return [r for r in self.results if r.status == STATUS_SUCCEEDED]

    @property
    def failed(self) -> list[SourceResult]:
        return [r for r in self.results if r.status == STATUS_FAILED]

    @property
    def skipped(self) -> list[SourceResult]:
        return [r for r in self.results if r.status == STATUS_SKIPPED]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def select_specs(names: Sequence[str] | None) -> list[SourceSpec]:
    """Resolve a list of source names to their specs, in registry order.

    Parameters
    ----------
    names
        Source names to select. ``None`` or empty selects every registered
        source (the default "run-all" behaviour).

    Returns
    -------
    list[SourceSpec]
        Selected specs, de-duplicated and ordered as in :data:`REGISTRY` (so
        F11 still precedes any ``needs_f11`` source).

    Raises
    ------
    ValueError
        If any requested name is not registered.
    """
    if not names:
        return list(REGISTRY)

    unknown = [n for n in names if n not in REGISTRY_BY_NAME]
    if unknown:
        valid = ", ".join(spec.name for spec in REGISTRY)
        raise ValueError(f"Unknown source(s): {', '.join(unknown)}. Registered sources: {valid}.")

    requested = set(names)
    return [spec for spec in REGISTRY if spec.name in requested]


def _invoke(
    spec: SourceSpec,
    *,
    force_download: bool,
    f11_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Call ``spec.fetch`` with the orchestrator-injected keywords."""
    kwargs: dict[str, object] = dict(spec.kwargs)
    kwargs["force_download"] = force_download
    if spec.needs_f11 and f11_frame is not None:
        kwargs["f11_meetings"] = f11_frame
    return spec.fetch(**kwargs)


def refresh(
    specs: Sequence[SourceSpec],
    *,
    force_download: bool = True,
    fail_fast: bool = False,
) -> RefreshSummary:
    """Refresh each source in ``specs`` with per-source error isolation.

    One source failing (network / WAF 403 / parse error) does not abort the run
    unless ``fail_fast`` is set: the failure is logged with its traceback, and
    the run continues. An F11-dependent source whose F11 dependency failed
    earlier in this run is skipped rather than attempted.

    Parameters
    ----------
    specs
        Source specs to refresh, in order. F11 should precede any
        ``needs_f11`` source (``select_specs`` guarantees this).
    force_download
        Passed through to every ``fetch`` as ``force_download``. ``False``
        reuses cached raw snapshots.
    fail_fast
        If True, re-raise the first source exception instead of isolating it.

    Returns
    -------
    RefreshSummary
        Per-source results plus the run exit code.
    """
    results: list[SourceResult] = []
    f11_frame: pd.DataFrame | None = None
    f11_failed = False

    for spec in specs:
        if spec.needs_f11 and f11_failed:
            logger.warning("Skipping {!r}: its F11 dependency failed earlier this run.", spec.name)
            results.append(
                SourceResult(
                    name=spec.name,
                    status=STATUS_SKIPPED,
                    error="F11 dependency failed",
                )
            )
            continue

        logger.info("Refreshing {!r} (force_download={}) ...", spec.name, force_download)
        start = time.perf_counter()
        try:
            df = _invoke(spec, force_download=force_download, f11_frame=f11_frame)
            if spec.materialise is not None:
                spec.materialise(df)
        except Exception as exc:  # noqa: BLE001 — isolation is the whole point
            duration = time.perf_counter() - start
            logger.opt(exception=True).error("Source {!r} failed: {}", spec.name, exc)
            if spec.name == F11_SOURCE_NAME:
                f11_failed = True
            results.append(
                SourceResult(
                    name=spec.name,
                    status=STATUS_FAILED,
                    error=repr(exc),
                    duration_s=duration,
                )
            )
            if fail_fast:
                raise
            continue

        duration = time.perf_counter() - start
        if spec.name == F11_SOURCE_NAME:
            f11_frame = df
        results.append(
            SourceResult(
                name=spec.name,
                status=STATUS_SUCCEEDED,
                rows=len(df),
                duration_s=duration,
            )
        )
        logger.success("Refreshed {!r}: {} rows in {:.1f}s", spec.name, len(df), duration)

    summary = RefreshSummary(results=results)
    _log_summary(summary)
    return summary


def _log_summary(summary: RefreshSummary) -> None:
    """Emit the succeeded / failed / skipped summary via loguru."""
    logger.info(
        "Refresh summary: {} succeeded, {} failed, {} skipped (of {} attempted).",
        len(summary.succeeded),
        len(summary.failed),
        len(summary.skipped),
        len(summary.results),
    )
    for result in summary.failed:
        logger.error("  FAILED  {}: {}", result.name, result.error)
    for result in summary.skipped:
        logger.warning("  SKIPPED {}: {}", result.name, result.error)


def format_registry() -> str:
    """Render the registry as a human-readable listing for ``--list``."""
    lines = ["Registered data sources:"]
    for spec in REGISTRY:
        tags = []
        if spec.needs_f11:
            tags.append("needs F11")
        tags.append("materialises" if spec.materialise is not None else "raw-only")
        lines.append(f"  {spec.name}  [{', '.join(tags)}]")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``rba.data.refresh`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="rba.data.refresh",
        description="Re-pull registered RBA data sources with error isolation.",
    )
    parser.add_argument(
        "--source",
        "--only",
        action="append",
        dest="sources",
        metavar="NAME",
        help="Refresh only this source (repeatable). Default: all sources.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_registry",
        help="Print the registered sources and exit.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Reuse cached raw snapshots (force_download=False).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first source failure instead of isolating it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_registry:
        logger.info("\n{}", format_registry())
        return 0

    try:
        specs = select_specs(args.sources)
    except ValueError as exc:
        parser.error(str(exc))

    summary = refresh(
        specs,
        force_download=not args.no_download,
        fail_fast=args.fail_fast,
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
