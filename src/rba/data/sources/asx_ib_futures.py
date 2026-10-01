"""ASX 30-day Interbank Cash Rate Futures (IB) — market-implied cash rate.

The ASX 30-day Interbank Cash Rate Futures contract (ticker family
``IB``) is the standard market-implied proxy for the RBA cash rate. The
contract month settles to ``100 - average(daily cash rate, contract
month)``, so:

    implied_cash_rate(% pa) = 100 - settlement_price

The full implied curve across the next ~18 contract months is what the
ASX RBA Rate Tracker page (https://www.asx.com.au/markets/trade-our-
derivatives-market/futures-market/rba-rate-tracker) renders, and is the
canonical source for the model's market-implied baseline (CHECKLIST §6)
and for the hit-rate-vs-market evaluation metric (CHECKLIST §8).

Why MattCowgill/cash-rate-scraper, not direct ASX endpoints?
------------------------------------------------------------
ASX does not publish a free long-history feed of daily IB settlement
prices. Probed alternatives and outcomes (2026-05):

- ``asx.com.au/asx/1/futures/series/IB`` — **404**, endpoint not live.
- ``asx.com.au/.../interest-rate-derivatives-settlement-history.html`` —
  one settlement row per active contract, **single snapshot only** (no
  daily history).
- ``asx.com.au/data/trt/rate_tracker_calc.htm`` — methodology doc, not
  data.
- Yahoo Finance per-contract symbols (e.g. ``IBSM5.AX``) — endpoint
  responds but returns **zero historical observations**; no continuous
  front-month ticker for the ASX IB chain exists on Yahoo.

`MattCowgill/cash-rate-scraper <https://github.com/MattCowgill/cash-
rate-scraper>`_ is a GitHub repository (MIT licensed) that runs a daily
GitHub Action calling the ASX MarkitDigital JSON endpoint
``https://asx.api.markitdigital.com/asx-research/1.0/derivatives/
interest-rate/IB/futures`` (the same endpoint the Rate Tracker page
consumes), parses the implied curve, and commits a daily CSV plus an
aggregated ``combined_data/all_data.Rds`` to the repo. We pull the
aggregated Rds file (one ~32 KB download per refresh covering ~25,000
rows across ~1,100 trade dates) via ``pyreadr``.

Coverage and limitations
------------------------
- **Start**: 2022-04-21 (the day the upstream scraper started running).
- **End**: latest GitHub Action run, refreshed daily.
- **Gap**: 2022-07-01 → 2022-07-20 (ASX site change broke the upstream
  scraper).
- **Excluded trade dates** (filtered by upstream as known-bad):
  2022-08-06/07/08, 2022-12-29/30, 2023-01-18/24/31, 2023-02-02.
- Pre-2022 history is **not available** from any free source; the
  market-implied baseline is therefore evaluable only over the
  2022-04-21 → present window. This is a known limitation documented in
  CONTEXT.md.

Upstream schema
---------------
``combined_data/all_data.Rds`` is a tibble with three columns produced
by `R/scrape_cash_rate.R <https://github.com/MattCowgill/cash-rate-
scraper/blob/main/R/scrape_cash_rate.R>`_:

============  ======================================================
Column        Content
------------  ------------------------------------------------------
``date``      Contract expiry month, floored to first-of-month
              (e.g. ``2025-12-01`` for the Dec-25 contract).
``cash_rate`` Implied cash rate (% pa) for that contract month, ==
              ``100 - settlement_price``.
``scrape_date`` Trade date the settlement was reported for (the
              ``datePreviousSettlement`` field in the MarkitDigital
              JSON payload — i.e. the trading day whose EOD settle
              the row represents).
============  ======================================================

pyreadr loads ``date`` and ``scrape_date`` as Python ``object`` dtype
(string ``YYYY-MM-DD``); we coerce to ``datetime64[ns]``.

Publication-date convention
---------------------------
ASX publishes 30-day IB futures settlement prices at end-of-day after
the 4:30pm Sydney close (typically by ~6pm AEST). The settlement for
trade date T is therefore publicly known by end of day T AEST, before
any RBA meeting on T+1 (RBA meetings start with morning Board
discussion and announce at 14:30 AEST). We set
``publication_date == observation_date``: the implied curve as at trade
date T is fully available to any market participant by midnight T AEST,
and joining features on ``publication_date < meeting_date`` enforces
the conservative "strictly before" rule for any meeting on T+1 or
later.

Note that ``_CALENDAR_VALIDATION_FLOOR`` is still set to 1993-01-01 to
match every other source module — but unlike the macro series, IB
coverage begins 2022-04-21, so every observation we ingest is
post-floor. There is no NaT-publication-date region for this source.

Output schema
-------------
``fetch()`` returns a long-format ``pandas.DataFrame`` with columns:

- ``observation_date`` (datetime64[ns]) — trade date.
- ``publication_date`` (datetime64[ns]) — same as ``observation_date``;
  see convention above.
- ``series_id`` (object) — contract code ``ib_YYYY_MM`` for contract
  expiry month YYYY-MM-01.
- ``value`` (float64) — settlement price, computed as
  ``100 - upstream_cash_rate`` to keep the raw cache faithful to the
  exchange-traded settlement convention. A contract's implied rate is
  recovered trivially as ``100 - value``; :func:`derive_meeting_implied`
  turns it into the implied post-meeting rate for each meeting.

Running this module as ``__main__`` also materialises a wide-format CSV
at ``data/external/asx_ib_futures.csv`` keyed by ``trade_date`` with
one column per contract month plus a ``n_contracts`` count.

Side effects
------------
``fetch()`` writes under ``data/raw/asx_ib_futures/``:

- ``<YYYY-MM-DD>__all_data.Rds`` — verbatim Rds bytes per refresh.
- ``_metadata.json`` — provenance manifest: URL, SHA-256, download
  timestamp (UTC), byte count, observation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd
import pyreadr

from rba.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR

SOURCE_NAME = "asx_ib_futures"
SOURCE_URL = "https://github.com/MattCowgill/cash-rate-scraper/raw/main/combined_data/all_data.Rds"

# All ingested observations are post-1993 (upstream coverage begins
# 2022-04-21). The floor exists for symmetry with other source modules;
# every row this module yields has a populated publication_date.
_CALENDAR_VALIDATION_FLOOR = pd.Timestamp("1993-01-01")

# Upstream coverage start. Used in the docstring and as a sanity-check
# floor — if fetch() ever yields an observation_date earlier than this,
# the upstream data has materially changed and the user should re-read
# the README.
_UPSTREAM_COVERAGE_START = pd.Timestamp("2022-04-21")


@dataclass(frozen=True)
class AsxIbContract:
    """One IB contract month identified by its expiry."""

    series_id: str
    expiry_month_start: pd.Timestamp

    @classmethod
    def from_expiry_month(cls, expiry_month_start: pd.Timestamp) -> "AsxIbContract":
        return cls(
            series_id=f"ib_{expiry_month_start.year:04d}_{expiry_month_start.month:02d}",
            expiry_month_start=expiry_month_start,
        )


def fetch(*, force_download: bool = True) -> pd.DataFrame:
    """Pull the IB futures implied curve history, transform to long format.

    Parameters
    ----------
    force_download
        If True (default), refresh from the upstream GitHub repository.
        If False, reuse the most recent dated snapshot under
        ``data/raw/asx_ib_futures/`` and only download when none is
        present.

    Returns
    -------
    pandas.DataFrame
        See module docstring. Sorted ascending by ``(observation_date,
        series_id)``.

    Shapes
    ------
    Returns: (n_obs, 4) with one row per (trade_date, contract_month).
    """
    dest_dir = RAW_DATA_DIR / SOURCE_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path, entry, raw_bytes = _resolve_snapshot(dest_dir, force_download=force_download)

    upstream = _load_rds(raw_bytes, snapshot_path)
    long_df = _parse_upstream_dataframe(upstream)
    entry["observations"] = int(len(long_df))

    _write_manifest(dest_dir, [entry])

    long_df = _attach_publication_dates(long_df)
    return long_df.sort_values(["observation_date", "series_id"]).reset_index(drop=True)


def _parse_upstream_dataframe(upstream: pd.DataFrame) -> pd.DataFrame:
    """Convert the upstream MattCowgill tibble to the canonical long format.

    Parameters
    ----------
    upstream
        Frame loaded from ``combined_data/all_data.Rds`` with columns
        ``date`` (contract expiry month, str ``YYYY-MM-DD`` from
        pyreadr), ``cash_rate`` (implied % pa = ``100 - settlement``),
        and ``scrape_date`` (trade date, str ``YYYY-MM-DD``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``observation_date`` (datetime64[ns], trade date),
        ``series_id`` (object, ``ib_YYYY_MM``), ``value`` (float64,
        settlement price = ``100 - cash_rate``). Drops any row missing
        ``date``, ``cash_rate``, or ``scrape_date``.

    Shapes
    ------
    Returns: (n_obs, 3) with one row per (trade_date, contract_month).
    """
    required = {"date", "cash_rate", "scrape_date"}
    missing = required - set(upstream.columns)
    if missing:
        raise ValueError(
            f"Upstream Rds frame missing required column(s): {sorted(missing)}. "
            f"Got columns: {list(upstream.columns)}. Schema may have changed."
        )

    df = upstream.dropna(subset=["date", "cash_rate", "scrape_date"]).copy()

    expiry = pd.to_datetime(df["date"], errors="coerce")
    trade = pd.to_datetime(df["scrape_date"], errors="coerce")
    rate = pd.to_numeric(df["cash_rate"], errors="coerce")

    valid = expiry.notna() & trade.notna() & rate.notna()
    expiry = expiry[valid]
    trade = trade[valid]
    rate = rate[valid]

    if expiry.empty:
        raise ValueError(
            "Parsed zero IB futures observations from upstream Rds; "
            "schema may have changed or the file is empty."
        )

    series_ids = expiry.dt.strftime("ib_%Y_%m")

    out = pd.DataFrame(
        {
            "observation_date": trade.dt.as_unit("ns").to_numpy(),
            "series_id": series_ids.to_numpy(),
            "value": (100.0 - rate.to_numpy()),
        }
    )
    return out.reset_index(drop=True)


def _attach_publication_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Set publication_date == observation_date for every IB observation.

    The settlement for trade date T is published at ASX EOD (~6pm AEST
    same day) and is publicly known by midnight T AEST. There is no
    separate release calendar; the rule is trivial and lives here.

    Defensively drops any pre-existing ``publication_date`` column so a
    stale value from a re-merge cannot leak through.
    """
    out = df.drop(columns=["publication_date"], errors="ignore").copy()
    out["publication_date"] = out["observation_date"]

    in_window = out["observation_date"] >= _CALENDAR_VALIDATION_FLOOR
    unmatched = out[in_window & out["publication_date"].isna()]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} ASX IB futures observation(s) on/after "
            f"{_CALENDAR_VALIDATION_FLOOR.date()} are missing a publication date."
        )

    return out[["observation_date", "publication_date", "series_id", "value"]]


def _resolve_snapshot(
    dest_dir: Path,
    *,
    force_download: bool,
) -> tuple[Path, dict[str, object], bytes]:
    """Return (snapshot path, manifest entry, raw bytes) for the Rds file."""
    existing = sorted(dest_dir.glob("[0-9]" * 4 + "-??-??__all_data.Rds"))
    if existing and not force_download:
        path = existing[-1]
        raw_bytes = path.read_bytes()
        logger.info("Reusing existing ASX IB snapshot at {}", path)
        entry: dict[str, object] = {
            "url": SOURCE_URL,
            "snapshot_filename": path.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "downloaded_at_utc": None,
            "reused": True,
        }
        return path, entry, raw_bytes
    return _download(dest_dir)


def _download(dest_dir: Path) -> tuple[Path, dict[str, object], bytes]:
    """Download the upstream Rds file, save dated snapshot, return path + entry."""
    today = datetime.now(timezone.utc).date().isoformat()
    snapshot_path = dest_dir / f"{today}__all_data.Rds"
    logger.info("Downloading {} -> {}", SOURCE_URL, snapshot_path)

    # GitHub raw / object storage requires a non-empty User-Agent — unlike the
    # RBA endpoints. urllib's default UA is empty, so set one explicitly.
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "rba-cash-rate/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 — public GitHub URL
        raw_bytes = response.read()
    snapshot_path.write_bytes(raw_bytes)

    entry: dict[str, object] = {
        "url": SOURCE_URL,
        "snapshot_filename": snapshot_path.name,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    return snapshot_path, entry, raw_bytes


def _load_rds(raw_bytes: bytes, snapshot_path: Path) -> pd.DataFrame:
    """Read the Rds payload into a pandas DataFrame via pyreadr.

    pyreadr requires a file path (cannot read from bytes). The bytes are
    rewritten to ``snapshot_path`` here only when the caller is the
    cache-reuse branch (in which case the bytes already match); the
    download branch has already written them. The cost is a no-op
    overwrite, kept for clarity.
    """
    if not snapshot_path.exists() or snapshot_path.read_bytes() != raw_bytes:
        snapshot_path.write_bytes(raw_bytes)
    result = pyreadr.read_r(str(snapshot_path))
    if not result:
        raise ValueError(
            f"pyreadr returned no R objects from {snapshot_path}; Rds file is empty or corrupt."
        )
    df = next(iter(result.values()))
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"Expected a DataFrame from pyreadr, got {type(df).__name__}. "
            f"Upstream all_data.Rds schema may have changed."
        )
    return df


def _write_manifest(dest_dir: Path, manifest: list[dict[str, object]]) -> None:
    metadata_path = dest_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote provenance manifest to {}", metadata_path)


# ---------------------------------------------------------------------------
# Meeting-implied rate derivation
# ---------------------------------------------------------------------------

#: ``method`` value: the meeting-month contract, day-weight unwound.
METHOD_DAY_WEIGHTED = "day_weighted"
#: ``method`` value: the following month's contract, read directly.
METHOD_NEXT_MONTH = "next_month"

_IMPLIED_COLUMNS = [
    "meeting_date",
    "lookup_date",
    "contract_series_id",
    "method",
    "contract_rate",
    "implied_cash_rate",
]


@dataclass(frozen=True)
class _ContractChoice:
    """The contract that prices one meeting, and how to read it."""

    contract: AsxIbContract
    method: str
    days_in_month: int
    days_before: int
    days_after: int


def _choose_contract(
    meeting_date: pd.Timestamp,
    *,
    min_post_meeting_share: float,
) -> _ContractChoice:
    """Pick the contract that prices ``meeting_date`` and the rule to read it with.

    A decision announced on the meeting date takes effect the following day, so
    within the meeting month days ``1..meeting_day`` settle at the pre-meeting
    rate and the remaining ``days_after`` settle at the post-meeting rate. A
    meeting on the last day of the month leaves ``days_after == 0``.
    """
    days_in_month = int(meeting_date.days_in_month)
    days_before = int(meeting_date.day)
    days_after = days_in_month - days_before
    month_start = meeting_date.normalize().replace(day=1)

    if days_after / days_in_month >= min_post_meeting_share:
        expiry, method = month_start, METHOD_DAY_WEIGHTED
    else:
        expiry, method = month_start + pd.offsets.MonthBegin(1), METHOD_NEXT_MONTH
    return _ContractChoice(
        contract=AsxIbContract.from_expiry_month(expiry),
        method=method,
        days_in_month=days_in_month,
        days_before=days_before,
        days_after=days_after,
    )


def _post_meeting_rate(contract_rate: float, prior_rate: float, choice: _ContractChoice) -> float:
    """Turn a contract's implied rate into the implied post-meeting rate.

    The next-month contract sits wholly after the decision, so its rate is the
    answer. The meeting-month contract is a day-weighted average of the pre- and
    post-meeting rates, which is solved for the post-meeting rate. A NaN
    ``prior_rate`` propagates: the average cannot be unwound without it.
    """
    if choice.method == METHOD_NEXT_MONTH:
        return contract_rate
    weighted_sum = choice.days_in_month * contract_rate - choice.days_before * prior_rate
    return weighted_sum / choice.days_after


def derive_meeting_implied(
    long_df: pd.DataFrame,
    meetings_df: pd.DataFrame,
    *,
    lookback_business_days: int = 1,
    min_post_meeting_share: float = 0.5,
) -> pd.DataFrame:
    """Derive the market-implied **post-meeting** cash rate per RBA meeting.

    For each meeting ``M``, look up the implied curve as of ``M -
    lookback_business_days`` (i.e. the trading day strictly before the
    meeting), pick the contract that prices the decision, and report the
    cash rate the market expects to prevail after it.

    Which contract, and why
    -----------------------
    An IB contract settles to the *average* daily cash rate over its
    contract month, and a decision announced on ``M`` takes effect the
    next day. The meeting-month contract therefore mixes two rates: days
    ``1..M`` at the pre-meeting rate and the remaining days at the
    post-meeting rate. How much of the decision it carries depends on
    where in the month the meeting falls, so the rule is chosen by the
    share of the month that falls after the decision takes effect:

    - **Share >= ``min_post_meeting_share``** (meeting in the first half
      of the month by default): read the meeting-month contract and
      unwind the average (:data:`METHOD_DAY_WEIGHTED`)::

          post = (days_in_month * contract_rate - days_before * prior_rate) / days_after

    - **Share below the threshold** (second half of the month): read the
      *following* month's contract directly (:data:`METHOD_NEXT_MONTH`).
      That month sits wholly after the decision, whereas the unwinding
      would divide by a handful of days and magnify the half-basis-point
      price tick into noise. A meeting on the 29th of a 30-day month
      moves its own contract by under one tick even when a 25 bp move is
      fully priced.

    Known approximations
    --------------------
    - ``prior_rate_pct`` is the cash rate *target*, while contracts settle
      on the *traded* interbank rate, which has sat a few basis points
      from target. The unwinding scales that basis by at most
      ``1 / min_post_meeting_share``.
    - The next-month contract also prices any meeting held within that
      month. Board meetings are at least five weeks apart, so a
      second-half meeting is never followed by one early in the next
      month and the overlap is at most the last days of the contract.

    Parameters
    ----------
    long_df
        Output of :func:`fetch` (or an in-memory equivalent). Must
        contain ``observation_date``, ``series_id``, ``value`` columns.
    meetings_df
        Meeting frame. Must contain ``observation_date`` (the meeting
        announcement date) and ``prior_rate_pct`` (the cash rate target
        going into the meeting, percent).
    lookback_business_days
        Number of business days strictly before the meeting to use as
        the lookup date. Default ``1`` ensures the curve used was
        publicly known before the meeting starts (no leakage).
    min_post_meeting_share
        Minimum share of the meeting month that must fall after the
        decision takes effect for the meeting-month contract to be used.
        Must lie in ``(0, 1]``. Default ``0.5``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``meeting_date`` (datetime64[ns]) — input meeting
        date; ``lookup_date`` (datetime64[ns]) — the trade date the
        curve was read from; ``contract_series_id`` (object) — the
        ``ib_YYYY_MM`` contract read; ``method`` (object) —
        :data:`METHOD_DAY_WEIGHTED` or :data:`METHOD_NEXT_MONTH`;
        ``contract_rate`` (float64) — ``100 - settlement_price`` of that
        contract; and ``implied_cash_rate`` (float64) — the implied
        post-meeting cash rate.

        Meetings for which no observation exists within a 7-business-
        day fallback window before the meeting (or the chosen contract
        is not quoted on that day) yield NaN in both rates and NaT in
        the lookup_date. A day-weighted meeting with a NaN
        ``prior_rate_pct`` keeps its ``contract_rate`` but yields a NaN
        ``implied_cash_rate``.

    Raises
    ------
    ValueError
        If a required column is missing from either frame, or
        ``min_post_meeting_share`` is outside ``(0, 1]``.

    Shapes
    ------
    Returns: (n_meetings, 6).
    """
    missing_meeting = {"observation_date", "prior_rate_pct"} - set(meetings_df.columns)
    if missing_meeting:
        raise ValueError(
            f"meetings_df missing column(s): {sorted(missing_meeting)}. It must carry "
            "'observation_date' (the meeting date) and 'prior_rate_pct' (the cash rate "
            "target going into the meeting)."
        )
    needed = {"observation_date", "series_id", "value"}
    missing = needed - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df missing column(s): {sorted(missing)}")
    if not 0.0 < min_post_meeting_share <= 1.0:
        raise ValueError(
            f"min_post_meeting_share must lie in (0, 1]; got {min_post_meeting_share!r}."
        )

    available_trade_dates = pd.Index(sorted(long_df["observation_date"].unique()))
    by_trade = {
        d: g.set_index("series_id")["value"].to_dict()
        for d, g in long_df.groupby("observation_date")
    }

    bday = pd.tseries.offsets.BusinessDay()
    fallback_window = 7  # business days

    meeting_dates = pd.to_datetime(meetings_df["observation_date"])
    prior_rates = pd.to_numeric(meetings_df["prior_rate_pct"], errors="coerce")

    out_rows: list[dict[str, object]] = []
    for meeting_date, prior_rate in zip(meeting_dates, prior_rates, strict=True):
        target = meeting_date - lookback_business_days * bday
        lookup_date = _latest_trade_date_on_or_before(
            target, available_trade_dates, max_lookback_bdays=fallback_window
        )
        choice = _choose_contract(meeting_date, min_post_meeting_share=min_post_meeting_share)
        contract_id = choice.contract.series_id

        contract_rate = float("nan")
        implied = float("nan")
        if lookup_date is not None:
            settlement = by_trade.get(lookup_date, {}).get(contract_id)
            if settlement is not None:
                contract_rate = float(100.0 - settlement)
                implied = _post_meeting_rate(contract_rate, float(prior_rate), choice)

        out_rows.append(
            {
                "meeting_date": meeting_date,
                "lookup_date": lookup_date if lookup_date is not None else pd.NaT,
                "contract_series_id": contract_id,
                "method": choice.method,
                "contract_rate": contract_rate,
                "implied_cash_rate": implied,
            }
        )
    out = pd.DataFrame(out_rows, columns=_IMPLIED_COLUMNS)
    out["meeting_date"] = pd.to_datetime(out["meeting_date"]).astype("datetime64[ns]")
    out["lookup_date"] = pd.to_datetime(out["lookup_date"]).astype("datetime64[ns]")
    out["contract_rate"] = out["contract_rate"].astype(float)
    out["implied_cash_rate"] = out["implied_cash_rate"].astype(float)
    return out


def _latest_trade_date_on_or_before(
    target: pd.Timestamp,
    available: pd.Index,
    *,
    max_lookback_bdays: int,
) -> pd.Timestamp | None:
    """Return the latest trade date on/before ``target``, within fallback.

    Walks back at most ``max_lookback_bdays`` business days from
    ``target`` looking for a trade date present in ``available``. Returns
    None if none is found (e.g. meeting falls inside the 2022-07-01 →
    2022-07-20 upstream gap, or earlier than 2022-04-21).
    """
    bday = pd.tseries.offsets.BusinessDay()
    cursor = target
    for _ in range(max_lookback_bdays + 1):
        if cursor in available:
            return cursor
        cursor = cursor - bday
    # Final fallback: any available trade date strictly before target.
    earlier = available[available <= target]
    if len(earlier) == 0:
        return None
    candidate = earlier[-1]
    # Guard against running too far back (e.g. meeting before upstream coverage).
    if (target - candidate) > pd.Timedelta(days=max_lookback_bdays + 7):
        return None
    return candidate


# ---------------------------------------------------------------------------
# Wide-format CSV materialisation (called from __main__)
# ---------------------------------------------------------------------------


def _to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ``fetch()`` output to a (trade_date × contract) wide frame.

    Parameters
    ----------
    long_df
        Output of :func:`fetch`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``trade_date`` (datetime64[ns]), one float column per
        ``ib_YYYY_MM`` contract present in the data (sorted by expiry
        month), and ``n_contracts`` (int) — the count of non-null
        contracts quoted on that trade date.

    Shapes
    ------
    Returns: (n_trade_dates, 2 + n_contracts).
    """
    wide = long_df.pivot_table(
        index="observation_date",
        columns="series_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"observation_date": "trade_date"})

    contract_cols = sorted(c for c in wide.columns if c.startswith("ib_"))
    wide["n_contracts"] = wide[contract_cols].notna().sum(axis=1).astype("int64")

    return (
        wide[["trade_date", *contract_cols, "n_contracts"]]
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    long_df = fetch()
    wide_df = _to_wide(long_df)
    dest = EXTERNAL_DATA_DIR / "asx_ib_futures.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(dest, index=False, date_format="%Y-%m-%d")
    logger.info(
        "Wrote ASX IB futures ({} trade dates, {} contracts) to {}",
        len(wide_df),
        len([c for c in wide_df.columns if c.startswith("ib_")]),
        dest,
    )
