"""Cross-checks for the cleaned RBA decisions frame.

The primary scraper pulls from ``/statistics/cash-rate/``, which embeds links
to each meeting's media release (when one was issued). This module verifies
that every *rate-change* row has a media-release URL — RBA has issued a
release for every announced change since 1990, so a missing URL on a change
row points at either a parser bug or an RBA page-layout regression.

The network check (``check_urls=True``) issues HEAD requests against each
``statement_url`` to confirm the page exists. It's opt-in because it's slow
and flaky, and the value over the static check is marginal in steady state.
"""

from __future__ import annotations

from collections.abc import Iterable
import urllib.error
import urllib.request

from loguru import logger
import pandas as pd

RBA_BASE_URL = "https://www.rba.gov.au"


def cross_check_media_releases(
    decisions: pd.DataFrame,
    *,
    check_urls: bool = False,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Return rows that fail cross-check against the media releases archive.

    Parameters
    ----------
    decisions
        Output of :func:`rba.data.preprocess_target.build_decisions`. Must
        contain ``observation_date``, ``rate_change_bps``, and ``statement_url``.
    check_urls
        If True, HEAD-request each non-null ``statement_url`` and flag any
        that don't return 2xx. Off by default — pure network check.
    timeout
        Per-request HTTP timeout (seconds). Ignored when ``check_urls=False``.

    Returns
    -------
    pandas.DataFrame
        Subset of ``decisions`` rows that failed any check, plus a
        ``failure_reason`` column. Empty frame ⇒ all good.

    Shapes
    ------
    Returns: (n_failing_rows, n_input_cols + 1).
    """
    required = {"observation_date", "rate_change_bps", "statement_url"}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions is missing required columns: {sorted(missing)}")

    failures: list[pd.Series] = []

    # Check 1: every rate change has a media-release URL.
    change_rows = decisions["rate_change_bps"] != 0
    missing_url = decisions["statement_url"].isna()
    bad_static = decisions.loc[change_rows & missing_url].assign(
        failure_reason="rate change has no statement_url"
    )
    if len(bad_static):
        failures.append(bad_static)

    if check_urls:
        urls_to_check = decisions.loc[~missing_url, "statement_url"].unique().tolist()
        broken = _head_check(urls_to_check, timeout=timeout)
        if broken:
            broken_rows = decisions[decisions["statement_url"].isin(broken)].assign(
                failure_reason="statement_url returns non-2xx"
            )
            failures.append(broken_rows)

    if not failures:
        logger.info(
            "Cross-check passed: {} decisions, {} rate changes, {} statement URLs.",
            len(decisions),
            int(change_rows.sum()),
            int((~missing_url).sum()),
        )
        return decisions.iloc[0:0].assign(failure_reason=pd.Series(dtype="object"))

    return pd.concat(failures, ignore_index=True)


def _head_check(urls: Iterable[str], *, timeout: float) -> list[str]:
    """Return URLs that do not return a 2xx status to a HEAD request."""
    broken: list[str] = []
    for url in urls:
        absolute = url if url.startswith("http") else f"{RBA_BASE_URL}{url}"
        request = urllib.request.Request(absolute, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        except urllib.error.URLError as exc:
            logger.warning("HEAD {} -> network error: {}", absolute, exc)
            broken.append(url)
            continue
        if not 200 <= status < 300:
            logger.warning("HEAD {} -> {}", absolute, status)
            broken.append(url)
    return broken
