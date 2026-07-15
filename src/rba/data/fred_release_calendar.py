"""FRED release calendar — algorithmic publication-date rules.

Three publication-date rules cover the 10 series this module knows about
(6 from :mod:`rba.data.sources.fred_global_signals`, 4 from
:mod:`rba.data.sources.commodity_prices`):

================================  ================  ===============================
Logical series ID                 FRED series ID    Publication rule
--------------------------------  ----------------  -------------------------------
``us_headline_cpi`` (monthly SA)  ``CPIAUCSL``      ``observation_date + 20 days``
``us_core_cpi``     (monthly SA)  ``CPILFESL``      ``observation_date + 20 days``
``us_fed_funds``    (daily)       ``DFF``           ``+ 1 US business day``
``us_10y_treasury`` (daily)       ``DGS10``         ``+ 1 US business day``
``us_dxy_broad``    (daily)       ``DTWEXBGS``      ``+ 1 US business day``
``us_vix``          (daily)       ``VIXCLS``        same day (CBOE 4:15pm ET close)
``iron_ore_spot``   (monthly)     ``PIORECRUSDM``   ``observation_date + 20 days``
``copper_spot``     (monthly)     ``PCOPPUSDM``     ``observation_date + 20 days``
``brent_crude``     (daily)       ``DCOILBRENTEU``  ``+ 1 US business day``
``wti_crude``       (daily)       ``DCOILWTICO``    ``+ 1 US business day``
================================  ================  ===============================

"US business day" = Mon-Fri excluding US federal holidays per
:class:`pandas.tseries.holiday.USFederalHolidayCalendar` (New Year's
Day, Martin Luther King Jr. Day, Presidents' Day, Memorial Day,
Juneteenth, Independence Day, Labor Day, Columbus Day, Veterans Day,
Thanksgiving, Christmas — with Mondayisation where applicable). VIX
follows CBOE closures (federal holidays + Good Friday); the
publication rule is unaffected because the FRED CSV already emits an
empty cell for any non-trading day, so no observation_date ever lands
on a closure.

Monthly flat-offset rationale
-----------------------------
Two upstreams currently use the ``us_monthly_flat`` rule: the US BLS
CPI series and the IMF Primary Commodity Prices series shipped by
:mod:`rba.data.sources.commodity_prices` (iron ore / copper). Worst
observed lags:

- BLS CPI: ~10–15 days from end-of-reference-month (sampled releases:
  Sep 2024 ref → Oct 11 release = 11 days; Oct 2024 → Nov 13 = 13 days;
  Nov 2024 → Dec 11 = 11 days; Dec 2024 → Jan 15 2025 = 15 days).
- IMF Primary Commodity Prices: ~5 days from end-of-reference-month
  (the bulletin is typically released in the first week of the
  following month).

A flat **+ 20 calendar days from period-end** is conservative for both
— it never claims an upstream was available *earlier* than reality.
The 5–15 day buffer is small enough to preserve realism (no RBA
meeting after the actual release would have its monthly feature
marked as unseen by mistake) while keeping us strictly inside the
leakage envelope. This matches the WPI / Building-Approvals / TVD
flat-fallback pattern used elsewhere in this codebase when a scraped
per-release calendar isn't available. Future contributors who need
release-date precision can replace this with an ALFRED-derived
first-release calendar (FRED archives the exact first-release date
per observation, walkable via the ``realtime_start``/``realtime_end``
parameters); the upgrade path is the same as the scheduled
vintage-accumulation upgrade noted for the broader vintage policy in
CONTEXT.md.

VIX same-day rationale
----------------------
CBOE publishes the VIX close at 4:15pm US Eastern Time, so the value
is public on the **same trading day**. RBA meetings announce at
2:30pm AEST (Sydney) which is roughly 12:30am ET the same calendar
day, so the VIX same-day publication still occurs after the RBA
decision in walltime. Downstream alignment-layer joins
(``publication_date < meeting_date``) treat same-day publication as
not-yet-known and exclude it conservatively — so this convention is
correct even with the timezone offset.

Overrides
---------
``_OVERRIDES`` is keyed by ``(series_id, observation_date)`` for known
deviations from the algorithmic rule (e.g. BLS reschedules around a
US federal-government shutdown). Empty by default; populate as
exceptions are confirmed.

Usage
-----
- ``fred_publication_date(observation_date, series_id)`` — single lookup.
- ``attach_fred_publication_dates(long_df)`` — vectorised stamp over a
  long-format DataFrame with ``observation_date`` and ``series_id``.
- ``build_fred_release_calendar(start_date, end_date)`` — diagnostic
  full long-format calendar across every series and rule.
- ``uv run python -m rba.data.fred_release_calendar`` — materialises
  the calendar to ``data/external/fred_release_dates.csv``.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from loguru import logger
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from rba.config import EXTERNAL_DATA_DIR

# Flat conservative offset for monthly upstreams without a scrapable
# per-release calendar (currently BLS CPI and IMF Primary Commodity
# Prices). Worst observed release lag from end-of-reference-month is
# ~15 days (BLS CPI); +20 days adds a 5-day buffer without distorting
# realism, and applies cleanly to IMF commodity series whose typical
# lag is shorter (~5 days).
_MONTHLY_FLAT_OFFSET_DAYS = 20

# Logical series_id -> publication-rule key. Three rules:
#   "us_daily_t_plus_1" — next US business day after observation_date.
#   "us_same_day"       — observation_date itself.
#   "us_monthly_flat"   — observation_date + _MONTHLY_FLAT_OFFSET_DAYS.
# This mapping is the single source of truth shared between every
# source module's _attach_publication_dates helper and the diagnostic
# builder. Both fred_global_signals and commodity_prices contribute
# entries to this dict.
_RULES: dict[str, str] = {
    # fred_global_signals
    "us_headline_cpi": "us_monthly_flat",
    "us_core_cpi": "us_monthly_flat",
    "us_fed_funds": "us_daily_t_plus_1",
    "us_10y_treasury": "us_daily_t_plus_1",
    "us_dxy_broad": "us_daily_t_plus_1",
    "us_vix": "us_same_day",
    # commodity_prices
    "iron_ore_spot": "us_monthly_flat",
    "copper_spot": "us_monthly_flat",
    "brent_crude": "us_daily_t_plus_1",
    "wti_crude": "us_daily_t_plus_1",
}

# Hand-verified per-(series, observation-date) deviations. Empty by
# default; populate as known exceptions are observed (e.g. BLS
# reschedules around a US federal-government shutdown).
_OVERRIDES: dict[tuple[str, date], date] = {}


@lru_cache(maxsize=64)
def _us_federal_holidays(year: int) -> frozenset[date]:
    """Return the observed US federal holiday dates for a calendar year.

    Uses :class:`pandas.tseries.holiday.USFederalHolidayCalendar` which
    handles Mondayisation natively (New Year's Day falling on Sun is
    observed Mon, etc.) and emits MLK Day from 1986, Juneteenth from
    2021, Veterans Day from 1978 (post-Uniform-Monday-Holiday-Act
    rollback). Cached because pandas builds the holiday index lazily
    on every call.
    """
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=pd.Timestamp(f"{year}-01-01"), end=pd.Timestamp(f"{year}-12-31"))
    return frozenset(h.date() for h in holidays)


def _next_us_business_day(d: date) -> date:
    """Return the first weekday strictly after ``d`` that is not a US
    federal holiday.
    """
    candidate = d + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in _us_federal_holidays(candidate.year):
        candidate += timedelta(days=1)
    return candidate


def fred_publication_date(observation_date: date, series_id: str) -> date:
    """Return the FRED publication date for one observation.

    Applies, in priority order:

    1. ``_OVERRIDES`` lookup keyed by ``(series_id, observation_date)``.
    2. Rule lookup via ``_RULES[series_id]`` — one of:
       ``us_daily_t_plus_1`` (next US business day),
       ``us_same_day`` (observation_date itself),
       ``us_monthly_flat`` (observation_date + 20 days).

    Parameters
    ----------
    observation_date
        Trade date (for daily series) or reference month-end (for
        monthly series).
    series_id
        Logical series ID from this module's ``_RULES`` mapping.

    Returns
    -------
    datetime.date
        Publication date for the observation.
    """
    key = (series_id, observation_date)
    if key in _OVERRIDES:
        return _OVERRIDES[key]
    rule = _RULES.get(series_id)
    if rule is None:
        raise ValueError(
            f"Unknown FRED series_id {series_id!r}; add it to _RULES in "
            "rba.data.fred_release_calendar before requesting a publication date."
        )
    if rule == "us_daily_t_plus_1":
        return _next_us_business_day(observation_date)
    if rule == "us_same_day":
        return observation_date
    if rule == "us_monthly_flat":
        return observation_date + timedelta(days=_MONTHLY_FLAT_OFFSET_DAYS)
    raise ValueError(
        f"Unknown FRED publication rule {rule!r} for series_id {series_id!r}; "
        "extend fred_publication_date when adding a new rule."
    )


def attach_fred_publication_dates(long_df: pd.DataFrame) -> pd.DataFrame:
    """Stamp ``publication_date`` onto a long-format FRED observations frame.

    Parameters
    ----------
    long_df
        Long-format ``pandas.DataFrame`` carrying at minimum
        ``observation_date`` (datetime64[ns]) and ``series_id``
        (object). Other columns pass through unchanged. Any pre-existing
        ``publication_date`` column is dropped defensively before the
        new column is computed.

    Returns
    -------
    pandas.DataFrame
        Input columns plus ``publication_date`` (datetime64[ns]) in the
        canonical column order
        ``[observation_date, publication_date, series_id, value]``
        when those four columns are present.

    Shapes
    ------
    Returns: (n_obs, n_in_cols [+1 if publication_date was absent]).
    """
    df = long_df.drop(columns=["publication_date"], errors="ignore").copy()
    if df.empty:
        df["publication_date"] = pd.Series(dtype="datetime64[ns]")
    else:
        pub_dates = [
            fred_publication_date(ts.date(), sid)
            for ts, sid in zip(df["observation_date"], df["series_id"])
        ]
        df["publication_date"] = pd.to_datetime(pub_dates).as_unit("ns")

    canonical = ["observation_date", "publication_date", "series_id", "value"]
    if all(col in df.columns for col in canonical):
        passthrough = [c for c in df.columns if c not in canonical]
        return df[canonical + passthrough]
    return df


def build_fred_release_calendar(
    start_date: date | pd.Timestamp = date(1947, 1, 1),
    end_date: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a diagnostic long-format calendar across every series and rule.

    For daily-rule series the calendar walks every weekday (Mon-Fri) in
    the window. For monthly-CPI series the calendar walks every
    month-end. Holiday filtering at the observation-date end is *not*
    applied — FRED simply emits a blank cell for a non-trading day, and
    a blank observation never reaches the publication-date stamp.
    Including those rows here keeps the calendar predictable and
    matches the AGB-yields-calendar convention.

    Parameters
    ----------
    start_date
        First observation_date to include. Defaults to ``1947-01-01``
        (the start of CPIAUCSL, the oldest FRED series this module
        covers).
    end_date
        Last observation_date to include. Defaults to ``today + 7
        days`` so the calendar always carries a small forward buffer.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``series_id`` (object) — logical series ID.
        - ``observation_date`` (datetime64[ns]) — trade date (daily) or
          reference month-end (monthly).
        - ``publication_date`` (datetime64[ns]) — algorithmic FRED
          release date.

    Shapes
    ------
    Returns: (n_daily_weekdays * 4 + n_months * 2, 3).
    """
    start = pd.Timestamp(start_date).date()
    if end_date is None:
        end_d = date.today() + timedelta(days=7)
    else:
        end_d = pd.Timestamp(end_date).date()

    rows: list[dict[str, object]] = []

    daily_series_ids = [sid for sid, rule in _RULES.items() if rule != "us_monthly_flat"]
    d = start
    while d <= end_d:
        if d.weekday() < 5:
            for sid in daily_series_ids:
                rows.append(
                    {
                        "series_id": sid,
                        "observation_date": d,
                        "publication_date": fred_publication_date(d, sid),
                    }
                )
        d += timedelta(days=1)

    monthly_series_ids = [sid for sid, rule in _RULES.items() if rule == "us_monthly_flat"]
    months = pd.date_range(start=start, end=end_d, freq="ME")
    for ts in months:
        month_end = ts.date()
        for sid in monthly_series_ids:
            rows.append(
                {
                    "series_id": sid,
                    "observation_date": month_end,
                    "publication_date": fred_publication_date(month_end, sid),
                }
            )

    df = pd.DataFrame(rows)
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.as_unit("ns")
    df["publication_date"] = pd.to_datetime(df["publication_date"]).dt.as_unit("ns")
    return df.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


if __name__ == "__main__":
    calendar_df = build_fred_release_calendar()
    dest = EXTERNAL_DATA_DIR / "fred_release_dates.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(dest, index=False)
    logger.info(
        "Wrote FRED release calendar ({} rows across {} series) to {}",
        len(calendar_df),
        calendar_df["series_id"].nunique(),
        dest,
    )
