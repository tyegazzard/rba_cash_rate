# CONTEXT.md — RBA Cash Rate Prediction

This file orients Claude Code (and any other agent) inside this repository. Read it before making non-trivial changes.

## What this project is
A machine learning pipeline that predicts Reserve Bank of Australia (RBA) cash rate decisions, supporting multiple target formulations (3-class, multi-class with magnitudes, regression on Δrate, regression on level, ordinal). Includes a Streamlit dashboard for before-meeting predictions. Built as a portfolio project demonstrating end-to-end ML craft.

## Tech stack
- **Python**: 3.11+
- **Env manager**: uv (single lockfile; `pyproject.toml` is source of truth)
- **Data**: pandas, polars (where speed matters), pyarrow
- **ML**: scikit-learn, XGBoost, LightGBM, mord (ordinal logistic), xRFM
- **NLP**: sentence-transformers, transformers (HF), Loughran-McDonald lexicon
- **Tracking**: MLflow (local, `./mlruns`)
- **Dashboard**: Streamlit
- **Testing**: pytest (+ pytest-cov for coverage)
- **Layout**: cookiecutter-data-science

## Repo structure
```
rba-cash-rate-prediction/
├── data/
│   ├── raw/           # IMMUTABLE — original downloads, never edit
│   ├── interim/       # cleaned but pre-feature
│   ├── processed/     # model-ready feature frames
│   └── external/      # third-party reference data
├── notebooks/         # 0.X-eda-*, 1.X-features-*, 2.X-modelling-*, 3.X-evaluation-*
├── src/rba/
│   ├── data/
│   │   ├── sources/   # one module per source (rba_f11, abs_cpi, asx_futures, ...)
│   │   ├── refresh.py # orchestrates re-pulling all sources
│   │   └── align.py   # point-in-time alignment to meeting frame
│   ├── features/      # lags.py, rolling.py, regime.py, text.py
│   ├── models/        # baselines.py, sklearn_wrappers.py, xgb.py, lgbm.py, ordinal.py, xrfm.py
│   ├── validation/    # cv.py (WalkForwardSplit), metrics.py, calibration.py
│   ├── viz/
│   └── config/        # __init__.py (constants), targets.yaml, features.yaml, models.yaml
├── tests/             # mirrors src/ structure
├── streamlit_app/
│   └── main.py
├── reports/
├── mlruns/
├── pyproject.toml
├── CONTEXT.md
└── README.md
```

## Common commands
```bash
# Setup
uv sync                                  # install all deps from lockfile
uv add <package>                         # add a runtime dep
uv add --dev <package>                   # add a dev dep
uv run python -m rba.<module>            # run a module

# MLflow
uv run mlflow ui --backend-store-uri ./mlruns   # tracking UI on :5000

# Testing
uv run pytest                            # full suite
uv run pytest tests/features/            # one folder
uv run pytest -k test_no_leakage         # match by name
uv run pytest --cov=src/rba              # with coverage

# Notebook
uv run jupyter lab

# Streamlit
uv run streamlit run streamlit_app/main.py

# Data refresh
uv run python -m rba.data.refresh        # re-pulls all sources, hashes, logs
```

## CRITICAL INVARIANTS — never violate these

### 1. Point-in-time correctness
Every feature value at meeting date `t` must be derivable using only information published *strictly before* `t`. The ingest layer in `src/rba/data/sources/` must record both:
- `observation_date` — the date the value applies to (e.g. CPI for Q3 2024)
- `publication_date` — the date the value became publicly known

Features join on `publication_date <= meeting_date`. If a source doesn't expose publication dates, log it explicitly and approximate conservatively (assume publication delay).

**CPI, Labour Force (LFS), Wage Price Index (WPI), GDP, RBA D1/D2, RBA E1/E2, RBA I2 Commodity Prices, RBA F2 ACGB Yields, ABS Building Approvals, and ABS Total Value of Dwellings** each have their own release-calendar module under `src/rba/data/`. **FRED global signals** and **commodity prices (iron ore / copper / Brent / WTI)** share a single `fred_release_calendar.py` module — both source modules pull from the FRED endpoint and the calendar's `_RULES` dict carries one entry per logical series across both sources (10 entries total as of 2026-05-27). **RBA F1 BBSW rates** reuses the F2 / AGB-yields release calendar directly (identical +1 AU business day rule — see the BBSW source-module note below the calendar list). **ASX 30-day IB futures**, **RBA F11.1 AUD exchange rates**, and **S&P/ASX 200 equity indices** intentionally do *not* use a release calendar — see the trivial-rule notes below the calendar list.

- `cpi_release_calendar.py` — last Wednesday of the month following the reference quarter (ABS Cat. 6401.0). Keyed by `reference_quarter_end`.
- `lfs_release_calendar.py` — era-aware rule for ABS Cat. 6202.0 keyed by `reference_month_end`:
  - 1993 – Dec 2015 reference months: 2nd Thursday of the following month.
  - Jan 2016 – present reference months: 3rd Thursday of the following month (McCarthy Review change).
  - Any December reference month: 4th Thursday of the following January (permanent exception, both eras).
- `wpi_release_calendar.py` — **scraped** rather than algorithmic, because the WPI (ABS Cat. 6345.0) has no clean release rule (mix of 2nd/3rd Wednesdays plus at least one Tuesday on record; observed lag 43–52 days). The scraper hits the ABS WPI release page per `(quarter, year)` slug and persists the original "Released" date to `data/external/wpi_release_dates.csv`. The post-redesign ABS URL pattern resolves back to 2019-Q3 — earlier quarters retain no calendar entry and the source applies a flat 60-day fallback (which still covers the worst observed 52-day lag with a buffer).
- `gdp_release_calendar.py` — **scraped** because the post-2003 "first Wednesday of the third month after quarter-end" rule (verified for 2019-Q2 → 2025-Q4) does **not** extend back to 1993: pre-2003 releases show non-Wednesday slots (Tue/Fri observed), 2-month rather than 3-month lags, and 2nd/3rd-Wednesday releases. The scraper hits two URL patterns to cover the full inflation-targeting era — the legacy `/ausstats/abs@.nsf/PreviousProducts/5206.0Main%20Features1<Mon>%20<YYYY>` namespace for 1993-Q1 through 2019-Q1, and the modern `/statistics/economy/national-accounts/.../<mon>-<yyyy>` namespace for 2019-Q2 onward — and persists original release dates to `data/external/gdp_release_dates.csv` (132 quarters as of 2026-Q2).
- `rba_d_release_calendar.py` — **algorithmic** for the RBA D1/D2 Financial Aggregates schedule, keyed by `reference_month_end`. Rule: publication is the last Mon-Fri of the month following the reference month. Verified against 14 RBA-archived release pages spanning 2001-07 → 2026-03 (the earliest reference month with a scrapable `/statistics/frequency/fin-agg/` page is 2001; pre-2001 pages return HTTP 404). Three hand-verified `_OVERRIDES` cover the known deviations: 2013-02 and 2024-02 are Good-Friday-on-month-end collisions where the RBA shifted to Thu Mar 28; 2025-11 is an early Dec 19 2025 release during the Direct-to-APRA (D2A) reporting-system decommissioning. Add further overrides as the D2A transition resolves and more exceptions are observed.
- `abs_ba_release_calendar.py` — **scraped** because the ABS Building Approvals release (Cat. 8731.0) has no clean algorithmic rule: across an 11-month sample spanning Sep 2020 → Mar 2026 the release landed days 1-8 of the second following month on every weekday Mon-Thu, with no era pattern (e.g. Mar 2023 ref → Mon May 8; Feb 2024 ref → Thu Apr 4; Nov 2025 ref → Wed Jan 7 2026). The scraper hits the modern `/statistics/industry/building-and-construction/building-approvals-australia/<mon>-<yyyy>` URL per reference month and persists original release dates to `data/external/abs_ba_release_dates.csv` keyed by `reference_month_end`. The modern URL space resolves back to Dec 2019 only (Nov 2019 / earlier 404); pre-floor observations fall back to a flat `+ 40` day offset in `abs_building_approvals`. Each row carries a `source` enum column (`abs_page` / `archive_org` / `inferred`) for provenance.
- `abs_tvd_release_calendar.py` — **scraped** because the ABS Total Value of Dwellings (Cat. 6432.0) release lands on a Tuesday but the *week* alternates between the 1st and 2nd Tuesday of the third month after quarter-end (verified across 2022-Q1 → 2025-Q4). The scraper hits `/statistics/economy/price-indexes-and-inflation/total-value-dwellings/<mon>-quarter-<yyyy>` and persists release dates to `data/external/abs_tvd_release_dates.csv` keyed by `reference_quarter_end`. The modern URL space resolves back to 2022-Q1 only; the RPPI-era quarters (2003-Q3 → 2021-Q4) the source module pulls as a splice baseline fall back to a flat `+ 76` day offset in `abs_total_value_dwellings`. Same `source` enum as `abs_ba_release_calendar`.
- `rba_i2_release_calendar.py` — **algorithmic** for the RBA I2 Index of Commodity Prices, keyed by `reference_month_end`. Rule: first business day on/after the 1st of the month following the reference month, where "business day" means Mon-Fri excluding a conservative AU-national + NSW (RBA is Sydney-based) public-holiday set (New Year's Day with Mondayisation, Australia Day Mondayised, Good Friday, Easter Monday, Anzac Day, Queen's/Sovereign's Birthday 2nd Mon June, NSW Labour Day 1st Mon Oct, Christmas/Boxing Day Mondayised). Verified against 14 Wayback-Machine `Publication date` headers spanning 2015-04 → 2023-06 plus the current live header (15 total samples), including a verified Easter shift to Tue Apr 3 2018 (Sun Apr 1 = Easter Sunday, Mon Apr 2 = Easter Monday). `_OVERRIDES` empty by default; populate as deviations are observed.
- `agb_yields_release_calendar.py` — **algorithmic** for RBA F2 Capital Market Yields – Government Bonds (daily ACGB benchmark yields), keyed by `observation_date` (the trade date — F2 is daily, not monthly). Rule: next AU business day after the trade date, where "business day" means Mon-Fri excluding the same AU/NSW public-holiday set used by `rba_i2_release_calendar` (the helper is imported from there). F2 publishes the prior session's closing yields the next morning around 11:30am AEST. Verified against 5 Wayback-Machine snapshots of `f2-data.csv` spanning 2018-01 → 2018-08 (every sample shows `Publication date` header = last data row date + 1 BDay exactly). `_OVERRIDES` empty by default; populate as deviations are observed. The current vintage F2 starts 2013-05-20 so every ingested observation is well post-floor — no NaT region for this source.
- `rba_e_release_calendar.py` — **scraped** because the RBA refreshes the E1 / E2 Household Balance Sheet tables in place with no public release-date archive, and the timing distribution (Fri/Tue/Sat, with Christmas pull-backs and Easter push-forwards) does not fit a clean algorithmic rule. The harvester pulls two sources: (1) the Wayback Machine CDX API to enumerate unique snapshots of `rba.gov.au/statistics/tables/csv/e2-data.csv` and extract the `Publication date` header + last data row from each, and (2) the live RBA CSV's `Publication date` header for the most recent release. Materialised to `data/external/rba_e_release_dates.csv` keyed by `reference_quarter_end`. Same `source` enum as the other scraped calendars (`rba_page` for the live header, `archive_org` for Wayback rows, `inferred` for overrides). **Coverage is partial** (~13% of E2 observations have scraped publication dates): the RBA does not publish a per-release archive and Wayback's crawl density is opportunistic, so most pre-2014 quarters and many post-2014 quarters fall through to the `rba_e2_household_ratios` source module's `+ 95` day flat-offset fallback. The ≥95% non-inferred target documented for other calendars is not achievable for E2 and is explicitly accepted here.
- `westpac_mi_release_calendar.py` — **algorithmic** for the Westpac-Melbourne Institute Consumer Sentiment release, keyed by `reference_month_end`. Rule: 2nd Wednesday of the reference month (the survey is conducted in the first week of the reference month and the headline released ~2 weeks later — verified for several recent releases, e.g. Apr 2026 ref → 2026-04-08). `_OVERRIDES` empty by default; populate as Westpac-MI reschedules. Same pattern as `lfs_release_calendar`. Unusual property: `publication_date < reference_month_end` for this series (the survey is released *during* the reference month), which is a faithful representation of the release timing rather than a leakage indicator.
- `nab_release_calendar.py` — **algorithmic** for the NAB Monthly Business Survey, keyed by `reference_month_end`. Rule: 2nd Tuesday of the month following the reference month (verified for the May 2025 reference month, released 2025-06-10). `_OVERRIDES` empty by default. Same pattern as `lfs_release_calendar`. We do not scrape: the NAB site retains only a ~5-month rolling window of release pages live (older URLs return HTTP 404, verified May 2026), so a scraped calendar would not cover the inflation-targeting training window and would be no more accurate than the algorithm.
- `fred_release_calendar.py` — **algorithmic** for both the FRED global signals module and the commodity prices module, keyed by `(series_id, observation_date)` because three rules coexist across the 10 logical series ingested across the two modules. Rules: (1) `us_daily_t_plus_1` (`us_fed_funds` / `us_10y_treasury` / `us_dxy_broad` / `brent_crude` / `wti_crude`) → next US business day after the trade date, where "US business day" means Mon-Fri excluding US federal holidays per `pandas.tseries.holiday.USFederalHolidayCalendar` (New Year's, MLK, Presidents', Memorial, Juneteenth, Independence, Labor, Columbus, Veterans, Thanksgiving, Christmas — with Mondayisation); (2) `us_same_day` (`us_vix`) → same calendar day (CBOE publishes the VIX close at 4:15pm ET — still strictly after every RBA 2:30pm AEST decision in walltime; downstream `publication_date < meeting_date` joins treat same-day as not-yet-known); (3) `us_monthly_flat` (`us_headline_cpi` / `us_core_cpi` / `iron_ore_spot` / `copper_spot`) → `observation_date + 20 calendar days` (flat conservative fallback — BLS releases monthly CPI 10-15 days after end of reference month, IMF Primary Commodity Prices typically ~5 days; +20 adds a small buffer without distorting realism, matches the WPI / Building-Approvals / TVD flat-fallback pattern; upgrade path to ALFRED-derived first-release dates documented in the calendar module docstring). `_OVERRIDES` keyed by `(series_id, observation_date)`, empty by default; populate as exceptions are observed (e.g. BLS reschedules around a US federal-government shutdown).
- `rba_minutes_release_calendar.py` — **algorithmic** for the RBA Board minutes (a *text* source, not a numerical series), keyed by `meeting_date` (the Board meeting's announcement / final day). Rule: the **second Tuesday strictly after the meeting** — `meeting_date + 14 days` for the regular Tuesday cadence, and the second calendar Tuesday after a non-Tuesday meeting (the COVID out-of-cycle Thursday meeting 2020-03-19 → 2020-03-31, +12 days). Unlike the monthly calendars, the meeting frame is irregular, so `build_rba_minutes_release_calendar(meeting_dates)` takes the meeting dates from F11 rather than enumerating them. Verified against `<meta name="dc.date">` headers on 195 regular-regime minutes pages (2008-02 → 2026-03): 175 match `+14d` exactly, the COVID meeting matches `+12d`, and the remaining ~19 divergent `dc.date` values are CMS artifacts (lag 0 or negative) the rule correctly overrides — which is why `dc.date` is used only as a warn-only soft cross-check in `rba_minutes.py`, never as the canonical value. `_OVERRIDES` (keyed by `meeting_date`) empty by default. Used by the text source `rba_minutes.py` below; see also its source-module note.

The respective source modules (`abs_cpi.py`, `abs_labour_force.py`, `abs_wpi.py`, `abs_gdp.py`, `rba_d.py`, `rba_e2_household_ratios.py`, `rba_i2_commodity_prices.py`, `agb_yields.py`, `bbsw_rates.py` (reusing `agb_yields_release_calendar`), `abs_building_approvals.py`, `abs_total_value_dwellings.py`, `westpac_mi_consumer_sentiment.py`, `nab_business_survey.py`, `fred_global_signals.py`, `commodity_prices.py` (reusing `fred_release_calendar`)) merge the calendar onto every observation in their `_attach_publication_dates` helpers. Known public-holiday / rescheduling exceptions go in each calendar's `_OVERRIDES` dict. Each source defines its own `_CALENDAR_VALIDATION_FLOOR`; observations on/after the floor must match the calendar and raise on miss, observations before the floor either retain `NaT` (CPI/LFS/GDP/RBA-D/RBA-I2/Westpac-MI/NAB/AGB-yields) or fall back to a flat conservative offset (WPI/Building Approvals/TVD/RBA-E).

- `asx_ib_futures.py` — **no release calendar.** ASX publishes 30-day IB futures EOD settlement prices same trading day (~6pm AEST after the 4:30pm close), so the publication-date rule is trivial: `publication_date == observation_date` (= the trade date). The rule lives directly in the source module's `_attach_publication_dates` helper rather than a separate calendar module. Source data comes from the `MattCowgill/cash-rate-scraper` GitHub repo (MIT, daily-refreshed Rds file scraped from the ASX MarkitDigital JSON endpoint the Rate Tracker page consumes — no free direct-from-ASX historical feed exists). **Coverage is 2022-04-21 → present** with a documented gap 2022-07-01 → 2022-07-20 (ASX site change); pre-2022 market-implied history is unavailable from any free source and the market-implied baseline is therefore evaluable only on the post-2022 window. Schema is per-`(trade_date, contract_month)` with `value = 100 - upstream_cash_rate` (i.e. raw settlement price) to keep the cached layer faithful to the exchange-traded convention. The companion helper `derive_meeting_implied(long_df, meetings_df, lookback_business_days=1)` joins the curve to the F11 meeting frame and returns the naïve implied cash rate (= `100 - settlement_price`) for the meeting-month contract as of T-1 business day. The naïve implied rate is an average of pre- and post-meeting cash rates within the contract month, not the isolated post-meeting rate; a separate decomposition helper can be added later as a backwards-compatible extension.

- `asx_200.py` — **no release calendar.** S&P/ASX 200 daily index (`^AXJO`) plus four sector sub-indices (Financials `^AXFJ`, Materials `^AXMJ`, Resources `^AXJR`, Energy `^AXEJ`). ASX cash market closes at 16:00 AEST/AEDT after the CSPA and EOD index values are public by ~16:30 market-local, so the publication-date rule is trivial: `publication_date == observation_date` (= the trade date). The rule lives directly in the source module's `_attach_publication_dates` helper rather than a separate calendar module. **Source choice**: Yahoo Finance v8 chart JSON endpoint (`query1.finance.yahoo.com/v8/finance/chart/<ticker>`) reached via plain `urllib.request` with the system SSL context — no third-party dependency, one ~700 KB JSON per ticker for full history. Probed alternatives rejected: ASX direct returns single-snapshot only, Stooq is now captcha-gated and requires a per-user API key, `yfinance` uses `curl_cffi` which bypasses the Windows cert store on this machine. **Coverage**: XJO from 1992-11-23 (the full inflation-targeting era), sector sub-indices from 2013-03-06/08 (the S&P/GICS reorganisation). **No pre-2013 splice** for the sector sub-indices — the All Ordinaries is a broad-market index, not a sector cut, so substituting it onto the Financials column would change the concept mid-series (same trade-off the project applied in reverse for RPPI/TVD and F11.1, where splice partners shared methodology). **Timezone gotcha**: Yahoo's daily-bar timestamps are at 10:00 *market-local* Sydney time as Unix seconds; during AEDT (~half the year) this is the previous UTC calendar day, so trade dates must be derived via `tz_convert('Australia/Sydney').floor('D').tz_localize(None)` — interpreting them via UTC `.date()` is off by one day. Sector tickers also emit `close: null` placeholder rows on ASX holidays, dropped at parse time.
 
- `bbsw_rates.py` — **reuses `agb_yields_release_calendar`.** RBA F1 (daily Bank Bill Swap Rate fixings: 1m / 3m / 6m — `FIRMMBAB30D` / `FIRMMBAB90D` / `FIRMMBAB180D`) publishes the prior session's closes the next morning around 11:30am AEST, so the publication-date rule is **next AU business day after the trade date** — *identical* to F2 / ACGB yields. Rather than ship a sister `bbsw_release_calendar.py`, the source imports `build_agb_yields_release_calendar` directly; if F1 and F2 ever diverge (no known case), the calendar can be forked. Verified against 10 Wayback Machine snapshots of `f1-data.csv` spanning 2018-01 → 2023-06: every snapshot shows `Publication date` header = last populated BBSW row's date + 1 BDay exactly. Coverage is **spliced**: the historical archive `/statistics/tables/xls-hist/f01dhist.xls` carries 1976-04 → 2010-12-31 daily (3m BAB from 1976-04-07; 1m / 6m both from 1995-01-03), the live CSV (`f1-data.csv`) carries 2011-01-04 → present. The XLS uses the same metadata layout as the live CSV with the date column read as `datetime64` rather than DD-MMM-YYYY strings; overlap at the splice boundary is resolved by `drop_duplicates(keep="last")` so the live-CSV vintage wins. Two **documented vintage shifts** that are *not* splice-adjusted: (1) administrator change Jan 2017 (AFMA → ASX) — the `Source` row in the historical archive says `AFMA`, the live CSV says `ASX`, level series is continuous; (2) ASX methodology change May 2018 from panel-mean ("NBBO") fixing to VWAP from actual transacted trades — a documented level break around 2018-05-21 that should be absorbed by spread/slope features (BBSW − cash, 3m-1m, 6m-3m) or flagged with a regime dummy at the methodology date, never restated. F1 also carries the daily cash rate, OIS (1m/3m/6m) and Treasury Notes (1m/3m/6m) under the same metadata layout — out of scope for this module (daily cash-rate semantics differ from the event-driven F11 decision series; OIS / Treasury Notes can ship as a separate `rba_f1.py` module later).

- `aud_exchange_rates.py` — **no release calendar.** RBA F11.1 (daily AUD exchange rates: USD, TWI, JPY, EUR, GBP, CNY, NZD — the SoMP-core set) publishes the 4.00pm Sydney close fix same trading day, so the publication-date rule is trivial: `publication_date == observation_date` (= the trade date). The rule lives directly in the source module's `_attach_publication_dates` helper rather than a separate calendar module. Verified against 5 Wayback Machine snapshots of `f11.1-data.csv` spanning 2023-03 → 2025-11: the `Publication date` header equals the latest data row's date exactly (zero business-day lag), every time. Coverage is **spliced**: 8 dated XLS archives under `/statistics/tables/xls-hist/` (`1991-1994.xls` → `2018-2022.xls`) provide pre-2023 history, the live CSV (`f11.1-data.csv`) carries 2023-01 → present. The eight files use the same metadata layout as the live CSV with the date column read as `datetime64` rather than DD-MMM-YYYY strings; overlap at the splice boundary is resolved by `drop_duplicates(keep="last")` so the live-CSV vintage wins. Coverage start is 1991-01-02 for every series except EUR (1999-01-04, euro inception). AED, ZAR, and SDR excluded — AED/ZAR frozen at `Publication date=01-Oct-2024` (same stale-feature failure mode as the dropped ABS Retail Trade), SDR is an IMF-sourced basket index rather than a market FX rate.

- `commodity_prices.py` — **uses `fred_release_calendar` (shared with `fred_global_signals.py`).** Four global commodity series from the FRED unauthenticated CSV endpoint (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`): iron ore `PIORECRUSDM` (IMF Primary Commodity Prices, 62% Fe China import spot, USD/dry MT, monthly, 1992-01 → present), copper `PCOPPUSDM` (IMF, LME Grade A cathode, USD/MT, monthly, 1992-01 → present), Brent crude `DCOILBRENTEU` (EIA Europe spot FOB, USD/bbl, daily, 1987-05 → present), WTI crude `DCOILWTICO` (EIA Cushing OK spot FOB, USD/bbl, daily, 1986-01 → present). **Why these and not the licensed alternatives**: SGX 62% Fe iron-ore futures (the market-watched daily benchmark) and LME-A copper daily are licensed; CME HG=F copper futures are a different basis (HG cathode, US-warehouse-delivered) that would silently change the conceptual series. The IMF monthly averages are the safe free-data choice for terms-of-trade features that already evaluate at monthly granularity. **Logical series-ID naming**: dropped the `us_` prefix used by `fred_global_signals` (these are global commodities, not US indicators) — series IDs are `iron_ore_spot` / `copper_spot` / `brent_crude` / `wti_crude`. Same FRED parser layout, same month-end conversion for monthly series, same +1 US BDay rule for daily series (EIA publishes prior session's closes next US business morning). Wide-format output is **split** into `data/external/commodity_prices_daily.csv` (Brent + WTI keyed by `trade_date`) and `commodity_prices_monthly.csv` (iron ore + copper keyed by `reference_month_end`) following the FRED-globals split — same reasoning (avoid forward-filling monthly values onto daily rows in the cached layer). **Oil-benchmark choice**: both Brent and WTI ingested so the Brent-WTI spread is available as a global-vs-US supply feature; trivial extension cost over Brent-only. **Vintage policy** same as fred_global_signals: stored values are the current FRED vintage; IMF restates commodity series ~annually on benchmark revisions.

- `fred_global_signals.py` — **uses `fred_release_calendar`.** First non-Australian macro source. Six series from the FRED unauthenticated CSV endpoint (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>`): US headline CPI `CPIAUCSL` (1947-01 → present, monthly SA), US core CPI ex food/energy `CPILFESL` (1957-01 → present, monthly SA), Fed funds effective daily `DFF` (1954-07 → present), US 10y CMT `DGS10` (1962-01 → present), Fed Nominal Broad USD Index `DTWEXBGS` (2006-01 → present), CBOE VIX close `VIXCLS` (1990-01 → present). FRED's CSV layout is simple — `observation_date,<SERIES_ID>` header, `YYYY-MM-DD` dates, UTF-8 (no metadata rows like RBA tables), blank cells for missing values (US federal holidays on daily series; Good Friday too for VIX following CBOE closures). First source to mix monthly + daily frequencies in one module — `observation_date` is the trade date for daily series and the **reference month-end** for monthly series (FRED stores monthly observations at month-start per BLS convention; the parser converts to month-end to match the project-wide period-end schema convention used by `abs_cpi`, `abs_labour_force`, `westpac_mi_consumer_sentiment`, etc.). Wide-format output is **split** into two CSVs to avoid forward-filling monthly values onto daily rows in the cached layer: `data/external/fred_global_signals_daily.csv` keyed by `trade_date` (DFF / DGS10 / DTWEXBGS / VIXCLS), and `data/external/fred_global_signals_monthly.csv` keyed by `reference_month_end` (CPIAUCSL / CPILFESL). **DXY substitution caveat**: the licensed ICE Dollar Index (the actual "DXY") is *not* on FRED — `DTWEXBGS` (the Fed's free broad-basket analog, ~26 currencies including EM) is shipped instead. Correlation with DXY is ≥ 0.95 historically at quarterly/monthly granularity but diverges during EM-stress episodes; documented in the source-module docstring. **CPI flat-offset rationale**: the BLS-CPI release-date rule isn't a clean algorithm (releases land on Tue/Wed/Thu at various days 10-15 after end of reference month) and ALFRED-derived first-release dates would require either a free API key or hundreds of per-vintage CSV pulls — overkill for v1. The `+20 calendar days from period-end` flat fallback is conservative (5-10 day buffer over the worst observed lag) and never claims a CPI release was available earlier than reality. Future contributors needing precision can replace with an ALFRED-derived calendar; the upgrade path mirrors the scheduled-vintage-accumulation upgrade documented under the broader vintage policy.

For any future ABS series we ingest, the fallback is a per-source conservative `observation_date + N days` offset until a similar release-rule module is added. Each source module defines its own offset constant local to that module — do not centralise. Prefer building an algorithmic calendar when the ABS publishes to a documented rule, or a scraped calendar (cf. WPI / GDP) when not; only fall back to a flat offset for series whose release pages are not scrapable.

**Text-data schema convention.** Every numerical source above shares the long-format `[observation_date, publication_date, series_id, value]` shape because each observation is a single numeric reading anchored to a recurring series. Text releases don't fit — each statement is a unique document with no recurring `series_id` and the value is a string rather than a float. Text sources therefore use a **decision-keyed wide schema** — one row per Board meeting with columns `[decision_date, publication_date, governor, title, url, sha256, raw_html_filename, body_text, paragraphs]` — joining 1:1 on `decision_date` to the F11 meeting frame (and indirectly to every numerical source via the meeting frame). The materialised artifact is **Parquet** rather than CSV: bodies span ~10-50 paragraphs with embedded newlines / `&nbsp;` / quotation marks, all of which fight CSV escaping; Parquet also preserves the `paragraphs: list[str]` column natively and gives ~10× smaller files on disk. The first text source below (`rba_media_releases.py`) sets this contract; the three remaining text scrapers (RBA minutes, Statement on Monetary Policy, Governor speeches) will inherit it.

- `rba_media_releases.py` — **no release calendar; F11-driven crawl.** Post-decision media-release statements ("Statement by ...") from `https://www.rba.gov.au/media-releases/<YYYY>/{mr,jmr}-<YY|YYYY>-<NN>.html` (the `jmr-` prefix marks joint releases with Treasury; year 2000 uniquely uses a 4-digit year segment, every other year uses `YY`). Publication-date rule is trivial (`publication_date == decision_date`: the RBA announces at 14:30 AEST and the statement is on the website the same moment), so no separate calendar module — the rule lives directly in `fetch()` / `_validate_cross_check`. **F11-driven, not yearly-archive-scraped**: `rba_f11.fetch()` already populates `statement_url` on every meeting row that issued a release (NaN on pre-Apr-2010 hold meetings — pre-2010 the Board only issued statements at rate-change meetings; from Apr 2010 every meeting gets a statement), giving an exact 1:1 expected set with no per-era title-regex maintenance and no risk of false positives from non-monetary-policy "Statement by ..." releases (regulatory, payments, appointments). Cross-check enforces three invariants: (1) every F11 row on/after 1993-01-01 with a `statement_url` yields exactly one media-release row; (2) every row's parsed `publication_date` (from `<span itemprop="datePublished">`) equals its `decision_date`; (3) no extras leak through. `_KNOWN_MISSING_DECISIONS` allowlist starts empty — missing statements are bugs, not data conditions. Three title eras observed: 1993 → ~Apr-2010 (Fraser / Macfarlane / early-Stevens; rate-change meetings only) → `"Statement by the Governor, Mr <Surname>: <topic>"`; ~Apr-2010 → Jan-2024 (post-template-standardisation Stevens / Lowe / early-Bullock) → `"Statement by <FirstName Surname>, Governor: Monetary Policy Decision - <Month YYYY>"`; Feb-2024 → present (post-governance-reform; separate Monetary Policy Board) → `"Statement by the Reserve Bank Board: Monetary Policy Decision - <Month YYYY>"`. Governor attribution is best-effort regex on the title; regex misses log a warning and yield `governor=None` rather than raising. Body extraction prefers the modern `<div class="rss-mr-content">` body wrapper (~2018+ releases) and falls back to walking every `<p>` under `<div id="content">` with ancestor-class exclusion for contact-box / aside / related-links blocks (1990s layout). Whitespace is normalised: NBSP (`\xa0`, common in "4.35 per cent" formatting) collapsed to single space; tabs / multi-newlines collapsed. **Network gotcha**: the RBA WAF returns HTTP 403 for requests carrying a custom `User-Agent` header but accepts the default `Python-urllib` UA — the reverse of the GitHub raw endpoint pattern used by `asx_ib_futures.py`. The source therefore passes the URL string directly to `urlopen` and never wraps it in a `Request` with headers. Wide-format output at `data/external/rba_media_releases.parquet`.
- `rba_minutes.py` — **algorithmic release calendar; F11-driven crawl.** Board-meeting minutes (~30-50 paragraphs each) from `https://www.rba.gov.au/monetary-policy/rba-board-minutes/<YYYY>/<slug>.html` (modern slug `<YYYY-MM-DD>` = meeting date; pre-2009 slug `<DDMMYYYY>`). **Inherits the decision-keyed wide schema** set by `rba_media_releases.py` (see the Text-data schema convention above) with two differences: the `governor` column is **dropped** (minutes are a Board document with no individual attribution), and `publication_date > decision_date` (minutes are released ~14 days *after* the announcement — the no-leakage sign flips). **F11-driven, not archive-scraped**: `rba_f11.fetch()` already populates `minutes_url`; `_select_targets` filters on `minutes_url.notna()`, which naturally excludes the structural "most recent meeting has no minutes for ~2 weeks" gap (a NaN row is not a target, never a miss — do not allow-list it). **Publication date is algorithmic** via `rba_minutes_release_calendar.py` (second Tuesday strictly after the meeting; see calendar note in Invariant #1), **not scraped**: minutes pages carry no `itemprop="datePublished"`, and the only date marker `<meta name="dc.date">` is ~10% CMS-artifact garbage, so `dc.date` is scraped only as a **warn-only soft cross-check** in `_attach_publication_dates` (a plausible divergence — `dc.date` strictly after the announcement but ≠ the rule — logs a warning; an implausible `dc.date` ≤ `decision_date` is ignored silently). **Validation floor is 2008-02-05** (`_CALENDAR_VALIDATION_FLOOR`), the start of the regular publication regime — *not* the project-wide 1993-01-01. The 2006-10 → 2007-11 minutes exist in F11 (`minutes_url` populated) but were backfilled and only published ~2007-12-05, so the second-Tuesday rule does not describe them; they are excluded by the floor rather than carried with overrides. Title is the uniform `<h1>` (`"Minutes of the Monetary Policy Meeting of the Reserve Bank Board"` every era). Body extraction walks every `<h2>`/`<p>` under `<div id="content">` in document order with the same ancestor-class exclusion as the media-release module (related-links rails / contact box / tiles), **capturing `<h2>` section headings as standalone entries in `paragraphs`** interleaved with their following `<p>` content (downstream features can split / filter on heading prefixes); trailing navigation headings (`Related Information` / `Related Content`) are dropped. Cross-check enforces: (1) every in-window F11 row with a `minutes_url` yields exactly one minutes row; (2) no extras; (3) `publication_date > decision_date` for every row. `_KNOWN_MISSING_DECISIONS` allowlist starts empty. Same **RBA WAF gotcha** as media releases (default `Python-urllib` UA, no custom `User-Agent`). Wide-format output at `data/external/rba_minutes.parquet`; coverage 2008-02-05 → present (195 meetings as of 2026-Q2), lag distribution `{+14d Tuesday meetings, +12d the single COVID Thursday meeting}`.
- `rba_somp.py` — **no release calendar; archive-index-driven crawl (NOT F11-driven).** Quarterly **Statement on Monetary Policy** (Feb/May/Aug/Nov) — a long multi-chapter document of which v1 captures the **Overview page only** (the highest-signal, cross-era-comparable summary). **Inherits the decision-keyed wide schema** (see the Text-data schema convention above) with the same two minutes-style differences — `governor` dropped (SoMP is a Bank document), and `publication_date` not tied to `decision_date` by a fixed sign — but diverges from the first two text scrapers on **two** axes the schema convention paragraph flagged: (1) **quarterly, not per-meeting** — the SoMP joins the meeting frame **sparsely** (~4 rows/year; absent for most meetings), and (2) **no `f11.somp_url` driver** — the crawl enumerates issues from the SoMP archive index (`/publications/smp/`) and maps each `<YYYY>/<mon>` issue to the Board decision in that same calendar month (`decision_date` == F11 publication_date), validating the mapping rather than an exact F11 URL set. **Publication date is scraped, not algorithmic**: SoMP pages carry no `itemprop="datePublished"`, but unlike the minutes' `<meta name="dc.date">` (~10% CMS garbage) the SoMP `dc.date` is reliable on every era probed, so it is the canonical `publication_date` — no calendar module ships. **The publication-vs-decision sign is mixed by era** (so the cross-check asserts no fixed sign, unlike media releases `==` and minutes `>`): same-day as the decision from 2024 (8-meeting cadence), `+3d`-typical/`+24d`-November in the ~2008–2023 era, and *before* the decision in the pre-2008 era (e.g. Feb 2006 SoMP 2006-02-07 < decision 2006-02-08); genuine gaps span −14d to +24d, so the warn-only soft-check flags only `|publication_date − decision_date| > 31d` (a cross-month dc.date artifact / mis-map). **Validation floor is 2006-02-01** (`_CALENDAR_VALIDATION_FLOOR`), the first quarterly SoMP published as browsable HTML chapters — *not* the project-wide 1993-01-01. The quarterly SoMP began **1997**, but the **1997–2005 issues are PDF-only** (per-month HTML folders 403/404; year pages link only `boxes.html` + PDFs) and excluded from v1 as a documented coverage limitation (a PDF-extraction path could reach back to 1997 later). Overview page filename has three era forms — `overview.html` (most), `00-overview.html` (a few mid-era), `intro.html` (2006–2014 "Introduction") — resolved by trying each in order; body extraction walks `<h2>`/`<p>` under `<div id="content">` with ancestor-class exclusion (the minutes set plus the SoMP `nav-publication-contents` / `publication-related-links` rails), capturing `<h2>` key-message headings as standalone interleaved entries (the 2015+/2024-redesign Overview uses them; pre-2015 is flat `<p>`) and dropping trailing `Related Information`/`Related Content` nav headings. Cross-check raises on a SoMP issue with no output row, an output row not derived from the index, a `decision_date` absent from F11 (phantom mapping), or two issues mapping to the same `decision_date` (collision); `_KNOWN_MISSING_DECISIONS` allowlist starts empty. Same **RBA WAF gotcha** (default `Python-urllib` UA). Wide-format output at `data/external/rba_somp.parquet`; coverage 2006-02-08 → present (82 issues as of 2026-Q2).

**Vintage policy.** Stored values are the **current vintage** at the time the source was last downloaded, *not* the original first-release value. ABS routinely revises historical observations (seasonal re-estimation, methodology changes, late source data) and we do *not* reconstruct prior vintages. This is a known deviation from strict real-time correctness: for revised periods, the value next to `publication_date` is *not* exactly the number the RBA board saw on that date. The bias is small for series with light revisions (headline CPI) and larger for heavily-revised series (LFS sub-aggregates, GDP). Acknowledged as a limitation in [README.md](README.md); potential upgrade paths (scheduled vintage accumulation, real-time test-window evaluation) are tracked in [CHECKLIST.md](CHECKLIST.md).

**Excluded series.** Some macro candidates have been considered and deliberately dropped from the feature set. Document the *why* here so the same dataset is not re-introduced silently by a future contributor.

- **ABS Retail Trade (Cat. 8501.0)** — dropped 2026-05-21. The ABS discontinued the monthly Retail Trade, Australia release after the Jun 2025 reference month. Coverage on the SDMX `RT` dataflow is frozen at 1982-04 → 2025-06 (monthly nominal) and 1983-Q3 → 2025-Q2 (quarterly chain-volume). Including a frozen series would mean every prediction the model makes after Jul 2025 carries the *same* retail values regardless of how much time has passed since the last real datapoint — increasingly stale information that the model would learn to weight as if fresh. The ABS-designated successor is the Monthly Household Spending Indicator (`HSI_M` dataflow), which measures a different concept (consumer card-spending across all merchant types, not retail-store turnover) and is out of scope for v1. If a future iteration adds HSI_M, it should be a *new* source module — do not splice it onto the discontinued retail history.

- **ICE U.S. Dollar Index (the licensed "DXY")** — substituted out 2026-05-27 when `fred_global_signals.py` shipped. The 6-currency ICE DXY basket (EUR / JPY / GBP / CAD / SEK / CHF) is a licensed product and not redistributed via FRED. The Fed's free `DTWEXBGS` ("Nominal Broad U.S. Dollar Index", ~26 currencies including EM) is shipped instead — same conceptual signal (USD strength against trading partners) with broader coverage. Historical correlation with the licensed DXY is ≥ 0.95 at quarterly / monthly granularity but the broad index moves more than DXY during EM-stress episodes (e.g. 2013 taper tantrum, 2015 CNY devaluation), so the two are not interchangeable for episode-level analysis. If a future iteration wants the licensed series, it should be a *separate* paid-source module — do not silently splice the two indices.

- **RBNZ Official Cash Rate (NZ peer policy rate)** — dropped 2026-05-27. Was scoped as a "Pacific peer cross-check" feature on the rationale that AU and NZ central banks face similar shocks and RBNZ has historically led the RBA on cycle turns. Two reasons for dropping: **(1) likely small marginal weight** — the global-rate-cycle signal the project actually needs (USD strength, US Fed path spillovers, terms-of-trade) is already carried by `us_fed_funds` / `us_10y_treasury` / `us_dxy_broad` in `fred_global_signals.py` and the four global commodities in `commodity_prices.py`. RBNZ OCR adds a regional cross-check that is highly correlated with the global signals already in the feature set, so feature-importance lift is expected to be small after the existing global block is conditioned on. **(2) feasibility difficulty** — no clean free source exists: the RBNZ landing pages probed (`/statistics/series/exchange-and-interest-rates/official-cash-rate-decisions-and-current-rate` and `/statistics/key-statistics/policy-rates`) returned HTTP 404 (URL space appears to have moved); the only working FRED candidate (`IRSTCI01NZM156N`, OECD NZ Immediate Rates < 24 Hours) is the overnight market rate rather than the OCR decision series itself and is frozen at 2024-12 (~17 months stale as of probe date), which is the same stale-feature failure mode that disqualified ABS Retail Trade and AED / ZAR FX rates. The combination — modest expected lift over the existing US/global block, plus no clean low-effort source — does not justify ingest cost for v1. If a future iteration finds a current, free, decision-keyed RBNZ feed and feature analysis shows the existing global block leaves residual signal unexplained, this can be revisited as a *new* source module.

### 2. No random k-fold cross-validation
This is time-series. Always use:
- `sklearn.model_selection.TimeSeriesSplit`, OR
- The custom `WalkForwardSplit` in `src/rba/validation/cv.py`.

If you find yourself reaching for `KFold` or `train_test_split(..., shuffle=True)`, **stop**. Sole exception: shuffling within a *single training window* for batch SGD; never across windows.

### 3. No leakage in rolling/expanding stats
- `pd.Series.rolling(window).mean()` is past-only by default — keep it that way.
- Verify no `center=True` slips in.
- `.expanding()` is fine.
- Z-scores must use *rolling* mean/std, not whole-series mean/std.
- Scalers (`StandardScaler` etc.): fit on training window only, transform test.

### 4. Imputation hygiene
- Never impute using whole-series statistics.
- Add `<col>_is_missing` indicator alongside any imputation.
- Prefer letting trees handle NaN natively over imputing.
- Imputation must be wrapped in a `Pipeline` so it's fit per-fold.

### 5. Reproducibility
- `RANDOM_SEED = 12` defined in `src/rba/config/__init__.py`; import everywhere.
- All versions pinned in `pyproject.toml` lockfile.
- Hash raw data on download; record hash + URL + timestamp in `data/raw/<source>/_metadata.json`.

## Model interface contract

Every model in `src/rba/models/` (including baselines) implements:

```python
from typing import Protocol
import numpy as np
import pandas as pd

class Model(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "Model": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Required for classification; raises NotImplementedError for regressors."""
        ...

    @property
    def feature_names_(self) -> list[str]: ...
```

Baselines (`majority`, `persistence`, `taylor_rule`, `market_implied`) implement the same interface so they're swappable into any evaluation loop.

## Adding a new feature
1. Write builder in `src/rba/features/<group>.py` as a pure function: `def build_<name>(df: pd.DataFrame) -> pd.DataFrame`.
2. Annotate input/output shapes in numpy-style docstring.
3. Add unit test in `tests/features/test_<group>.py` covering:
   - Output shape / dtype
   - **Leakage test**: feature at row `t` does not depend on rows where `publication_date > meeting_date_t`.
   - NaN handling at series start (lags should produce NaNs, not zeros).
4. Register in `src/rba/config/features.yaml` with name, dependencies, default params.

## Adding a new model
1. Create `src/rba/models/<name>.py` implementing the `Model` protocol.
2. Register in `src/rba/models/__init__.py` model registry.
3. Add config entry in `src/rba/config/models.yaml` with default hyperparameters and search space.
4. Add unit test in `tests/models/test_<name>.py` covering fit/predict/predict_proba on a tiny synthetic dataset.

## Adding a new data source
1. Create `src/rba/data/sources/<source>.py` exposing `def fetch() -> pd.DataFrame`.
2. Output must have `observation_date`, `publication_date`, plus value columns. Document schema in module docstring.
3. Save raw to `data/raw/<source>/<YYYY-MM-DD>.parquet`.
4. Hash the raw bytes; log to `data/raw/<source>/_metadata.json`.
5. Register in `src/rba/data/refresh.py` orchestration.

## Coding conventions
- **Type hints**: required on public functions.
- **Docstrings**: numpy-style; for any function returning arrays/frames, include a `Shapes` section.
- **Imports**: absolute (`from rba.features import lags`); grouped (stdlib / third-party / local).
- **Logging**: `loguru` via `from loguru import logger` (single global logger; sink configured in `src/rba/config/__init__.py`). Never `print` outside notebooks.
- **Configs**: YAML for experiment configs; runtime constants in `src/rba/config/__init__.py`.
- **DataFrames**: pandas by default; polars where pandas is the bottleneck, with explicit conversion at module boundaries.
- **No magic numbers**: name constants in config.

## Common pitfalls (watch for these)
- Joining on calendar date instead of publication date → **leakage**.
- `.shift(k)` after a `groupby` that drops the index → silent corruption.
- Class imbalance + accuracy as the only metric → meaningless results.
- Hyperparameter tuning on the held-out test set → biased generalisation estimate.
- `random_state=None` anywhere → reproducibility broken.
- Re-fitting a scaler/imputer on the test split → leakage.
- Forgetting the meeting cadence changed Feb 2024 (11/yr → 8/yr): use meeting-indexed time, add gap-since-last-meeting feature.
- Forward-fill across long gaps without an "age of last reading" feature → stale values look fresh.
- Using whole-series z-scores instead of rolling z-scores.

## Conventions for Claude Code in this repo
- Always run `uv run pytest` after non-trivial changes touching `src/`.
- Before writing data-loading code, check whether the source is already implemented in `src/rba/data/sources/`.
- Follow the "Adding a new ..." sections to the letter when extending features / models / sources.
- When proposing a CV or evaluation change, verify it doesn't violate Invariants 1–3.
- **Default to complete files. If a snippet must be partial, say so explicitly.**
- When showing pandas/numpy ops, comment shapes inline (e.g. `# X: (n_meetings, n_features)`).
- For experiments: log to MLflow with run name `<target>_<model>_<feature_set>_<timestamp>`.

## Useful references
- RBA F1.1 (decisions): https://www.rba.gov.au/statistics/tables/
- RBA media releases: https://www.rba.gov.au/media-releases/
- ASX 30-day futures (RBA Rate Tracker): https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker
- ABS data: https://www.abs.gov.au/
- FRED (US data): https://fred.stlouisfed.org/
- ALFRED (Archival FRED — per-observation first-release vintages): https://alfred.stlouisfed.org/
- IMF Primary Commodity Prices (iron ore / copper monthly source): https://www.imf.org/en/Research/commodity-prices
- EIA Petroleum & Other Liquids (Brent / WTI daily spot source): https://www.eia.gov/dnav/pet/
- Tomasz Woźniak's RBA forecasts (benchmark): https://forecasting-cash-rate.github.io/
- yukit-k/centralbank_analysis (FOMC NLP analog): https://github.com/yukit-k/centralbank_analysis
- kakeith/op-fed (hawkish/dovish labelled corpus): https://github.com/kakeith/op-fed
