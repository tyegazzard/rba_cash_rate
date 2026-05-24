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

**CPI, Labour Force (LFS), Wage Price Index (WPI), GDP, RBA D1/D2, RBA E1/E2, ABS Building Approvals, and ABS Total Value of Dwellings** each have their own release-calendar module under `src/rba/data/`:

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
- `rba_e_release_calendar.py` — **scraped** because the RBA refreshes the E1 / E2 Household Balance Sheet tables in place with no public release-date archive, and the timing distribution (Fri/Tue/Sat, with Christmas pull-backs and Easter push-forwards) does not fit a clean algorithmic rule. The harvester pulls two sources: (1) the Wayback Machine CDX API to enumerate unique snapshots of `rba.gov.au/statistics/tables/csv/e2-data.csv` and extract the `Publication date` header + last data row from each, and (2) the live RBA CSV's `Publication date` header for the most recent release. Materialised to `data/external/rba_e_release_dates.csv` keyed by `reference_quarter_end`. Same `source` enum as the other scraped calendars (`rba_page` for the live header, `archive_org` for Wayback rows, `inferred` for overrides). **Coverage is partial** (~13% of E2 observations have scraped publication dates): the RBA does not publish a per-release archive and Wayback's crawl density is opportunistic, so most pre-2014 quarters and many post-2014 quarters fall through to the `rba_e2_household_ratios` source module's `+ 95` day flat-offset fallback. The ≥95% non-inferred target documented for other calendars is not achievable for E2 and is explicitly accepted here.
- `westpac_mi_release_calendar.py` — **algorithmic** for the Westpac-Melbourne Institute Consumer Sentiment release, keyed by `reference_month_end`. Rule: 2nd Wednesday of the reference month (the survey is conducted in the first week of the reference month and the headline released ~2 weeks later — verified for several recent releases, e.g. Apr 2026 ref → 2026-04-08). `_OVERRIDES` empty by default; populate as Westpac-MI reschedules. Same pattern as `lfs_release_calendar`. Unusual property: `publication_date < reference_month_end` for this series (the survey is released *during* the reference month), which is a faithful representation of the release timing rather than a leakage indicator.
- `nab_release_calendar.py` — **algorithmic** for the NAB Monthly Business Survey, keyed by `reference_month_end`. Rule: 2nd Tuesday of the month following the reference month (verified for the May 2025 reference month, released 2025-06-10). `_OVERRIDES` empty by default. Same pattern as `lfs_release_calendar`. We do not scrape: the NAB site retains only a ~5-month rolling window of release pages live (older URLs return HTTP 404, verified May 2026), so a scraped calendar would not cover the inflation-targeting training window and would be no more accurate than the algorithm.

The respective source modules (`abs_cpi.py`, `abs_labour_force.py`, `abs_wpi.py`, `abs_gdp.py`, `rba_d.py`, `rba_e2_household_ratios.py`, `abs_building_approvals.py`, `abs_total_value_dwellings.py`, `westpac_mi_consumer_sentiment.py`, `nab_business_survey.py`) merge the calendar onto every observation in their `_attach_publication_dates` helpers. Known public-holiday / rescheduling exceptions go in each calendar's `_OVERRIDES` dict. Each source defines its own `_CALENDAR_VALIDATION_FLOOR`; observations on/after the floor must match the calendar and raise on miss, observations before the floor either retain `NaT` (CPI/LFS/GDP/RBA-D/Westpac-MI/NAB) or fall back to a flat conservative offset (WPI/Building Approvals/TVD/RBA-E).

For any future ABS series we ingest, the fallback is a per-source conservative `observation_date + N days` offset until a similar release-rule module is added. Each source module defines its own offset constant local to that module — do not centralise. Prefer building an algorithmic calendar when the ABS publishes to a documented rule, or a scraped calendar (cf. WPI / GDP) when not; only fall back to a flat offset for series whose release pages are not scrapable.

**Vintage policy.** Stored values are the **current vintage** at the time the source was last downloaded, *not* the original first-release value. ABS routinely revises historical observations (seasonal re-estimation, methodology changes, late source data) and we do *not* reconstruct prior vintages. This is a known deviation from strict real-time correctness: for revised periods, the value next to `publication_date` is *not* exactly the number the RBA board saw on that date. The bias is small for series with light revisions (headline CPI) and larger for heavily-revised series (LFS sub-aggregates, GDP). Acknowledged as a limitation in [README.md](README.md); potential upgrade paths (scheduled vintage accumulation, real-time test-window evaluation) are tracked in [CHECKLIST.md](CHECKLIST.md).

**Excluded series.** Some macro candidates have been considered and deliberately dropped from the feature set. Document the *why* here so the same dataset is not re-introduced silently by a future contributor.

- **ABS Retail Trade (Cat. 8501.0)** — dropped 2026-05-21. The ABS discontinued the monthly Retail Trade, Australia release after the Jun 2025 reference month. Coverage on the SDMX `RT` dataflow is frozen at 1982-04 → 2025-06 (monthly nominal) and 1983-Q3 → 2025-Q2 (quarterly chain-volume). Including a frozen series would mean every prediction the model makes after Jul 2025 carries the *same* retail values regardless of how much time has passed since the last real datapoint — increasingly stale information that the model would learn to weight as if fresh. The ABS-designated successor is the Monthly Household Spending Indicator (`HSI_M` dataflow), which measures a different concept (consumer card-spending across all merchant types, not retail-store turnover) and is out of scope for v1. If a future iteration adds HSI_M, it should be a *new* source module — do not splice it onto the discontinued retail history.

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
- Tomasz Woźniak's RBA forecasts (benchmark): https://forecasting-cash-rate.github.io/
- yukit-k/centralbank_analysis (FOMC NLP analog): https://github.com/yukit-k/centralbank_analysis
- kakeith/op-fed (hawkish/dovish labelled corpus): https://github.com/kakeith/op-fed
