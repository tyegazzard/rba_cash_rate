# RBA Cash Rate Prediction — what got built

A concise inventory of **what** exists and a one-line **how** for each. For the full
spec and architectural detail see [CONTEXT.md](CONTEXT.md); for progress see
[CHECKLIST.md](CHECKLIST.md); for the data catalog see [DATA.md](DATA.md).

**Goal:** Predict the next RBA Board cash-rate decision as 3-class (cut / hold / hike)
for the upcoming meeting. Inflation-targeting era onward (1993+); the prediction point
is the announcement date (2:30pm decision), so `meeting_date ≡ publication_date`.

**Stack:** Python 3.11, `uv` env, pandas 3. Tooling: ruff + mypy + pytest + pre-commit
+ GitHub Actions CI. `RANDOM_SEED=12`. MLflow is wired but **upstream-blocked** (mlflow
caps pandas<3) → local artifacts under `reports/` are the durable record.

## Data layer (`src/rba/data/sources/`, `refresh.py`, `inventory.py`)
- **19 numeric sources (120 series) — these are what the model runs on:** RBA F11
  (target/decisions), ABS macro (CPI, labour force, WPI, GDP, building approvals,
  dwellings), RBA aggregates (D credit, E household ratios, I2 commodities), market
  (ASX 30-day IB futures, AGB yields, AUD/FX, ASX200, BBSW), global via FRED (US CPI,
  fed funds, 10y, DXY, VIX, iron ore/copper/oil).
- **4 text scrapers — built, but not part of the v1 model** (media releases, minutes,
  SoMP, speeches). Each has a full parser + cross-check; see Feature engineering for
  why none reach `X`. Current crawl coverage: minutes 197/197 and SoMP 82/82 complete;
  media releases 34/230 and speeches 22/1,340 — both aborted mid-crawl on unhandled
  `datePublished` layout variants (multi-day date ranges; a few pages with the element
  absent) and need a parser fix plus a re-run. None are materialised to
  `data/external/*.parquet` by `refresh.py`; that write only happens when a text module
  is run as `__main__`.
- **How:** each source `fetch()` writes an immutable raw snapshot + hashed
  `_metadata.json` (provenance). `refresh.py` orchestrates all sources with per-source
  error isolation. `inventory.py` catalogs them → generates `DATA.md` +
  `inventory.parquet`. Each series records observation_date AND publication_date.

## Preprocessing / point-in-time alignment (`src/rba/data/align.py`)
- **How:** as-of join of every numeric source onto the meeting frame with a **strict
  `publication_date < meeting_date`** rule (no same-day leakage). Emits per-series
  `<sid>`, `<sid>_age_days` (staleness), `<sid>_is_missing`. Adds regime dummies
  (governor eras, GFC, COVID, forward guidance, 2024 cadence change) and
  gap-since-last-meeting. Output: `data/processed/master.parquet` (hashed). Leakage
  guarded by synthetic-future-injection tests.

## Feature engineering (`src/rba/features/`)
- Builders: lags, rolling (mean/std/z-score/min/max/EWMA, past-only), changes (Δ/%Δ/YoY),
  target-lags. The point-in-time level columns and `regime_*` dummies come from
  `align.py` and pass through untouched.
- **No text features in v1.** The Loughran-McDonald lexicon scorer is implemented and
  unit-tested (`features/text/lexicon.py`), but it produces a standalone artifact only:
  `text` sits in `build.py`'s `DEFERRED_GROUPS`, so point-in-time alignment of text onto
  the meeting frame was never built and no `lm_*` column exists in `features.parquet`.
  (Embeddings / custom hawkish-dovish dict also deferred.)
- **How:** YAML-driven builders → `features.parquet` (681 model columns, all numeric)
  + a feature-version hash. `feature_columns()` is the single leakage-free X definition
  (excludes outcome columns).

## Baselines + models (`src/rba/models/`, `models.yaml`)
- **Baselines:** majority-class, persistence, Taylor rule (literature + fitted),
  market-implied (from ASX futures).
- **Models:** logistic (+L1/L2), random forest, XGBoost, LightGBM, SVM (linear/RBF),
  ordinal logistic (mord), MLP; voting + stacking ensembles.
- **How:** all implement one `Model` protocol (fit/predict/predict_proba/
  feature_names_), built from `models.yaml` via a `build_model()` factory — swappable
  into one eval harness.

## Validation & evaluation (`src/rba/validation/`)
- **How:** expanding `WalkForwardSplit`; an untouchable held-out test window (39
  meetings from 2022-05, deliberately spanning a full hike→hold→cut cycle). Optuna
  hyperparameter search + threshold tuning + SMOTE + per-fold feature selection — all
  on the **dev** portion only, never the test set. Metrics: accuracy, balanced
  accuracy, macro-F1, log-loss, Brier, calibration/ECE, and **hit-rate-vs-market**.
  `report.py` runs everything on the held-out window → `reports/results.md`,
  `holdout_metrics.csv`, and persists the refit best model to
  `reports/best_model/*.joblib`.

## Key result
- **The market-implied baseline (ASX futures) wins** — balanced accuracy ≈0.656; best
  learned model is LightGBM ≈0.624; no model beats the market. Honest null result: on
  a rate-tracked market the futures curve already prices the macro signal the models
  try to learn. Balanced accuracy is the headline metric (the test window is only
  ~51% hold, so raw accuracy is misleading).

## Deployment — before-meeting prediction
`src/rba/data/meeting_schedule.py`, `align.append_future_meeting`, `src/rba/models/predict_next.py`
- **How:** `meeting_schedule.py` holds the hand-verified forward RBA meeting dates
  (validated against F11). `append_future_meeting()` adds a leakage-safe undecided row
  (outcome cols NaN, prior_rate = last decided rate) *before* the point-in-time join.
  `predict_next.py` loads the `best_model/` joblib (first consumer), builds the
  next-meeting feature row, reindexes to the model's features, predicts, adds the
  market-implied comparison + top global feature importances + a data-vintage summary,
  and appends to `reports/predictions/predictions.jsonl`. CLI:
  `python -m rba.models.predict_next`. Runs on cached data by default; `--refresh`
  pulls live. Run manually (no scheduling/automation).

## Dashboard (`streamlit_app/main.py`)
- **How:** Streamlit app over `predict_next_meeting()` — next-meeting prediction,
  hike/hold/cut probability distribution, model-vs-market comparison, top drivers, and
  a held-out track-record panel (leaderboard + confusion matrix). The prediction is
  `st.cache_data`-memoised. Run: `uv run streamlit run streamlit_app/main.py`.

## Known limitations
- Market baseline beats the models (the honest finding). Small test window (N≈39).
  Vintage bias (current-vintage macro, not the real-time values the Board saw).
  Taylor-rule inputs are proxies. **No text signal in the model** — scrapers and the LM
  scorer exist, but text→meeting-frame alignment is deferred and two of the four crawls
  are incomplete, so the models see numeric features only. MLflow upstream-blocked.
  Meeting schedule needs manual 2027+ dates appended when the RBA publishes them.
  `DATA.md` is stale (predates the last crawl) — regenerate with
  `python -m rba.data.inventory`; note it reports document sources from the
  unmaterialised `data/external/*.parquet`, so the four text rows read blank.
