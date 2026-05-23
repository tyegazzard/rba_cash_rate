# RBA cash rate prediction — project checklist

Progress: 0 / 138 (0%)

## 1. Problem definition

- [x] Lock target formulations: 3-class, multi-class with magnitudes, Δrate regression, level regression, ordinal
- [x] Confirm forecast horizon: next meeting only
- [x] Confirm scope: announcement (primary), surprise vs market (stretch)
- [x] Confirm history start: 1993-01 (inflation-targeting era)
- [x] Define success metrics per target type (accuracy, F1, log-loss, Brier, hit rate vs market)
- [x] Define held-out test window covering ≥1 hike and ≥1 cut cycle

## 2. Repo & environment setup

### GitHub & local repo

- [x] Create GitHub repo (public): rba-cash-rate-prediction
- [x] Add MIT or Apache 2.0 LICENSE
- [x] Add .gitignore (Python + data/raw/* + mlruns/* + .venv + .ipynb_checkpoints)
- [x] Clone locally and set up branch protection on main
- [x] Add CONTEXT.md to repo root (for Claude Code)
- [x] Add README.md skeleton with problem statement and quickstart

### Cookiecutter scaffold

- [x] Run cookiecutter-data-science (or ccds) and merge into repo
- [x] Adjust folder names to match CONTEXT.md spec
- [x] Create src/rba/{data,features,models,validation,viz,config} package skeleton
- [x] Create tests/ mirroring src/ structure
- [x] Create streamlit_app/ folder with placeholder main.py

### uv environment

- [x] Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)
- [x] Run uv init and confirm pyproject.toml created
- [x] Pin Python 3.11+ in pyproject.toml
- [x] Add core deps: pandas, numpy, scikit-learn, pyarrow, pyyaml
- [x] Add ML deps: xgboost, lightgbm, mord
- [x] Add NLP deps: sentence-transformers, transformers, torch
- [x] Add tracking: mlflow
- [x] Add dashboard: streamlit, plotly
- [x] Add dev deps: pytest, pytest-cov, ruff, mypy, jupyterlab
- [x] Run uv sync and commit uv.lock
- [x] Verify uv run pytest works on empty test

### Tooling & CI

- [x] Configure ruff (linting + formatting) in pyproject.toml
- [x] Configure mypy (basic strictness)
- [x] Add pre-commit hooks (ruff, mypy, pytest)
- [x] Add GitHub Actions: lint + tests on push
- [x] Set RANDOM_SEED = 12 in src/rba/config/__init__.py
- [x] Add MLflow tracking URI config (./mlruns)

## 3. Data collection & EDA

### RBA decisions (target)

- [x] Implement src/rba/data/sources/rba_f11.py — pull cash rate decision history
- [x] Validate: every decision has date, prior rate, new rate, change
- [x] Cross-check against RBA media releases archive
- [x] Hash raw download to data/raw/rba_f11/_metadata.json

### Macro features

- [x] Implement ABS CPI source (headline, trimmed mean, weighted median)
- [x] Implement ABS labour force source (unemployment, underemployment, participation)
- [x] Implement ABS wage price index source
- [x] Implement ABS GDP source
- [~] ~~Implement ABS retail trade source~~ — **dropped 2026-05-21**. ABS discontinued Cat. 8501.0 (Retail Trade, Australia) after the Jun 2025 reference month. Coverage would be 1982-04 → 2025-06 only, which means every prediction the model makes after Jul 2025 would have a frozen, increasingly stale retail feature — useless for forward decisions. The ABS-designated successor (Monthly Household Spending Indicator, dataflow `HSI_M`) covers a different concept (consumer spending across all channels, not retail-store turnover) and is out of scope for v1. See CONTEXT.md "Excluded series" for details.
- [x] Implement RBA D1/D2 (credit aggregates) source
- [x] Implement RBA E2 (household balance-sheet ratios) source — see section below for module-level checklist.
- [x] Implement housing source (CoreLogic or ABS dwelling prices, building approvals) — shipped two modules: `abs_building_approvals` (NSA from ABS BA_GCCSA + SA/trend from RBA H3) and `abs_total_value_dwellings` (TVD value/count/mean + 8-capital medians + transfer counts + legacy RPPI 8-cap index + a derived spliced index with QoQ-boundary validation). Each ships a scraped release calendar with the `source` provenance enum (`abs_page`/`archive_org`/`inferred`). CoreLogic was out of scope (licensed/gated); the user-directed splice replaces RPPI past 2021-Q4 with TVD-mean growth and documents the unstratified-median caveat in the source docstring.
- [ ] Implement business/consumer sentiment (NAB, Westpac-MI)
- [ ] Implement RBA index of commodity prices source
- [ ] For each: record observation_date AND publication_date

### RBA E2 (household balance-sheet ratios)

- [x] Confirm the E2 spreadsheet structure — CSV-only at `e2-data.csv` (the `e02hist.xls` URL from the brief 404s; `e02hist.xlsx` exists with extra `Notes` sheet but the data matches the CSV exactly).
- [x] Identify the exact series IDs — `BHFDDIT` (household debt-to-income), `BHFDDIH` (housing debt-to-income), `BHFDDIO` (owner-occupier housing debt-to-income). The brief's `BHFIDR`/`BHFHDR` codes are not in the current E2 table.
- [x] Drop the interest-paid-to-income ratio — RBA removed the series from E2 in Feb 2023. The E13 replacement (`LPHTICRI`) is housing-only and only covers 2009-Q1 onward; out of scope for v1.
- [x] Add owner-occupier housing debt-to-income (`BHFDDIO`) — RBA-calculated, pre-spliced across breaks. Useful third leverage ratio.
- [x] Drop the `series_break_indicator` column — no published E2 break list to populate it; `BHFDDIO` is internally pre-spliced; no in-scope methodology breaks affect the three ratios pulled.
- [x] Sample ~10 historical release dates — 20 unique (reference_quarter, publication_date) pairs harvested from the Wayback Machine spanning 2014-Q4 → 2025-Q1, plus the live CSV header for 2025-Q4. Releases vary across Fri/Tue/Sat with Christmas pull-backs and Easter push-forwards. No clean algorithmic rule fits.
- [x] Implement `rba_e_release_calendar.py` — scrapes Wayback CDX for `e2-data.csv` snapshots + reads the live CSV's `Publication date` header. Source enum `{rba_page, archive_org, inferred}` matches existing scraped calendars. Coverage is partial: ~13% of E2 observations have scraped publication dates; the rest fall back to a `+ 95` day flat offset (slightly longer than the worst observed lag of ~92 days) — documented limitation since the RBA does not publish a per-release archive.
- [x] Implement `rba_e2_household_ratios.py` source module — pulls the three series from `e2-data.csv`, caches verbatim CSV bytes to `data/raw/rba_e/`, writes `_metadata.json` provenance, attaches publication dates via the calendar with flat-offset fallback, exposes a long-format `fetch()` and an `__main__` block that materialises the wide CSV at `data/external/rba_e2_household_ratios.csv` with columns `reference_quarter_end / reference_label / household_debt_to_income / housing_debt_to_income / owner_occupier_housing_debt_to_income / release_date / source`.
- [x] Write tests mirroring `rba_d` patterns — synthetic-CSV `_parse` tests (extract / drop-null / missing-RBA-ID / empty-series), `_to_quarter_end` parametric, `_attach_publication_dates` scraped + flat-offset + NaT-pre-1993, no-future-leakage invariant (`publication_date > observation_date + 7 days`), `_to_wide` schema + source population, registry assertions, hand-verified known-value sanity for 2025-Q4 / 2014-Q4 / 1995-Q4. Calendar tests cover modern + legacy CSV parsing, latin-1 fallback, materialised-CSV loader, override precedence/extension, and rba_page-precedence-over-archive_org.
- [x] Update `CONTEXT.md` Invariant #1 with the new calendar entry (`rba_e_release_calendar.py`, source enum) and document the partial-coverage limitation.
- [x] Update `README.md` data-vintage section to include RBA E2 alongside the existing macro sources.

### Market data

- [ ] Implement ASX 30-day futures source (market-implied cash rate)
- [ ] Implement Australian Government Bond yields (2y, 5y, 10y)
- [ ] Implement AUD/USD and AUD TWI source
- [ ] Implement ASX 200 source (incl. financials sub-index)
- [ ] Implement BBSW rates source

### Global signals

- [ ] Implement FRED source for US CPI, Fed funds rate, US 10y, DXY, VIX
- [ ] Implement iron ore / copper / oil price sources
- [ ] Implement RBNZ OCR source (cross-check Pacific peer)

### Text data

- [ ] Implement RBA media release scraper (post-decision statements)
- [ ] Implement RBA minutes scraper (released 2 weeks after meeting)
- [ ] Implement RBA Statement on Monetary Policy scraper
- [ ] Implement Governor speech scraper
- [ ] Store full text + metadata (date, type, governor, URL)

### Source orchestration

- [ ] Create `src/rba/data/refresh.py` skeleton (CLI entry point, per-source error handling, logging)
- [ ] Register `rba.data.sources.rba_f11.fetch` in refresh.py
- [ ] Register `rba.data.sources.abs_cpi.fetch` in refresh.py

### Data inventory & catalog

Single generated source of truth: what data we have, where it lives, when it was last refreshed. Built from the three existing canonical sources (per-source `SERIES` registries in code, per-source `_metadata.json`, per-source release-calendar CSVs) — no hand-maintained spreadsheets.

- [ ] Create `src/rba/data/inventory.py` — import every `SERIES` tuple from `src/rba/data/sources/*.py` and every release-calendar module
- [ ] Walk `data/raw/<source>/_metadata.json` files; join provenance (URL, SHA-256, downloaded_at_utc, observations) onto each series
- [ ] Compute last `observation_date` per series from the latest raw snapshot
- [ ] Materialise `DATA.md` at repo root — Markdown table, checked in, diffable in PRs, renders on GitHub. Columns: `series_id | source_module | dataflow/table | source_url | raw_dir | snapshot_filename | observations | last_observation_date | release_calendar_module`
- [ ] Materialise `data/external/inventory.parquet` — machine-readable counterpart for downstream tooling
- [ ] Add `--xlsx` CLI flag (`uv add openpyxl`) that emits an optional spreadsheet copy for teammates who prefer Excel ergonomics — generated artifact, not the source of truth
- [ ] Add CI test (`tests/data/test_inventory.py`) asserting every registered `SERIES` entry has a matching `_metadata.json` row — catches "added a series but forgot to refresh raw"
- [ ] Document the inventory contract in CONTEXT.md (single command to regenerate, what gets checked in vs ignored)

### EDA notebooks

- [ ] 0.1-eda-target-distribution.ipynb (class balance, decision frequency)
- [ ] 0.2-eda-macro-series.ipynb (trends, missingness, regime breaks)
- [ ] 0.3-eda-market-vs-actual.ipynb (futures-implied vs realised)
- [ ] 0.4-eda-text-corpus.ipynb (length, vocab, governor differences)
- [ ] 0.5-eda-regime-changes.ipynb (governor eras, COVID, cadence change)

## 4. Data preprocessing

- [ ] Build src/rba/data/align.py — point-in-time join to meeting frame
- [ ] For each source: assert publication_date ≤ meeting_date in join
- [ ] Write tests/data/test_no_leakage.py with synthetic future-data injection
- [ ] Implement gap-since-last-meeting feature
- [ ] Add governor / GFC / COVID / forward-guidance / cadence-change regime dummies
- [ ] Build _is_missing indicator columns for every imputed feature
- [ ] Document imputation choice per series in src/rba/config/features.yaml
- [ ] Save processed master frame to data/processed/master.parquet
- [ ] Hash processed frame and log to MLflow as artifact

## 5. Feature engineering

### Numerical features

- [ ] Implement src/rba/features/lags.py with configurable lag horizons
- [ ] Implement src/rba/features/rolling.py (mean, std, z-score, min, max, EWMA)
- [ ] Implement src/rba/features/changes.py (Δ, %Δ, YoY)
- [ ] Implement src/rba/features/surprises.py where consensus available
- [ ] Verify all rolling windows are past-only (test with center=False)
- [ ] Add unit tests for shape, NaN handling at series start, and leakage

### Text features

- [ ] Implement Loughran-McDonald lexicon scoring
- [ ] Implement custom hawkish/dovish dictionary (collect, label ~50 terms)
- [ ] Implement off-the-shelf sentence embeddings (all-MiniLM-L6-v2 baseline)
- [ ] Optional: fine-tune sentiment model on labelled RBA/FOMC data
- [ ] Cache embeddings to data/processed/embeddings/ keyed by document hash

### Feature pipeline

- [ ] Build src/rba/features/build.py — orchestrates all feature builders from YAML config
- [ ] Add feature versioning (hash of config + code) for reproducibility
- [ ] Generate feature importance report (mutual info, correlation with target)

## 6. Baseline models

- [ ] Implement Model protocol in src/rba/models/base.py
- [ ] Implement majority-class baseline
- [ ] Implement persistence baseline (predict last decision)
- [ ] Implement Taylor rule baseline (with literature default coefficients, then fitted)
- [ ] Implement market-implied baseline (from ASX 30-day futures)
- [ ] All baselines pass the same evaluation harness as ML models
- [ ] Document baseline performance as floor for all subsequent work

## 7. Model development

### Validation harness

- [ ] Implement WalkForwardSplit in src/rba/validation/cv.py
- [ ] Implement metrics module: accuracy, balanced accuracy, macro-F1, log-loss, Brier, confusion matrix
- [ ] Implement calibration plots (reliability diagrams)
- [ ] Implement hit-rate-vs-market metric
- [ ] Wrap evaluation in a function that logs everything to MLflow

### Models

- [ ] Logistic regression (baseline ML model)
- [ ] L1 / L2 regularised logistic regression
- [ ] Random forest
- [ ] XGBoost
- [ ] LightGBM
- [ ] SVM (linear + RBF kernels)
- [ ] Ordinal logistic regression (mord)
- [ ] MLP (sklearn or small PyTorch)
- [ ] xRFM
- [ ] Each model registered in src/rba/config/models.yaml with default + search space

### Tuning & imbalance

- [ ] Use class_weight='balanced' as default for classifiers
- [ ] Run threshold-tuning experiments per classifier
- [ ] Run SMOTE experiment (compare against class-weight)
- [ ] Hyperparameter search (Optuna or sklearn) within walk-forward CV only
- [ ] Never tune on the held-out test set

## 8. Model evaluation

- [ ] Run all models against held-out test window
- [ ] Compare against all 4 baselines with paired metrics table
- [ ] Generate confusion matrix per model per target
- [ ] Generate calibration plot per probabilistic model
- [ ] Compute hit rate vs market-implied baseline
- [ ] Conduct error analysis: when does the best model fail? (regime, cycle phase, surprise meetings)
- [ ] Build ensemble (stacking or voting) and compare against individual best
- [ ] Write reports/results.md summarising findings
- [ ] Save best model artifacts to MLflow registry

## 9. Deployment & monitoring

### Streamlit dashboard

- [ ] Design dashboard layout (next-meeting prediction, history, feature importances)
- [ ] Implement streamlit_app/main.py with model loading from MLflow
- [ ] Show probability distribution across hike/hold/cut
- [ ] Show comparison vs market-implied probabilities
- [ ] Show top driving features for current prediction
- [ ] Add historical accuracy panel
- [ ] Test locally: uv run streamlit run streamlit_app/main.py

### Before-meeting predictions

- [ ] Schedule data refresh script (cron or GitHub Actions) before each meeting
- [ ] Implement predict-next-meeting CLI command
- [ ] Log every pre-meeting prediction with timestamp
- [ ] After each meeting: log actual outcome and update accuracy tracker

### Hosting & polish

- [ ] Deploy dashboard to Streamlit Community Cloud (free)
- [ ] Add custom domain (optional)
- [ ] Write final README with screenshots, methodology, results, limitations
- [ ] Tag v1.0.0 release on GitHub
- [ ] Optional: write blog-style writeup (Distill format)
- [ ] Optional: submit to Tomasz Woźniak's RBA forecast survey for credible benchmark
