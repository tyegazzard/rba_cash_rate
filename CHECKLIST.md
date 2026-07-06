# RBA cash rate prediction — project checklist

Progress: 88 / 153 (58%)

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
- [~] ~~Implement ABS retail trade source~~ — dropped 2026-05-21 (ABS discontinued Cat. 8501.0 after Jun 2025; see CONTEXT.md "Excluded series")
- [x] Implement RBA D1/D2 (credit aggregates) source
- [x] Implement RBA E2 (household balance-sheet ratios) source
- [x] Implement housing source (ABS building approvals + total value of dwellings)
- [x] Implement business/consumer sentiment (NAB, Westpac-MI)
- [x] Implement RBA index of commodity prices source
- [x] For each: record observation_date AND publication_date

### Market data

- [x] Implement ASX 30-day futures source (market-implied cash rate)
- [x] Implement Australian Government Bond yields (2y, 5y, 10y)
- [x] Implement AUD/USD, TWI, JPY, EUR, GBP, CNY, NZD exchange rates
- [x] Implement ASX 200 source (incl. financials sub-index)
- [x] Implement BBSW rates source

### Global signals

- [x] Implement FRED source for US CPI, Fed funds rate, US 10y, DXY, VIX
- [x] Implement iron ore / copper / oil price sources

### Text data

- [x] Implement RBA media release scraper (post-decision statements)
- [x] Implement RBA minutes scraper (released 2 weeks after meeting)
- [x] Implement RBA Statement on Monetary Policy scraper
- [x] Implement Governor speech scraper
- [x] Store full text + metadata (date, type, governor, URL)

### Source orchestration

- [x] Create src/rba/data/refresh.py skeleton (CLI entry point, per-source error handling, logging)
- [x] Register rba.data.sources.rba_f11.fetch in refresh.py
- [x] Register rba.data.sources.abs_cpi.fetch in refresh.py

### Data inventory & catalog

Single generated source of truth: what data we have, where it lives, when it was last refreshed.

- [x] Create src/rba/data/inventory.py (import every SERIES tuple + release-calendar module)
- [x] Walk data/raw/<source>/_metadata.json and join provenance onto each series
- [x] Compute last observation_date per series from the latest raw snapshot
- [x] Materialise DATA.md at repo root (checked-in Markdown table)
- [x] Materialise data/external/inventory.parquet (machine-readable counterpart)
- [x] Add --xlsx CLI flag emitting an optional spreadsheet copy
- [x] Add CI test asserting every registered SERIES has a matching _metadata.json row
- [x] Document the inventory contract in CONTEXT.md

### EDA notebooks

- [ ] 0.1-eda-target-distribution.ipynb (class balance, decision frequency)
- [ ] 0.2-eda-macro-series.ipynb (trends, missingness, regime breaks)
- [ ] 0.3-eda-market-vs-actual.ipynb (futures-implied vs realised)
- [ ] 0.4-eda-text-corpus.ipynb (length, vocab, governor differences)
- [ ] 0.5-eda-regime-changes.ipynb (governor eras, COVID, cadence change)

## 4. Data preprocessing

- [x] Build src/rba/data/align.py — point-in-time join to meeting frame
- [x] For each source: assert publication_date ≤ meeting_date in join
- [x] Write tests/data/test_no_leakage.py with synthetic future-data injection
- [x] Implement gap-since-last-meeting feature
- [x] Add governor / GFC / COVID / forward-guidance / cadence-change regime dummies
- [x] Build _is_missing indicator columns for every imputed feature
- [x] Document imputation choice per series in src/rba/config/features.yaml
- [x] Save processed master frame to data/processed/master.parquet
- [x] Hash processed frame and log to MLflow as artifact (hash + local manifest done; MLflow upstream-blocked)

## 5. Feature engineering

### Numerical features

- [x] Implement src/rba/features/lags.py with configurable lag horizons
- [x] Implement src/rba/features/rolling.py (mean, std, z-score, min, max, EWMA)
- [x] Implement src/rba/features/changes.py (Δ, %Δ, YoY)
- [x] Implement src/rba/features/surprises.py where consensus available
- [x] Implement src/rba/features/target_lags.py (past values of the target as features)
- [x] Verify all rolling windows are past-only (test with center=False)
- [x] Add unit tests for shape, NaN handling at series start, and leakage

### Text features

- [x] Implement Loughran-McDonald lexicon scoring
- [~] Implement custom hawkish/dovish dictionary (collect, label ~50 terms) (deferred)
- [~] Implement off-the-shelf sentence embeddings (all-MiniLM-L6-v2 baseline) (deferred)
- [~] Optional: fine-tune sentiment model on labelled RBA/FOMC data (deferred)
- [~] Cache embeddings to data/processed/embeddings/ keyed by document hash (deferred)

### Feature pipeline

- [x] Build src/rba/features/build.py — orchestrates all feature builders from YAML config
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
