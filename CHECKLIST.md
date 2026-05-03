# RBA cash rate prediction — project checklist

Progress: 0 / 130 (0%)

## 1. Problem definition

- [ ] Lock target formulations: 3-class, multi-class with magnitudes, Δrate regression, level regression, ordinal
- [ ] Confirm forecast horizon: next meeting only
- [ ] Confirm scope: announcement (primary), surprise vs market (stretch)
- [ ] Confirm history start: 1993-01 (inflation-targeting era)
- [ ] Define success metrics per target type (accuracy, F1, log-loss, Brier, hit rate vs market)
- [ ] Define held-out test window covering ≥1 hike and ≥1 cut cycle
- [ ] Write project README sketch (problem, approach, success criteria)

## 2. Repo & environment setup

### GitHub & local repo

- [ ] Create GitHub repo (public): rba-cash-rate-prediction
- [ ] Add MIT or Apache 2.0 LICENSE
- [ ] Add .gitignore (Python + data/raw/* + mlruns/* + .venv + .ipynb_checkpoints)
- [ ] Clone locally and set up branch protection on main
- [ ] Add CONTEXT.md to repo root (for Claude Code)
- [ ] Add README.md skeleton with problem statement and quickstart

### Cookiecutter scaffold

- [ ] Run cookiecutter-data-science (or ccds) and merge into repo
- [ ] Adjust folder names to match CONTEXT.md spec
- [ ] Create src/rba/{data,features,models,validation,viz,config} package skeleton
- [ ] Create tests/ mirroring src/ structure
- [ ] Create streamlit_app/ folder with placeholder main.py

### uv environment

- [ ] Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)
- [ ] Run uv init and confirm pyproject.toml created
- [ ] Pin Python 3.11+ in pyproject.toml
- [ ] Add core deps: pandas, numpy, scikit-learn, pyarrow, pyyaml
- [ ] Add ML deps: xgboost, lightgbm, mord
- [ ] Add NLP deps: sentence-transformers, transformers, torch
- [ ] Add tracking: mlflow
- [ ] Add dashboard: streamlit, plotly
- [ ] Add dev deps: pytest, pytest-cov, ruff, mypy, jupyterlab
- [ ] Run uv sync and commit uv.lock
- [ ] Verify uv run pytest works on empty test

### Tooling & CI

- [ ] Configure ruff (linting + formatting) in pyproject.toml
- [ ] Configure mypy (basic strictness)
- [ ] Add pre-commit hooks (ruff, mypy, pytest)
- [ ] Add GitHub Actions: lint + tests on push
- [ ] Set RANDOM_SEED = 42 in src/rba/config/__init__.py
- [ ] Add MLflow tracking URI config (./mlruns)

## 3. Data collection & EDA

### RBA decisions (target)

- [ ] Implement src/rba/data/sources/rba_f11.py — pull cash rate decision history
- [ ] Validate: every decision has date, prior rate, new rate, change
- [ ] Cross-check against RBA media releases archive
- [ ] Hash raw download to data/raw/rba_f11/_metadata.json

### Macro features

- [ ] Implement ABS CPI source (headline, trimmed mean, weighted median)
- [ ] Implement ABS labour force source (unemployment, underemployment, participation)
- [ ] Implement ABS wage price index source
- [ ] Implement ABS GDP source
- [ ] Implement ABS retail trade source
- [ ] Implement RBA D1/D2 (credit aggregates) source
- [ ] Implement housing source (CoreLogic or ABS dwelling prices, building approvals)
- [ ] Implement business/consumer sentiment (NAB, Westpac-MI)
- [ ] Implement RBA index of commodity prices source
- [ ] For each: record observation_date AND publication_date

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
