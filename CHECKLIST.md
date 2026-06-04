# RBA cash rate prediction — project checklist

Progress: 55 / 152 (36%)

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
- [x] Implement RBA E2 (household balance-sheet ratios) source — `rba_e2_household_ratios` pulls `BHFDDIT`/`BHFDDIH`/`BHFDDIO` (total / housing / owner-occupier household debt-to-income) from `e2-data.csv`; caches verbatim CSV bytes + `_metadata.json` provenance under `data/raw/rba_e/` and materialises a wide CSV at `data/external/rba_e2_household_ratios.csv`. Ships scraped `rba_e_release_calendar.py` (Wayback CDX + live CSV `Publication date` header, `{rba_page,archive_org,inferred}` source enum) covering ~13% of observations; the rest fall back to a +95-day flat offset (worst observed lag ~92 days — no RBA per-release archive exists). Interest-paid-to-income ratio dropped (removed from E2 Feb 2023; E13 replacement housing-only, post-2009, out of scope for v1). `series_break_indicator` column dropped (no published break list; `BHFDDIO` is pre-spliced by RBA).
- [x] Implement housing source (CoreLogic or ABS dwelling prices, building approvals) — shipped two modules: `abs_building_approvals` (NSA from ABS BA_GCCSA + SA/trend from RBA H3) and `abs_total_value_dwellings` (TVD value/count/mean + 8-capital medians + transfer counts + legacy RPPI 8-cap index + a derived spliced index with QoQ-boundary validation). Each ships a scraped release calendar with the `source` provenance enum (`abs_page`/`archive_org`/`inferred`). CoreLogic was out of scope (licensed/gated); the user-directed splice replaces RPPI past 2021-Q4 with TVD-mean growth and documents the unstratified-median caveat in the source docstring.
- [x] Implement business/consumer sentiment (NAB, Westpac-MI)
- [x] Implement RBA index of commodity prices source
- [x] For each: record observation_date AND publication_date

### Market data

- [x] Implement ASX 30-day futures source (market-implied cash rate) — `asx_ib_futures` pulls the daily implied curve from the `MattCowgill/cash-rate-scraper` GitHub repo (MIT, daily-refreshed `combined_data/all_data.Rds` scraped from the ASX MarkitDigital JSON endpoint the Rate Tracker page consumes — no free direct-from-ASX historical feed exists). Long-format `[observation_date, publication_date, series_id, value]` where `series_id` = `ib_YYYY_MM` (contract expiry month) and `value` = `100 - upstream_cash_rate` (raw settlement price). Publication-date rule is trivial (`publication_date == observation_date`, ~6pm AEST same-day EOD), so no separate release calendar. Coverage 2022-04-21 → present with a documented gap 2022-07-01 → 2022-07-20 (ASX site change); pre-2022 history unavailable from any free source — market-implied baseline evaluable on post-2022 window only. Companion `derive_meeting_implied(long_df, meetings_df, lookback_business_days=1)` joins the curve to the F11 meeting frame and returns the naïve implied rate for the meeting-month contract as of T-1 business day. New dep: `pyreadr`.
- [x] Implement Australian Government Bond yields (2y, 5y, 10y)
- [x] Implement AUD/USD, TWI, JPY, EUR, GBP, CNY, NZD exchange rates
- [x] Implement ASX 200 source (incl. financials sub-index)
- [x] Implement BBSW rates source

### Global signals

- [x] Implement FRED source for US CPI, Fed funds rate, US 10y, DXY, VIX
- [x] Implement iron ore / copper / oil price sources

### Text data

- [x] Implement RBA media release scraper (post-decision statements) — `rba_media_releases` drives the crawl from `rba_f11.statement_url` (no yearly-archive title regex), pulls the post-decision "Statement by ..." HTML per Board meeting on/after 1993-01-01, and emits a decision-keyed wide frame `[decision_date, publication_date, governor, title, url, sha256, raw_html_filename, body_text, paragraphs]` materialised to `data/external/rba_media_releases.parquet`. **No release calendar** — publication-date rule is trivial (`publication_date == decision_date`; announce 14:30 AEST, statement live the same moment). Cross-check raises if any F11 statement_url has no matching media-release row, if any media row has `publication_date != decision_date`, or if any media row has no matching F11 row (`_KNOWN_MISSING_DECISIONS` allowlist starts empty). Three title eras handled (Fraser/Macfarlane/early-Stevens `"Statement by the Governor, Mr <Surname>: ..."`; ~Apr-2010 → Jan-2024 Stevens/Lowe/early-Bullock `"Statement by <Name>, Governor: Monetary Policy Decision - <Month YYYY>"`; Feb-2024+ post-governance-reform `"Statement by the Reserve Bank Board: ..."`); governor attribution is best-effort regex on the title. Body extraction prefers the modern `<div class="rss-mr-content">` wrapper, falls back to ancestor-class-filtered `<p>` under `<div id="content">` for pre-2018 layout, normalises NBSP / tabs / multi-newline whitespace. Network gotcha documented: the RBA WAF rejects custom `User-Agent` headers — the default `Python-urllib` UA is required. **Decision-keyed wide + parquet is the contract the three remaining text scrapers (minutes, SoMP, Governor speeches) will inherit** — see CONTEXT.md "Text-data schema convention".
- [x] Implement RBA minutes scraper (released 2 weeks after meeting) — `rba_minutes` drives the crawl from `rba_f11.minutes_url` (F11-driven, no archive scrape), pulls each Board-meeting minutes document on/after 2008-02-05 and emits the inherited decision-keyed wide frame `[decision_date, publication_date, title, url, sha256, raw_html_filename, body_text, paragraphs]` (drops the `governor` column media releases had — minutes are a Board document) materialised to `data/external/rba_minutes.parquet` (195 meetings). **Algorithmic release calendar** `rba_minutes_release_calendar.py` (rule: second Tuesday strictly after the meeting → `+14d` for Tuesday meetings, `+12d` for the COVID Thursday meeting 2020-03-19; `_OVERRIDES` empty): minutes pages have no `itemprop="datePublished"` and the only `<meta name="dc.date">` marker is ~10% CMS-artifact garbage, so `dc.date` is scraped only as a warn-only soft cross-check. **Publication history & floor (2008-02-05, NOT 1993):** the RBA published *no* board minutes before December 2007 — pre-2007 monetary-policy communication ran through the post-decision media-release statements (rate-change meetings only until Apr 2010; see `rba_media_releases`), the quarterly Statement on Monetary Policy, and Governor speeches / parliamentary testimony. The minutes regime began Dec 2007, and the initial release **backfilled** the prior ~14 months in two batches (`dc.date` 2007-12-05 and 2007-12-18) reaching back only to the **October 2006** meeting — so `minutes_url` is `NaN` for every meeting before 2006-10-04 (0 rows) and the 14 backfilled 2006-10 → 2008-01 meetings, though present in F11, carry a real publication date of ~2007-12 (not meeting+14d) and are therefore excluded by the floor rather than mis-dated by the algorithmic rule. `<h2>` section headings captured as standalone interleaved entries in `paragraphs`; trailing `Related Information`/`Related Content` nav headings dropped; same RBA-WAF default-UA gotcha as media releases. Cross-check raises on F11 mismatch / extras / `publication_date <= decision_date`; `minutes_url.notna()` filter naturally excludes the most-recent-meeting "no minutes yet" gap (never a miss). See CONTEXT.md `rba_minutes.py` source-module note + the calendar bullet in Invariant #1.
- [x] Implement RBA Statement on Monetary Policy scraper — `rba_somp` drives the crawl from the **SoMP archive index** (`/publications/smp/`, NOT F11 — there is no `f11.somp_url`), enumerates each `<YYYY>/<mon>` quarterly issue (Feb/May/Aug/Nov) on/after 2006-02-01, captures the **Overview page only** (highest-signal cross-era summary), and emits the inherited decision-keyed wide frame `[decision_date, publication_date, title, url, sha256, raw_html_filename, body_text, paragraphs]` (drops `governor` like minutes — SoMP is a Bank document) materialised to `data/external/rba_somp.parquet` (82 issues, 2006-02-08 → 2026-05-05). **Quarterly + sparse**: joins the meeting frame ~4×/year, absent for most meetings; each issue is mapped to the F11 decision in its **same calendar month** (`decision_date` == F11 publication_date) and the cross-check validates that mapping (raises on no-output-row / extra-not-in-index / phantom decision absent from F11 / two issues colliding on one decision_date; `_KNOWN_MISSING_DECISIONS` empty). **No release calendar — publication date is scraped**: SoMP pages have no `itemprop="datePublished"`, but `<meta name="dc.date">` is reliable across every era (unlike the minutes' ~10%-artifact dc.date), so it is the canonical `publication_date`. **Mixed publication-vs-decision sign by era** (so the cross-check asserts *no* fixed sign, unlike media-releases `==` / minutes `>`): same-day from 2024 (8-meeting cadence), `+3d`-typical / `+24d`-November in ~2008–2023, and *before* the decision pre-2008 (Feb 2006 SoMP 2006-02-07 < decision 2006-02-08); genuine gaps span −14d…+24d, so the warn-only soft-check fires only at `|gap| > 31d` (cross-month dc.date artifact). **Floor 2006-02-01 (HTML era), NOT 1993**: the quarterly SoMP began 1997 but 1997–2005 are PDF-only (per-month HTML 403/404) — excluded from v1 as a documented coverage limitation (PDF path could extend back later). Overview filename has three era forms (`overview.html` / `00-overview.html` / `intro.html`) resolved by trying each; body walks `<h2>`/`<p>` under `<div id="content">` with the minutes ancestor-exclusion set plus `nav-publication-contents`/`publication-related-links`, capturing `<h2>` key-message headings as standalone interleaved entries and dropping trailing nav headings. Same RBA-WAF default-UA gotcha. See CONTEXT.md `rba_somp.py` source-module note + the Text-data schema convention paragraph.
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
