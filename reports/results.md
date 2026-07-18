# RBA cash-rate prediction — §8 model evaluation (held-out test)

_Generated 2026-07-18T00:41:22+00:00. Held-out test window: 39 meetings on/after `TEST_WINDOW_START` (cut=3 / hold=20 / hike=16)._

## Method

- **Split**: `dev_test_split` at `TEST_WINDOW_START` — dev is every meeting before the window; the test window is scored exactly once.
- **Test-prediction shape**: expanding walk-forward *through* the test window — each test meeting is predicted by a model trained on all strictly-earlier meetings (dev + earlier test meetings). Hyper-parameters and decision thresholds are frozen from the dev-only tuning campaign; only the training data expands. No test row informs its own prediction.
- **Tuning** (dev only): Optuna hyper-search over each model's `models.yaml` `search_space` scored by inner walk-forward CV, then per-class decision-threshold weights fit on dev out-of-fold probabilities. Persisted to `reports/tuned/<model>.json`.
- **Why balanced accuracy, not raw accuracy**: the test window is 51% hold (much more balanced than the ~77% full-history prior — it deliberately spans a full hike→hold→cut cycle), so the majority-class floor here is only ~51%. Balanced accuracy / macro-F1 / log-loss are the honest metrics; raw accuracy is reported but not ranked on.

## Headline — paired metrics table

Ranked by **balanced accuracy (argmax)** — the robust primary metric. `bal_acc_tuned` applies each model's dev-tuned decision thresholds; on this N≈39 window it helps some models and hurts others, so it is reported but not ranked on. `market_lift` is the model's accuracy minus the market baseline's on the covered meetings.

| model | kind | tuned | n | accuracy | balanced_accuracy | macro_f1 | log_loss | brier_score | bal_acc_tuned | market_lift |
|---|---|---|---|---|---|---|---|---|---|---|
| market_implied | baseline | False | 38 | 0.7895 | 0.6561 | 0.6777 | — | — | — | — |
| lightgbm_classifier | model | True | 39 | 0.7436 | 0.6236 | 0.6687 | 1.2795 | 0.4779 | 0.6111 | -0.0526 |
| ordinal_logistic | model | True | 39 | 0.7179 | 0.6069 | 0.612 | 3.1534 | 0.5671 | 0.6028 | -0.0526 |
| svm_classifier | model | True | 39 | 0.6923 | 0.5903 | 0.5937 | 0.736 | 0.4167 | 0.4653 | -0.0789 |
| logistic_regression | model | True | 39 | 0.641 | 0.5569 | 0.5569 | 1.7794 | 0.6159 | 0.5903 | -0.1316 |
| voting_ensemble | ensemble | False | 39 | 0.641 | 0.5528 | 0.5706 | 0.9721 | 0.5037 | — | -0.1316 |
| xgboost_classifier | model | True | 39 | 0.6154 | 0.5444 | 0.5379 | 0.8561 | 0.5307 | 0.625 | -0.1842 |
| stacking_ensemble | ensemble | False | 39 | 0.5641 | 0.5153 | 0.475 | 1.7867 | 0.5729 | — | -0.2105 |
| mlp_classifier | model | True | 39 | 0.5641 | 0.4861 | 0.5276 | 2.5054 | 0.7205 | 0.4778 | -0.2368 |
| persistence | baseline | False | 39 | 0.641 | 0.4667 | 0.4667 | 12.9387 | 0.7179 | — | -0.1316 |
| random_forest_classifier | model | True | 39 | 0.6154 | 0.4458 | 0.4271 | 0.8628 | 0.5386 | 0.3222 | -0.1842 |
| majority_class | baseline | False | 39 | 0.5128 | 0.3333 | 0.226 | 1.2404 | 0.7361 | — | -0.2895 |


**Key finding — the market-implied baseline (`market_implied`, balanced accuracy 0.6561) is the single best predictor on the test window; no learned model beats it (every `market_lift` is negative).** On a market as liquid and rate-tracked as the RBA cash rate, the 30-day futures price already impounds the macro signal the models are trying to learn — the honest result for a portfolio project, not a disappointment.


**Best learned model: `lightgbm_classifier`** (balanced accuracy 0.624, tuned-threshold 0.611).

## Hit rate vs market-implied baseline

Paired on the meetings the ASX 30-day-futures curve covers (from 2022-04). `lift` = model − market accuracy on the same meetings.

| model | n_compared | model_hit_rate | market_hit_rate | lift | only_model_correct | only_market_correct |
|---|---|---|---|---|---|---|
| lightgbm_classifier | 38 | 0.7368 | 0.7895 | -0.0526 | 0.1579 | 0.2105 |
| ordinal_logistic | 38 | 0.7368 | 0.7895 | -0.0526 | 0.1053 | 0.1579 |
| svm_classifier | 38 | 0.7105 | 0.7895 | -0.0789 | 0.1316 | 0.2105 |
| persistence | 38 | 0.6579 | 0.7895 | -0.1316 | 0.0789 | 0.2105 |
| voting_ensemble | 38 | 0.6579 | 0.7895 | -0.1316 | 0.1316 | 0.2632 |
| logistic_regression | 38 | 0.6579 | 0.7895 | -0.1316 | 0.1316 | 0.2632 |
| random_forest_classifier | 38 | 0.6053 | 0.7895 | -0.1842 | 0.1316 | 0.3158 |
| xgboost_classifier | 38 | 0.6053 | 0.7895 | -0.1842 | 0.1579 | 0.3421 |
| stacking_ensemble | 38 | 0.5789 | 0.7895 | -0.2105 | 0.1579 | 0.3684 |
| mlp_classifier | 38 | 0.5526 | 0.7895 | -0.2368 | 0.1053 | 0.3421 |
| majority_class | 38 | 0.5 | 0.7895 | -0.2895 | 0.0263 | 0.3158 |

## Confusion matrices


### `lightgbm_classifier` (argmax)

| true \ pred | cut | hike | hold |
|---|---|---|---|
| **cut** | 1 | 0 | 2 |
| **hike** | 0 | 11 | 5 |
| **hold** | 0 | 3 | 17 |


### `majority_class` (argmax)

| true \ pred | cut | hike | hold |
|---|---|---|---|
| **cut** | 0 | 0 | 3 |
| **hike** | 0 | 0 | 16 |
| **hold** | 0 | 0 | 20 |


### `market_implied` (argmax)

| true \ pred | cut | hike | hold |
|---|---|---|---|
| **cut** | 1 | 0 | 2 |
| **hike** | 0 | 11 | 5 |
| **hold** | 1 | 0 | 18 |


_Full per-model confusion matrices are in each run's `manifest.json` under `reports/runs/`._

## Calibration (reliability)

Per-class expected calibration error (ECE, lower = better-calibrated probabilities):

| model | ece_cut | ece_hike | ece_hold |
|---|---|---|---|
| logistic_regression | 0.0969 | 0.2456 | 0.3494 |
| random_forest_classifier | 0.0992 | 0.0526 | 0.1139 |
| xgboost_classifier | 0.1245 | 0.1872 | 0.2181 |
| lightgbm_classifier | 0.0764 | 0.201 | 0.2699 |
| svm_classifier | 0.0514 | 0.1721 | 0.2487 |
| mlp_classifier | 0.0494 | 0.336 | 0.3754 |
| ordinal_logistic | 0.1004 | 0.1862 | 0.2865 |
| voting_ensemble | 0.0889 | 0.2301 | 0.2798 |
| stacking_ensemble | 0.0736 | 0.2004 | 0.283 |
| majority_class | 0.0353 | 0.3014 | 0.2661 |
| persistence | 0.1538 | 0.2051 | 0.359 |

Reliability diagrams (one panel per class):

- `logistic_regression` → [`figures/calibration_logistic_regression.html`](figures/calibration_logistic_regression.html)
- `random_forest_classifier` → [`figures/calibration_random_forest_classifier.html`](figures/calibration_random_forest_classifier.html)
- `xgboost_classifier` → [`figures/calibration_xgboost_classifier.html`](figures/calibration_xgboost_classifier.html)
- `lightgbm_classifier` → [`figures/calibration_lightgbm_classifier.html`](figures/calibration_lightgbm_classifier.html)
- `svm_classifier` → [`figures/calibration_svm_classifier.html`](figures/calibration_svm_classifier.html)
- `mlp_classifier` → [`figures/calibration_mlp_classifier.html`](figures/calibration_mlp_classifier.html)
- `ordinal_logistic` → [`figures/calibration_ordinal_logistic.html`](figures/calibration_ordinal_logistic.html)
- `voting_ensemble` → [`figures/calibration_voting_ensemble.html`](figures/calibration_voting_ensemble.html)
- `stacking_ensemble` → [`figures/calibration_stacking_ensemble.html`](figures/calibration_stacking_ensemble.html)
- `majority_class` → [`figures/calibration_majority_class.html`](figures/calibration_majority_class.html)
- `persistence` → [`figures/calibration_persistence.html`](figures/calibration_persistence.html)

## Ensemble vs best individual

Best individual model: `lightgbm_classifier` (balanced accuracy 0.624).

| ensemble | balanced_accuracy | macro_f1 | vs_best_individual |
|---|---|---|---|
| voting_ensemble | 0.5528 | 0.5706 | -0.0708 |
| stacking_ensemble | 0.5153 | 0.475 | -0.1083 |

## Error analysis — `lightgbm_classifier`

Overall: accuracy 0.744, balanced accuracy 0.624, 10/39 meetings missed.


**By actual decision (recall):**

| true_class | support | n_correct | recall |
|---|---|---|---|
| cut | 3 | 1 | 0.3333 |
| hike | 16 | 11 | 0.6875 |
| hold | 20 | 17 | 0.85 |

**By regime (accuracy on vs off the flag):**

| regime | n_active | acc_active | n_inactive | acc_inactive |
|---|---|---|---|---|
| regime_gov_lowe | 16 | 0.625 | 23 | 0.8261 |
| regime_gov_bullock | 23 | 0.8261 | 16 | 0.625 |
| regime_post2024_cadence | 20 | 0.85 | 19 | 0.6316 |

**By cycle phase:**

| phase | n | accuracy |
|---|---|---|
| easing | 7 | 0.8571 |
| on-hold | 13 | 0.5385 |
| tightening | 19 | 0.8421 |

**On market-surprise meetings** (where the market-implied call missed):

| bucket | n | model_accuracy |
|---|---|---|
| market_expected | 30 | 0.7333 |
| market_surprise | 8 | 0.75 |

**Every missed meeting:**

| meeting_date | y_true | y_pred | cycle_phase | active_regimes |
|---|---|---|---|---|
| 2022-05-03 | hike | hold | on-hold | regime_gov_lowe |
| 2022-06-07 | hike | hold | on-hold | regime_gov_lowe |
| 2022-07-05 | hike | hold | on-hold | regime_gov_lowe |
| 2023-04-04 | hold | hike | tightening | regime_gov_lowe |
| 2023-07-04 | hold | hike | tightening | regime_gov_lowe |
| 2023-08-01 | hold | hike | tightening | regime_gov_lowe |
| 2023-11-07 | hike | hold | on-hold | regime_gov_bullock |
| 2025-02-18 | cut | hold | on-hold | regime_gov_bullock, regime_post2024_cadence |
| 2025-08-12 | cut | hold | easing | regime_gov_bullock, regime_post2024_cadence |
| 2026-02-03 | hike | hold | on-hold | regime_gov_bullock, regime_post2024_cadence |

## Taylor-rule baseline (regression / directional footing)

The Taylor rule predicts a rate *level*, not a class, so it is scored on level-regression footing (predict `new_rate_pct`) rather than forced into the class table. Inputs are **proxies** (see Limitations).


- Level RMSE: 3.371 pp · MAE: 2.316 pp · R²: -10.459

- Derived hike/hold/cut accuracy vs the actual decision: 0.462 (direction calls on test: {'hike': 32, 'hold': 5, 'cut': 2}). The proxy Taylor rule over-prescribes hikes on this window, so its directional signal is weak.

## Limitations & honest caveats

- **Test window is small (N≈39) and idiosyncratic** — one hike→hold→cut cycle. Balanced-accuracy differences of a few points are within sampling noise; treat the ranking as indicative, not decisive.
- **Market baseline coverage** begins 2022-04, so ~1 test meeting has no futures quote and drops from the paired hit-rate comparison.
- **Taylor-rule inputs are proxies**: `cpi_headline_yoy` is a trimmed-mean-CPI year-over-year built from the meeting-aligned index, and `output_gap` is a negative unemployment gap (Okun-style) vs a rolling trend — not a true potential-output gap. The Taylor numbers are directional, not authoritative.
- **MLflow is upstream-blocked** on this stack (mlflow caps `pandas<3`; the project runs pandas 3). Local artifacts under `reports/` are the durable record; the registry write degraded gracefully (`import_failed:ImportError`).
- **Vintage bias**: features use current-vintage macro data, not the exact real-time values the Board saw (see CONTEXT.md 'Vintage policy').

## Artifacts

_Paths are relative to `reports/` (where this file lives)._

- Paired metrics table (CSV): `holdout_metrics.csv`
- Per-run manifests / predictions / calibration: `runs/<run_id>/`
- Tuned configs: `tuned/<model>.json`
- Best-model artifact: `best_model/`
- Calibration figure `logistic_regression`: [`figures/calibration_logistic_regression.html`](figures/calibration_logistic_regression.html)
- Calibration figure `random_forest_classifier`: [`figures/calibration_random_forest_classifier.html`](figures/calibration_random_forest_classifier.html)
- Calibration figure `xgboost_classifier`: [`figures/calibration_xgboost_classifier.html`](figures/calibration_xgboost_classifier.html)
- Calibration figure `lightgbm_classifier`: [`figures/calibration_lightgbm_classifier.html`](figures/calibration_lightgbm_classifier.html)
- Calibration figure `svm_classifier`: [`figures/calibration_svm_classifier.html`](figures/calibration_svm_classifier.html)
- Calibration figure `mlp_classifier`: [`figures/calibration_mlp_classifier.html`](figures/calibration_mlp_classifier.html)
- Calibration figure `ordinal_logistic`: [`figures/calibration_ordinal_logistic.html`](figures/calibration_ordinal_logistic.html)
- Calibration figure `voting_ensemble`: [`figures/calibration_voting_ensemble.html`](figures/calibration_voting_ensemble.html)
- Calibration figure `stacking_ensemble`: [`figures/calibration_stacking_ensemble.html`](figures/calibration_stacking_ensemble.html)
- Calibration figure `majority_class`: [`figures/calibration_majority_class.html`](figures/calibration_majority_class.html)
- Calibration figure `persistence`: [`figures/calibration_persistence.html`](figures/calibration_persistence.html)

