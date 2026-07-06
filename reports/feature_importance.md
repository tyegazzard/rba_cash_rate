# Feature importance report

Descriptive, **in-sample** importance of each engineered feature against the
RBA cash-rate decision target. These are whole-sample statistics for feature
triage — **not** a model-evaluation metric and not walk-forward performance
(see CONTEXT.md Invariants #2/#3, which govern *modelling*).

## Targets

- **Correlation** (Pearson / Spearman) vs `rate_change_bps` — the continuous
  Δrate target (`targets.yaml` `delta_regression`).
- **Mutual information** vs the 3-class direction target (`targets.yaml` `three_class`: sign of Δ → cut `-1` / hold `0` / hike `+1`),
  which captures non-linear dependence correlation misses.

## Provenance

- Features scored: **660**
- Target-defined meetings: **364**
- Feature version: `fcad83d9319dba272c0a81d0c2a5bdc262a70002182762b82ff352c8253a24eb`
- Git commit: `04683be06f5bfd2bcc625016f4cb6d8aac9f9220`
- Full table: `feature_importance.csv` (all features)

## Top 30 features by mutual information

| rank | feature | group | n_obs | pearson_r | spearman_r | mutual_info |
| ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| 1 | `credit_total_inc_fin_yoy_growth_sa_lag_3` | lag | 53 | 0.4979 | 0.6227 | 0.4743 |
| 2 | `credit_total_inc_fin_yoy_growth_sa_lag_2` | lag | 54 | 0.5326 | 0.6393 | 0.4337 |
| 3 | `credit_total_inc_fin_yoy_growth_sa_lag_1` | lag | 55 | 0.5252 | 0.6199 | 0.3925 |
| 4 | `credit_total_inc_fin_yoy_growth_sa` | level | 56 | 0.5171 | 0.6038 | 0.3615 |
| 5 | `credit_total_inc_fin_sa` | level | 68 | 0.1352 | 0.2142 | 0.2978 |
| 6 | `agb_yield_10y_diff_12` | change | 125 | 0.6297 | 0.5503 | 0.2933 |
| 7 | `credit_business_inc_fin_sa` | level | 68 | 0.1020 | 0.2132 | 0.2894 |
| 8 | `wpi_total_hourly_excl_bonuses_all_sectors_sa_lag_12` | lag | 295 | 0.0546 | 0.0160 | 0.2577 |
| 9 | `business_conditions_roll_6_max` | rolling | 311 | 0.2987 | 0.3678 | 0.2349 |
| 10 | `agb_yield_10y_roll_3_max` | rolling | 135 | 0.2838 | 0.2999 | 0.2280 |
| 11 | `xmj` | level | 139 | 0.3640 | 0.3489 | 0.2247 |
| 12 | `wpi_total_hourly_excl_bonuses_all_sectors_sa_lag_1` | lag | 306 | 0.0561 | 0.0284 | 0.2164 |
| 13 | `wpi_total_hourly_excl_bonuses_all_sectors_sa_lag_6` | lag | 301 | 0.0585 | 0.0273 | 0.2138 |
| 14 | `wpi_total_hourly_excl_bonuses_all_sectors_sa_lag_3` | lag | 304 | 0.0568 | 0.0274 | 0.2119 |
| 15 | `wpi_total_hourly_excl_bonuses_all_sectors_sa` | level | 307 | 0.0559 | 0.0281 | 0.2106 |
| 16 | `trimmed_mean_index_lag_12` | lag | 349 | 0.0297 | 0.0405 | 0.2103 |
| 17 | `tvd_dwelling_count_thousand` | level | 152 | 0.3403 | 0.3331 | 0.2086 |
| 18 | `trimmed_mean_index_roll_6_max` | rolling | 356 | 0.0351 | 0.0412 | 0.2080 |
| 19 | `trimmed_mean_index_roll_3_mean` | rolling | 359 | 0.0432 | 0.0510 | 0.2064 |
| 20 | `wpi_total_hourly_excl_bonuses_public_sa` | level | 307 | 0.0580 | 0.0281 | 0.2041 |
| 21 | `wpi_total_hourly_excl_bonuses_all_sectors_sa_lag_2` | lag | 305 | 0.0579 | 0.0295 | 0.2033 |
| 22 | `trimmed_mean_index_roll_6_min` | rolling | 356 | 0.0298 | 0.0407 | 0.2028 |
| 23 | `housing_median_price_attached_melbourne_aud_thousand` | level | 240 | 0.1302 | 0.0866 | 0.2026 |
| 24 | `weighted_median_index` | level | 361 | 0.0423 | 0.0512 | 0.2024 |
| 25 | `wpi_total_hourly_excl_bonuses_private_sa` | level | 307 | 0.0553 | 0.0280 | 0.2017 |
| 26 | `trimmed_mean_index_roll_3_ewma` | rolling | 361 | 0.0431 | 0.0507 | 0.2016 |
| 27 | `trimmed_mean_index_roll_12_max` | rolling | 350 | 0.0359 | 0.0416 | 0.2014 |
| 28 | `trimmed_mean_index_roll_3_max` | rolling | 359 | 0.0448 | 0.0514 | 0.2008 |
| 29 | `trimmed_mean_index_roll_6_mean` | rolling | 356 | 0.0320 | 0.0406 | 0.2007 |
| 30 | `trimmed_mean_index_roll_6_ewma` | rolling | 361 | 0.0419 | 0.0507 | 0.2000 |
