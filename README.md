# RBA Cash Rate Prediction
A machine learning pipeline that predicts RBA target cash rate deicions before each meeting, benchmarked against the ASX futures market.

## Hero Visual / Screenshots TODO


## Problem Statement TODO


## Approach TODO


## Results TODO


## Limitations

### Data vintage / revisions
Macro features (ABS CPI, Labour Force, Wage Price Index, GDP, Building Approvals, Total Value of Dwellings; RBA D1/D2 credit aggregates; RBA E2 household balance-sheet ratios) use the **current ABS / RBA vintage** at the time the source was last downloaded — not the original first-release value. ABS routinely revises historical observations through re-estimated seasonal adjustment, methodology updates, and late source data, and the RBA passes those revisions through to its household-ratio tables. The pipeline does not reconstruct prior vintages, so for any revised period, the feature value next to its `publication_date` is not exactly what the RBA board would have seen on that date.

The bias is light for series with minor revisions (headline CPI) and larger for heavily-revised series (LFS sub-aggregates, GDP). This deviates from strict real-time correctness but is accepted as a known limitation, with the trade-off being a far simpler data pipeline.

Planned mitigations (see [CHECKLIST.md](./CHECKLIST.md)):
- Scheduled vintage accumulation going forward — capture each ABS release as it lands so the held-out test window can be evaluated against real-time data even though the training period remains current-vintage.
- Optional manual back-population of first-vintage values for the held-out test window from RBA Statement on Monetary Policy historical tables.

For full methodology details see [CONTEXT.md — Invariant #1](./CONTEXT.md).

### Excluded macro series
ABS Retail Trade (Cat. 8501.0) was considered and **dropped**. The ABS discontinued the release after the Jun 2025 reference month; any model using it for forward predictions would carry a permanently frozen feature whose staleness grows with every meeting. The ABS-designated successor (Monthly Household Spending Indicator) measures a different concept and is not used in v1. See [CONTEXT.md — Excluded series](./CONTEXT.md) for the full rationale.


## Repo Structure TODO


## Quick start TODO


## RoadMap / Progress TODO


## Licence + Credits TODO

### Acknowledgements
- RBA
- ABS
- TODO

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Progress
See [CHECKLIST.md](./CHECKLIST.md) for the full project plan.

## Project Organization

--------

