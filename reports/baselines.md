# Baseline performance floors

Reference floors against which every §7 ML model is judged. Numbers here are
computed directly from the raw F11 cash-rate decision history and the
`three_class` target (sign of `rate_change_bps` → cut / hold / hike). This is
whole-sample descriptive accounting to establish the meaningful zero, **not** a
walk-forward evaluation — the walk-forward numbers land in `reports/results.md`
when the §7 harness ships.

## Provenance

- Target: `targets.yaml` → `three_class`, sign of `rate_change_bps`.
- Source: `rba.data.preprocess_target.build_decisions` of `rba.data.sources.rba_f11.fetch()`.
- Coverage: **1993-02-03 → 2026-06-17**, **365 meetings** (post-1993, inflation-targeting era).
- All floors below are full-sample descriptive statistics computed directly on
  that decision series.

## Class balance

| class | count | share |
| :--- | ---: | ---: |
| hold | 281 | 76.99% |
| hike | 43 | 11.78% |
| cut | 41 | 11.23% |

The RBA holds ~77% of meetings, so "hold" is the dominant class by a wide
margin. Any classifier that fails to beat 76.99% accuracy has learnt nothing.

## Baselines

Four floors, ordered by increasing informational sophistication. Each is
implemented in [`rba.models.baselines`](../src/rba/models/baselines.py) and
registered in [`models.yaml`](../src/rba/config/models.yaml).

### `MajorityClass` — the base-rate floor

Predicts `hold` for every meeting; `predict_proba` returns the empirical prior
(deliberately not one-hot, so log-loss and Brier are finite meaningful floors).

| metric | value |
| :--- | ---: |
| accuracy | **0.7699** |
| log-loss (entropy of prior) | **0.6989** |
| Brier score | **0.3808** |

This is the **hardest structural floor to beat on accuracy** — hold dominates
so completely that a no-feature classifier already gets 3-out-of-4 right. A
model that beats MajorityClass on accuracy but not on macro-F1 or log-loss is
just re-discovering the base rate; the real work is at the cycle turns.

### `Persistence` — "predict the last decision"

Predicts the label of the most recent training meeting. `predict_proba` is
one-hot on the persisted class (log-loss saturates on wrong calls but sklearn
clips it to a finite upper bound).

| metric | value |
| :--- | ---: |
| accuracy | **0.7143** |
| accuracy after hold | 0.8143 |
| accuracy after cut | 0.2927 |
| accuracy after hike | 0.4651 |
| eligible meetings | 364 |

Persistence is **worse than MajorityClass on plain accuracy** because it
reverses on ~19% of hold-followed-by-hold meetings without ever recouping the
loss on cycle turns (where it typically follows the previous action right up
until the direction changes and then misses badly — 29% after cuts, 47% after
hikes). But it captures the correct behaviour *within* a policy stretch and is
the natural floor for the "does this model understand transition dynamics?"
question — which is what the §7 evaluation actually cares about at cycle turns.

### `TaylorRule` — the economic prior

Predicts the prescribed **rate level** from `i = α + φ_π · π + φ_y · y_gap`.
Two modes: literature defaults (Taylor's `(1.5, 0.5)`, RBA-appropriate
`neutral_rate = 3.0` / `inflation_target = 2.5`) or OLS fit on the training
window.

**Not yet quantitatively evaluated**: the inflation and output-gap features
that feed this baseline are not materialised in the current master frame
(`cpi_headline_yoy` is aspirational per [features.yaml](../src/rba/config/features.yaml)
and `output_gap` needs a builder to be added). Historical Australian cash-rate
paths depart meaningfully from Taylor prescriptions during the GFC,
forward-guidance era (2020–2021), and COVID emergency response — so this
baseline is expected to be an "informed prior" that ML models should beat on
average but not necessarily at regime shifts, rather than a tight lower bound.

Metrics will be reported here once the harness in §7 lands and the
inflation/output-gap features are wired in.

### `MarketImplied` — the hardest floor

Reads the ASX 30-day IB-futures implied cash rate (naïve
`100 − settlement_price` of the meeting-month contract at T−1, per
[`derive_meeting_implied`](../src/rba/data/sources/asx_ib_futures.py)) and
maps `Δ = implied_rate − current_rate` to hike/hold/cut via the single-move
linear decomposition (`P(hike) = clip(Δ / 25 bp, 0, 1)`, etc.).

**Coverage caveat**: ASX IB-futures data begins 2022-04-21, so this baseline
is only evaluable on the **post-2022 window: 39 meetings** (of which 20 hold /
16 hike / 3 cut). That window's hike share is **41%** — the tightening cycle
is over-represented relative to the full-history balance — so the
market-implied floor competes on a genuinely different distribution than the
three preceding baselines. Comparing to MajorityClass and Persistence on the
same 39-meeting window is the meaningful benchmark.

Metrics will be reported here once the harness in §7 evaluates on the
post-2022 window.

This is the **strongest signal floor**: an ML model that fails to beat the
market has not found information beyond what every rates trader already
priced in. It is also the only baseline that admits calibrated probabilities
(the futures-implied decomposition is the closest thing to a market
consensus on P(hike)).

## What ML models must clear

At minimum, on the `three_class` target:

1. **Accuracy > 76.99%** (`MajorityClass` on full history) — otherwise the
   model isn't beating "always hold."
2. **Log-loss < 0.6989** (`MajorityClass`) — otherwise the model isn't
   improving on the base-rate probability distribution.
3. **Macro-F1 well above the MajorityClass floor** (which is ≈ 0.29 for a
   constant-`hold` predictor). Macro-F1 is the primary signal for cycle-turn
   competence.
4. **Beating `MarketImplied` accuracy on the post-2022 window** — the
   strongest evidence of genuine signal.

The `Persistence` accuracy (71.43%) is not a lower bound for §7 models; it's a
diagnostic. A model matching Persistence *within* hold streaks but beating it
at cycle turns is the profile §7 aims for.

## Regenerating this document

The measured floors in this document are recomputed directly from the raw F11
decision history — no harness or master frame required. The one-shot Python:

```python
from rba.data.sources import rba_f11
from rba.data.preprocess_target import build_decisions
import numpy as np

tgt = build_decisions(rba_f11.fetch())
tgt = tgt[tgt["observation_date"] >= "1993-01-01"].sort_values("observation_date").reset_index(drop=True)
three_class = np.sign(tgt["rate_change_bps"]).map({-1: "cut", 0: "hold", 1: "hike"})
# … see the "Provenance" section for exact metric definitions.
```

The `TaylorRule` / `MarketImplied` rows above are populated once the §7
harness materialises the missing features (`output_gap` in particular) and
runs the walk-forward evaluation on the covered windows.
