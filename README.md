# Customer Purchase Decision Engine

Predicting who will buy is the easy half. This project is about the other half:
turning a probability into a marketing decision that can defend its own cost,
and setting it up so the next cycle produces causal evidence instead of another
correlation.

Built on [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
(UCI), ~1M transactions, Dec 2009 – Dec 2011.

---

## The problem with most repeat-purchase projects

They stop at the model and bolt on a rule:

```
P(purchase) > 0.5  ->  SEND DISCOUNT
```

That fails for a reason you can prove with arithmetic. It targets the customers
most likely to buy anyway and pays them to do what they were going to do. The
higher the score, the more money it burns.

The fix is not a better model. It is a decision layer underneath it.

---

## The framework

```
  PREDICTION      What is likely to happen?
       |
  EXPLANATION     What behaviour sits behind that prediction?
       |
  STATE           What situation is this customer actually in?
       |
  ECONOMICS       What would an intervention have to achieve to pay for itself?
       |
  DECISION        Which action, or none?
       |
  PRIORITISATION  Who gets attention first, given finite capacity?
       |
  EXPERIMENT      Who is held out, so the next cycle can measure cause?
```

Each stage is a separate notebook and a separate function. When margin changes
or the media budget is cut, the states stay identical and only the actions move.

---

## The equation everything rests on

Let `d` be the incremental conversion an action causes, in probability points.
Contacting a customer is worth it when:

```
d × AOV × margin  >  contact_cost  +  (P + d) × AOV × incentive_rate
────────────────     ────────────     ──────────────────────────────
margin gained        fixed cost       incentive paid to EVERYONE who
from extra orders                     converts, including the P who
                                      would have bought anyway
```

Solved for `d`:

```
d_breakeven = (contact_cost + P × AOV × incentive_rate)
              ────────────────────────────────────────
                   AOV × (margin − incentive_rate)
```

Three things fall straight out of it:

1. **`incentive_rate = 0` → the bar is tiny.** Owned channels almost always
   clear it. Recommendation beats discount by default.
2. **`P` sits in the numerator whenever there is an incentive.** The more
   certain a customer is to buy, the higher the uplift a discount must produce
   to pay for itself. That is deadweight cost, quantified — and it is exactly
   why `P > 0.5 → discount` is backwards.
3. **`incentive_rate ≥ margin` → `d_breakeven` is infinite.** No uplift can
   ever justify it. Impossible, not merely unprofitable.

The engine reports **required uplift** next to **assumed uplift** for every
recommendation, so anyone reading the output can see which number is measured
and which is a guess.

---

## Customer states

State is a description, not an instruction. Precedence is deliberate:

| State | Rule | Why it sits there |
|---|---|---|
| `SERVICE_RISK` | return rate ≥ 40%, ≥ 2 returns | A service problem is not a marketing problem. Outranks everything. |
| `NEW_UNPROVEN` | exactly one order | Not churned — unmeasured. No rhythm to compare against. |
| `ACTIVE_*` | recency ≤ 30d **or** within own rhythm | Behaving normally for them. |
| `COOLING_*` | 1–2.5× their own average gap | Slowing, not stopped. |
| `LAPSING_*` | beyond 2.5× their gap, ≤ 180d | Genuinely drifting. |
| `DORMANT` | > 180d, below value threshold | Cheap presence only. |

`*_VALUABLE` vs `*_STANDARD` splits at the 70th percentile of historic spend,
computed per snapshot.

**`recency_ratio` is the feature that makes this work.** It is days since last
order divided by that customer's *own* average gap. Sixty days of silence is
alarming for a weekly buyer and completely normal for a quarterly one. Raw
recency treats them identically and is wrong for both.

---

## What this project claims, and what it does not

| | |
|---|---|
| **Can estimate** | who is likely to purchase in the next 30 days |
| **Can infer** | who warrants different marketing attention |
| **Can compute** | what an intervention must achieve to pay for itself |
| **Cannot prove** | that any intervention caused a purchase |

Online Retail II has no channel, no offer, no control group and no response
data. No amount of modelling fixes a missing variable.

So the project ends where the evidence ends — with a designed experiment. Every
treated cell reserves a 10% control via deterministic hashing. That costs 10% of
reach and it is the only reason next cycle can compute:

```
uplift = repeat_rate(TREATMENT) − repeat_rate(CONTROL)
```

Which is the training data an uplift model needs. The question then changes from
*who will buy* to *who will buy because we acted* — and those two rankings are
not the same list.

### Language used in the outputs

- `priority_score` is a **prioritisation heuristic**, not a profit forecast.
  "I ranked attention using predicted purchase likelihood, customer value and
  the cost of each intervention." Not "the model calculated expected profit."
- `expected_organic_revenue` is expected value under the model. It is
  **not incremental** and is deliberately named to prevent that reading.
- `assumed_uplift` is flagged `[Guessing]` wherever it appears. It is a prior
  standing in for evidence that does not exist yet.

---

## Modelling decisions worth knowing about

**Temporal snapshots, not a single cutoff.** Twenty monthly snapshots, features
strictly before each cutoff, target strictly inside the following 30 days.
Notebook 02 asserts the leakage checks rather than assuming them.

**No `class_weight="balanced"`.** Balanced weighting barely moves ranking and
systematically inflates predicted probabilities. That is harmless if the output
is only ever sorted — but the decision layer multiplies it by order value and
margin. An inflated probability produces an inflated business case. Imbalance is
handled in the *metric* (PR-AUC, lift), not the loss function. Notebook 03
demonstrates the size of the distortion.

**Calibration by Platt scaling.** Strictly monotone, so ROC-AUC, PR-AUC and lift
are provably unchanged; two parameters, so it barely overfits a small validation
window. Isotonic is available but is only *weakly* monotone — it creates ties
that can move PR-AUC, and it flatters itself when fitted and measured on the
same data.

**Baselines first.** Recency-only, frequency-only and an RFM score. A model that
does not beat "sort by recency" is not worth its maintenance cost.

**Known caveat:** the split is clean in time, not in customers. The same
customer appears in train and validation at different snapshots. This matches
deployment — you score customers you have seen before — but the numbers describe
performance on known customers, not new ones.

---

## Two things the framework gets wrong

Stated here rather than buried, because both point somewhere useful.

**Uplift is treated as constant across P.** Since `P` sits in the numerator,
discounts are cheapest to justify on the *lowest*-probability customers — which,
taken literally, sends offers to the people least likely to respond. The
equation is fine; the flat-uplift assumption feeding it is not. Real uplift is
near zero at both extremes and highest in the middle. This is the strongest
argument for uplift modelling, and it fell out of the arithmetic rather than
being asserted.

**Near-free channels make the economic test almost vacuous.** An owned email
costs ~0.02 to send, so the break-even uplift is a few hundredths of a
percentage point and nearly everyone clears it. For owned channels the binding
constraint is not money, it is attention — a cost that never appears in a
per-send model. Hence the contact-fatigue cap in notebook 07.

---

## Repository layout

```
customer-purchase-decision-engine/
├── data/
│   ├── raw/            online_retail_II.xlsx  (not committed)
│   ├── processed/
│   └── features/       cached snapshot dataset
├── notebooks/
│   ├── 01_business_problem_and_data.py     target definition, data audit, limits
│   ├── 02_customer_feature_engineering.py  snapshots, leakage checks, features
│   ├── 03_purchase_prediction.py           baselines, models, calibration
│   ├── 04_model_evaluation.py              ranking, lift, reliability, stability
│   ├── 05_customer_explanation.py          importance, reason codes, causal traps
│   ├── 06_marketing_strategy.py            states, economics, decisions
│   ├── 07_customer_prioritisation.py       capacity, budget, holdout, power
│   ├── 08_decision_engine.py               end-to-end runner
│   └── 09_video_assets.py                  charts + Excel workbook for presenting
├── src/
│   ├── engine.py       all reusable logic
│   ├── charts.py       every figure, themed in one place
│   └── workbook.py     the presentation workbook, incl. a live break-even sheet
├── scripts/
│   └── make_sample_data.py   synthetic data with the same schema
├── outputs/
│   └── charts/         PNGs at 200 DPI, numbered in presentation order
└── README.md
```

Notebooks explain. `src/engine.py` does the work. Eight notebooks with
duplicated code becomes unmaintainable by notebook four.

---

## Running it

```bash
pip install -r requirements.txt

# Optional: verify the pipeline before downloading the real 150MB file
python scripts/make_sample_data.py

# Put online_retail_II.xlsx in data/raw/, then:
python notebooks/08_decision_engine.py

# Or walk through the reasoning:
python notebooks/01_business_problem_and_data.py
python notebooks/02_customer_feature_engineering.py
# ... etc

# Charts and the presentation workbook:
python notebooks/09_video_assets.py
```

Snapshots are cached to `data/features/` after the first build. Pass `--rebuild`
to force a refresh.

---

## Outputs

| File | Contents |
|---|---|
| `customer_scores.csv` | probability, features, state, per-customer reason codes |
| `marketing_actions.csv` | chosen action, required vs assumed uplift, margin of safety, experiment arm |
| `priority_customers.csv` | top 500 selected contacts, ranked |
| `budget_summary.csv` | cost per action and the uplift it must deliver |
| `experiment_design.csv` | cell sizes and minimum detectable effect |
| `calibration.csv` | predicted vs observed by decile |
| `ranking_metrics.csv` | precision, recall and lift by population depth |
| `stability_by_snapshot.csv` | performance per test period |
| `feature_importance.csv` | permutation importance |

A row in `marketing_actions.csv` reads as a complete argument:

> Customer 12668 is `LAPSING_VALUABLE`. Purchase probability 0.31, historic
> spend 10,527, 167 days since last order — 2.8× their own average gap.
> Recommended action: high-value win-back. It needs 0.94pp of incremental
> conversion to break even; we assume 7pp, a margin of safety of 7.4×.
> Assigned to TREATMENT.

Which is a decision a marketer can accept, reject, or argue with — rather than
a score they are asked to trust.

---

## Presenting it

`notebooks/09_video_assets.py` regenerates seventeen charts into
`outputs/charts/`, numbered in presentation order, plus
`outputs/video_tables.xlsx`.

Three of the charts carry the argument:

- **`10_deadweight_curve.png`** — required break-even uplift against P, per action.
  Zero-incentive channels stay flat and cheap; discounts climb steeply as the
  customer becomes more certain to buy. The one chart that proves
  `P > 0.5 → discount` is backwards.
- **`11_customer_state_map.png`** — every customer plotted on overdue-ness
  (against their *own* rhythm) versus value, coloured by state. Shows the
  segmentation as a map rather than a table.
- **`13_state_action_matrix.png`** — which state receives which action.
  `ACTIVE_VALUABLE` gets loyalty and never a discount; `SERVICE_RISK` gets no
  campaign at all.

Sheet `05 Break-even calculator` in the workbook holds **live Excel formulas**.
Change the margin, contact cost or discount rate and the whole verdict column
recalculates — including the case where the discount exceeds the margin and no
uplift can ever pay for it. Blue cells are inputs; black cells are formulas.

The script also prints a "numbers to say out loud" block at the end, so the
figures quoted on camera come from the run rather than from memory.

---

## Roadmap

1. **Measure uplift.** Run one cycle, read the holdout, replace `assumed_uplift`
   with a measured number per cell.
2. **Model uplift, not propensity.** Two-model or transformed-outcome approach
   on the cells that have accumulated enough data.
3. **Uplift that varies with P.** Fixes the limitation above directly.
4. **Margin per order.** Currently a flat 35% assumption. Real category margin
   would change which actions clear the bar.
5. **Contact fatigue as a modelled cost**, not a hard cap.