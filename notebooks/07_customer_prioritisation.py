"""
07 — Customer prioritisation and experiment design

Notebook 06 decides which action each customer deserves. That produces
more recommended contacts than any real team can execute.

This notebook answers the two questions that follow:

  Given a finite budget, who actually gets contacted?
  And how do we set this up so that next cycle we KNOW whether it worked?

The second one is the point of the whole project.

Run:  python notebooks/07_customer_prioritisation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    FEATURES,
    GROSS_MARGIN_RATE,
    CONTROL_HOLDOUT_SHARE,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    CalibratedModel,
    create_decision_output,
    allocate_capacity,
    budget_summary,
    assign_holdout,
    experiment_design_summary,
    OUTPUTS,
)

pd.set_option("display.width", 175)


model_data = build_or_load_snapshots()
train, validation, test = temporal_split(model_data)

pipeline = build_logistic_model().fit(train[FEATURES], train["purchased_again_30d"])
model = CalibratedModel(pipeline).fit_calibration(
    validation[FEATURES], validation["purchased_again_30d"]
)

scored = test.copy()
scored["purchase_probability"] = model.predict_proba(test[FEATURES])
decided = create_decision_output(scored, margin_rate=GROSS_MARGIN_RATE)


# ==========================================================================
# 1. THE PRIORITISATION SCORE, NAMED HONESTLY
# ==========================================================================

print(
    """
WHAT priority_score IS AND IS NOT

    priority_score = expected_net_margin
                   = d x AOV x margin
                     - contact_cost
                     - (P + d) x AOV x incentive_rate

where d is the ASSUMED uplift for the chosen action.

This is a prioritisation heuristic. It is not a profit forecast, and the
distinction is not pedantry — d is a guess, so the absolute number is a
guess multiplied by a probability. What survives that is the ORDERING,
because the same assumption applies to every customer in a cell.

Say this: "I ranked attention using predicted purchase likelihood,
customer value, and the cost of each intervention."

Do not say: "the model calculated expected profit." It did not.

A common alternative is P x log1p(monetary). It ranks sensibly but has no
units, so it cannot be checked against a budget. Expected net margin has
units, which means it can be wrong in a way you can detect.
"""
)


# ==========================================================================
# 2. CAPACITY
# ==========================================================================

budgets = {
    "HIGH_VALUE_WINBACK": 200,      # human time
    "PAID_RETARGETING": 1500,       # media budget
    "TARGETED_INCENTIVE": 3000,     # margin exposure
    "SERVICE_REVIEW": 150,          # operations capacity
}

print("CAPACITY CONSTRAINTS")
for key, value in budgets.items():
    print(f"  {key:<24} {value:>6,} contacts")
print("  (owned channels unconstrained; fatigue cap of 1 action per cycle)")

allocated = allocate_capacity(decided, budgets=budgets, fatigue_cap=1)

recommended = (decided["recommended_action"] != "NO_ACTION").sum()
contacted = allocated["contacted"].sum()

print(f"\n  Recommended : {recommended:,}")
print(f"  Contacted   : {contacted:,}")
print(f"  Constrained out : {recommended - contacted:,}")

print(
    """
Allocation is greedy on expected net margin within each action, which is
optimal for a simple per-action count constraint.

It stops being optimal the moment constraints interact — a shared budget
across paid channels, or an offer that cannibalises another. At that
point this becomes a knapsack problem and needs an actual solver. Worth
knowing where the simple approach stops being right.
"""
)


# ==========================================================================
# 3. BUDGET
# ==========================================================================

print("BUDGET FOR THIS CYCLE\n")

budget = budget_summary(allocated)
if budget.empty:
    print("  nothing selected")
else:
    print(
        budget[
            [
                "action", "channel", "customers", "fixed_cost",
                "expected_incentive_cost", "total_cost",
                "median_required_uplift_pp",
                "expected_net_margin_if_assumption_holds",
            ]
        ].round(2).to_string(index=False)
    )

    print(
        f"""
  Total cost: {budget['total_cost'].sum():,.0f}

Notice how much of the spend is incentive rather than delivery. Owned
channels reach large numbers for almost nothing; the money concentrates
in the small number of cells where margin is being given away.

median_required_uplift_pp is the column to argue about in a planning
meeting. It converts "should we run this campaign" into "do we believe
this campaign can move conversion by X percentage points" — a question
with a checkable answer.
"""
    )


# ==========================================================================
# 4. THE HOLDOUT
# ==========================================================================

allocated = assign_holdout(allocated, share=CONTROL_HOLDOUT_SHARE)

print("EXPERIMENT ARMS\n")
print(allocated["experiment_arm"].value_counts().to_string())

print(
    f"""
Every treated cell reserves {CONTROL_HOLDOUT_SHARE:.0%} as CONTROL: the
recommendation is logged, the contact is not sent.

Assignment is a deterministic hash of customer id and action, so the same
customer lands in the same arm on every run. No accidental
re-randomisation between cycles, and no need to store an assignment table.

This is the most valuable behaviour in the repo, and it costs
{CONTROL_HOLDOUT_SHARE:.0%} of reach. Without it, next cycle you can
report that treated customers bought more — which tells you nothing,
because you selected them for being likely to buy. Selection effect,
not treatment effect.
"""
)


# ==========================================================================
# 5. CAN THE EXPERIMENT DETECT WHAT IT NEEDS TO?
# ==========================================================================

print("EXPERIMENT POWER\n")

design = experiment_design_summary(allocated)
if design.empty:
    print("  no cells")
else:
    print(design.round(3).to_string(index=False))

    print(
        """
detectable_uplift_pp is the smallest true effect this cell could detect
at 80% power and 5% significance, given its size.

The check that matters: compare it against the assumed uplift for that
action. If the cell cannot detect an effect as small as the one you are
assuming, running it produces a number you cannot interpret either way.

Three ways out, in order of preference:
  1. Run the cell for several cycles and pool the results.
  2. Enlarge the cell by loosening its eligibility.
  3. Accept it as directional and say so out loud.

Doing this arithmetic BEFORE the campaign is the difference between an
experiment and a post-hoc story.
"""
    )


# ==========================================================================
# 6. WHAT TO MEASURE NEXT CYCLE
# ==========================================================================

print(
    """
NEXT CYCLE

With arms assigned, the measurement becomes trivial:

    uplift = repeat_rate(TREATMENT) - repeat_rate(CONTROL)

per action cell. That single number does three things:

  1. Replaces assumed_uplift with a measured one, so every break-even
     threshold in notebook 06 becomes evidence-based.
  2. Kills actions that do not work, which is the cheapest win available.
  3. Produces the training data for uplift modelling — where the question
     changes from "who will buy" to "who will buy BECAUSE we acted".

That last shift is the real destination. A propensity model ranks
customers by how likely they are to buy. An uplift model ranks them by how
much difference you make — and those two rankings are not the same list.
The customers most likely to buy are often precisely the ones you affect
least.

This dataset cannot get there on its own. The holdout is how you start
generating data that can.
"""
)

allocated.to_csv(OUTPUTS / "allocation_detail.csv", index=False)
if not budget.empty:
    budget.to_csv(OUTPUTS / "budget_summary.csv", index=False)
if not design.empty:
    design.to_csv(OUTPUTS / "experiment_design.csv", index=False)

print(f"Saved allocation and experiment tables to {OUTPUTS}")
