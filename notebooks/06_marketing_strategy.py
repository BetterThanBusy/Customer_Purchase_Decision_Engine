"""
06 — Marketing strategy

This is the notebook that separates the project from a churn tutorial.

The strategy most of these projects implement:

    P(purchase) > 0.5  ->  SEND DISCOUNT

That fails for a specific, provable reason. It targets the customers
most likely to buy anyway, and pays them to do it. The higher the
probability, the more money it wastes.

The structure used here instead:

    P(purchase) + RECENCY vs OWN RHYTHM + VALUE + TREND
    + RETURN SIGNAL + LIFECYCLE
              |
              v
        CUSTOMER STATE          what situation is this?
              |
              v
        ACTION ECONOMICS        what would fixing it have to achieve?
              |
              v
        DECISION                which action, or none
              |
              v
        PRIORITISATION          who first, given finite capacity
              |
              v
        EXPERIMENT              who is held out, so we can learn

Run:  python notebooks/06_marketing_strategy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    FEATURES,
    ACTION_CATALOGUE,
    GROSS_MARGIN_RATE,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    CalibratedModel,
    create_decision_output,
    required_uplift,
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


# ==========================================================================
# 1. THE BREAK-EVEN EQUATION
# ==========================================================================

print(
    f"""
THE ONE PIECE OF MATHS THAT DRIVES EVERY DECISION

Let d be the incremental conversion the action causes, in probability
points. Contacting a customer is worth it when:

    d x AOV x margin  >  contact_cost  +  (P + d) x AOV x incentive_rate
    ----------------     ------------     ------------------------------
    margin you gain      fixed cost       incentive paid to EVERYONE who
    from extra orders                     converts, including the P who
                                          would have bought anyway

Solve for d:

    d_breakeven = (contact_cost + P x AOV x incentive_rate)
                  / (AOV x (margin - incentive_rate))

Three consequences fall straight out of it:

  1. incentive_rate = 0  ->  the numerator is just contact_cost, and the
     bar is tiny. Owned channels almost always clear it.

  2. P appears in the NUMERATOR whenever there is an incentive. The more
     certain a customer is to buy, the higher the uplift the discount
     must produce to pay for itself. That is the deadweight cost, and
     it is why "target the high scorers with an offer" is backwards.

  3. incentive_rate >= margin  ->  the denominator goes non-positive and
     d_breakeven is infinite. No uplift can ever justify it. Impossible,
     not merely unprofitable.

Margin assumption in use: {GROSS_MARGIN_RATE:.0%}. [Guessing] — it is an
input, not a finding. Change it in src/engine.py and every threshold in
this notebook moves.
"""
)


# ==========================================================================
# 2. DEADWEIGHT, DEMONSTRATED
# ==========================================================================

print("DEADWEIGHT COST OF A 15% INCENTIVE, BY PURCHASE PROBABILITY\n")

incentive = ACTION_CATALOGUE["TARGETED_INCENTIVE"]
example_aov = float(np.nanmedian(scored["aov"]))

rows = []
for p in [0.05, 0.15, 0.30, 0.50, 0.70, 0.85]:
    needed = float(required_uplift(np.array([p]), np.array([example_aov]), incentive)[0])
    rows.append(
        {
            "P(purchase)": p,
            "subsidy_to_organic_buyers": p * example_aov * incentive.incentive_rate,
            "required_uplift_pp": needed * 100,
            "assumed_uplift_pp": incentive.assumed_uplift * 100,
            "verdict": "worth it" if incentive.assumed_uplift >= needed else "wasteful",
        }
    )

deadweight = pd.DataFrame(rows)
print(deadweight.round(3).to_string(index=False))

print(
    f"""
At a median AOV of {example_aov:,.0f}, the discount is defensible at low
probabilities and indefensible at high ones. The exact crossover moves
with AOV and margin, but the direction never does.

The naive "P > 0.5 -> send discount" rule targets precisely the right-hand
side of this table.
"""
)


# ==========================================================================
# 3. CUSTOMER STATE
# ==========================================================================

decided = create_decision_output(scored, margin_rate=GROSS_MARGIN_RATE)

print("CUSTOMER STATES\n")

state_profile = (
    decided.groupby("customer_state")
    .agg(
        customers=("customer_id", "size"),
        mean_probability=("purchase_probability", "mean"),
        actual_repeat_rate=("purchased_again_30d", "mean"),
        median_recency=("recency_days", "median"),
        median_recency_ratio=("recency_ratio", "median"),
        median_monetary=("monetary", "median"),
        median_return_rate=("return_rate", "median"),
    )
    .sort_values("customers", ascending=False)
)
print(state_profile.round(3).to_string())

print(
    """
Precedence in assign_customer_state(), and why each rule sits where it does:

  1. SERVICE_RISK       high return rate with at least two return events.
                        A service problem is not a marketing problem, and
                        a discount aimed at it makes the returns worse.
                        Outranks everything.

  2. NEW_UNPROVEN       exactly one order. Not churned — unmeasured. You
                        have no rhythm to compare against, so any lapse
                        judgement about them is invented.

  3. ACTIVE / COOLING / LAPSING
                        split by recency_ratio, not raw recency. 60 days
                        silent is alarming for a weekly buyer and entirely
                        normal for a quarterly one. Raw recency treats
                        them identically and is wrong for both.

  4. VALUABLE vs STANDARD
                        the 70th percentile of monetary value, computed
                        per snapshot. It decides which channels are even
                        on the table — a 1.50 human touch is rational for
                        one group and absurd for the other.

Note that state is a DESCRIPTION, not an instruction. That separation is
deliberate: when margin changes or the media budget is cut, the states
stay identical and only the actions move.
"""
)


# ==========================================================================
# 4. ACTIONS
# ==========================================================================

print("ACTION CATALOGUE\n")

catalogue = pd.DataFrame(
    [
        {
            "action": a.key,
            "channel": a.channel,
            "contact_cost": a.contact_cost,
            "incentive_rate": a.incentive_rate,
            "assumed_uplift_pp": a.assumed_uplift * 100,
            "objective": a.objective,
        }
        for a in ACTION_CATALOGUE.values()
    ]
)
print(catalogue.to_string(index=False))

print(
    """
Every action is described by what it COSTS, not what it is called. That
is what makes them comparable.

assumed_uplift is flagged [Guessing] everywhere it appears. It is a prior
standing in for evidence that does not exist yet. The engine does not hide
it inside a formula — it prints it next to the required uplift so anyone
reading the output can see exactly which number is measured and which is
assumed. Every completed experiment replaces one of these guesses.
"""
)

print("\nDECISIONS TAKEN\n")

decision_profile = (
    decided.groupby("recommended_action")
    .agg(
        customers=("customer_id", "size"),
        mean_probability=("purchase_probability", "mean"),
        median_required_uplift_pp=("required_uplift", lambda s: s.median() * 100),
        median_margin_of_safety=("margin_of_safety", "median"),
        total_expected_net_margin=("expected_net_margin", "sum"),
    )
    .sort_values("customers", ascending=False)
)
print(decision_profile.round(3).to_string())

no_action_share = (decided["recommended_action"] == "NO_ACTION").mean()
print(
    f"""
NO_ACTION share: {no_action_share:.1%}

A decision engine that never recommends doing nothing is not making
decisions. These are customers for whom no eligible action clears its own
break-even bar at their probability and order value.
"""
)


# ==========================================================================
# 4b. TWO LIMITATIONS THE MATHS EXPOSES
# ==========================================================================

print(
    """
TWO THINGS THIS FRAMEWORK GETS WRONG, STATED PLAINLY

LIMITATION 1 — uplift is treated as constant across P.

  Look at where TARGETED_INCENTIVE lands. The break-even equation puts P
  in the numerator, so discounts are cheapest to justify on the LOWEST
  probability customers. Taken literally, that sends offers to the people
  least likely to respond to anything.

  The equation is not wrong. The assumption feeding it is. Real uplift is
  not flat across P — it is usually near zero at both extremes (the
  certain buyer needs nothing; the dead customer responds to nothing) and
  highest somewhere in the middle. This code assumes one uplift number per
  action, which is a placeholder.

  This is the single strongest argument for uplift modelling, and it fell
  out of the arithmetic rather than being asserted. Until uplift is
  measured per segment, treat the low-P incentive recommendations as an
  artefact and cap the action by state eligibility, as it is here.

LIMITATION 2 — near-free channels make the economic test almost vacuous.

  An owned email costs about 0.02 to send. Against any realistic AOV, the
  break-even uplift is a few hundredths of a percentage point, so almost
  every customer clears the bar. The economics stop binding.

  Which means: for owned channels the real constraint is not money, it is
  attention. Send everything to everyone and open rates collapse — a cost
  that never appears in a per-send cost model. That is why notebook 07
  applies a contact-fatigue cap rather than relying on break-even alone.

  Cost-based decisioning works where the cost is real. Where it is not,
  the binding constraint has to be modelled explicitly.
"""
)


# ==========================================================================
# 5. STATE x ACTION
# ==========================================================================

print("STATE x ACTION\n")
print(
    pd.crosstab(decided["customer_state"], decided["recommended_action"]).to_string()
)

print(
    """
Worth checking a few cells by hand:

  ACTIVE_VALUABLE     should skew to loyalty and cross-sell, NOT discount.
                      These customers are already coming back. Anything
                      you pay them is a subsidy on demand you had.

  COOLING_VALUABLE    reactivation before incentive. Try the free channel
                      first; the offer is a fallback, not an opener.

  LAPSING_VALUABLE    the segment worth real money. A customer with high
                      historic value and a low current score is not the
                      same as a low-value customer with a low score, even
                      though a pure prediction model scores them alike.
                      This is the clearest case where prediction alone
                      gives the wrong answer.

  NEW_UNPROVEN        second-purchase journey. The objective is habit
                      formation, not immediate revenue.

  SERVICE_RISK        no campaign at all. Route to operations.
"""
)


# ==========================================================================
# 6. WORKED CUSTOMERS
# ==========================================================================

print("WORKED EXAMPLES\n")

columns = [
    "customer_id", "customer_state", "purchase_probability", "recency_days",
    "recency_ratio", "frequency", "monetary", "aov", "return_rate",
    "recommended_action", "required_uplift", "margin_of_safety",
    "expected_net_margin",
]

for state in ["ACTIVE_VALUABLE", "LAPSING_VALUABLE", "NEW_UNPROVEN", "SERVICE_RISK"]:
    subset = decided[decided["customer_state"] == state]
    if subset.empty:
        continue
    print(f"--- {state} ---")
    print(
        subset.nlargest(2, "monetary")[columns].round(3).to_string(index=False)
    )
    print()

decided.to_csv(OUTPUTS / "decision_detail.csv", index=False)
print(f"Saved decision detail to {OUTPUTS / 'decision_detail.csv'}")
