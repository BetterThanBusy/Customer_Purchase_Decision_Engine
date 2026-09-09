"""
05 — Customer explanation

Two different questions, routinely confused:

  GLOBAL   which features does the model rely on?
  LOCAL    why did THIS customer get THIS score?

Neither answers a third question people assume they have answered:
what would happen if we changed the feature. That is a causal question
and nothing in this notebook addresses it.

Run:  python notebooks/05_customer_explanation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    FEATURES,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    CalibratedModel,
    permutation_importance_table,
    linear_contributions,
    top_reason_codes,
    OUTPUTS,
)

pd.set_option("display.width", 170)


model_data = build_or_load_snapshots()
train, validation, test = temporal_split(model_data)

X_train, y_train = train[FEATURES], train["purchased_again_30d"]
X_val, y_val = validation[FEATURES], validation["purchased_again_30d"]
X_test, y_test = test[FEATURES], test["purchased_again_30d"]

pipeline = build_logistic_model().fit(X_train, y_train)
model = CalibratedModel(pipeline).fit_calibration(X_val, y_val)


# ==========================================================================
# 1. GLOBAL — PERMUTATION IMPORTANCE
# ==========================================================================

print("GLOBAL IMPORTANCE (permutation, scored on average precision)\n")

importance = permutation_importance_table(
    lambda frame: model.predict_proba(frame), X_test, y_test, n_repeats=5
)
print(importance.round(5).to_string(index=False))

print(
    """
Read as: how much PR-AUC is lost when this column is shuffled.

Two cautions that apply to every permutation importance table ever made:

  1. Correlated features share credit. recency_days and recency_ratio
     measure overlapping things, so shuffling one leaves the other to
     carry the signal, and both look less important than they are.
  2. "Important to the model" is not "important to the customer". If a
     feature is important only because it proxies something you cannot
     act on, its importance is not actionable.
"""
)


# ==========================================================================
# 2. GLOBAL — DIRECTION
# ==========================================================================

print("COEFFICIENT DIRECTION (standardised, log-odds)\n")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

coefficients = pd.DataFrame(
    {
        "feature": [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()],
        "coefficient": classifier.coef_[0],
    }
)
coefficients["odds_multiplier"] = np.exp(coefficients["coefficient"])

top = pd.concat(
    [
        coefficients.nlargest(8, "coefficient"),
        coefficients.nsmallest(8, "coefficient"),
    ]
)
print(top.round(4).to_string(index=False))

print(
    """
Features are standardised, so a coefficient reads as: the change in
log-odds from a one standard deviation move in that feature, holding the
others fixed. odds_multiplier is the same thing in multiplicative form.

"Holding the others fixed" is the load-bearing phrase, and in real
customer data the others are never fixed.
"""
)


# ==========================================================================
# 3. LOCAL — PER-CUSTOMER REASON CODES
# ==========================================================================

print("LOCAL REASON CODES\n")

contributions = linear_contributions(pipeline, X_test)
reasons = top_reason_codes(contributions, top_k=3)

scored = test.copy()
scored["purchase_probability"] = model.predict_proba(X_test)
scored["model_reasons"] = reasons.values

sample_columns = [
    "customer_id", "purchase_probability", "recency_days",
    "recency_ratio", "frequency", "monetary", "model_reasons",
]

print("Highest scoring:")
print(scored.nlargest(5, "purchase_probability")[sample_columns].round(3).to_string(index=False))

print("\nLowest scoring:")
print(scored.nsmallest(5, "purchase_probability")[sample_columns].round(3).to_string(index=False))

print(
    """
Each reason is an exact decomposition of the log-odds:

    contribution = standardised feature value x coefficient

Sum them, add the intercept, apply the logistic function, and you recover
the score precisely. This is not an approximation like SHAP sampling —
for a linear model it is arithmetic.

It still is not a cause. "-recency_days" means recency pushed this score
down. It does not mean contacting them tomorrow will push it back up.
"""
)


# ==========================================================================
# 4. THE RETURNS TRAP
# ==========================================================================

print("A WORKED EXAMPLE OF CORRELATION MISREAD AS CAUSATION\n")

has_returns = model_data["return_orders"] > 0

comparison = (
    model_data.assign(has_returns=has_returns)
    .groupby("has_returns")
    .agg(
        customers=("customer_id", "size"),
        repeat_rate=("purchased_again_30d", "mean"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        median_recency=("recency_days", "median"),
    )
)
print(comparison.round(3).to_string())

if has_returns.any() and (~has_returns).any():
    gap = (
        comparison.loc[True, "repeat_rate"] - comparison.loc[False, "repeat_rate"]
    )
    print(f"\n  Raw repeat-rate gap: {gap:+.3f}")

print(
    """
The tempting headline: "customers who return items buy more often".

The problem: look at median_frequency and median_monetary in the same
table. Customers with returns are the heavier buyers. Buying more creates
more opportunities to return, and buying more also predicts buying again.
Frequency drives both sides.

So the gap is at least partly composition, not effect. Test it by
comparing within frequency bands rather than across the whole base:
"""
)

bands = pd.cut(
    model_data["frequency"],
    bins=[0, 1, 2, 4, 8, np.inf],
    labels=["1", "2", "3-4", "5-8", "9+"],
)

within = (
    model_data.assign(has_returns=has_returns, band=bands)
    .groupby(["band", "has_returns"], observed=True)["purchased_again_30d"]
    .agg(["size", "mean"])
    .unstack()
)
print(within.round(3).to_string())

print(
    """
If the gap shrinks or reverses inside frequency bands, most of the raw
difference was frequency wearing a returns costume.

The defensible statement is: returns are a behavioural signal worth
investigating, and a high return rate warrants a service review rather
than another campaign. That is why SERVICE_RISK outranks every marketing
state in notebook 06 — not because returns cause loyalty, but because
sending a discount to someone with a product-fit problem solves the wrong
problem.
"""
)

importance.to_csv(OUTPUTS / "feature_importance.csv", index=False)
coefficients.to_csv(OUTPUTS / "model_coefficients.csv", index=False)
print(f"Saved explanation tables to {OUTPUTS}")
