"""
04 — Model evaluation

A model does not become trustworthy because the algorithm is
sophisticated. It becomes trustworthy when you test what it predicts,
when it predicts it, and how well the probabilities behave.

Five questions:

  ROC-AUC     can it rank buyers above non-buyers at all?
  PR-AUC      is the ranking good where it matters, at the top?
  Lift        how much better than random is a targeted list?
  Brier       do the probabilities mean what they say?
  Stability   does it hold in every period, or only on average?

Run:  python notebooks/04_model_evaluation.py
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
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    build_gradient_boosting_model,
    CalibratedModel,
    evaluate_predictions,
    ranking_metrics,
    calibration_table,
    stability_by_snapshot,
    OUTPUTS,
)

pd.set_option("display.width", 160)


model_data = build_or_load_snapshots()
train, validation, test = temporal_split(model_data)

X_train, y_train = train[FEATURES], train["purchased_again_30d"]
X_val, y_val = validation[FEATURES], validation["purchased_again_30d"]
X_test, y_test = test[FEATURES], test["purchased_again_30d"]

candidates = {
    "logistic_regression": build_logistic_model(),
    "gradient_boosting": build_gradient_boosting_model(),
}

scores = {}
for name, pipeline in candidates.items():
    pipeline.fit(X_train, y_train)
    scores[name] = evaluate_predictions(y_val, pipeline.predict_proba(X_val)[:, 1])["pr_auc"]

best_name = max(scores, key=scores.get)
model = CalibratedModel(candidates[best_name]).fit_calibration(X_val, y_val)

probability = model.predict_proba(X_test)


# ==========================================================================
# 1. HEADLINE
# ==========================================================================

print(f"MODEL: {best_name}, sigmoid-calibrated on the validation window\n")

metrics = evaluate_predictions(y_test, probability)
for key, value in metrics.items():
    print(f"  {key:<18}: {value:.4f}")

print(
    """
  base_rate         what you would get by picking at random
  roc_auc           overall ranking quality
  pr_auc            ranking quality at the top, where campaigns live
  brier_score       squared error of the probabilities. Lower is better,
                    but it is bounded by the base rate — compare it to
                    base_rate * (1 - base_rate), not to zero.
"""
)

reference_brier = metrics["base_rate"] * (1 - metrics["base_rate"])
skill = 1 - metrics["brier_score"] / reference_brier
print(
    f"  Brier of a no-skill constant predictor : {reference_brier:.4f}"
    f"\n  Brier skill score                      : {skill:.4f}"
    "\n  (0 = no better than predicting the base rate for everyone)"
)


# ==========================================================================
# 2. LIFT — THE METRIC A MARKETER RECOGNISES
# ==========================================================================

print("\nRANKING PERFORMANCE")
ranking = ranking_metrics(y_test, probability)
print(ranking.round(4).to_string(index=False))

top_decile = ranking[ranking["population_percent"] == 0.10].iloc[0]
print(
    f"""
Read the 10% row as a sentence a marketer can act on:

  Contact the top 10% of customers and you reach
  {top_decile['buyers_found']:,.0f} of the {int(y_test.sum()):,} customers who
  would buy — {top_decile['recall']:.1%} of all buyers — at a hit rate
  {top_decile['lift']:.1f}x better than contacting at random.

Lift decays as you go deeper. That decay curve is the budget argument:
it shows where an extra contact stops being worth its cost.
"""
)


# ==========================================================================
# 3. CALIBRATION, AND WHAT IT COSTS IN MONEY
# ==========================================================================

print("CALIBRATION (predicted vs observed, by decile)")
reliability = calibration_table(y_test, probability)
print(reliability.round(4).to_string(index=False))

raw = model.predict_proba_raw(X_test)
aov = test["aov"].to_numpy()

value_calibrated = (probability * aov * GROSS_MARGIN_RATE).sum()
value_raw = (raw * aov * GROSS_MARGIN_RATE).sum()
value_actual = (np.asarray(y_test) * aov * GROSS_MARGIN_RATE).sum()

print(
    f"""
WHY THIS MATTERS IN CURRENCY

Expected 30-day gross margin across the test population, at
{GROSS_MARGIN_RATE:.0%} margin:

  using calibrated probabilities : {value_calibrated:,.0f}
  using raw probabilities        : {value_raw:,.0f}
  actually observed              : {value_actual:,.0f}

Calibration error is not an academic concern. It is the gap between the
forecast you present and the money that arrives. Every threshold in the
decision layer is set against these numbers.
"""
)


# ==========================================================================
# 4. STABILITY
# ==========================================================================

print("STABILITY BY SNAPSHOT")
stability = stability_by_snapshot(test, probability)
print(stability.round(4).to_string(index=False))

if len(stability) > 1:
    spread = stability["roc_auc"].max() - stability["roc_auc"].min()
    print(
        f"""
ROC-AUC spread across test periods: {spread:.4f}

A single average number hides period-to-period variance. If the spread is
wide, the honest claim is "it works in some months", not "it works".
Seasonality, promotional calendars and one-off events all show up here
before they show up in a campaign post-mortem.
"""
    )


# ==========================================================================
# 5. WHAT THE NUMBER 0.82 ACTUALLY MEANS
# ==========================================================================

print(
    """
A SCORE OF 0.82 DOES NOT MEAN "THIS CUSTOMER WILL BUY"

It means: among customers historically resembling this one on these
features, roughly 82% went on to place an order in the following 30 days
— assuming the future resembles the training period.

Three assumptions are doing work in that sentence:

  "resembling this one on these features"  the feature set is incomplete.
      No channel, no offer, no service history, no competitor activity.

  "roughly 82%"  only true if calibration holds, which is why the
      reliability table above is not optional.

  "assuming the future resembles the training period"  the stability
      table is the only evidence offered for that, and it covers two
      periods.

State the assumptions, then act anyway. That is different from either
pretending to certainty or refusing to decide.
"""
)

reliability.to_csv(OUTPUTS / "calibration.csv", index=False)
stability.to_csv(OUTPUTS / "stability_by_snapshot.csv", index=False)
ranking.to_csv(OUTPUTS / "ranking_metrics.csv", index=False)
print(f"Saved evaluation tables to {OUTPUTS}")
