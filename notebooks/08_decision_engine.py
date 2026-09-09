"""
08 — Decision engine (end-to-end runner)

Notebooks 01-07 explain each layer. This one runs all of them and writes
the files a marketer would actually open.

    python notebooks/08_decision_engine.py
    python notebooks/08_decision_engine.py --rebuild
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    FEATURES,
    ACTION_CATALOGUE,
    OUTPUTS,
    GROSS_MARGIN_RATE,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    build_gradient_boosting_model,
    CalibratedModel,
    BASELINES,
    evaluate_predictions,
    ranking_metrics,
    calibration_table,
    stability_by_snapshot,
    permutation_importance_table,
    linear_contributions,
    top_reason_codes,
    create_decision_output,
    allocate_capacity,
    budget_summary,
    assign_holdout,
    experiment_design_summary,
)


def header(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="path to the retail xlsx")
parser.add_argument("--rebuild", action="store_true", help="rebuild snapshot cache")
args = parser.parse_args()


# ==========================================================================
# 1. DATA AND SNAPSHOTS
# ==========================================================================

header("CUSTOMER PURCHASE DECISION ENGINE")

print("\nBuilding snapshots...")
model_data = build_or_load_snapshots(data_path=args.data, rebuild=args.rebuild)

print(f"\nSnapshot rows : {len(model_data):,}")
print(f"Snapshots     : {model_data['snapshot_date'].nunique()}")
print(f"Customers     : {model_data['customer_id'].nunique():,}")
print(f"Base rate     : {model_data['purchased_again_30d'].mean():.4f}")

train, validation, test = temporal_split(model_data)

print("\nTemporal split")
for name, frame in [("Train", train), ("Validation", validation), ("Test", test)]:
    if frame.empty:
        print(f"  {name:<11}: EMPTY")
        continue
    print(
        f"  {name:<11}: {frame['snapshot_date'].min().date()} -> "
        f"{frame['snapshot_date'].max().date()}  "
        f"({len(frame):,} rows, base rate {frame['purchased_again_30d'].mean():.3f})"
    )

X_train, y_train = train[FEATURES], train["purchased_again_30d"]
X_val, y_val = validation[FEATURES], validation["purchased_again_30d"]
X_test, y_test = test[FEATURES], test["purchased_again_30d"]


# ==========================================================================
# 2. BASELINES FIRST
# ==========================================================================
# If the model cannot beat "sort by recency", it does not deserve to ship.

header("BASELINES (validation window)")

baseline_rows = []
for name, fn in BASELINES.items():
    score = fn(X_val)
    # Rescale to [0, 1] only so the metric functions accept it. These are
    # rankings, not probabilities — Brier and log-loss would be meaningless.
    scaled = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-9)
    metrics = evaluate_predictions(y_val, scaled)
    baseline_rows.append(
        {"model": name, "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]}
    )

baselines = pd.DataFrame(baseline_rows)
print(baselines.to_string(index=False))


# ==========================================================================
# 3. MODEL SELECTION ON VALIDATION
# ==========================================================================

header("MODEL SELECTION (validation window)")

candidates = {
    "logistic_regression": build_logistic_model(),
    "gradient_boosting": build_gradient_boosting_model(),
}

selection_rows = []
fitted = {}

for name, pipeline in candidates.items():
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    probability = pipeline.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, probability)
    metrics["model"] = name
    selection_rows.append(metrics)

selection = pd.DataFrame(selection_rows)[
    ["model", "roc_auc", "pr_auc", "brier_score", "calibration_gap"]
]
print(selection.to_string(index=False))

best_name = selection.sort_values("pr_auc", ascending=False).iloc[0]["model"]
print(f"\nSelected on PR-AUC: {best_name}")

# Calibrate the winner on the validation window, then leave test untouched.
model = CalibratedModel(fitted[best_name]).fit_calibration(X_val, y_val)


# ==========================================================================
# 4. TEST PERFORMANCE
# ==========================================================================

header("TEST PERFORMANCE")

raw_probability = model.predict_proba_raw(X_test)
probability = model.predict_proba(X_test)

before = evaluate_predictions(y_test, raw_probability)
after = evaluate_predictions(y_test, probability)

comparison = pd.DataFrame([before, after], index=["uncalibrated", "calibrated"])
print(comparison[["roc_auc", "pr_auc", "brier_score", "mean_predicted", "calibration_gap"]].to_string())
print("\nRanking is unchanged by calibration. Probability accuracy is not.")

print("\nRANKING")
ranking = ranking_metrics(y_test, probability)
print(ranking.to_string(index=False))

print("\nCALIBRATION (predicted vs observed)")
reliability = calibration_table(y_test, probability)
print(reliability.to_string(index=False))

print("\nSTABILITY BY SNAPSHOT")
stability = stability_by_snapshot(test, probability)
print(stability.to_string(index=False))


# ==========================================================================
# 5. EXPLANATION
# ==========================================================================

header("EXPLANATION")

importance = permutation_importance_table(
    lambda frame: model.predict_proba(frame), X_test, y_test, n_repeats=3
)
print(importance.head(10).to_string(index=False))

reasons = None
if best_name == "logistic_regression":
    contributions = linear_contributions(model.pipeline, X_test)
    reasons = top_reason_codes(contributions, top_k=3)


# ==========================================================================
# 6. DECISION LAYER
# ==========================================================================

header("DECISION LAYER")

scored = test.copy()
scored["purchase_probability"] = probability
if reasons is not None:
    scored["model_reasons"] = reasons.values

scored = create_decision_output(scored, margin_rate=GROSS_MARGIN_RATE)

print("\nCUSTOMER STATE")
print(scored["customer_state"].value_counts().to_string())

print("\nRECOMMENDED ACTION")
print(scored["recommended_action"].value_counts().to_string())

print("\nREQUIRED UPLIFT TO BREAK EVEN, BY ACTION (percentage points)")
print(
    scored[scored["recommended_action"] != "NO_ACTION"]
    .groupby("recommended_action")["required_uplift"]
    .describe()[["count", "25%", "50%", "75%"]]
    .mul([1, 100, 100, 100])
    .round(2)
    .to_string()
)


# ==========================================================================
# 7. CAPACITY AND EXPERIMENT
# ==========================================================================

header("CAPACITY AND EXPERIMENT DESIGN")

budgets = {
    "HIGH_VALUE_WINBACK": 200,      # human time is the scarce resource
    "PAID_RETARGETING": 1500,       # media budget
    "TARGETED_INCENTIVE": 3000,     # margin exposure
    "SERVICE_REVIEW": 150,          # operations capacity
}

allocated = allocate_capacity(scored, budgets=budgets, fatigue_cap=1)
allocated = assign_holdout(allocated)

print("\nBUDGET")
budget = budget_summary(allocated)
if budget.empty:
    print("  no customers selected for contact")
else:
    print(
        budget[
            [
                "action", "channel", "customers", "fixed_cost",
                "expected_incentive_cost", "total_cost",
                "median_required_uplift_pp",
            ]
        ].round(2).to_string(index=False)
    )

print("\nEXPERIMENT CELLS")
design = experiment_design_summary(allocated)
if design.empty:
    print("  no cells")
else:
    print(design.round(3).to_string(index=False))
    print(
        "\nAny cell whose detectable_uplift_pp exceeds its assumed uplift "
        "cannot prove its own case. Either grow the cell or run it longer."
    )


# ==========================================================================
# 8. OUTPUTS
# ==========================================================================

header("WRITING OUTPUTS")

customer_columns = [
    "customer_id", "snapshot_date", "purchase_probability",
    "recency_days", "recency_ratio", "frequency", "monetary", "aov",
    "distinct_products", "recent_orders", "recent_revenue",
    "previous_orders", "previous_revenue", "revenue_trend",
    "return_orders", "return_rate", "lifecycle", "customer_state",
    "expected_organic_revenue", "expected_organic_margin",
]
if "model_reasons" in allocated.columns:
    customer_columns.append("model_reasons")

marketing_columns = [
    "customer_id", "snapshot_date", "purchase_probability", "monetary", "aov",
    "customer_state", "recommended_action", "action_label", "action_objective",
    "required_uplift", "assumed_uplift", "margin_of_safety",
    "expected_net_margin", "priority_band", "contacted", "experiment_arm",
    "execute", "decision_reason",
]

priority_columns = [
    "customer_id", "snapshot_date", "purchase_probability", "priority_score",
    "priority_band", "monetary", "aov", "recency_days", "frequency",
    "customer_state", "recommended_action", "action_label",
    "required_uplift", "margin_of_safety", "experiment_arm", "decision_reason",
]

allocated[customer_columns].sort_values(
    "purchase_probability", ascending=False
).to_csv(OUTPUTS / "customer_scores.csv", index=False)

allocated[marketing_columns].sort_values(
    ["priority_band", "expected_net_margin"], ascending=[False, False]
).to_csv(OUTPUTS / "marketing_actions.csv", index=False)

allocated[allocated["contacted"]].sort_values(
    "priority_score", ascending=False
).head(500)[priority_columns].to_csv(OUTPUTS / "priority_customers.csv", index=False)

importance.to_csv(OUTPUTS / "feature_importance.csv", index=False)
reliability.to_csv(OUTPUTS / "calibration.csv", index=False)
ranking.to_csv(OUTPUTS / "ranking_metrics.csv", index=False)
stability.to_csv(OUTPUTS / "stability_by_snapshot.csv", index=False)
if not budget.empty:
    budget.to_csv(OUTPUTS / "budget_summary.csv", index=False)
if not design.empty:
    design.to_csv(OUTPUTS / "experiment_design.csv", index=False)

for name in sorted(p.name for p in OUTPUTS.glob("*.csv")):
    print(f"  outputs/{name}")

print(
    "\nWhat these files claim: who is likely to buy, and which action clears "
    "its own break-even bar under a stated assumption."
    "\nWhat they do not claim: that any action caused a purchase. "
    "The control arm exists so the next cycle can."
)
