"""
09 — Video assets

Regenerates every chart and the companion Excel workbook used to present
this project. Self-contained: it refits the model rather than depending on
which notebooks you ran last.

    python notebooks/09_video_assets.py

Outputs land in outputs/charts/ (PNG, 200 DPI, beige background) and
outputs/video_tables.xlsx.

Chart numbering follows the video running order, so they drop into a
timeline in filename order.
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
    CONTROL_HOLDOUT_SHARE,
    BASELINES,
    OUTPUTS,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    build_gradient_boosting_model,
    build_preprocessor,
    CalibratedModel,
    evaluate_predictions,
    calibration_table,
    stability_by_snapshot,
    permutation_importance_table,
    create_decision_output,
    allocate_capacity,
    budget_summary,
    assign_holdout,
    experiment_design_summary,
)

from src import charts  # noqa: E402
from src.workbook import build_workbook  # noqa: E402

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


CHARTS = OUTPUTS / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

charts.apply_theme()
saved = []


def emit(number, name, fig):
    path = CHARTS / f"{number:02d}_{name}.png"
    charts.save(fig, path)
    saved.append(path)
    print(f"  {path.name}")


# ==========================================================================
# REBUILD EVERYTHING
# ==========================================================================

print("Rebuilding model and decisions...\n")

model_data = build_or_load_snapshots()
train, validation, test = temporal_split(model_data)

X_train, y_train = train[FEATURES], train["purchased_again_30d"]
X_val, y_val = validation[FEATURES], validation["purchased_again_30d"]
X_test, y_test = test[FEATURES], test["purchased_again_30d"]

# Baselines
baseline_scores = []
for name, fn in BASELINES.items():
    score = fn(X_val)
    scaled = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-9)
    metrics = evaluate_predictions(y_val, scaled)
    baseline_scores.append({"model": name, "pr_auc": metrics["pr_auc"]})

# Candidate models
candidates = {
    "logistic_regression": build_logistic_model(),
    "gradient_boosting": build_gradient_boosting_model(),
}

model_scores = []
fitted = {}
for name, pipeline in candidates.items():
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    metrics = evaluate_predictions(y_val, pipeline.predict_proba(X_val)[:, 1])
    model_scores.append({"model": name, "pr_auc": metrics["pr_auc"]})

best_name = max(model_scores, key=lambda r: r["pr_auc"])["model"]
model = CalibratedModel(fitted[best_name]).fit_calibration(X_val, y_val)

probability = model.predict_proba(X_test)

# The calibration trap, measured rather than asserted
balanced = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced",
                                          random_state=42)),
    ]
).fit(X_train, y_train)

mean_balanced = float(balanced.predict_proba(X_val)[:, 1].mean())
mean_unweighted = float(fitted["logistic_regression"].predict_proba(X_val)[:, 1].mean())
observed_rate = float(y_val.mean())

# Evaluation tables
test_metrics = evaluate_predictions(y_test, probability)
reliability = calibration_table(y_test, probability)
stability_frame = stability_by_snapshot(test, probability)
importance = permutation_importance_table(
    lambda frame: model.predict_proba(frame), X_test, y_test, n_repeats=3
)

# Decision layer
scored = test.copy()
scored["purchase_probability"] = probability
decided = create_decision_output(scored, margin_rate=GROSS_MARGIN_RATE)

budgets = {
    "HIGH_VALUE_WINBACK": 200,
    "PAID_RETARGETING": 1500,
    "TARGETED_INCENTIVE": 3000,
    "SERVICE_REVIEW": 150,
}
allocated = assign_holdout(
    allocate_capacity(decided, budgets=budgets, fatigue_cap=1),
    share=CONTROL_HOLDOUT_SHARE,
)
budget = budget_summary(allocated)
design = experiment_design_summary(allocated)

median_aov = float(np.nanmedian(decided["aov"]))


# ==========================================================================
# CHARTS
# ==========================================================================

print("\nCHARTS\n")

emit(1, "score_distribution", charts.score_distribution(decided))
emit(2, "snapshot_concept", charts.snapshot_concept())
emit(3, "base_rate_drift", charts.base_rate_drift(model_data))
emit(4, "baselines_vs_model", charts.baselines_vs_model(baseline_scores, model_scores))
emit(5, "calibration_trap", charts.calibration_trap(observed_rate, mean_balanced, mean_unweighted))
emit(6, "gains_and_lift", charts.gains_and_lift(y_test, probability))
emit(7, "calibration_curve", charts.calibration_curve(reliability))
emit(8, "stability", charts.stability(stability_frame))
emit(9, "feature_importance", charts.feature_importance(importance))
emit(10, "deadweight_curve", charts.deadweight_curve(ACTION_CATALOGUE, median_aov, GROSS_MARGIN_RATE))
emit(11, "customer_state_map", charts.customer_state_map(decided))
emit(12, "state_profile", charts.state_profile(decided))
emit(13, "state_action_matrix", charts.state_action_matrix(decided))
emit(14, "action_economics", charts.action_economics(decided, ACTION_CATALOGUE))
if not budget.empty:
    emit(15, "budget_allocation", charts.budget_allocation(budget))
if not design.empty:
    emit(16, "experiment_power", charts.experiment_power(design, ACTION_CATALOGUE))
emit(17, "decision_framework", charts.decision_framework())


# ==========================================================================
# EXCEL WORKBOOK
# ==========================================================================

print("\nWORKBOOK\n")

workbook_path = build_workbook(
    path=OUTPUTS / "video_tables.xlsx",
    decided=decided,
    allocated=allocated,
    budget=budget,
    design=design,
    reliability=reliability,
    stability_frame=stability_frame,
    importance=importance,
    baseline_scores=baseline_scores,
    model_scores=model_scores,
    test_metrics=test_metrics,
    best_name=best_name,
    median_aov=median_aov,
    observed_rate=observed_rate,
    mean_balanced=mean_balanced,
    mean_unweighted=mean_unweighted,
)
print(f"  {workbook_path.name}")


# ==========================================================================
# TALKING POINTS
# ==========================================================================

top_ten_recall = float(
    np.cumsum(np.asarray(y_test)[np.argsort(-probability)])[
        int(len(y_test) * 0.10) - 1
    ] / np.asarray(y_test).sum()
)

state_counts = decided["customer_state"].value_counts()
action_counts = decided["recommended_action"].value_counts()

print(
    f"""
========================================================================
NUMBERS TO SAY OUT LOUD
========================================================================

OPENING
  {len(decided):,} customer-snapshots scored. Base rate {test_metrics['base_rate']:.1%}.
  Contact the top 10% and you reach {top_ten_recall:.0%} of all buyers.

THE MODEL
  {best_name}, sigmoid-calibrated. ROC-AUC {test_metrics['roc_auc']:.3f},
  PR-AUC {test_metrics['pr_auc']:.3f}.
  Best simple rule to beat: {max(baseline_scores, key=lambda r: r['pr_auc'])['model']}
  at PR-AUC {max(r['pr_auc'] for r in baseline_scores):.3f}.

THE TRAP
  class_weight='balanced' predicts {mean_balanced:.1%} average probability
  against an observed {observed_rate:.1%} — {mean_balanced / max(observed_rate, 1e-9):.1f}x too high.

SEGMENTATION
  {len(state_counts)} customer states. Largest: {state_counts.index[0]}
  at {state_counts.iloc[0]:,} ({state_counts.iloc[0] / len(decided):.0%}).
  ACTIVE_VALUABLE: {state_counts.get('ACTIVE_VALUABLE', 0):,}
  LAPSING_VALUABLE: {state_counts.get('LAPSING_VALUABLE', 0):,}
  SERVICE_RISK: {state_counts.get('SERVICE_RISK', 0):,}

THE DECISION
  NO_ACTION on {action_counts.get('NO_ACTION', 0):,} customers
  ({action_counts.get('NO_ACTION', 0) / len(decided):.0%}). Doing nothing is a decision.
  Discount recommended to {action_counts.get('TARGETED_INCENTIVE', 0):,} customers —
  because the break-even maths refuses to subsidise people who were buying anyway.

THE EXPERIMENT
  {(allocated['experiment_arm'] == 'TREATMENT').sum():,} treated,
  {(allocated['experiment_arm'] == 'CONTROL').sum():,} held back as control.
  That {CONTROL_HOLDOUT_SHARE:.0%} is the only reason next cycle can measure cause.
========================================================================
"""
)