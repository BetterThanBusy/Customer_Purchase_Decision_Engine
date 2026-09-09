"""
03 — Purchase prediction

Three questions, in this order:

  1. Can a model beat the rule a marketer would use anyway?
  2. Does a more complex model beat a simple one?
  3. Are the probabilities usable as numbers, or only as a ranking?

Question 3 is the one usually skipped, and it is the one that decides
whether a decision layer can be built on top.

Run:  python notebooks/03_purchase_prediction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    FEATURES,
    BASELINES,
    build_or_load_snapshots,
    temporal_split,
    build_logistic_model,
    build_gradient_boosting_model,
    build_preprocessor,
    CalibratedModel,
    evaluate_predictions,
    ranking_metrics,
)

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

pd.set_option("display.width", 160)


model_data = build_or_load_snapshots()
train, validation, test = temporal_split(model_data)

X_train, y_train = train[FEATURES], train["purchased_again_30d"]
X_val, y_val = validation[FEATURES], validation["purchased_again_30d"]

print("TEMPORAL SPLIT")
for name, frame in [("train", train), ("validation", validation), ("test", test)]:
    print(
        f"  {name:<11} {frame['snapshot_date'].min().date()} -> "
        f"{frame['snapshot_date'].max().date()}  "
        f"{len(frame):>7,} rows  base rate {frame['purchased_again_30d'].mean():.4f}"
    )

print(
    """
Validation exists so that model choice and calibration never touch the
test window. Without it, "we picked the best model on test" is just
overfitting with extra steps.

One honest caveat: the same customer appears in train and validation at
different snapshot dates. The split is clean in time, not in customers.
This is standard for repeat-purchase modelling and it matches deployment
(you score customers you have seen before) — but it does mean these
numbers describe performance on known customers, not on new ones.
"""
)


# ==========================================================================
# 1. BASELINES
# ==========================================================================

print("BASELINES (validation)")

rows = []
for name, fn in BASELINES.items():
    score = fn(X_val)
    scaled = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-9)
    metrics = evaluate_predictions(y_val, scaled)
    rows.append({"model": name, "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]})

baselines = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print(baselines.round(4).to_string(index=False))

best_baseline = baselines.iloc[0]
print(
    f"\nBar to clear: {best_baseline['model']} at PR-AUC "
    f"{best_baseline['pr_auc']:.4f}."
    "\nA model that does not beat this is not worth its maintenance cost."
)


# ==========================================================================
# 2. MODELS
# ==========================================================================

print("\nMODEL COMPARISON (validation)")

candidates = {
    "logistic_regression": build_logistic_model(),
    "gradient_boosting": build_gradient_boosting_model(),
}

fitted = {}
results = []

for name, pipeline in candidates.items():
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    metrics = evaluate_predictions(y_val, pipeline.predict_proba(X_val)[:, 1])
    metrics["model"] = name
    results.append(metrics)

comparison = pd.DataFrame(results)[
    ["model", "roc_auc", "pr_auc", "brier_score", "log_loss", "calibration_gap"]
]
print(comparison.round(4).to_string(index=False))

best_name = comparison.sort_values("pr_auc", ascending=False).iloc[0]["model"]
print(f"\nSelected on PR-AUC: {best_name}")

print(
    """
PR-AUC rather than ROC-AUC, because the positive class is a minority and
the job is ranking a small group to the top. ROC-AUC rewards correctly
ordering the vast, uninteresting middle of the distribution — which no
marketer will ever act on.
"""
)


# ==========================================================================
# 3. THE class_weight="balanced" TRAP
# ==========================================================================

print("THE class_weight='balanced' TRAP\n")

weighted = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42
            ),
        ),
    ]
).fit(X_train, y_train)

unweighted = fitted["logistic_regression"]

p_weighted = weighted.predict_proba(X_val)[:, 1]
p_unweighted = unweighted.predict_proba(X_val)[:, 1]

trap = pd.DataFrame(
    [
        evaluate_predictions(y_val, p_weighted),
        evaluate_predictions(y_val, p_unweighted),
    ],
    index=["class_weight=balanced", "unweighted"],
)[["roc_auc", "pr_auc", "brier_score", "mean_predicted", "base_rate", "calibration_gap"]]

print(trap.round(4).to_string())

overstatement = p_weighted.mean() / max(y_val.mean(), 1e-9)
print(
    f"""
Balanced weighting predicts an average purchase probability of
{p_weighted.mean():.3f} against an observed rate of {y_val.mean():.3f}
— roughly {overstatement:.1f}x too high.

Ranking is essentially untouched. Calibration is destroyed.

That is fine if the model output is only ever sorted. It is not fine
here, because notebook 06 multiplies this probability by order value and
margin to decide what to spend. A 4x inflated probability produces a 4x
inflated business case.

Conclusion: handle class imbalance in the METRIC (PR-AUC, lift), not in
the loss function.
"""
)


# ==========================================================================
# 4. CALIBRATION
# ==========================================================================

print("CALIBRATION (fitted on validation)\n")

for method in ["sigmoid", "isotonic"]:
    model = CalibratedModel(fitted[best_name], method=method).fit_calibration(
        X_val, y_val
    )
    raw = model.predict_proba_raw(X_val)
    cal = model.predict_proba(X_val)

    print(
        f"  {method:<10} brier {evaluate_predictions(y_val, raw)['brier_score']:.5f}"
        f" -> {evaluate_predictions(y_val, cal)['brier_score']:.5f}"
        f"   pr_auc {evaluate_predictions(y_val, raw)['pr_auc']:.4f}"
        f" -> {evaluate_predictions(y_val, cal)['pr_auc']:.4f}"
    )

print(
    """
  sigmoid   strictly monotone, so ranking metrics are provably identical.
            Two parameters, so it barely overfits a small window.
  isotonic  more flexible, but only weakly monotone — it creates ties,
            and ties can move PR-AUC. Fitted and measured on the same
            window it will look better than it is.

Default: sigmoid. Switch to isotonic only with a large, separate
calibration set.
"""
)

print("RANKING ON VALIDATION")
final = CalibratedModel(fitted[best_name]).fit_calibration(X_val, y_val)
print(ranking_metrics(y_val, final.predict_proba(X_val)).round(4).to_string(index=False))
