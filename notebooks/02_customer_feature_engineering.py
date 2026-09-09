"""
02 — Customer feature engineering

One customer produces one row per snapshot date, describing them as they
were on that date. Twenty snapshots means the model learns "what does a
customer about to buy look like" across twenty different months, not just
one arbitrary end-of-data moment.

The rule that governs everything here:

    Features come from strictly before the cutoff.
    The target comes from strictly after it.
    Nothing crosses.

Run:  python notebooks/02_customer_feature_engineering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    HORIZON_DAYS,
    NUMERIC_FEATURES,
    SNAPSHOT_DATES,
    resolve_data_path,
    load_retail_data,
    clean_transactions,
    create_orders,
    create_snapshot,
    build_customer_features,
    add_target,
    build_or_load_snapshots,
)

pd.set_option("display.width", 160)


# ==========================================================================
# ONE SNAPSHOT, INSPECTED
# ==========================================================================

path = resolve_data_path()
raw = load_retail_data(path)
purchases, returns = clean_transactions(raw)
orders = create_orders(purchases)

cutoff = "2011-06-01"
print(f"Inspecting a single snapshot at {cutoff}\n")

past_orders, future_orders, past_returns, past_lines = create_snapshot(
    orders, returns, purchases, cutoff, HORIZON_DAYS
)

print(f"  Orders before cutoff        : {len(past_orders):,}")
print(f"  Orders in the {HORIZON_DAYS}-day horizon : {len(future_orders):,}")
print(f"  Return lines before cutoff  : {len(past_returns):,}")

customers = build_customer_features(past_orders, past_returns, past_lines, cutoff)
customers = add_target(customers, future_orders)

print(f"\n  Customers in snapshot       : {len(customers):,}")
print(f"  Base rate                   : {customers['purchased_again_30d'].mean():.4f}")


# ==========================================================================
# LEAKAGE CHECKS
# ==========================================================================
# A leakage check is worth more than a feature. Run it every time the
# feature set changes.

print("\nLEAKAGE CHECKS")

cutoff_ts = pd.Timestamp(cutoff)

checks = [
    (
        "no feature-side order on or after cutoff",
        (past_orders["order_date"] < cutoff_ts).all(),
    ),
    (
        "no target-side order before cutoff",
        (future_orders["order_date"] >= cutoff_ts).all(),
    ),
    (
        "target-side orders stay inside the horizon",
        (
            future_orders["order_date"]
            < cutoff_ts + pd.Timedelta(days=HORIZON_DAYS)
        ).all(),
    ),
    (
        "recency is never negative",
        (customers["recency_days"] >= 0).all(),
    ),
    (
        "tenure is never shorter than recency",
        (customers["tenure_days"] >= customers["recency_days"]).all(),
    ),
]

for label, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

assert all(passed for _, passed in checks), "leakage check failed"


# ==========================================================================
# WHAT EACH FEATURE IS FOR
# ==========================================================================

print(
    """
FEATURE GROUPS

  RFM core          recency_days, frequency, monetary, aov
                    The baseline every retailer already has.

  breadth           products, distinct_products, total_items
                    Distinct SKUs, not order count. A customer who buys
                    forty different things is a different animal from one
                    who reorders the same item forty times.

  rhythm            interpurchase_mean_days, recency_ratio
                    recency_ratio = days since last order / this customer's
                    own average gap. This is the feature that separates a
                    genuinely lapsing customer from a naturally infrequent
                    one. 60 days silent is alarming for a weekly buyer and
                    completely normal for a quarterly buyer. Raw recency
                    cannot tell those apart. This can.

  momentum          recent vs previous 30 days, order_velocity,
                    revenue_velocity, revenue_trend
                    Direction of travel, not just position.

  friction          return_orders, return_rate
                    Kept deliberately. See notebook 05 before drawing any
                    conclusion from it.
"""
)


# ==========================================================================
# FULL DATASET
# ==========================================================================

print("Building the full snapshot dataset...\n")
model_data = build_or_load_snapshots(rebuild=True, snapshot_dates=SNAPSHOT_DATES)

print(f"\nRows      : {len(model_data):,}")
print(f"Customers : {model_data['customer_id'].nunique():,}")
print(f"Snapshots : {model_data['snapshot_date'].nunique()}")

print("\nBASE RATE DRIFT OVER TIME")
drift = (
    model_data.groupby("snapshot_date")
    .agg(customers=("customer_id", "size"), base_rate=("purchased_again_30d", "mean"))
    .round(4)
)
print(drift.to_string())

print(
    """
The base rate is not constant. That is the reason for a temporal split
rather than a random one: a random split lets the model learn from a
future period to predict a past one, and quietly inflates every metric.
"""
)


# ==========================================================================
# SEPARATION: WHICH FEATURES ACTUALLY DIFFER BY OUTCOME
# ==========================================================================

print("FEATURE SEPARATION (median by outcome)")

separation = (
    model_data.groupby("purchased_again_30d")[NUMERIC_FEATURES]
    .median()
    .T
    .rename(columns={0: "no_repeat", 1: "repeat"})
)
separation["ratio"] = (
    separation["repeat"] / separation["no_repeat"].replace(0, np.nan)
)
print(separation.sort_values("ratio", ascending=False).round(3).to_string())

print(
    """
Read this as a hypothesis list, not a result. A median difference is a
univariate association. It says nothing about what survives once the
other twenty features are in the room, which is what notebook 03 tests.
"""
)

print("\nMISSINGNESS")
missing = model_data[NUMERIC_FEATURES].isna().mean()
print(missing[missing > 0].round(4).to_string() or "  none")

print(
    """
revenue_trend and recency_ratio are missing by construction, not by
accident: a customer with no prior-period revenue has no trend, and a
one-order customer has no average gap. Imputing the median would invent a
rhythm they have never demonstrated. The gradient boosting model handles
these natively; the linear model imputes and the effect is absorbed into
the frequency features.
"""
)
