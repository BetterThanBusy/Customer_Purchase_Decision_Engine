"""
01 — Business problem and data

The question this project answers:

    Given everything a retailer knows about a customer today, which
    customers deserve which marketing attention, and what would that
    attention have to achieve to be worth its cost?

Note what that is not. It is not "who will buy". "Who will buy" is an
input to the decision, not the decision itself. Most repeat-purchase
projects stop at the input and call it a strategy.

Run:  python notebooks/01_business_problem_and_data.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.engine import (  # noqa: E402
    HORIZON_DAYS,
    resolve_data_path,
    load_retail_data,
    clean_transactions,
    create_orders,
    DATA_PROCESSED,
)

pd.set_option("display.width", 140)


# ==========================================================================
# THE TARGET, DEFINED BEFORE ANY MODELLING
# ==========================================================================

print(
    f"""
TARGET DEFINITION
-----------------
For a customer observed at a cutoff date T:

    purchased_again_30d = 1 if they place at least one valid order in
                          [T, T + {HORIZON_DAYS} days), else 0

Why {HORIZON_DAYS} days and not 90 or 180?

  * It matches a marketing cycle. A campaign planned this month is
    judged this month.
  * A longer horizon inflates the base rate and makes the model look
    better while making it less useful — almost everyone buys eventually.
  * Shorter horizons are noisier than the intervention lag allows.

The horizon is a choice with consequences, not a default. Changing it
changes the base rate, the model, and every decision downstream.
"""
)


# ==========================================================================
# LOAD AND AUDIT
# ==========================================================================

path = resolve_data_path()
print(f"Loading {path.name}...")

raw = load_retail_data(path)

print(f"\nRaw rows        : {len(raw):,}")
print(f"Date range      : {raw['invoicedate'].min()} -> {raw['invoicedate'].max()}")
print(f"Distinct SKUs   : {raw['stockcode'].nunique():,}")
print(f"Countries       : {raw['country'].nunique()}")

print("\nDATA QUALITY")
audit = pd.DataFrame(
    {
        "issue": [
            "missing customer_id",
            "cancellations (invoice starts with C)",
            "negative quantity",
            "zero or negative price",
        ],
        "rows": [
            int(raw["customer_id"].isna().sum()),
            int(raw["is_cancellation"].sum()),
            int((raw["quantity"] < 0).sum()),
            int((raw["price"] <= 0).sum()),
        ],
    }
)
audit["share"] = (audit["rows"] / len(raw)).round(4)
print(audit.to_string(index=False))

print(
    """
Decisions taken on these rows:

  missing customer_id  -> dropped. No customer, no customer-level model.
                          This silently removes a large chunk of revenue,
                          so any revenue figure here is understated.
  cancellations        -> KEPT, as a separate signal. A return means the
                          customer engaged, received, and reacted. That is
                          behavioural information, not noise.
  price <= 0           -> dropped. Adjustments and admin lines, not demand.
"""
)


# ==========================================================================
# CLEAN AND SUMMARISE
# ==========================================================================

purchases, returns = clean_transactions(raw)
orders = create_orders(purchases)

print("AFTER CLEANING")
print(f"  Purchase lines  : {len(purchases):,}")
print(f"  Return lines    : {len(returns):,}")
print(f"  Orders          : {len(orders):,}")
print(f"  Customers       : {orders['customer_id'].nunique():,}")

per_customer = orders.groupby("customer_id").agg(
    orders=("invoice", "nunique"),
    revenue=("order_value", "sum"),
)

print("\nORDERS PER CUSTOMER")
print(per_customer["orders"].describe().round(2).to_string())

one_time = (per_customer["orders"] == 1).mean()
print(f"\n  One-order customers: {one_time:.1%} of the base")

top_decile_share = (
    per_customer["revenue"].nlargest(max(1, int(len(per_customer) * 0.1))).sum()
    / per_customer["revenue"].sum()
)
print(f"  Top 10% of customers hold {top_decile_share:.1%} of revenue")

print(
    """
Both numbers shape the strategy:

  * A large one-order population means a large group with almost no
    behavioural history. Scoring them low and calling them churned is a
    modelling artefact, not a finding. They get their own treatment.
  * Concentrated revenue means uniform treatment is wasteful. The value
    spread is the reason a decision layer exists at all.
"""
)


# ==========================================================================
# WHAT THIS DATASET CANNOT DO
# ==========================================================================

print(
    """
WHAT THE DATA CONTAINS
  customer, invoice, date, product, quantity, price, country

WHAT IT DOES NOT CONTAIN
  was this customer emailed?      which offer did they receive?
  did they open or click?         was there a control group?
  what did the contact cost?      what margin did the order carry?

CONSEQUENCE — and this is the honest boundary of the project:

  CAN estimate    who is likely to purchase in the next 30 days
  CAN infer       who warrants different marketing attention
  CAN compute     what an intervention must achieve to pay for itself
  CANNOT prove    that any intervention caused a purchase

The last line is not a weakness to hide. It defines the roadmap:
this project ends with a designed experiment, and the experiment is
what turns the next cycle into causal evidence.
"""
)

orders.to_csv(DATA_PROCESSED / "orders.csv", index=False)
print(f"Saved orders to {DATA_PROCESSED / 'orders.csv'}")
