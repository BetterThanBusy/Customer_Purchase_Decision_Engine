"""
Generate a small synthetic file with the Online Retail II schema.

Purpose: verify the pipeline runs end to end before you download the real
150MB Excel file. The numbers are meaningless. The shape is not.

    python scripts/make_sample_data.py
    python notebooks/08_decision_engine.py --data data/raw/sample_retail.xlsx
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "sample_retail.xlsx"

rng = np.random.default_rng(7)

N_CUSTOMERS = 1200
COUNTRIES = ["United Kingdom"] * 12 + ["Germany", "France", "EIRE", "Spain", "Netherlands"]
STOCKCODES = [f"{n}" for n in range(10000, 10400)]

START = pd.Timestamp("2009-12-01")
END = pd.Timestamp("2011-12-09")

rows = []

for customer in range(N_CUSTOMERS):
    customer_id = 12000 + customer
    country = COUNTRIES[rng.integers(len(COUNTRIES))]

    # Heterogeneous customers: a few very frequent, a long tail of one-offs.
    intensity = rng.gamma(shape=1.1, scale=3.0)
    n_orders = max(1, int(rng.poisson(intensity) + rng.integers(0, 2)))

    first = START + pd.Timedelta(days=int(rng.integers(0, 600)))
    gap_mean = float(rng.uniform(20, 140))

    order_date = first
    for order in range(n_orders):
        if order_date > END:
            break

        invoice = f"5{customer:04d}{order:02d}"
        n_lines = int(rng.integers(1, 12))

        for _ in range(n_lines):
            rows.append(
                {
                    "Invoice": invoice,
                    "StockCode": STOCKCODES[rng.integers(len(STOCKCODES))],
                    "Description": "SAMPLE ITEM",
                    "Quantity": int(rng.integers(1, 25)),
                    "InvoiceDate": order_date,
                    "Price": round(float(rng.gamma(2.0, 2.0)) + 0.5, 2),
                    "Customer ID": customer_id,
                    "Country": country,
                }
            )

        # Occasional return, more likely for heavier buyers.
        if rng.random() < 0.06 + 0.02 * min(n_orders, 5):
            rows.append(
                {
                    "Invoice": f"C{invoice}",
                    "StockCode": STOCKCODES[rng.integers(len(STOCKCODES))],
                    "Description": "SAMPLE ITEM",
                    "Quantity": -int(rng.integers(1, 6)),
                    "InvoiceDate": order_date + pd.Timedelta(days=int(rng.integers(1, 20))),
                    "Price": round(float(rng.gamma(2.0, 2.0)) + 0.5, 2),
                    "Customer ID": customer_id,
                    "Country": country,
                }
            )

        order_date = order_date + pd.Timedelta(
            days=max(1, int(rng.exponential(gap_mean)))
        )

    # A slice of customers with no identity, as in the real file.
    if rng.random() < 0.05:
        rows.append(
            {
                "Invoice": f"6{customer:05d}",
                "StockCode": STOCKCODES[0],
                "Description": "SAMPLE ITEM",
                "Quantity": 3,
                "InvoiceDate": first,
                "Price": 2.5,
                "Customer ID": np.nan,
                "Country": country,
            }
        )

frame = pd.DataFrame(rows)

split = frame["InvoiceDate"] < pd.Timestamp("2010-12-09")

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as writer:
    frame[split].to_excel(writer, sheet_name="Year 2009-2010", index=False)
    frame[~split].to_excel(writer, sheet_name="Year 2010-2011", index=False)

print(f"Wrote {len(frame):,} rows to {OUT}")
