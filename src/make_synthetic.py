"""Generate a file shaped like Online Retail II, for testing the pipeline.

Never debug a pipeline and a dataset at the same time. Run everything on
synthetic data first; when it works end to end, point DATA_PATH at the
real workbook. Any error that appears then is a data problem, not a
code problem.
"""

from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = (["United Kingdom"] * 89 + ["Germany"] * 4 + ["France"] * 3
             + ["EIRE"] * 2 + ["Netherlands", "Spain"])


def generate(n_customers: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days
    rows, invoice_no = [], 500000

    for cid in range(12346, 12346 + n_customers):
        propensity = rng.beta(1.5, 3.0)
        n_orders = max(1, int(rng.poisson(1 + propensity * 26)))
        day = int(rng.integers(0, max(1, span - 40)))
        gap = max(4, int(190 * (1 - propensity) + 12 + rng.normal(0, 10)))
        wealth = float(np.exp(rng.normal(3.3, 1.0)))
        country = COUNTRIES[rng.integers(0, len(COUNTRIES))]
        returner = rng.random() < 0.35

        for _ in range(n_orders):
            if day >= span:
                break
            odate = start + pd.Timedelta(days=day, hours=int(rng.integers(8, 19)))
            invoice_no += 1
            season = 1.5 if odate.month in (11, 12) else 1.0
            lines = max(1, int(rng.poisson(5)))
            inv = str(invoice_no)
            for _ in range(lines):
                rows.append({
                    "Invoice": inv,
                    "StockCode": str(rng.integers(20000, 25500)),
                    "Description": "SYNTHETIC ITEM",
                    "Quantity": int(max(1, rng.poisson(9))),
                    "InvoiceDate": odate,
                    "Price": round(float(abs(rng.normal(wealth / 9, 1.1))) + 0.35, 2),
                    "Customer ID": float(cid),
                    "Country": country,
                })
            if returner and rng.random() < 0.25:
                rows.append({
                    "Invoice": "C" + str(invoice_no),
                    "StockCode": str(rng.integers(20000, 25500)),
                    "Description": "SYNTHETIC ITEM",
                    "Quantity": -int(max(1, rng.poisson(4))),
                    "InvoiceDate": odate + pd.Timedelta(days=int(rng.integers(1, 20))),
                    "Price": round(float(abs(rng.normal(wealth / 9, 1.1))) + 0.35, 2),
                    "Customer ID": float(cid),
                    "Country": country,
                })
            day += max(3, int(rng.normal(gap / season, gap * 0.3)))

    df = pd.DataFrame(rows)
    n = len(df)
    df.loc[rng.choice(n, int(n * 0.22), replace=False), "Customer ID"] = np.nan
    df.loc[df.sample(frac=0.004, random_state=seed).index, "Price"] = 0.0
    return df.sort_values("InvoiceDate").reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthetic_retail.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"{len(df):,} rows -> {out}")
    print(f"{df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    print(f"customers: {df['Customer ID'].nunique():,}")
