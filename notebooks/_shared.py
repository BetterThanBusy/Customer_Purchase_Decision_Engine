"""Shared setup for every analysis script. Import this first."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

DATA = ROOT / "data" / "raw" / "online_retail_II.xlsx"
if not DATA.exists():
    DATA = ROOT / "data" / "raw" / "synthetic_retail.csv"

OUT = ROOT / "outputs"
CACHE = ROOT / "data" / "processed"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def load_or_cache():
    """Load once, cache as parquet. Excel parsing is slow; do it once."""
    from src.engine import clean_transactions, create_orders, load_retail_data
    cp, cr = CACHE / "purchases.parquet", CACHE / "returns.parquet"
    if cp.exists() and cr.exists():
        return pd.read_parquet(cp), pd.read_parquet(cr)
    df = load_retail_data(DATA)
    purchases, returns, audit = clean_transactions(df)
    purchases.to_parquet(cp, index=False)
    returns.to_parquet(cr, index=False)
    print("cached:", audit)
    return purchases, returns
