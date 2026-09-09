"""
Customer Purchase Decision Engine — core library.

The notebooks explain the analysis. This module holds the logic they share.

Layers, in order:

    DATA        raw transactions -> clean purchase lines + return lines
    SNAPSHOT    point-in-time customer views, strictly no future information
    MODEL       P(purchase in next 30 days), calibrated
    STATE       what kind of customer is this, right now
    ECONOMICS   what would an intervention have to achieve to pay for itself
    DECISION    which action, or none
    CAPACITY    who actually gets contacted, given a finite budget
    EXPERIMENT  who is held out, so next cycle can measure causality

Nothing in here estimates causal uplift. The dataset cannot support that.
What it does instead is make the assumption explicit and solvable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)


# ==========================================================================
# 1. CONFIGURATION
# ==========================================================================

HORIZON_DAYS = 30
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "aov",
    "total_items",
    "products",
    "distinct_products",
    "tenure_days",
    "orders_per_month",
    "recent_orders",
    "recent_revenue",
    "previous_orders",
    "previous_revenue",
    "order_velocity",
    "revenue_velocity",
    "revenue_trend",
    "return_orders",
    "return_rate",
    "interpurchase_mean_days",
    "recency_ratio",
]

CATEGORICAL_FEATURES = [
    "country",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Business assumptions. These are inputs, not findings.
# Change them here rather than burying them in a notebook.
GROSS_MARGIN_RATE = 0.35          # [Guessing] typical online retail gross margin
CONTROL_HOLDOUT_SHARE = 0.10      # share of each treated cell reserved as control


# ==========================================================================
# 2. LOAD
# ==========================================================================

def load_retail_data(path):
    """Load every sheet of Online Retail II into one frame."""
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df["invoice"] = df["invoice"].astype(str)
    df["stockcode"] = df["stockcode"].astype(str)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    df["is_cancellation"] = df["invoice"].str.startswith("C")
    df["line_value"] = df["quantity"] * df["price"]

    return df


# ==========================================================================
# 3. CLEAN
# ==========================================================================

def clean_transactions(df):
    """
    Split into genuine purchase lines and return/cancellation lines.

    Returns are kept, not discarded. A return is behaviour: it tells you
    the customer engaged, received something, and reacted to it.
    """
    data = df.copy()
    data = data[data["customer_id"].notna()].copy()
    data = data[data["invoicedate"].notna()].copy()
    data = data[data["price"] > 0].copy()

    data["customer_id"] = data["customer_id"].astype(int).astype(str)

    purchases = data[
        (~data["is_cancellation"])
        & (data["quantity"] > 0)
        & (data["line_value"] > 0)
    ].copy()

    returns = data[
        data["is_cancellation"] | (data["quantity"] < 0)
    ].copy()

    return purchases, returns


def create_orders(purchases):
    """Collapse transaction lines into one row per customer-invoice."""
    orders = (
        purchases
        .groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            order_date=("invoicedate", "min"),
            order_value=("line_value", "sum"),
            order_items=("quantity", "sum"),
            products_in_order=("stockcode", "nunique"),
            country=("country", "first"),
        )
    )
    return orders[orders["order_value"] > 0].copy()


# ==========================================================================
# 4. SNAPSHOTS
# ==========================================================================

def create_snapshot(orders, returns, purchases, cutoff_date, horizon_days=HORIZON_DAYS):
    """
    Cut the world at cutoff_date.

    Anything before it can become a feature.
    Anything in [cutoff, cutoff + horizon) can only become the target.
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)

    past_orders = orders[orders["order_date"] < cutoff_date].copy()
    future_orders = orders[
        (orders["order_date"] >= cutoff_date)
        & (orders["order_date"] < horizon_end)
    ].copy()
    past_returns = returns[returns["invoicedate"] < cutoff_date].copy()
    past_lines = purchases[purchases["invoicedate"] < cutoff_date].copy()

    return past_orders, future_orders, past_returns, past_lines


def build_customer_features(past_orders, past_returns, past_lines, cutoff_date):
    """One row per customer, describing them as at cutoff_date."""
    cutoff_date = pd.Timestamp(cutoff_date)

    customer = (
        past_orders
        .groupby("customer_id")
        .agg(
            first_order=("order_date", "min"),
            last_order=("order_date", "max"),
            frequency=("invoice", "nunique"),
            monetary=("order_value", "sum"),
            total_items=("order_items", "sum"),
            products=("products_in_order", "sum"),
            country=("country", "first"),
        )
    )

    # Product breadth from line data: how many *different* things they buy.
    distinct_products = (
        past_lines
        .groupby("customer_id")
        .agg(distinct_products=("stockcode", "nunique"))
    )
    customer = customer.join(distinct_products, how="left")
    customer["distinct_products"] = customer["distinct_products"].fillna(0)

    # --- timing -----------------------------------------------------------
    customer["recency_days"] = (cutoff_date - customer["last_order"]).dt.days
    customer["tenure_days"] = (cutoff_date - customer["first_order"]).dt.days

    customer["aov"] = customer["monetary"] / customer["frequency"].replace(0, np.nan)
    customer["orders_per_month"] = customer["frequency"] / np.maximum(
        customer["tenure_days"] / 30.0, 1.0
    )

    # Mean gap between orders, and how overdue they are relative to it.
    # recency_ratio > 1 means this customer is quieter than their own norm.
    customer["interpurchase_mean_days"] = np.where(
        customer["frequency"] > 1,
        customer["tenure_days"] / np.maximum(customer["frequency"] - 1, 1),
        np.nan,
    )
    customer["recency_ratio"] = (
        customer["recency_days"] / customer["interpurchase_mean_days"]
    )

    # --- momentum ---------------------------------------------------------
    recent_start = cutoff_date - pd.Timedelta(days=30)
    previous_start = cutoff_date - pd.Timedelta(days=60)

    recent = past_orders[past_orders["order_date"] >= recent_start]
    previous = past_orders[
        (past_orders["order_date"] >= previous_start)
        & (past_orders["order_date"] < recent_start)
    ]

    recent_stats = recent.groupby("customer_id").agg(
        recent_orders=("invoice", "nunique"),
        recent_revenue=("order_value", "sum"),
    )
    previous_stats = previous.groupby("customer_id").agg(
        previous_orders=("invoice", "nunique"),
        previous_revenue=("order_value", "sum"),
    )

    customer = customer.join(recent_stats, how="left").join(previous_stats, how="left")

    activity_columns = [
        "recent_orders",
        "recent_revenue",
        "previous_orders",
        "previous_revenue",
    ]
    customer[activity_columns] = customer[activity_columns].fillna(0)

    customer["order_velocity"] = customer["recent_orders"] - customer["previous_orders"]
    customer["revenue_velocity"] = (
        customer["recent_revenue"] - customer["previous_revenue"]
    )
    customer["revenue_trend"] = np.where(
        customer["previous_revenue"] > 0,
        customer["recent_revenue"] / customer["previous_revenue"],
        np.nan,
    )

    # --- returns ----------------------------------------------------------
    return_stats = past_returns.groupby("customer_id").agg(
        return_orders=("invoice", "nunique")
    )
    customer = customer.join(return_stats, how="left")
    customer["return_orders"] = customer["return_orders"].fillna(0)
    customer["return_rate"] = (
        customer["return_orders"] / (customer["frequency"] + customer["return_orders"])
    ).fillna(0)

    # --- lifecycle label (descriptive only, not a decision) ---------------
    customer["lifecycle"] = np.select(
        [
            customer["frequency"] == 1,
            customer["recency_days"] <= 30,
            customer["recency_days"] <= 90,
            customer["recency_days"] <= 180,
        ],
        ["NEW", "ACTIVE", "COOLING", "AT_RISK"],
        default="DORMANT",
    )

    return customer


def add_target(customer, future_orders):
    repeat_customers = set(future_orders.loc[future_orders["order_value"] > 0, "customer_id"])
    customer["purchased_again_30d"] = customer.index.isin(repeat_customers).astype(int)
    return customer


def build_snapshot_dataset(orders, returns, purchases, snapshot_dates,
                           horizon_days=HORIZON_DAYS, verbose=True):
    """Stack many point-in-time snapshots into one modelling table."""
    frames = []

    for date in snapshot_dates:
        past_orders, future_orders, past_returns, past_lines = create_snapshot(
            orders, returns, purchases, date, horizon_days
        )

        if past_orders.empty:
            continue

        customers = build_customer_features(
            past_orders, past_returns, past_lines, date
        )
        customers = add_target(customers, future_orders)
        customers["snapshot_date"] = pd.Timestamp(date)

        frames.append(customers.reset_index())

        if verbose:
            print(
                f"  {date}: {len(customers):>6,} customers  "
                f"base rate {customers['purchased_again_30d'].mean():.3f}"
            )

    return pd.concat(frames, ignore_index=True)


def filter_modelling_rows(model_data):
    """Drop rows that cannot form a valid customer view."""
    out = model_data.copy()
    out = out[out["monetary"] > 0]
    out = out[out["frequency"] > 0]
    return out.copy()


# ==========================================================================
# 5. BASELINES
# ==========================================================================
# A model is only worth deploying if it beats the rule a marketer would
# have used anyway. These are those rules.

def baseline_recency(X):
    """More recent = more likely. The single strongest naive rule."""
    return -X["recency_days"].astype(float).to_numpy()


def baseline_frequency(X):
    return X["frequency"].astype(float).to_numpy()


def baseline_rfm(X):
    """Classic RFM: equal-weight decile score across R, F, M."""
    r = (-X["recency_days"]).rank(pct=True)
    f = X["frequency"].rank(pct=True)
    m = X["monetary"].rank(pct=True)
    return ((r + f + m) / 3).to_numpy()


BASELINES = {
    "recency_only": baseline_recency,
    "frequency_only": baseline_frequency,
    "rfm_score": baseline_rfm,
}


# ==========================================================================
# 6. MODELS
# ==========================================================================

def build_preprocessor():
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_logistic_model():
    """
    Deliberately NOT class_weight='balanced'.

    Balanced weighting inflates predicted probabilities. It barely helps
    ranking and it wrecks calibration — and the decision layer downstream
    multiplies these probabilities by money. Handle imbalance by evaluating
    with PR-AUC and lift, not by distorting the output.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gradient_boosting_model():
    """Non-linear challenger. Handles NaN natively, so no imputation."""
    categorical_index = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]
    return Pipeline(
        steps=[
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    categorical_features=categorical_index,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            )
        ]
    )


def build_model():
    """Default model used by the runner."""
    return build_logistic_model()


class CalibratedModel:
    """
    Wrap a fitted pipeline with an isotonic calibrator fitted on a
    held-out validation window.

    Two methods, and the difference matters more than it looks:

      "sigmoid"   Platt scaling. A one-parameter logistic fit on the
                  log-odds. STRICTLY monotone, so ROC-AUC, PR-AUC and
                  lift are provably unchanged. Two parameters, so it
                  barely overfits a small validation window. Default.

      "isotonic"  Non-parametric, more flexible, usually lower Brier
                  when you have plenty of validation data. But it is only
                  WEAKLY monotone: it flattens regions into ties, and
                  those ties can nudge PR-AUC down. It is not free.

    Either way, what changes is whether "0.30" actually means 30%. That
    matters here because the decision layer spends money against it.
    """

    def __init__(self, pipeline, method="sigmoid"):
        self.pipeline = pipeline
        self.method = method
        self.calibrator = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit_calibration(self, X_val, y_val):
        raw = self.pipeline.predict_proba(X_val)[:, 1]
        y_val = np.asarray(y_val, dtype=float)

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(raw, y_val)
        else:
            self.calibrator = LogisticRegression().fit(
                self._logit(raw).reshape(-1, 1), y_val
            )

        return self

    def predict_proba_raw(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        raw = self.predict_proba_raw(X)

        if self.calibrator is None:
            return raw

        if self.method == "isotonic":
            return self.calibrator.predict(raw)

        return self.calibrator.predict_proba(
            self._logit(raw).reshape(-1, 1)
        )[:, 1]


# ==========================================================================
# 7. EVALUATION
# ==========================================================================

def evaluate_predictions(y_true, probability):
    y_true = np.asarray(y_true)
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)

    return {
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "mean_predicted": float(probability.mean()),
        "calibration_gap": float(probability.mean() - y_true.mean()),
    }


def ranking_metrics(y_true, probability, percentages=(0.01, 0.05, 0.10, 0.20, 0.50)):
    """
    The metric a marketer actually uses: if I contact the top X%,
    how many buyers do I capture, and how much better is that than random.
    """
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    base_rate = y_true.mean()
    total_buyers = y_true.sum()
    order = np.argsort(-probability)

    rows = []
    for pct in percentages:
        n = max(1, int(len(y_true) * pct))
        top_idx = order[:n]
        found = int(y_true[top_idx].sum())
        precision = found / n

        rows.append(
            {
                "population_percent": pct,
                "customers": n,
                "buyers_found": found,
                "precision": precision,
                "recall": found / total_buyers if total_buyers else np.nan,
                "lift": precision / base_rate if base_rate > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def calibration_table(y_true, probability, bins=10):
    """Reliability table: predicted vs observed, by probability decile."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(probability)}
    )
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)

    table = (
        frame.groupby("bin")
        .agg(
            customers=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index()
    )
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def stability_by_snapshot(test_frame, probability, target="purchased_again_30d"):
    """Does performance hold in every test period, or only on average?"""
    frame = test_frame.copy()
    frame["_p"] = probability

    rows = []
    for date, group in frame.groupby("snapshot_date"):
        if group[target].nunique() < 2:
            continue
        rows.append(
            {
                "snapshot_date": date,
                "customers": len(group),
                "base_rate": group[target].mean(),
                "roc_auc": roc_auc_score(group[target], group["_p"]),
                "pr_auc": average_precision_score(group[target], group["_p"]),
                "brier_score": brier_score_loss(group[target], group["_p"]),
            }
        )

    return pd.DataFrame(rows)


# ==========================================================================
# 8. EXPLANATION
# ==========================================================================

def permutation_importance_table(predict_fn, X, y, scorer=average_precision_score,
                                 n_repeats=5, random_state=RANDOM_STATE):
    """
    Model-agnostic permutation importance.

    Scored on average precision, because that is the metric that reflects
    how the model is actually used: ranking a minority class.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    baseline = scorer(y, predict_fn(X))
    rows = []

    for column in X.columns:
        drops = []
        original = X[column].to_numpy(copy=True)

        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(original)
            drops.append(baseline - scorer(y, predict_fn(shuffled)))

        rows.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def linear_contributions(pipeline, X):
    """
    Per-customer contribution of each feature, for a linear pipeline.

    contribution = transformed_value * coefficient

    This is an exact decomposition of the log-odds — not an approximation
    and not a causal claim. It answers "what pushed this score up or down",
    nothing more.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    matrix = preprocessor.transform(X)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    return pd.DataFrame(matrix * coefficients, columns=names, index=X.index)


def top_reason_codes(contributions, top_k=3):
    """Turn the contribution matrix into a short human-readable reason string."""
    values = contributions.to_numpy()
    names = np.array(contributions.columns)

    order = np.argsort(-np.abs(values), axis=1)[:, :top_k]

    reasons = []
    for row_index, cols in enumerate(order):
        parts = []
        for col in cols:
            raw_name = names[col].split("__", 1)[-1]
            direction = "+" if values[row_index, col] > 0 else "-"
            parts.append(f"{direction}{raw_name}")
        reasons.append(" | ".join(parts))

    return pd.Series(reasons, index=contributions.index, name="model_reasons")


# ==========================================================================
# 9. CUSTOMER STATE
# ==========================================================================
# State is not an action. State is a description of the situation.
# Separating them is the whole point: the same state can justify a
# different action once cost, margin or capacity changes.

CUSTOMER_STATES = [
    "SERVICE_RISK",
    "NEW_UNPROVEN",
    "ACTIVE_VALUABLE",
    "ACTIVE_STANDARD",
    "COOLING_VALUABLE",
    "COOLING_STANDARD",
    "LAPSING_VALUABLE",
    "LAPSING_STANDARD",
    "DORMANT",
]


def assign_customer_state(row):
    """
    Precedence matters and is deliberate:

      1. A service problem is not a marketing problem. It outranks everything.
      2. A one-order customer has no behavioural history. Do not call them
         churned — you have no evidence either way.
      3. After that: split by how overdue they are relative to their own
         rhythm, then by whether they are worth spending real money on.
    """
    if row["return_rate"] >= 0.40 and row["return_orders"] >= 2:
        return "SERVICE_RISK"

    if row["frequency"] == 1:
        return "NEW_UNPROVEN"

    valuable = row["monetary"] >= row["value_threshold"]

    # How overdue is this customer against their own normal gap?
    ratio = row.get("recency_ratio", np.nan)
    if pd.isna(ratio):
        ratio = row["recency_days"] / 45.0

    if row["recency_days"] <= 30 or ratio <= 1.0:
        return "ACTIVE_VALUABLE" if valuable else "ACTIVE_STANDARD"

    if ratio <= 2.5 and row["recency_days"] <= 90:
        return "COOLING_VALUABLE" if valuable else "COOLING_STANDARD"

    if row["recency_days"] <= 180:
        return "LAPSING_VALUABLE" if valuable else "LAPSING_STANDARD"

    return "DORMANT" if not valuable else "LAPSING_VALUABLE"


# ==========================================================================
# 10. ACTION CATALOGUE AND ECONOMICS
# ==========================================================================

@dataclass(frozen=True)
class Action:
    """
    A marketing action, described by what it costs rather than what it is
    called.

    contact_cost    fixed cost per customer contacted (production, media,
                    agency, human time), in currency units
    incentive_rate  discount or credit given, as a share of AOV. This is
                    paid to EVERYONE who converts — including the people
                    who would have bought anyway.
    assumed_uplift  [Guessing] plausible incremental conversion, in
                    percentage points. This is a prior, not a measurement.
                    Replace it with experiment results as they arrive.
    """
    key: str
    label: str
    channel: str
    contact_cost: float
    incentive_rate: float
    assumed_uplift: float
    objective: str
    eligible_states: tuple = field(default=())
    # Some actions are not campaigns and must not be judged on marketing ROI.
    # A service review has no uplift by design — its return is fewer returns
    # and a retained customer, neither of which this model measures. Sending
    # it through the break-even test would always reject it, which is how a
    # service problem quietly becomes a marketing decision.
    bypass_economics: bool = False


ACTION_CATALOGUE = {
    a.key: a
    for a in [
        Action(
            key="NO_ACTION",
            label="No action",
            channel="none",
            contact_cost=0.0,
            incentive_rate=0.0,
            assumed_uplift=0.0,
            objective="Spend nothing. Let organic demand happen.",
        ),
        Action(
            key="OWNED_RECOMMENDATION",
            label="Owned recommendation",
            channel="email/app",
            contact_cost=0.02,
            incentive_rate=0.0,
            assumed_uplift=0.010,
            objective="Expand basket and category without buying demand.",
            eligible_states=("ACTIVE_VALUABLE", "ACTIVE_STANDARD", "COOLING_STANDARD"),
        ),
        Action(
            key="LOYALTY_CROSS_SELL",
            label="Loyalty / early access",
            channel="email + curation",
            contact_cost=0.12,
            incentive_rate=0.0,
            assumed_uplift=0.020,
            objective="Raise basket value on customers already likely to buy.",
            eligible_states=("ACTIVE_VALUABLE",),
        ),
        Action(
            key="SECOND_PURCHASE_JOURNEY",
            label="Second-purchase journey",
            channel="email sequence",
            contact_cost=0.10,
            incentive_rate=0.05,
            assumed_uplift=0.030,
            objective="Convert a first order into a habit, not a one-off.",
            eligible_states=("NEW_UNPROVEN",),
        ),
        Action(
            key="REACTIVATION_SEQUENCE",
            label="Reactivation sequence",
            channel="email sequence",
            contact_cost=0.15,
            incentive_rate=0.0,
            assumed_uplift=0.025,
            objective="Re-engage before cooling becomes dormancy. No discount yet.",
            eligible_states=("COOLING_VALUABLE", "COOLING_STANDARD"),
        ),
        Action(
            key="TARGETED_INCENTIVE",
            label="Targeted incentive",
            channel="email + offer",
            contact_cost=0.15,
            incentive_rate=0.15,
            assumed_uplift=0.045,
            objective="Buy back demand where owned channels have failed.",
            eligible_states=("COOLING_VALUABLE", "LAPSING_STANDARD", "LAPSING_VALUABLE"),
        ),
        Action(
            key="HIGH_VALUE_WINBACK",
            label="High-value win-back",
            channel="paid + human + offer",
            contact_cost=1.50,
            incentive_rate=0.20,
            assumed_uplift=0.070,
            objective="Recover customers whose historic value justifies real spend.",
            eligible_states=("LAPSING_VALUABLE",),
        ),
        Action(
            key="PAID_RETARGETING",
            label="Paid retargeting",
            channel="paid media",
            contact_cost=0.90,
            incentive_rate=0.0,
            assumed_uplift=0.030,
            objective="Reach customers outside owned channels.",
            eligible_states=("LAPSING_VALUABLE", "COOLING_VALUABLE"),
        ),
        Action(
            key="SERVICE_REVIEW",
            label="Service / product-fit review",
            channel="operations",
            contact_cost=2.50,
            incentive_rate=0.0,
            assumed_uplift=0.0,
            objective="Fix the experience. This is not a campaign.",
            eligible_states=("SERVICE_RISK",),
            bypass_economics=True,
        ),
        Action(
            key="LOW_COST_NURTURE",
            label="Low-cost nurture",
            channel="owned only",
            contact_cost=0.02,
            incentive_rate=0.0,
            assumed_uplift=0.005,
            objective="Stay present at near-zero cost. Do not invest further.",
            eligible_states=("DORMANT", "LAPSING_STANDARD", "ACTIVE_STANDARD"),
        ),
    ]
}


def required_uplift(probability, aov, action, margin_rate=GROSS_MARGIN_RATE):
    """
    The break-even question, solved.

    Let d = incremental conversion caused by the action (in probability
    points). Contacting a customer is worth it when:

        d * aov * margin  >  contact_cost + (P + d) * aov * incentive_rate
        ------------------     ------------  -------------------------------
        incremental margin     fixed cost    incentive paid to ALL converters
                                             including the P who'd have
                                             bought anyway

    Rearranged:

        d_breakeven = (contact_cost + P * aov * incentive_rate)
                      / (aov * (margin - incentive_rate))

    Read it out loud and the strategy writes itself:

      * incentive_rate = 0  ->  break-even uplift is tiny. Owned channels
        almost always clear the bar.
      * high P + an incentive  ->  the numerator grows with P. The more
        certain a customer is to buy, the more expensive it is to discount
        them. That is the deadweight cost, quantified.
      * incentive_rate >= margin  ->  denominator goes non-positive. No
        uplift can ever pay for it. Impossible, not merely unprofitable.
    """
    probability = np.asarray(probability, dtype=float)
    aov = np.asarray(aov, dtype=float)

    denominator = aov * (margin_rate - action.incentive_rate)
    numerator = action.contact_cost + probability * aov * action.incentive_rate

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator > 0, numerator / denominator, np.inf)

    return result


def evaluate_action(scored, action, margin_rate=GROSS_MARGIN_RATE):
    """
    Score one action against every customer.

    margin_of_safety = assumed uplift / required uplift.

      > 1  the action clears its own bar under our assumption
      < 1  it does not, and no amount of enthusiasm changes that
    """
    probability = scored["purchase_probability"].to_numpy()
    aov = scored["aov"].to_numpy()

    needed = required_uplift(probability, aov, action, margin_rate)

    with np.errstate(divide="ignore", invalid="ignore"):
        safety = np.where(needed > 0, action.assumed_uplift / needed, np.inf)

    # Expected net margin at the assumed uplift. An assumption-conditional
    # number, not a forecast.
    d = action.assumed_uplift
    expected_net = (
        d * aov * margin_rate
        - action.contact_cost
        - (probability + d) * aov * action.incentive_rate
    )
    if action.key == "NO_ACTION":
        expected_net = np.zeros_like(aov)
        safety = np.zeros_like(aov)

    return pd.DataFrame(
        {
            "action": action.key,
            "required_uplift": needed,
            "assumed_uplift": d,
            "margin_of_safety": safety,
            "expected_net_margin": expected_net,
            "contact_cost": action.contact_cost,
            "incentive_cost": (probability + d) * aov * action.incentive_rate,
        },
        index=scored.index,
    )


def select_action(scored, catalogue=ACTION_CATALOGUE,
                  margin_rate=GROSS_MARGIN_RATE, min_margin_of_safety=1.0):
    """
    For each customer: consider every action their state allows, keep the
    ones that clear break-even, pick the best expected net margin.

    If nothing clears the bar, the answer is NO_ACTION. A decision engine
    that never says "do nothing" is not a decision engine.
    """
    results = {}
    eligible_mask = {}

    for key, action in catalogue.items():
        if key == "NO_ACTION":
            continue
        results[key] = evaluate_action(scored, action, margin_rate)
        if action.eligible_states:
            eligible_mask[key] = scored["customer_state"].isin(action.eligible_states)
        else:
            eligible_mask[key] = pd.Series(True, index=scored.index)

    keys = list(results.keys())

    net = np.column_stack([results[k]["expected_net_margin"].to_numpy() for k in keys])
    safety = np.column_stack([results[k]["margin_of_safety"].to_numpy() for k in keys])
    allowed = np.column_stack([eligible_mask[k].to_numpy() for k in keys])

    bypass = np.array([catalogue[k].bypass_economics for k in keys])

    # Economics gate everything except actions explicitly exempt from it.
    viable = allowed & (
        bypass[None, :] | ((safety >= min_margin_of_safety) & (net > 0))
    )

    # An exempt action always wins its eligible rows: it is answering a
    # different question from the one expected net margin asks.
    priority = np.where(viable & bypass[None, :], np.inf, np.where(viable, net, -np.inf))
    best_index = np.argmax(priority, axis=1)
    has_choice = viable.any(axis=1)

    chosen = np.where(has_choice, np.array(keys)[best_index], "NO_ACTION")
    rows = np.arange(len(scored))

    out = pd.DataFrame(index=scored.index)
    out["recommended_action"] = chosen
    out["expected_net_margin"] = np.where(has_choice, net[rows, best_index], 0.0)
    out["required_uplift"] = np.where(
        has_choice,
        np.column_stack(
            [results[k]["required_uplift"].to_numpy() for k in keys]
        )[rows, best_index],
        np.nan,
    )
    out["assumed_uplift"] = np.where(
        has_choice,
        np.array([catalogue[k].assumed_uplift for k in keys])[best_index],
        0.0,
    )
    out["margin_of_safety"] = np.where(has_choice, safety[rows, best_index], 0.0)
    out["contact_cost"] = np.where(
        has_choice,
        np.array([catalogue[k].contact_cost for k in keys])[best_index],
        0.0,
    )
    out["action_label"] = [
        catalogue[k].label for k in out["recommended_action"]
    ]
    out["action_objective"] = [
        catalogue[k].objective for k in out["recommended_action"]
    ]

    return out


# ==========================================================================
# 11. THE DECISION LAYER
# ==========================================================================

def create_decision_output(scored, margin_rate=GROSS_MARGIN_RATE,
                           value_quantile=0.70, catalogue=ACTION_CATALOGUE):
    """Attach state, economics, chosen action and prioritisation to scored rows."""
    scored = scored.copy()

    # Value threshold is relative to the population being scored, per snapshot.
    scored["value_threshold"] = scored.groupby("snapshot_date")["monetary"].transform(
        lambda s: s.quantile(value_quantile)
    )

    scored["customer_state"] = scored.apply(assign_customer_state, axis=1)

    decisions = select_action(scored, catalogue=catalogue, margin_rate=margin_rate)
    scored = pd.concat([scored, decisions], axis=1)

    # --- observed vs incremental: keep these visibly separate --------------
    # Expected value of demand we EXPECT ANYWAY. Not caused by marketing.
    scored["expected_organic_revenue"] = (
        scored["purchase_probability"] * scored["aov"]
    )
    scored["expected_organic_margin"] = (
        scored["expected_organic_revenue"] * margin_rate
    )

    # Prioritisation heuristic. Explicitly named as such.
    # It ranks attention. It does not forecast profit.
    scored["priority_score"] = scored["expected_net_margin"]

    scored["priority_band"] = pd.cut(
        scored["priority_score"].rank(pct=True),
        bins=[-0.01, 0.50, 0.80, 0.95, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    scored.loc[
        scored["recommended_action"] == "NO_ACTION", "priority_band"
    ] = "LOW"

    scored["decision_reason"] = scored.apply(_decision_reason, axis=1)

    return scored


def _decision_reason(row):
    action = row["recommended_action"]
    state = row["customer_state"]

    if action == "NO_ACTION":
        return (
            f"{state}: no eligible action clears break-even "
            f"at this probability and order value"
        )

    return (
        f"{state}: needs {row['required_uplift']*100:.2f}pp incremental "
        f"conversion to break even; assumed {row['assumed_uplift']*100:.1f}pp"
    )


# ==========================================================================
# 12. CAPACITY
# ==========================================================================

def allocate_capacity(scored, budgets, fatigue_cap=1):
    """
    Break-even says an action is worth doing. Capacity says whether you
    can actually do it.

    budgets: {"ACTION_KEY": max_contacts}. Missing keys are unconstrained.
    fatigue_cap: max actions per customer per cycle.

    Greedy allocation by expected net margin within each action, which is
    optimal for a per-action count constraint.
    """
    frame = scored.copy()
    frame["contacted"] = False
    frame["allocation_rank"] = np.nan

    candidates = frame[frame["recommended_action"] != "NO_ACTION"]

    for action_key, group in candidates.groupby("recommended_action"):
        ranked = group.sort_values("expected_net_margin", ascending=False)
        limit = budgets.get(action_key, len(ranked))
        selected = ranked.head(int(limit)).index

        frame.loc[selected, "contacted"] = True
        frame.loc[ranked.index, "allocation_rank"] = np.arange(1, len(ranked) + 1)

    if fatigue_cap is not None:
        # Scoped to one snapshot = one planning cycle. In production you
        # would widen this key to a rolling window (say 30 days) so the
        # cap binds across cycles, not just within one.
        counts = (
            frame[frame["contacted"]]
            .sort_values("expected_net_margin", ascending=False)
            .groupby(["snapshot_date", "customer_id"])
            .cumcount()
        )
        over_cap = counts[counts >= fatigue_cap].index
        frame.loc[over_cap, "contacted"] = False
        frame.loc[over_cap, "suppressed_reason"] = "contact_fatigue_cap"

    return frame


def budget_summary(allocated, catalogue=ACTION_CATALOGUE):
    """What this cycle costs, and what it must deliver to be worth it."""
    contacted = allocated[allocated["contacted"]]

    rows = []
    for action_key, group in contacted.groupby("recommended_action"):
        action = catalogue[action_key]
        rows.append(
            {
                "action": action_key,
                "label": action.label,
                "channel": action.channel,
                "customers": len(group),
                "fixed_cost": len(group) * action.contact_cost,
                "expected_incentive_cost": float(
                    (
                        (group["purchase_probability"] + action.assumed_uplift)
                        * group["aov"]
                        * action.incentive_rate
                    ).sum()
                ),
                "median_required_uplift_pp": float(
                    group["required_uplift"].median() * 100
                ),
                "expected_net_margin_if_assumption_holds": float(
                    group["expected_net_margin"].sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["total_cost"] = summary["fixed_cost"] + summary["expected_incentive_cost"]
    return summary.sort_values("total_cost", ascending=False).reset_index(drop=True)


# ==========================================================================
# 13. EXPERIMENT DESIGN
# ==========================================================================

def assign_holdout(allocated, share=CONTROL_HOLDOUT_SHARE, salt="cpde-v1"):
    """
    Reserve a control group inside every treated cell.

    Deterministic hashing, so the same customer lands in the same arm on
    every run — no accidental re-randomisation between cycles.

    This is the single most valuable line in the repo. Without it, next
    cycle you still cannot answer "did marketing cause anything".
    """
    frame = allocated.copy()

    def bucket(customer_id, action):
        digest = hashlib.md5(
            f"{salt}:{action}:{customer_id}".encode()
        ).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    draws = np.array(
        [
            bucket(cid, act)
            for cid, act in zip(frame["customer_id"], frame["recommended_action"])
        ]
    )

    frame["experiment_arm"] = np.where(
        ~frame["contacted"],
        "NOT_SELECTED",
        np.where(draws < share, "CONTROL", "TREATMENT"),
    )

    # Control gets the recommendation logged but not executed.
    frame["execute"] = frame["experiment_arm"] == "TREATMENT"

    return frame


def experiment_design_summary(frame):
    """Sample size per cell, and the effect it could actually detect."""
    rows = []
    for (action, arm), group in frame[
        frame["experiment_arm"] != "NOT_SELECTED"
    ].groupby(["recommended_action", "experiment_arm"]):
        rows.append(
            {
                "action": action,
                "arm": arm,
                "customers": len(group),
                "mean_probability": group["purchase_probability"].mean(),
            }
        )

    design = pd.DataFrame(rows)
    if design.empty:
        return design

    wide = design.pivot(
        index="action", columns="arm", values="customers"
    ).fillna(0)

    baseline = (
        design[design["arm"] == "CONTROL"]
        .set_index("action")["mean_probability"]
    )
    wide["baseline_rate"] = baseline

    # Detectable difference, two-proportion, alpha 0.05, power 0.80.
    n_c = wide.get("CONTROL", pd.Series(0, index=wide.index))
    n_t = wide.get("TREATMENT", pd.Series(0, index=wide.index))
    p = wide["baseline_rate"].fillna(0.15)

    with np.errstate(divide="ignore", invalid="ignore"):
        effective_n = np.where(
            (n_c > 0) & (n_t > 0), 1 / (1 / n_c + 1 / n_t), np.nan
        )
        mde = 2.8 * np.sqrt(p * (1 - p) / effective_n)

    wide["detectable_uplift_pp"] = mde * 100
    return wide.reset_index()


# ==========================================================================
# 14. PATHS
# ==========================================================================

from pathlib import Path  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES = PROJECT_ROOT / "data" / "features"
OUTPUTS = PROJECT_ROOT / "outputs"

for _directory in (DATA_PROCESSED, DATA_FEATURES, OUTPUTS):
    _directory.mkdir(parents=True, exist_ok=True)


def resolve_data_path(preferred="online_retail_II.xlsx", fallback="sample_retail.xlsx"):
    """Use the real dataset if present, otherwise the synthetic sample."""
    primary = DATA_RAW / preferred
    if primary.exists():
        return primary

    secondary = DATA_RAW / fallback
    if secondary.exists():
        print(f"[note] {preferred} not found — using {fallback}. Results are not real.")
        return secondary

    raise FileNotFoundError(
        f"Put online_retail_II.xlsx in {DATA_RAW}, or run scripts/make_sample_data.py"
    )


SNAPSHOT_DATES = [
    "2010-04-01", "2010-05-01", "2010-06-01", "2010-07-01",
    "2010-08-01", "2010-09-01", "2010-10-01", "2010-11-01",
    "2010-12-01", "2011-01-01", "2011-02-01", "2011-03-01",
    "2011-04-01", "2011-05-01", "2011-06-01", "2011-07-01",
    "2011-08-01", "2011-09-01", "2011-10-01", "2011-11-01",
]

TRAIN_END = pd.Timestamp("2011-07-01")
VALIDATION_START = pd.Timestamp("2011-08-01")
VALIDATION_END = pd.Timestamp("2011-09-01")
TEST_START = pd.Timestamp("2011-10-01")


def temporal_split(model_data):
    """
    Three windows, in time order. Validation exists so calibration and
    model choice never touch the test period.
    """
    train = model_data[model_data["snapshot_date"] <= TRAIN_END].copy()
    validation = model_data[
        (model_data["snapshot_date"] >= VALIDATION_START)
        & (model_data["snapshot_date"] <= VALIDATION_END)
    ].copy()
    test = model_data[model_data["snapshot_date"] >= TEST_START].copy()
    return train, validation, test


def build_or_load_snapshots(data_path=None, rebuild=False, snapshot_dates=None):
    """
    Cached snapshot construction. The Excel load is the slow part, so
    notebooks 02-08 read the cached parquet instead of redoing it.
    """
    parquet_cache = DATA_FEATURES / "customer_snapshots.parquet"
    csv_cache = DATA_FEATURES / "customer_snapshots.csv"

    if not rebuild:
        if parquet_cache.exists():
            return pd.read_parquet(parquet_cache)
        if csv_cache.exists():
            return pd.read_csv(
                csv_cache,
                parse_dates=["snapshot_date", "first_order", "last_order"],
                dtype={"customer_id": str},
            )

    path = data_path or resolve_data_path()
    df = load_retail_data(path)
    purchases, returns = clean_transactions(df)
    orders = create_orders(purchases)

    dates = snapshot_dates or SNAPSHOT_DATES
    model_data = build_snapshot_dataset(orders, returns, purchases, dates)
    model_data = filter_modelling_rows(model_data)

    try:
        model_data.to_parquet(parquet_cache, index=False)
    except (ImportError, ValueError):
        model_data.to_csv(csv_cache, index=False)

    return model_data
    

# ==========================================================================
# 15. THE PLAYBOOK
# ==========================================================================
# The action catalogue says what an action COSTS. This says what it IS —
# the actual sequence a marketer executes, what to avoid, and which metric
# judges it. Without this layer the engine outputs a label, and a label is
# not an instruction.
#
# "measure" matters more than it looks. Judging a loyalty programme on
# conversion rate makes it look useless, because those customers were
# converting anyway. Each state gets the metric its objective implies.

STATE_PLAYBOOK = {
    "SERVICE_RISK": {
        "who": "Returns 40%+ of what they order, across at least two orders.",
        "objective": "Fix the experience. Do not run a campaign.",
        "do": [
            "Pull their returns by SKU and category — look for concentration",
            "If one category dominates: sizing, description or quality issue, not a customer issue",
            "Route to service for outreach; suppress from all promotional sends this cycle",
            "Feed the SKU pattern back to merchandising",
        ],
        "avoid": "Any offer. A discount on a product that does not fit "
                 "increases return volume and the cost behind it.",
        "measure": "Return rate over the next 90 days. Not orders.",
    },
    "NEW_UNPROVEN": {
        "who": "Exactly one order. No rhythm yet — unmeasured, not churned.",
        "objective": "Turn one purchase into a habit.",
        "do": [
            "Day 2: how to get the most from what they bought. No sell",
            "Day 7: one complementary item tied to the exact SKU purchased",
            "Day 14: gentle category expansion",
            "Day 21: small first-repeat incentive ONLY if still silent",
        ],
        "avoid": "Discounting order two. It teaches a brand-new customer "
                 "that waiting is rewarded, and you pay for every order after.",
        "measure": "Second-order rate at 60 days.",
    },
    "ACTIVE_VALUABLE": {
        "who": "Buying on their own normal rhythm, top 30% by historic spend.",
        "objective": "Raise basket value without buying demand you already have.",
        "do": [
            "Early access to new stock before general release",
            "Curated cross-category recommendations from complementary SKUs",
            "Free-shipping or gift threshold instead of % off — protects margin, lifts basket",
            "Service perks: priority delivery, easier returns, named contact for the top decile",
        ],
        "avoid": "Percentage discounts. This is the group where the "
                 "deadweight cost is highest — you are paying them to do what they were doing.",
        "measure": "Items and value per order. Not conversion rate — "
                   "theirs is already high and will not move.",
    },
    "ACTIVE_STANDARD": {
        "who": "Buying on their own normal rhythm, everyone else.",
        "objective": "Widen the relationship at near-zero cost.",
        "do": [
            "Automated post-purchase recommendations",
            "Replenishment reminders timed to THEIR interpurchase gap, not a fixed 30 days",
            "Category expansion based on what similar customers buy next",
        ],
        "avoid": "Paid media and incentives. Owned channels already clear "
                 "the bar here; anything paid does not.",
        "measure": "Incremental items per order, and category breadth over two cycles.",
    },
    "COOLING_VALUABLE": {
        "who": "1–2.5x past their own average gap, high value. Slowing, not stopped.",
        "objective": "Re-engage before cooling becomes dormancy — and before you have to pay for it.",
        "do": [
            "Day 0: what's new in the categories they actually buy",
            "Day 7: personalised recommendation from their top category",
            "Day 14: paid retargeting — still no offer",
            "Day 21: a targeted incentive ONLY if all three failed",
        ],
        "avoid": "Leading with the discount. Open with an offer and you "
                 "train a valuable customer to stop buying until one arrives.",
        "measure": "Return to activity within 1.5x their own gap.",
    },
    "COOLING_STANDARD": {
        "who": "1–2.5x past their own average gap, standard value.",
        "objective": "Same sequence, owned channels only.",
        "do": [
            "Day 0: category refresh",
            "Day 10: single personalised recommendation",
            "Stop. Do not escalate to paid at this value level",
        ],
        "avoid": "Escalating spend. The value does not support a second "
                 "paid touch, and the break-even maths says so before you find out.",
        "measure": "Reactivation rate per pound of owned-channel cost.",
    },
    "LAPSING_VALUABLE": {
        "who": "Beyond 2.5x their own gap, high historic value. The segment that "
               "justifies the whole exercise.",
        "objective": "Win back a customer whose past value warrants real spend.",
        "do": [
            "Diagnose first: did they stop buying, or move category? The data distinguishes these",
            "Lead with what CHANGED — new range, faster delivery, a fixed problem — not a % off",
            "Human or concierge touch for the top decile; a real person, not a template",
            "Meaningful one-time offer only after two owned attempts have failed",
        ],
        "avoid": "Batching them with a low-value lapsed customer. A pure "
                 "propensity model scores them identically. They are not the same customer.",
        "measure": "Win-back rate AND 12-month value after their return. "
                   "First order value alone will understate it.",
    },
    "LAPSING_STANDARD": {
        "who": "Beyond 2.5x their own gap, standard value.",
        "objective": "One good attempt, then stop.",
        "do": [
            "Single batch reactivation with one clear offer",
            "Suppress for a full cycle if there is no response",
        ],
        "avoid": "Repeated attempts. Cost compounds every send; response "
                 "rate does not.",
        "measure": "Cost per reactivated customer.",
    },
    "DORMANT": {
        "who": "Over 180 days silent and below the value threshold.",
        "objective": "Stay present for almost nothing. Invest nothing further.",
        "do": [
            "Seasonal and newsletter only, cheapest owned channel",
            "Use this group as the test bed for creative — the downside is near zero",
            "Review list hygiene before deliverability suffers",
        ],
        "avoid": "Any paid spend. Also avoid deleting them — they are "
                 "your cheapest experiment population.",
        "measure": "Deliverability and list health. Revenue here is a bonus, not a target.",
    },
}