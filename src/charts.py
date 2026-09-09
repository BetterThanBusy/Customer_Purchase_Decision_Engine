"""
src/charts.py

Every figure used to present this project. Notebook 09 orchestrates; the
drawing lives here so charts can be regenerated without rerunning
analysis, and restyled in one place.

Palette is beige + burgundy. Charts are exported at 200 DPI on a solid
background so they drop straight into video without transparency issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


# ==========================================================================
# THEME
# ==========================================================================

BG = "#F7F1EA"          # cream — background, always
INK = "#000000"         # black — all text, titles, labels, ticks
BURGUNDY = "#51051E"    # the default mark. Structure.
TEAL = "#043C4D"        # the baseline, the comparison, the "before"
ORANGE = "#FF4500"      # the point of the chart. One idea per figure.

# Neutrals derived from cream — for muted marks only, never for text.
GREY = "#B3AAA0"
GRID = "#DED4C8"

# Tints, used only where a chart needs more marks than the palette has
# (segment maps). Every tint resolves to one of the three accents, so the
# semantic reading survives: teal = healthy, burgundy = drifting,
# orange = needs action now.
BURGUNDY_LIGHT = "#8A3049"
TEAL_LIGHT = "#3E7C8C"
ORANGE_LIGHT = "#FF8C5A"

STATE_COLOURS = {
    "ACTIVE_VALUABLE": TEAL,
    "ACTIVE_STANDARD": TEAL_LIGHT,
    "NEW_UNPROVEN": "#7FA0A8",
    "COOLING_VALUABLE": BURGUNDY,
    "COOLING_STANDARD": BURGUNDY_LIGHT,
    "LAPSING_VALUABLE": ORANGE,
    "LAPSING_STANDARD": ORANGE_LIGHT,
    "SERVICE_RISK": INK,
    "DORMANT": GREY,
}


def apply_theme():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _finish(fig, ax, subtitle=None):
    fig.tight_layout()
    if subtitle:
        fig.text(
            0.005, -0.035, subtitle,
            fontsize=10.5, color=INK, va="top", ha="left",
            linespacing=1.5,
        )
    return fig


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


# ==========================================================================
# 01 — OPENING: THE SCORE
# ==========================================================================

def score_distribution(scored):
    """Where the propensity scores actually sit. The opening shot."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    probability = scored["purchase_probability"]
    counts, edges, patches = ax.hist(probability, bins=40, color=BURGUNDY, edgecolor=BG)

    # Orange marks the only part of this distribution anyone will act on.
    cutoff = probability.quantile(0.90)
    for patch, left in zip(patches, edges[:-1]):
        if left >= cutoff:
            patch.set_facecolor(ORANGE)

    ax.axvline(cutoff, color=ORANGE, linewidth=1.8)
    ax.text(cutoff, max(counts) * 0.55, "  top 10%", fontsize=10.5,
            color=INK, weight="bold")

    base_rate = scored["purchased_again_30d"].mean()
    ax.axvline(base_rate, color=TEAL, linestyle="--", linewidth=1.8)
    ax.text(
        base_rate + 0.012, ax.get_ylim()[1] * 0.92,
        f"actual repeat rate {base_rate:.1%}\n(what random targeting gets you)",
        fontsize=10, color=INK,
    )

    ax.set_title("Every customer gets a score. Now what?")
    ax.set_xlabel("Probability of purchasing in the next 30 days")
    ax.set_ylabel("Customers")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.5)

    return _finish(
        fig, ax,
        "Most customers score low. A thin tail scores very high. "
        "The score alone does not tell you who to contact.",
    )


# ==========================================================================
# 02 — THE SNAPSHOT CONCEPT
# ==========================================================================

def snapshot_concept():
    """How a point-in-time customer view is constructed. No data needed."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    ax.add_patch(Rectangle((0.4, 1.5), 5.6, 0.9, facecolor=TEAL, edgecolor=INK, lw=1.2))
    ax.add_patch(Rectangle((6.0, 1.5), 2.2, 0.9, facecolor=ORANGE, edgecolor=INK, lw=1.2))
    ax.add_patch(Rectangle((8.2, 1.5), 1.4, 0.9, facecolor=BG, edgecolor=GREY, lw=1.0, ls="--"))

    ax.text(3.2, 1.95, "EVERYTHING BEFORE THE CUTOFF", ha="center", va="center",
            fontsize=11.5, weight="bold", color=BG)
    ax.text(3.2, 1.68, "becomes features", ha="center", va="center", fontsize=10, color=BG)

    ax.text(7.1, 1.95, "NEXT 30 DAYS", ha="center", va="center",
            fontsize=11.5, weight="bold", color=INK)
    ax.text(7.1, 1.68, "becomes the target", ha="center", va="center",
            fontsize=10, color=INK)

    ax.text(8.9, 1.95, "unseen", ha="center", va="center", fontsize=10, color=GREY)

    ax.plot([6.0, 6.0], [1.1, 2.75], color=INK, lw=2.4)
    ax.text(6.0, 2.85, "CUTOFF", ha="center", fontsize=11, weight="bold", color=INK)

    ax.text(
        0.4, 0.75,
        "One customer produces one row per cutoff date. Twenty monthly cutoffs means the model learns\n"
        "what a customer about to buy looks like across twenty different months — not one arbitrary moment.",
        fontsize=10.5, color=INK, va="top",
    )
    ax.text(
        0.4, 0.2,
        "Nothing crosses the line. That is the whole discipline.",
        fontsize=10.5, color=INK, weight="bold", va="top",
    )

    ax.set_title("The customer snapshot", loc="left", pad=6)
    fig.tight_layout()
    return fig


# ==========================================================================
# 03 — BASE RATE DRIFT
# ==========================================================================

def base_rate_drift(model_data):
    drift = (
        model_data.groupby("snapshot_date")
        .agg(customers=("customer_id", "size"), base_rate=("purchased_again_30d", "mean"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(drift["snapshot_date"], drift["base_rate"], color=BURGUNDY, lw=2.6,
            marker="o", markersize=5)
    ax.fill_between(drift["snapshot_date"], drift["base_rate"], color=BURGUNDY, alpha=0.12)

    ax.axhline(drift["base_rate"].mean(), color=TEAL, ls="--", lw=1.8)
    ax.text(drift["snapshot_date"].iloc[0], drift["base_rate"].mean() + 0.006,
            f"average {drift['base_rate'].mean():.1%}", fontsize=10, color=INK)

    # Orange on the spread, because the spread is the argument.
    for idx in [drift["base_rate"].idxmax(), drift["base_rate"].idxmin()]:
        row = drift.loc[idx]
        ax.scatter([row["snapshot_date"]], [row["base_rate"]], s=110,
                   color=ORANGE, zorder=6)
        ax.annotate(f"{row['base_rate']:.1%}",
                    xy=(row["snapshot_date"], row["base_rate"]),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=10.5, weight="bold", color=INK, ha="center")

    ax.set_title("The repeat-purchase rate is not constant")
    ax.set_ylabel("30-day repeat rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.5)
    fig.autofmt_xdate(rotation=45)

    return _finish(
        fig, ax,
        "Which is why the train/test split is by TIME, not at random. "
        "A random split lets the model see the future.",
    )


# ==========================================================================
# 04 — BASELINES VS MODEL
# ==========================================================================

def baselines_vs_model(baseline_scores, model_scores):
    frame = pd.concat(
        [
            pd.DataFrame(baseline_scores).assign(kind="baseline"),
            pd.DataFrame(model_scores).assign(kind="model"),
        ]
    ).sort_values("pr_auc")

    colours = [TEAL if k == "baseline" else BURGUNDY for k in frame["kind"]]
    colours[-1] = ORANGE  # the winner is the point of the chart

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(frame["model"], frame["pr_auc"], color=colours, edgecolor=BG)

    for bar, value in zip(bars, frame["pr_auc"]):
        ax.text(value + 0.006, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=10.5, color=INK)

    best_baseline = frame[frame["kind"] == "baseline"]["pr_auc"].max()
    ax.axvline(best_baseline, color=TEAL, ls="--", lw=1.8)
    ax.text(best_baseline, len(frame) - 0.35, "  the bar to clear",
            fontsize=10, color=INK, va="center")

    ax.set_title("Does the model beat the rule a marketer already uses?")
    ax.set_xlabel("PR-AUC on the validation window (higher is better)")
    ax.grid(axis="x", alpha=0.5)

    return _finish(
        fig, ax,
        "Teal = rules anyone can write in SQL. Burgundy = the models, orange = the winner. "
        "If the gap is small, say so.",
    )


# ==========================================================================
# 05 — THE CALIBRATION TRAP
# ==========================================================================

def calibration_trap(observed_rate, mean_balanced, mean_unweighted):
    fig, ax = plt.subplots(figsize=(10, 5.2))

    labels = ["What actually\nhappened", "class_weight\n= 'balanced'", "unweighted\n(used here)"]
    values = [observed_rate, mean_balanced, mean_unweighted]
    colours = [TEAL, ORANGE, BURGUNDY]

    bars = ax.bar(labels, values, color=colours, width=0.55, edgecolor=BG)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                f"{value:.1%}", ha="center", fontsize=13, weight="bold", color=INK)

    ax.axhline(observed_rate, color=TEAL, ls="--", lw=1.6)

    ax.set_title("The setting that quietly inflates your business case")
    ax.set_ylabel("Average predicted purchase probability")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.5)

    ratio = mean_balanced / max(observed_rate, 1e-9)
    return _finish(
        fig, ax,
        f"Balanced weighting overstates probability by {ratio:.1f}x. Harmless if you only sort the list.\n"
        "Fatal once you multiply it by order value to decide what to spend.",
    )


# ==========================================================================
# 06 — GAINS AND LIFT
# ==========================================================================

def gains_and_lift(y_true, probability):
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(probability))
    sorted_y = y_true[order]

    depth = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    gains = np.cumsum(sorted_y) / sorted_y.sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot(depth, gains, color=BURGUNDY, lw=2.8)
    ax.plot([0, 1], [0, 1], color=TEAL, ls="--", lw=1.8)
    ax.fill_between(depth, gains, depth, color=BURGUNDY, alpha=0.12)

    at_ten = gains[int(len(gains) * 0.10) - 1]
    ax.scatter([0.10], [at_ten], color=ORANGE, zorder=5, s=130)
    ax.annotate(
        f"top 10% of customers\ncaptures {at_ten:.0%} of all buyers",
        xy=(0.10, at_ten), xytext=(0.28, at_ten - 0.16),
        fontsize=10.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
    )

    ax.set_title("Cumulative gains")
    ax.set_xlabel("Share of customers contacted")
    ax.set_ylabel("Share of buyers reached")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.45)

    ax = axes[1]
    base_rate = y_true.mean()
    depths = [0.01, 0.05, 0.10, 0.20, 0.50]
    lifts = []
    for pct in depths:
        n = max(1, int(len(sorted_y) * pct))
        lifts.append(sorted_y[:n].mean() / base_rate)

    colours = [ORANGE if d == 0.10 else BURGUNDY for d in depths]
    bars = ax.bar([f"{int(d*100)}%" for d in depths], lifts, color=colours,
                  width=0.6, edgecolor=BG)
    for bar, value in zip(bars, lifts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.06,
                f"{value:.1f}x", ha="center", fontsize=12, weight="bold", color=INK)

    ax.axhline(1.0, color=TEAL, ls="--", lw=1.8)
    ax.text(len(depths) - 0.4, 1.06, "random targeting", fontsize=10, color=INK, ha="right")

    ax.set_title("Lift over random")
    ax.set_xlabel("Depth of the contact list")
    ax.set_ylabel("Times better than random")
    ax.grid(axis="y", alpha=0.45)

    fig.suptitle("What the ranking is actually worth", fontsize=16, weight="bold", color=INK)
    fig.tight_layout()
    return fig


# ==========================================================================
# 07 — CALIBRATION CURVE
# ==========================================================================

def calibration_curve(reliability):
    fig, ax = plt.subplots(figsize=(9.5, 6))

    ax.plot([0, 1], [0, 1], color=TEAL, ls="--", lw=1.8, label="perfectly calibrated")
    ax.plot(reliability["mean_predicted"], reliability["observed_rate"],
            color=BURGUNDY, lw=2.6, marker="o", markersize=7, label="this model")

    for _, row in reliability.iterrows():
        ax.plot([row["mean_predicted"], row["mean_predicted"]],
                [row["mean_predicted"], row["observed_rate"]],
                color=ORANGE, lw=2.6, zorder=1)

    ax.set_title("Does 30% actually mean 30%?")
    ax.set_xlabel("Predicted probability (decile average)")
    ax.set_ylabel("Observed repeat rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.45)

    return _finish(
        fig, ax,
        "Every orange line is a forecasting error. The decision layer multiplies these numbers by money,\n"
        "so the gap is not academic — it is the difference between the plan and the bank balance.",
    )


# ==========================================================================
# 08 — STABILITY
# ==========================================================================

def stability(stability_frame):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(stability_frame))
    labels = [d.strftime("%b %Y") for d in stability_frame["snapshot_date"]]

    ax.bar(x - 0.19, stability_frame["roc_auc"], width=0.36, color=BURGUNDY,
           label="ROC-AUC", edgecolor=BG)
    ax.bar(x + 0.19, stability_frame["pr_auc"], width=0.36, color=TEAL,
           label="PR-AUC", edgecolor=BG)

    for i, (roc, pr) in enumerate(zip(stability_frame["roc_auc"], stability_frame["pr_auc"])):
        ax.text(i - 0.19, roc + 0.012, f"{roc:.2f}", ha="center", fontsize=10, color=INK)
        ax.text(i + 0.19, pr + 0.012, f"{pr:.2f}", ha="center", fontsize=10, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Does it hold every month, or only on average?")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.45)

    return _finish(
        fig, ax,
        "A single average number hides period-to-period variance. "
        "If the spread is wide, the honest claim is 'it works in some months'.",
    )


# ==========================================================================
# 09 — FEATURE IMPORTEALCE
# ==========================================================================

def feature_importance(importance, top_n=10):
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10.5, 6))
    colours = [BURGUNDY] * len(top)
    colours[-1] = ORANGE  # top feature; the axis is reversed so it is drawn last
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color=colours, edgecolor=BG,
            error_kw=dict(ecolor=INK, lw=1.1, capsize=3))

    ax.set_title("What the model actually leans on")
    ax.set_xlabel("PR-AUC lost when this feature is shuffled")
    ax.grid(axis="x", alpha=0.45)

    return _finish(
        fig, ax,
        "Rhythm and recency dominate. Correlated features share credit, so this ranks\n"
        "what the model uses — not what a customer responds to.",
    )


# ==========================================================================
# 10 — DEADWEIGHT: THE CENTRAL CHART
# ==========================================================================

def deadweight_curve(catalogue, aov, margin_rate, actions=None):
    """
    Required break-even uplift as a function of P, per action.

    This is the chart the whole marketing argument rests on.
    """
    from src.engine import required_uplift

    actions = actions or [
        "OWNED_RECOMMENDATION",
        "PAID_RETARGETING",
        "TARGETED_INCENTIVE",
        "HIGH_VALUE_WINBACK",
    ]
    colours = {
        "OWNED_RECOMMENDATION": TEAL,       # the cheap comparison
        "PAID_RETARGETING": BURGUNDY,       # structure
        "TARGETED_INCENTIVE": ORANGE,       # the point
        "HIGH_VALUE_WINBACK": ORANGE_LIGHT,
    }

    probabilities = np.linspace(0.01, 0.95, 200)

    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    for key in actions:
        action = catalogue[key]
        needed = required_uplift(
            probabilities, np.full_like(probabilities, aov), action, margin_rate
        ) * 100

        ax.plot(probabilities, needed, lw=2.8, color=colours.get(key, GREY),
                label=f"{action.label}  ({action.incentive_rate:.0%} off, {action.contact_cost:.2f}/contact)")
        ax.axhline(action.assumed_uplift * 100, color=colours.get(key, GREY),
                   ls=":", lw=1.6, alpha=0.75)

    ax.set_ylim(0, 40)
    ax.set_xlim(0, 0.95)
    ax.set_title("Why you should not discount your best customers")
    ax.set_xlabel("Probability the customer buys anyway")
    ax.set_ylabel("Uplift the action must create to break even (pp)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.45)

    ax.text(
        0.50, 33,
        "Solid line = what the action MUST achieve.\nDotted line = what we assume it achieves.\n"
        "Above the dotted line, the action cannot pay for itself.",
        fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=BG, edgecolor=TEAL),
    )

    return _finish(
        fig, ax,
        f"At AOV {aov:,.0f} and {margin_rate:.0%} margin. A zero-incentive email stays flat and cheap.\n"
        "A discount gets more expensive the more certain the customer already is.",
    )


# ==========================================================================
# 11 — THE CUSTOMER STATE MAP
# ==========================================================================

def customer_state_map(decided, sample=4000, seed=7):
    """Where each state lives in rhythm x value space."""
    frame = decided.copy()
    frame["ratio"] = frame["recency_ratio"].fillna(
        frame["recency_days"] / 45.0
    ).clip(0.02, 12)
    frame["value"] = frame["monetary"].clip(lower=1)

    if len(frame) > sample:
        frame = frame.sample(sample, random_state=seed)

    fig, ax = plt.subplots(figsize=(12, 7))

    for state, group in frame.groupby("customer_state"):
        ax.scatter(
            group["ratio"], group["value"],
            s=16, alpha=0.55, edgecolors="none",
            color=STATE_COLOURS.get(state, GREY), label=state,
        )

    ax.axvline(1.0, color=INK, ls="--", lw=1.4)
    ax.axvline(2.5, color=INK, ls="--", lw=1.4)
    ax.text(1.02, ax.get_ylim()[1] * 0.55, "on their own\nnormal rhythm",
            fontsize=9.5, color=INK)
    ax.text(2.55, ax.get_ylim()[1] * 0.55, "2.5x overdue",
            fontsize=9.5, color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Customer states: overdue-ness against value")
    ax.set_xlabel("Days since last order  /  this customer's own average gap")
    ax.set_ylabel("Historic spend (log scale)")
    ax.legend(loc="lower right", fontsize=9, ncol=2, markerscale=2.2)
    ax.grid(alpha=0.35)

    return _finish(
        fig, ax,
        "The x-axis is the point: 60 days silent is alarming for a weekly buyer\n"
        "and completely normal for a quarterly one. Raw recency cannot tell them apart.",
    )


# ==========================================================================
# 12 — STATE PROFILE
# ==========================================================================

def state_profile(decided):
    profile = (
        decided.groupby("customer_state")
        .agg(
            customers=("customer_id", "size"),
            mean_probability=("purchase_probability", "mean"),
            median_monetary=("monetary", "median"),
        )
        .sort_values("customers", ascending=True)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    colours = [STATE_COLOURS.get(s, GREY) for s in profile.index]

    axes[0].barh(profile.index, profile["customers"], color=colours, edgecolor=BG)
    axes[0].set_title("How many", fontsize=13)
    for i, v in enumerate(profile["customers"]):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=9.5, color=INK)

    axes[1].barh(profile.index, profile["mean_probability"], color=colours, edgecolor=BG)
    axes[1].set_title("Likely to buy", fontsize=13)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    for i, v in enumerate(profile["mean_probability"]):
        axes[1].text(v, i, f" {v:.0%}", va="center", fontsize=9.5, color=INK)

    axes[2].barh(profile.index, profile["median_monetary"], color=colours, edgecolor=BG)
    axes[2].set_title("Worth (median spend)", fontsize=13)
    for i, v in enumerate(profile["median_monetary"]):
        axes[2].text(v, i, f" {v:,.0f}", va="center", fontsize=9.5, color=INK)

    for ax in axes:
        ax.grid(axis="x", alpha=0.4)

    fig.suptitle(
        "The same score means different things to different customers",
        fontsize=16, weight="bold", color=INK,
    )
    fig.tight_layout()
    return fig


# ==========================================================================
# 13 — STATE x ACTION
# ==========================================================================

def state_action_matrix(decided):
    matrix = pd.crosstab(decided["customer_state"], decided["recommended_action"])
    share = matrix.div(matrix.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(13, 6.5))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "btb", [BG, "#E8D9CE", GREY, BURGUNDY, "#2C0210"]
    )
    ax.imshow(share.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(share.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in share.columns], fontsize=9)
    ax.set_yticks(range(len(share.index)))
    ax.set_yticklabels(share.index, fontsize=10)

    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            count = matrix.values[i, j]
            if count == 0:
                continue
            ax.text(
                j, i, f"{count:,}", ha="center", va="center",
                fontsize=10,
                color="white" if share.values[i, j] > 0.55 else INK,
                weight="bold" if share.values[i, j] > 0.55 else "normal",
            )

    ax.set_title("State decides the action. Not the score.")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    return _finish(
        fig, ax,
        "ACTIVE_VALUABLE gets loyalty, never a discount. SERVICE_RISK gets no campaign at all.\n"
        "Change the margin assumption and this grid moves; the states do not.",
    )


# ==========================================================================
# 14 — ACTION ECONOMICS
# ==========================================================================

def action_economics(decided, catalogue):
    taken = decided[decided["recommended_action"] != "NO_ACTION"]

    summary = (
        taken.groupby("recommended_action")
        .agg(
            customers=("customer_id", "size"),
            median_required=("required_uplift", "median"),
        )
        .reset_index()
    )
    summary["assumed"] = [catalogue[a].assumed_uplift for a in summary["recommended_action"]]
    summary = summary.sort_values("median_required")

    y = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(11.5, 6))

    ax.hlines(y, summary["median_required"] * 100, summary["assumed"] * 100,
              color=GREY, lw=3, zorder=1)
    ax.scatter(summary["median_required"] * 100, y, s=110, color=BURGUNDY,
               zorder=3, label="required to break even (median)")
    ax.scatter(summary["assumed"] * 100, y, s=130, color=ORANGE, zorder=3,
               marker="D", label="assumed uplift  [Guessing]")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a.replace('_', ' ').title()}\n({n:,} customers)"
         for a, n in zip(summary["recommended_action"], summary["customers"])],
        fontsize=9.5,
    )
    ax.set_xlabel("Percentage points of incremental conversion")
    ax.set_title("Every recommendation shows its own break-even")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.45)

    return _finish(
        fig, ax,
        "Orange to the right of burgundy = the action clears its own bar. The gap is the margin of safety.\n"
        "The orange diamonds are assumptions, not measurements. The experiment replaces them.",
    )


# ==========================================================================
# 15 — BUDGET
# ==========================================================================

def budget_allocation(budget):
    frame = budget.sort_values("total_cost", ascending=True)
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(11.5, 6))

    ax.barh(y - 0.19, frame["fixed_cost"], height=0.36, color=BURGUNDY,
            label="delivery cost", edgecolor=BG)
    ax.barh(y - 0.19, frame["expected_incentive_cost"], height=0.36,
            left=frame["fixed_cost"], color=ORANGE,
            label="incentive given away", edgecolor=BG)
    ax.barh(y + 0.19, frame["expected_net_margin_if_assumption_holds"], height=0.36,
            color=TEAL, label="expected net margin (if assumption holds)", edgecolor=BG)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a.replace('_', ' ').title()}\n{int(n):,} contacts"
         for a, n in zip(frame["action"], frame["customers"])],
        fontsize=9.5,
    )
    ax.set_xlabel("Currency")
    ax.set_title("Where the money goes this cycle")
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="x", alpha=0.45)

    return _finish(
        fig, ax,
        "Owned channels reach thousands for almost nothing. The spend concentrates in the few cells\n"
        "where margin is being handed over — which is exactly where the break-even test bites.",
    )


# ==========================================================================
# 16 — EXPERIMENT POWER
# ==========================================================================

def experiment_power(design, catalogue):
    frame = design.dropna(subset=["detectable_uplift_pp"]).copy()
    frame["assumed_pp"] = [
        catalogue[a].assumed_uplift * 100 if a in catalogue else np.nan
        for a in frame["action"]
    ]
    frame = frame.sort_values("detectable_uplift_pp")

    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(11.5, 6))

    colours = [
        TEAL if d <= a else ORANGE
        for d, a in zip(frame["detectable_uplift_pp"], frame["assumed_pp"])
    ]

    ax.barh(y, frame["detectable_uplift_pp"], color=colours, edgecolor=BG, height=0.55)
    ax.scatter(frame["assumed_pp"], y, s=110, marker="D", color=INK, zorder=4,
               label="effect we are assuming")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a.replace('_', ' ').title()}\n{int(c):,} control / {int(t):,} treated"
         for a, c, t in zip(frame["action"], frame["CONTROL"], frame["TREATMENT"])],
        fontsize=9.5,
    )
    ax.set_xlabel("Smallest effect this cell can detect (pp, 80% power)")
    ax.set_title("Can the experiment prove its own case?")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.45)

    return _finish(
        fig, ax,
        "Orange = the cell is too small to detect the effect we are assuming. Running it produces\n"
        "a number nobody can interpret. Do this arithmetic BEFORE the campaign, not after.",
    )


# ==========================================================================
# 17 — THE FRAMEWORK
# ==========================================================================

def decision_framework():
    steps = [
        ("PREDICTION", "What is likely to happen?"),
        ("EXPLANATION", "What behaviour sits behind it?"),
        ("STATE", "What situation is this customer in?"),
        ("ECONOMICS", "What must an action achieve to pay for itself?"),
        ("DECISION", "Which action — or none?"),
        ("PRIORITISATION", "Who gets attention first?"),
        ("EXPERIMENT", "Did the action actually change anything?"),
    ]

    fig, ax = plt.subplots(figsize=(9, 10.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.45 + 0.6)
    ax.axis("off")

    for i, (title, question) in enumerate(steps):
        y = (len(steps) - i - 1) * 1.45 + 0.4
        is_last = i == len(steps) - 1

        ax.add_patch(
            FancyBboxPatch(
                (0.6, y), 8.8, 1.0,
                boxstyle="round,pad=0.06",
                facecolor=ORANGE if is_last else BURGUNDY,
                edgecolor="none",
            )
        )
        text_colour = INK if is_last else BG
        ax.text(1.0, y + 0.65, title, fontsize=13, weight="bold",
                color=text_colour, va="center")
        ax.text(1.0, y + 0.32, question, fontsize=10.5, color=text_colour,
                alpha=0.85, va="center")

        if i < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (5.0, y), (5.0, y - 0.42),
                    arrowstyle="-|>", mutation_scale=17, color=INK, lw=1.6,
                )
            )

    ax.text(
        5.0, len(steps) * 1.45 + 0.35,
        "From a score to a decision system",
        fontsize=16, weight="bold", color=INK, ha="center",
    )

    fig.tight_layout()
    return fig


# ==========================================================================
# 13 — THE PLAYBOOK (replaces the matrix as the primary segmentation asset)
# ==========================================================================

def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width))


def state_playbook_table(decided, playbook):
    """
    One row per state: who they are, what to do, what to avoid, how it is judged.

    The matrix shows how many customers land where. This shows what a
    marketer actually executes, which is the part a count cannot carry.
    """
    counts = decided["customer_state"].value_counts()
    actions = (
        decided.groupby("customer_state")["recommended_action"]
        .agg(lambda s: s.value_counts().index[0])
    )

    order = [s for s in [
        "SERVICE_RISK", "NEW_UNPROVEN", "ACTIVE_VALUABLE", "ACTIVE_STANDARD",
        "COOLING_VALUABLE", "COOLING_STANDARD", "LAPSING_VALUABLE",
        "LAPSING_STANDARD", "DORMANT",
    ] if s in counts.index]

    col_x = [0.0, 1.85, 4.35, 8.55, 11.9]
    col_w = [1.85, 2.50, 4.20, 3.35, 2.30]
    headers = ["State", "Who they are", "What to actually do", "What to avoid", "How it is judged"]

    rows = []
    for state in order:
        entry = playbook[state]
        do_text = "\n".join(f"• {_wrap(step, 52).replace(chr(10), chr(10) + '  ')}"
                            for step in entry["do"])
        rows.append(
            {
                "state": state,
                "count": counts[state],
                "action": actions[state],
                "who": _wrap(entry["who"], 30),
                "objective": _wrap(entry["objective"], 30),
                "do": do_text,
                "avoid": _wrap(entry["avoid"], 40),
                "measure": _wrap(entry["measure"], 28),
            }
        )

    heights = []
    for row in rows:
        lines = max(
            row["do"].count("\n") + 1,
            row["avoid"].count("\n") + 1,
            row["who"].count("\n") + row["objective"].count("\n") + 4,
            row["measure"].count("\n") + 1,
        )
        heights.append(lines * 0.235 + 0.36)

    total = sum(heights)
    fig, ax = plt.subplots(figsize=(16.5, total + 1.9))
    ax.set_xlim(0, sum(col_w))
    ax.set_ylim(0, total + 1.35)
    ax.axis("off")

    header_y = total + 0.42
    for x, w, text in zip(col_x, col_w, headers):
        ax.add_patch(Rectangle((x, header_y), w - 0.06, 0.5,
                               facecolor=BURGUNDY, edgecolor="none"))
        ax.text(x + 0.12, header_y + 0.25, text, fontsize=11.5, weight="bold",
                color="white", va="center")

    y = total
    for row, height in zip(rows, heights):
        y -= height
        colour = STATE_COLOURS.get(row["state"], GREY)

        ax.add_patch(Rectangle((0, y), sum(col_w) - 0.06, height - 0.07,
                               facecolor="#FBF6EF", edgecolor="#E0D2BF", lw=0.9))
        ax.add_patch(Rectangle((0, y), 0.13, height - 0.07,
                               facecolor=colour, edgecolor="none"))

        top = y + height - 0.28

        # State name in ink, not the accent colour — the pale state colours
        # are unreadable on beige. The colour lives on the left bar instead.
        ax.text(col_x[0] + 0.2, top, row["state"].replace("_", "\n"),
                fontsize=10.5, weight="bold", color=INK, va="top")
        ax.text(col_x[0] + 0.2, top - 0.62, f"{row['count']:,} customers",
                fontsize=9, color=INK, va="top")
        ax.text(col_x[0] + 0.2, top - 0.86,
                _wrap(row["action"].replace("_", " ").title(), 18),
                fontsize=8.5, color=GREY, va="top", style="italic")

        ax.text(col_x[1] + 0.12, top, row["who"], fontsize=9, color=INK, va="top")
        ax.text(col_x[1] + 0.12, top - (row["who"].count("\n") + 1) * 0.235 - 0.14,
                row["objective"], fontsize=9, color=INK, weight="bold", va="top")

        ax.text(col_x[2] + 0.12, top, row["do"], fontsize=9, color=INK,
                va="top", linespacing=1.45)

        ax.text(col_x[3] + 0.12, top, "NOT this", fontsize=8.5, weight="bold",
                color=ORANGE, va="top")
        ax.text(col_x[3] + 0.12, top - 0.3, row["avoid"], fontsize=9,
                color=INK, va="top")

        ax.text(col_x[4] + 0.12, top, row["measure"], fontsize=9,
                color=TEAL, va="top", weight="bold")

    ax.text(0, total + 1.15, "The playbook: what each segment actually gets",
            fontsize=17, weight="bold", color=INK, va="bottom")

    fig.text(
        0.005, -0.012,
        "The rightmost column is the one people skip. Judge a loyalty programme on conversion rate and it looks useless — "
        "those customers\nwere converting anyway. Each state is judged on the metric its objective implies.",
        fontsize=10.5, color=BURGUNDY, va="top", ha="left", linespacing=1.5,
    )
    fig.tight_layout()
    return fig


def state_playbook_card(state, decided, playbook):
    """A single state, full width — for cutting one segment at a time on screen."""
    entry = playbook[state]
    subset = decided[decided["customer_state"] == state]
    colour = STATE_COLOURS.get(state, GREY)

    action = (
        subset["recommended_action"].value_counts().index[0]
        if len(subset) else "NO_ACTION"
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 6.85), 10, 1.15, facecolor=colour, edgecolor="none"))
    ax.text(0.3, 7.62, state.replace("_", " "), fontsize=22, weight="bold",
            color="white", va="center")
    ax.text(0.3, 7.12, entry["who"], fontsize=10.5, color="white", alpha=0.9, va="center")

    if len(subset):
        stats = (
            f"{len(subset):,} customers   |   "
            f"avg propensity {subset['purchase_probability'].mean():.0%}   |   "
            f"median spend {subset['monetary'].median():,.0f}   |   "
            f"median {subset['recency_days'].median():.0f} days silent"
        )
        ax.text(9.7, 7.62, f"{len(subset) / len(decided):.0%}", fontsize=22,
                weight="bold", color="white", ha="right", va="center")
        ax.text(0.3, 6.45, stats, fontsize=10, color=INK, va="center")

    ax.text(0.3, 5.95, "OBJECTIVE", fontsize=9.5, weight="bold", color=GREY, va="center")
    ax.text(0.3, 5.6, entry["objective"], fontsize=13, weight="bold",
            color=INK, va="center")

    ax.text(0.3, 5.05, "WHAT TO ACTUALLY DO", fontsize=9.5, weight="bold",
            color=GREY, va="center")

    y = 4.65
    for step in entry["do"]:
        ax.add_patch(Rectangle((0.3, y - 0.05), 0.06, 0.3,
                               facecolor=colour, edgecolor="none"))
        ax.text(0.55, y + 0.1, _wrap(step, 88), fontsize=10.5, color=INK,
                va="center", linespacing=1.4)
        y -= 0.42 + 0.2 * _wrap(step, 88).count("\n")

    y -= 0.25
    ax.add_patch(
        FancyBboxPatch((0.3, y - 0.72), 9.4, 0.92, boxstyle="round,pad=0.05",
                       facecolor="#FFE8DE", edgecolor=ORANGE, lw=1.4)
    )
    ax.text(0.55, y - 0.02, "NOT THIS", fontsize=9.5, weight="bold",
            color=ORANGE, va="center")
    ax.text(0.55, y - 0.42, _wrap(entry["avoid"], 96), fontsize=10.5,
            color=INK, va="center", linespacing=1.4)

    ax.text(0.3, y - 1.15, "JUDGED ON", fontsize=9.5, weight="bold",
            color=GREY, va="center")
    ax.text(0.3, y - 1.5, entry["measure"], fontsize=11.5, color=TEAL,
            weight="bold", va="center")

    ax.text(9.7, 0.12, f"engine recommends: {action.replace('_', ' ').lower()}",
            fontsize=9.5, color=GREY, ha="right", style="italic")

    fig.tight_layout()
    return fig