"""Module 6: Inventory Decision Engine (Part A)."""
from typing import Dict, Any
import pandas as pd
from config.settings import (
    URGENT_PROB_THRESHOLD, REORDER_PROB_THRESHOLD, MONITOR_PROB_THRESHOLD,
    URGENT_COVERAGE_DAYS, REORDER_COVERAGE_DAYS, MONITOR_COVERAGE_DAYS,
)


def _classify_decision(row: pd.Series) -> tuple[str, str, int]:
    prob = float(row["Stockout_Probability"])
    cov = float(row["Inventory_Coverage_Days"])
    reorder_qty = int(row["Recommended_Reorder_Qty"])

    if (prob >= URGENT_PROB_THRESHOLD or cov < URGENT_COVERAGE_DAYS) and reorder_qty > 0:
        return (
            "URGENT REORDER",
            f"Critical stockout risk (P={prob:.2f} >= {URGENT_PROB_THRESHOLD} or coverage {cov:.2f}d < {URGENT_COVERAGE_DAYS}d). Immediate replenishment of {reorder_qty} units required.",
            1,
        )
    elif (prob >= REORDER_PROB_THRESHOLD or cov < REORDER_COVERAGE_DAYS) and reorder_qty > 0:
        return (
            "REORDER",
            f"Elevated stockout exposure (P={prob:.2f} >= {REORDER_PROB_THRESHOLD} or coverage {cov:.2f}d < {REORDER_COVERAGE_DAYS}d). Schedule replenishment of {reorder_qty} units.",
            2,
        )
    elif prob >= MONITOR_PROB_THRESHOLD or cov < MONITOR_COVERAGE_DAYS:
        return (
            "MONITOR",
            f"Moderate buffer (coverage {cov:.2f}d, stockout P={prob:.2f}). Monitor daily sales velocity closely.",
            3,
        )
    else:
        return (
            "HOLD",
            f"Healthy buffer (coverage {cov:.2f}d >= {MONITOR_COVERAGE_DAYS}d, stockout P={prob:.2f}). Hold new replenishment.",
            4,
        )


def generate_inventory_decisions(reorder_df: pd.DataFrame) -> pd.DataFrame:
    df = reorder_df.copy()
    results = df.apply(_classify_decision, axis=1)
    df["Decision"] = [r[0] for r in results]
    df["Decision_Rationale"] = [r[1] for r in results]
    df["Decision_Priority"] = [r[2] for r in results]
    return df.sort_values(["Decision_Priority", "Stockout_Probability"], ascending=[True, False]).reset_index(drop=True)


def summarize_decisions(decision_df: pd.DataFrame) -> Dict[str, Any]:
    counts = decision_df["Decision"].value_counts().to_dict()
    for cat in ["URGENT REORDER", "REORDER", "MONITOR", "HOLD"]:
        counts.setdefault(cat, 0)
    return {
        "total_items": int(len(decision_df)),
        "decision_counts": counts,
        "at_risk_count": int((decision_df["Risk_Category"] == "AT_RISK").sum()),
        "safe_count": int((decision_df["Risk_Category"] == "SAFE").sum()),
        "total_recommended_reorder_units": int(decision_df["Recommended_Reorder_Qty"].sum()),
        "total_estimated_reorder_cost": round(float(decision_df["Estimated_Reorder_Cost"].sum()), 2),
        "avg_coverage_days": round(float(decision_df["Inventory_Coverage_Days"].mean()), 2),
    }
