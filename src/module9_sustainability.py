"""Module 9: Sustainability Intelligence Layer (Part B — UN SDG 12)."""
import math
from typing import Dict, Any, Tuple
import pandas as pd
from config.settings import OVERSTOCK_COVERAGE_DAYS, OVERSTOCK_TARGET_MULTIPLIER

PERISHABILITY_WEIGHT = {
    "Groceries": 1.35,
    "Toys": 1.05,
    "Clothing": 1.10,
    "Electronics": 1.00,
    "Furniture": 0.95,
}


def analyze_sustainability(decision_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate inventory conditions from an operational sustainability perspective (UN SDG 12):
    1. Overstock Detection
    2. Inventory Efficiency Analysis
    3. Potential Waste Risk Identification
    4. Sustainability Recommendations
    Disclaimer: Inventory-based operational indicators, not measured physical waste or carbon reductions.
    """
    df = decision_df.copy()
    df["Optimal_Upper_Stock"] = (df["Target_Stock"] * OVERSTOCK_TARGET_MULTIPLIER).round(2)
    df["Excess_Stock_Units"] = (
        (df["Net_Available_Stock"] - df["Optimal_Upper_Stock"]).clip(lower=0).apply(lambda x: int(math.ceil(x)))
    )
    df["Is_Overstocked"] = (df["Inventory_Coverage_Days"] > OVERSTOCK_COVERAGE_DAYS) | (df["Excess_Stock_Units"] > 0)

    df["Inventory_Utilization_Pct"] = (
        (df["Forecast_7D_Demand"] / df["Net_Available_Stock"].clip(lower=1)) * 100.0
    ).clip(0, 100).round(2)

    waste_risks, suggestions = [], []
    for _, r in df.iterrows():
        cov = float(r["Inventory_Coverage_Days"])
        excess = int(r["Excess_Stock_Units"])
        cat_w = PERISHABILITY_WEIGHT.get(r["Category"], 1.0)
        weighted_cov = cov * cat_w

        if excess > 0 and weighted_cov >= 5.2:
            waste_risks.append("HIGH_WASTE_RISK")
            suggestions.append(
                f"Pause replenishment; rebalance or run targeted markdown on {excess} excess units ({cov:.2f}d coverage in {r['Category']})."
            )
        elif r["Is_Overstocked"] or weighted_cov >= 4.3:
            waste_risks.append("MODERATE_WASTE_RISK")
            suggestions.append(
                f"Hold new purchase orders; allow inventory to draw down from {cov:.2f}d toward balanced target."
            )
        else:
            waste_risks.append("OPTIMAL")
            suggestions.append("Stock aligned with demand velocity; maintain lean 1.2x safety buffer.")

    df["Potential_Waste_Risk"] = waste_risks
    df["Sustainability_Recommendation"] = suggestions

    overstock_cnt = int(df["Is_Overstocked"].sum())
    high_waste_cnt = int((df["Potential_Waste_Risk"] == "HIGH_WASTE_RISK").sum())
    mod_waste_cnt = int((df["Potential_Waste_Risk"] == "MODERATE_WASTE_RISK").sum())
    total_excess = int(df["Excess_Stock_Units"].sum())
    avg_util = round(float(df["Inventory_Utilization_Pct"].mean()), 2)
    sdg_score = round(max(0.0, min(100.0, 100.0 - (overstock_cnt * 0.7) - (high_waste_cnt * 1.1))), 1)

    summary = {
        "sdg_alignment": "UN Sustainable Development Goal 12 — Responsible Consumption and Production",
        "disclaimer": "These metrics are inventory-based operational indicators, not measured physical waste or carbon-emission reductions.",
        "total_items": int(len(df)),
        "overstock_count": overstock_cnt,
        "high_waste_risk_count": high_waste_cnt,
        "moderate_waste_risk_count": mod_waste_cnt,
        "total_excess_units": total_excess,
        "avg_utilization_pct": avg_util,
        "sdg12_efficiency_score": sdg_score,
    }
    return df, summary
