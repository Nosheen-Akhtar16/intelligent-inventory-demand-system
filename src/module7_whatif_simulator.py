"""Module 7: What-if Scenario Simulator (Part A)."""
import math
from typing import Dict, Any, Tuple
import pandas as pd
from config.settings import (
    DEFAULT_DEMAND_CHANGE_PCT, SAFETY_FACTOR,
    URGENT_PROB_THRESHOLD, REORDER_PROB_THRESHOLD, MONITOR_PROB_THRESHOLD,
    URGENT_COVERAGE_DAYS, REORDER_COVERAGE_DAYS, MONITOR_COVERAGE_DAYS,
)


def run_whatif_simulation(
    decision_df: pd.DataFrame,
    demand_change_pct: float = DEFAULT_DEMAND_CHANGE_PCT,
    safety_factor: float = SAFETY_FACTOR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    multiplier = 1.0 + (demand_change_pct / 100.0)
    sim = decision_df.copy()

    sim["Scenario_Demand_Change_Pct"] = float(demand_change_pct)
    sim["Scenario_Daily_Demand"] = (sim["Forecast_Daily_Demand"] * multiplier).round(2)
    sim["Scenario_7D_Demand"] = (sim["Forecast_7D_Demand"] * multiplier).round(2)

    sim["Scenario_Coverage_Days"] = (
        sim["Net_Available_Stock"] / sim["Scenario_Daily_Demand"].clip(lower=0.1)
    ).round(2)

    sim["Scenario_Shortage_Units"] = (
        (sim["Scenario_7D_Demand"] - sim["Net_Available_Stock"]).clip(lower=0).apply(lambda x: int(math.ceil(x)))
    )

    sim["Scenario_Target_Stock"] = (sim["Scenario_7D_Demand"] * safety_factor).round(2)
    sim["Scenario_Reorder_Qty"] = (
        (sim["Scenario_Target_Stock"] - sim["Net_Available_Stock"])
        .apply(lambda x: int(math.ceil(x)) if x > 0 else 0)
    )
    sim["Reorder_Qty_Delta"] = sim["Scenario_Reorder_Qty"] - sim["Recommended_Reorder_Qty"]
    sim["Coverage_Days_Delta"] = (sim["Scenario_Coverage_Days"] - sim["Inventory_Coverage_Days"]).round(2)

    if demand_change_pct >= 0:
        prob_shift = (demand_change_pct / 100.0) * 0.45 * (1.0 - sim["Stockout_Probability"])
        sim["Scenario_Stockout_Prob"] = (sim["Stockout_Probability"] + prob_shift).clip(0.0, 0.999).round(4)
    else:
        prob_shift = (abs(demand_change_pct) / 100.0) * 0.45 * sim["Stockout_Probability"]
        sim["Scenario_Stockout_Prob"] = (sim["Stockout_Probability"] - prob_shift).clip(0.001, 0.999).round(4)

    def _scenario_decision(r: pd.Series) -> str:
        p = float(r["Scenario_Stockout_Prob"])
        c = float(r["Scenario_Coverage_Days"])
        rq = int(r["Scenario_Reorder_Qty"])
        if (p >= URGENT_PROB_THRESHOLD or c < URGENT_COVERAGE_DAYS) and rq > 0:
            return "URGENT REORDER"
        elif (p >= REORDER_PROB_THRESHOLD or c < REORDER_COVERAGE_DAYS) and rq > 0:
            return "REORDER"
        elif p >= MONITOR_PROB_THRESHOLD or c < MONITOR_COVERAGE_DAYS:
            return "MONITOR"
        return "HOLD"

    sim["Scenario_Decision"] = sim.apply(_scenario_decision, axis=1)
    sim["Decision_Escalated"] = sim["Scenario_Decision"] != sim["Decision"]

    baseline_counts = sim["Decision"].value_counts().to_dict()
    scenario_counts = sim["Scenario_Decision"].value_counts().to_dict()
    for cat in ["URGENT REORDER", "REORDER", "MONITOR", "HOLD"]:
        baseline_counts.setdefault(cat, 0)
        scenario_counts.setdefault(cat, 0)

    comparison_summary = {
        "demand_change_pct": float(demand_change_pct),
        "baseline_urgent_count": baseline_counts["URGENT REORDER"],
        "scenario_urgent_count": scenario_counts["URGENT REORDER"],
        "baseline_reorder_count": baseline_counts["REORDER"],
        "scenario_reorder_count": scenario_counts["REORDER"],
        "baseline_decision_counts": baseline_counts,
        "scenario_decision_counts": scenario_counts,
        "baseline_total_reorder_units": int(sim["Recommended_Reorder_Qty"].sum()),
        "scenario_total_reorder_units": int(sim["Scenario_Reorder_Qty"].sum()),
        "additional_reorder_units_needed": int(sim["Reorder_Qty_Delta"].sum()),
        "items_with_projected_shortage": int((sim["Scenario_Shortage_Units"] > 0).sum()),
        "total_projected_shortage_units": int(sim["Scenario_Shortage_Units"].sum()),
        "items_escalated_count": int(sim["Decision_Escalated"].sum()),
        "baseline_avg_coverage_days": round(float(sim["Inventory_Coverage_Days"].mean()), 2),
        "scenario_avg_coverage_days": round(float(sim["Scenario_Coverage_Days"].mean()), 2),
    }

    return sim, comparison_summary
