"""Module 12: Agentic AI Inventory Decision Assistant (Part B — Tool Calling)."""
import re
from typing import Dict, Any, List, Optional
import pandas as pd
from src.module7_whatif_simulator import run_whatif_simulation


class InventoryAgenticAssistant:
    """
    Agentic AI Decision Assistant with structured planning & tool calling:
    - get_inventory_status()
    - forecast_demand()
    - predict_stockout_risk()
    - calculate_reorder_quantity()
    - run_what_if_simulation()
    - get_sustainability_insights()
    """

    def __init__(self, decision_df: pd.DataFrame, sustainability_df: pd.DataFrame):
        self.decision_df = decision_df
        self.sustainability_df = sustainability_df

    def get_inventory_status(self, store_id: Optional[str] = None, product_id: Optional[str] = None) -> Dict[str, Any]:
        df = self.decision_df
        if store_id:
            df = df[df["Store ID"] == store_id]
        if product_id:
            df = df[df["Product ID"] == product_id]
        return {
            "records_checked": int(len(df)),
            "total_on_hand": int(df["Current_Inventory"].sum()),
            "total_incoming": int(df["Incoming_Orders"].sum()),
            "avg_coverage_days": round(float(df["Inventory_Coverage_Days"].mean()), 2) if len(df) else 0.0,
        }

    def forecast_demand(self, store_id: Optional[str] = None, product_id: Optional[str] = None) -> Dict[str, Any]:
        df = self.decision_df
        if store_id:
            df = df[df["Store ID"] == store_id]
        if product_id:
            df = df[df["Product ID"] == product_id]
        return {
            "series_evaluated": int(len(df)),
            "total_7d_forecast_units": round(float(df["Forecast_7D_Demand"].sum()), 1),
            "avg_holdout_mae": round(float(df["Holdout_MAE"].mean()), 3) if len(df) else 0.0,
        }

    def predict_stockout_risk(self, store_id: Optional[str] = None, only_at_risk: bool = True) -> List[Dict[str, Any]]:
        df = self.decision_df
        if store_id:
            df = df[df["Store ID"] == store_id]
        if only_at_risk:
            df = df[df["Risk_Category"] == "AT_RISK"]
        df = df.sort_values("Stockout_Probability", ascending=False).head(10)
        return df[["Store ID", "Product ID", "Category", "Stockout_Probability", "Risk_Category", "Inventory_Coverage_Days"]].to_dict(orient="records")

    def calculate_reorder_quantity(self, store_id: Optional[str] = None, urgent_only: bool = True) -> List[Dict[str, Any]]:
        df = self.decision_df
        if store_id:
            df = df[df["Store ID"] == store_id]
        if urgent_only:
            df = df[df["Decision"].isin(["URGENT REORDER", "REORDER"])]
        df = df.sort_values(["Decision_Priority", "Recommended_Reorder_Qty"], ascending=[True, False]).head(10)
        return df[["Store ID", "Product ID", "Decision", "Current_Inventory", "Incoming_Orders", "Forecast_7D_Demand", "Recommended_Reorder_Qty"]].to_dict(orient="records")

    def run_what_if_simulation(self, demand_change_pct: float = 20.0, store_id: Optional[str] = None) -> Dict[str, Any]:
        df = self.decision_df[self.decision_df["Store ID"] == store_id] if store_id else self.decision_df
        _, summary = run_whatif_simulation(df, demand_change_pct=demand_change_pct)
        return summary

    def get_sustainability_insights(self, store_id: Optional[str] = None) -> Dict[str, Any]:
        df = self.sustainability_df[self.sustainability_df["Store ID"] == store_id] if store_id else self.sustainability_df
        over = df[df["Is_Overstocked"]].sort_values("Excess_Stock_Units", ascending=False).head(5)
        return {
            "overstocked_count": int(df["Is_Overstocked"].sum()),
            "total_excess_units": int(df["Excess_Stock_Units"].sum()),
            "top_overstocked": over[["Store ID", "Product ID", "Category", "Inventory_Coverage_Days", "Excess_Stock_Units", "Potential_Waste_Risk"]].to_dict(orient="records"),
        }

    def run(self, user_query: str) -> Dict[str, Any]:
        q_lower = user_query.lower()
        store_m = re.search(r"\bS00[1-5]\b", user_query.upper())
        prod_m = re.search(r"\bP00\d{2}\b", user_query.upper())
        store_id = store_m.group(0) if store_m else None
        product_id = prod_m.group(0) if prod_m else None

        plan = ["1. Understand user intent & extract store/product scope", "2. Call get_inventory_status() & forecast_demand()"]
        tool_trace = []

        inv_out = self.get_inventory_status(store_id=store_id, product_id=product_id)
        tool_trace.append({"tool": "get_inventory_status", "args": {"store_id": store_id, "product_id": product_id}, "output": inv_out})

        fc_out = self.forecast_demand(store_id=store_id, product_id=product_id)
        tool_trace.append({"tool": "forecast_demand", "args": {"store_id": store_id, "product_id": product_id}, "output": fc_out})

        if any(k in q_lower for k in ["what-if", "what if", "surge", "increase", "scenario", "%"]):
            pct_m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", user_query)
            pct = float(pct_m.group(1)) if pct_m else 20.0
            plan.append(f"3. Call run_what_if_simulation(demand_change_pct={pct})")
            sim_out = self.run_what_if_simulation(demand_change_pct=pct, store_id=store_id)
            tool_trace.append({"tool": "run_what_if_simulation", "args": {"demand_change_pct": pct, "store_id": store_id}, "output": sim_out})
            response = (
                f"### Agentic What-If Simulation Analysis ({pct:+.1f}% Demand Variation)\n"
                f"- **Scope Evaluated:** {inv_out['records_checked']} series (Baseline 7D Forecast: {fc_out['total_7d_forecast_units']:,.0f} units).\n"
                f"- **Urgent Reorder Escalation:** Increases from **{sim_out['baseline_urgent_count']}** to **{sim_out['scenario_urgent_count']}** items.\n"
                f"- **Additional Reorder Units Required:** **+{sim_out['additional_reorder_units_needed']:,} units** (Projected 7D Shortage: {sim_out['total_projected_shortage_units']:,} units across {sim_out['items_with_projected_shortage']} items).\n"
                f"- *Limitation: Simulation applies uniform percentage scaling; human approval is required prior to purchase order issuance.*"
            )
        elif any(k in q_lower for k in ["sustain", "overstock", "waste", "sdg", "excess"]):
            plan.append("3. Call get_sustainability_insights() for UN SDG-12 indicators")
            sus_out = self.get_sustainability_insights(store_id=store_id)
            tool_trace.append({"tool": "get_sustainability_insights", "args": {"store_id": store_id}, "output": sus_out})
            top_lines = "\n".join([
                f"- **{x['Store ID']} / {x['Product ID']}** ({x['Category']}): Coverage `{x['Inventory_Coverage_Days']}d`, Excess `{x['Excess_Stock_Units']}` units (`{x['Potential_Waste_Risk']}`)"
                for x in sus_out["top_overstocked"]
            ])
            response = (
                f"### Agentic Sustainability & Overstock Audit (UN SDG-12 Indicators)\n"
                f"- **Overstocked Items Flagged:** {sus_out['overstocked_count']} ({sus_out['total_excess_units']:,} excess units).\n"
                f"{top_lines}\n"
                f"- *Disclosure: Metrics reflect inventory-based operational indicators, not physical waste or carbon reductions.*"
            )
        else:
            plan.extend(["3. Call predict_stockout_risk()", "4. Call calculate_reorder_quantity()"])
            risk_out = self.predict_stockout_risk(store_id=store_id, only_at_risk=True)
            tool_trace.append({"tool": "predict_stockout_risk", "args": {"store_id": store_id, "only_at_risk": True}, "output": risk_out})
            reord_out = self.calculate_reorder_quantity(store_id=store_id, urgent_only=True)
            tool_trace.append({"tool": "calculate_reorder_quantity", "args": {"store_id": store_id, "urgent_only": True}, "output": reord_out})
            action_lines = "\n".join([
                f"- **{x['Store ID']} / {x['Product ID']}** [`{x['Decision']}`]: On-Hand `{x['Current_Inventory']}`, Incoming `{x['Incoming_Orders']}`, 7D Forecast `{x['Forecast_7D_Demand']}`, **Recommended Reorder: `{x['Recommended_Reorder_Qty']}` units**."
                for x in reord_out[:6]
            ])
            response = (
                f"### Agentic Replenishment & Stockout Risk Summary\n"
                f"- **Scope Evaluated:** {inv_out['records_checked']} items | **At-Risk Items:** {len(risk_out)} | **Avg Coverage:** {inv_out['avg_coverage_days']} days.\n"
                f"#### Priority Replenishment Recommendations (Safety Factor = 1.2)\n{action_lines}\n"
                f"- *Governance: Recommendations are decision-support outputs awaiting human manager sign-off.*"
            )

        return {"query": user_query, "plan": plan, "tool_trace": tool_trace, "response": response}
