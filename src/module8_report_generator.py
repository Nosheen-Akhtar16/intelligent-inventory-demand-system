"""Module 8: Automated Scenario Reports (Part A)."""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from config.settings import REPORTS_DIR


def generate_automated_scenario_report(
    dataset_profile: Dict[str, Any],
    validation_report: Dict[str, Any],
    forecast_metrics: Dict[str, Any],
    risk_metrics: Dict[str, Any],
    decision_df: pd.DataFrame,
    decision_summary: Dict[str, Any],
    sim_df: pd.DataFrame,
    sim_summary: Dict[str, Any],
    sustainability_summary: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    urgent_items = decision_df[decision_df["Decision"] == "URGENT REORDER"].head(15)

    lines = [
        "# StockPilot AI — Automated Inventory & Scenario Decision Report",
        f"**Generated Timestamp:** {timestamp}  ",
        f"**Dataset Snapshot Date:** {decision_df['Latest_Date'].iloc[0]}  ",
        "",
        "---",
        "",
        "## 1. Dataset & Validation Summary (Part A — Modules 1 & 2)",
        f"- **Total Records:** {dataset_profile['total_records']:,} ({dataset_profile['num_attributes']} attributes)",
        f"- **Store-Product Combinations:** {dataset_profile['unique_store_product_combinations']} ({len(dataset_profile['unique_stores'])} Stores × {len(dataset_profile['unique_products'])} Products)",
        f"- **Historical Window:** {dataset_profile['date_start']} to {dataset_profile['date_end']}",
        f"- **Data Quality Audit:** {validation_report['status']} (Negative Demand Forecasts repaired: {validation_report['negative_demand_forecast_repaired']}, Missing imputed: {validation_report['total_missing_values_detected']})",
        "",
        "## 2. Demand Forecasting Performance (Part A — Module 3: Holt-Winters)",
        f"- **Model:** {forecast_metrics['model_name']}",
        f"- **30-Day Holdout Overall MAE:** {forecast_metrics['overall_mae']:.3f} units/day",
        f"- **30-Day Holdout Overall RMSE:** {forecast_metrics['overall_rmse']:.3f} units/day",
        f"- **30-Day Holdout Overall sMAPE:** {forecast_metrics['overall_smape_pct']:.2f}%",
        "",
        "## 3. Stockout Risk Classification (Part A — Module 4: Logistic Regression)",
        f"- **Time-Aware Split Cutoff:** {risk_metrics['cutoff_date']} (Train: {risk_metrics['train_records']:,} | Test: {risk_metrics['test_records']:,})",
        f"- **Accuracy:** {risk_metrics['accuracy']*100:.2f}% | **Precision:** {risk_metrics['precision']*100:.2f}% | **Recall:** {risk_metrics['recall']*100:.2f}% | **F1-Score:** {risk_metrics['f1_score']*100:.2f}% | **ROC-AUC:** {risk_metrics['roc_auc']:.4f}",
        f"- **Current Snapshot Risk Split:** {decision_summary['at_risk_count']} `AT_RISK` vs {decision_summary['safe_count']} `SAFE` store-product series",
        "",
        "## 4. Inventory Decision Engine & Reorder Recommendations (Part A — Modules 5 & 6)",
        f"- **URGENT REORDER:** {decision_summary['decision_counts']['URGENT REORDER']} items",
        f"- **REORDER:** {decision_summary['decision_counts']['REORDER']} items",
        f"- **MONITOR:** {decision_summary['decision_counts']['MONITOR']} items",
        f"- **HOLD:** {decision_summary['decision_counts']['HOLD']} items",
        f"- **Total Recommended Reorder Quantity (Safety Factor = 1.2):** {decision_summary['total_recommended_reorder_units']:,} units (Est. Value: ${decision_summary['total_estimated_reorder_cost']:,.2f})",
        "",
        "### Top Priority Replenishment Actions (`URGENT REORDER`)",
        "| Store | Product | Category | Current Inv | Incoming | 7D Forecast | Coverage (Days) | Stockout Prob | Reorder Qty | Decision |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for _, r in urgent_items.iterrows():
        lines.append(
            f"| {r['Store ID']} | {r['Product ID']} | {r['Category']} | {int(r['Current_Inventory'])} | "
            f"{int(r['Incoming_Orders'])} | {r['Forecast_7D_Demand']:.1f} | {r['Inventory_Coverage_Days']:.2f} | "
            f"{r['Stockout_Probability']:.2f} | **{int(r['Recommended_Reorder_Qty'])}** | {r['Decision']} |"
        )

    lines.extend([
        "",
        f"## 5. What-If Demand Variation Scenario ({sim_summary['demand_change_pct']:+.1f}% Demand Shock — Module 7)",
        "| Metric | Baseline (Normal Demand) | What-If Scenario | Delta |",
        "| :--- | ---: | ---: | ---: |",
        f"| `URGENT REORDER` Count | {sim_summary['baseline_urgent_count']} | {sim_summary['scenario_urgent_count']} | {sim_summary['scenario_urgent_count'] - sim_summary['baseline_urgent_count']:+d} |",
        f"| `REORDER` Count | {sim_summary['baseline_reorder_count']} | {sim_summary['scenario_reorder_count']} | {sim_summary['scenario_reorder_count'] - sim_summary['baseline_reorder_count']:+d} |",
        f"| Total Reorder Units Needed | {sim_summary['baseline_total_reorder_units']:,} | {sim_summary['scenario_total_reorder_units']:,} | {sim_summary['additional_reorder_units_needed']:+,d} |",
        f"| Average Inventory Coverage (Days) | {sim_summary['baseline_avg_coverage_days']:.2f} | {sim_summary['scenario_avg_coverage_days']:.2f} | {sim_summary['scenario_avg_coverage_days'] - sim_summary['baseline_avg_coverage_days']:+.2f} |",
        f"| Items with Projected 7D Shortage | — | {sim_summary['items_with_projected_shortage']} | — |",
        f"| Total Projected Shortage Units | — | {sim_summary['total_projected_shortage_units']:,} | — |",
        "",
    ])

    if sustainability_summary:
        lines.extend([
            "## 6. Sustainability Intelligence Layer (Part B — Module 9: UN SDG-12 Indicators)",
            f"- **Overstocked Store-Product Items Flagged:** {sustainability_summary['overstock_count']} / {sustainability_summary['total_items']}",
            f"- **High Potential Waste Risk Items:** {sustainability_summary['high_waste_risk_count']}",
            f"- **Total Excess Inventory Units:** {sustainability_summary['total_excess_units']:,} units",
            f"- **Average Inventory Utilization Rate:** {sustainability_summary['avg_utilization_pct']:.1f}%",
            f"- **SDG-12 Inventory Efficiency Score:** {sustainability_summary['sdg12_efficiency_score']:.1f} / 100",
            f"- *Disclosure: {sustainability_summary['disclaimer']}*",
            "",
        ])

    lines.extend([
        "## 7. Responsible AI Governance Notice (Part B — Module 10)",
        "- All reorder quantities and risk alerts operate strictly as **human-in-the-loop decision support**. No automated purchase orders are executed without human approval.",
    ])

    report_md = "\n".join(lines)
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / "latest_scenario_report.md"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report_md, encoding="utf-8")
    return report_md
