"""Unified Service Orchestration for All 13 Modules (Part A + Part B)."""
from typing import Dict, Any
from config.settings import DB_PATH
from src.database import init_db, save_table, save_metadata, load_metadata
from src.module1_data_loader import load_raw_dataset, analyze_dataset_structure
from src.module2_preprocessing import preprocess_and_validate
from src.module3_forecasting import run_demand_forecasting
from src.module4_stockout_risk import train_and_predict_stockout_risk
from src.module5_reorder_engine import compute_reorder_recommendations
from src.module6_decision_engine import generate_inventory_decisions, summarize_decisions
from src.module7_whatif_simulator import run_whatif_simulation
from src.module8_report_generator import generate_automated_scenario_report
from src.module9_sustainability import analyze_sustainability
from src.module11_rag_copilot import InventoryRAGCopilot


def execute_full_pipeline() -> Dict[str, Any]:
    init_db(DB_PATH)
    raw_df = load_raw_dataset()
    profile = analyze_dataset_structure(raw_df)
    clean_df, val_report = preprocess_and_validate(raw_df)

    fc_df, holdout_df, fc_metrics = run_demand_forecasting(clean_df)
    risk_df, risk_metrics, model_artifacts = train_and_predict_stockout_risk(clean_df)
    reorder_df = compute_reorder_recommendations(fc_df, risk_df)
    decision_df = generate_inventory_decisions(reorder_df)
    decision_summary = summarize_decisions(decision_df)
    sim_df, sim_summary = run_whatif_simulation(decision_df, demand_change_pct=20.0)
    sus_df, sus_summary = analyze_sustainability(decision_df)

    save_table(decision_df, "inventory_decisions", DB_PATH)
    save_table(holdout_df, "forecast_holdout", DB_PATH)
    save_table(sus_df, "sustainability_analysis", DB_PATH)

    save_metadata("dataset_profile", profile, DB_PATH)
    save_metadata("validation_report", val_report, DB_PATH)
    save_metadata("forecast_metrics", fc_metrics, DB_PATH)
    save_metadata("risk_metrics", risk_metrics, DB_PATH)
    save_metadata("model_artifacts", model_artifacts, DB_PATH)
    save_metadata("decision_summary", decision_summary, DB_PATH)
    save_metadata("default_sim_summary", sim_summary, DB_PATH)
    save_metadata("sustainability_summary", sus_summary, DB_PATH)

    copilot = InventoryRAGCopilot()
    chunks_indexed = copilot.build_index()

    report_md = generate_automated_scenario_report(
        profile, val_report, fc_metrics, risk_metrics, decision_df, decision_summary, sim_df, sim_summary, sus_summary
    )
    save_metadata("latest_report_md", {"markdown": report_md}, DB_PATH)
    return {
        "status": "COMPLETED",
        "forecast_metrics": fc_metrics,
        "risk_metrics": risk_metrics,
        "decision_summary": decision_summary,
        "sim_summary": sim_summary,
        "sustainability_summary": sus_summary,
        "rag_chunks_indexed": chunks_indexed,
    }


def ensure_initialized() -> None:
    if not DB_PATH.exists() or not load_metadata("forecast_metrics", DB_PATH):
        execute_full_pipeline()
