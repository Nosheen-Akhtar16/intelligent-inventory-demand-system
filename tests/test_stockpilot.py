"""Comprehensive test suite covering Part A (Modules 1-8) and Part B (Modules 9-13)."""
from src.services import execute_full_pipeline
from src.database import load_table, load_metadata, record_hitl_decision, get_hitl_history
from src.module10_responsible_ai import explain_sku_prediction
from src.module11_rag_copilot import InventoryRAGCopilot
from src.module12_agentic_assistant import InventoryAgenticAssistant


def test_end_to_end_pipeline():
    res = execute_full_pipeline()
    assert res["status"] == "COMPLETED"
    assert res["forecast_metrics"]["series_count"] == 100
    assert res["risk_metrics"]["accuracy"] > 0.80

    dec_df = load_table("inventory_decisions")
    sus_df = load_table("sustainability_analysis")
    assert len(dec_df) == 100
    assert len(sus_df) == 100

    # Module 10 Explainability
    exp = explain_sku_prediction(
        dec_df.iloc[0],
        load_metadata("model_artifacts"),
        load_metadata("forecast_metrics"),
        load_metadata("risk_metrics"),
    )
    assert exp["human_in_the_loop"]["auto_po_execution"] is False

    # Module 11 RAG Copilot
    copilot = InventoryRAGCopilot()
    copilot.build_index()
    rag_res = copilot.answer_query("What is the reorder policy for Store S001?", dec_df)
    assert len(rag_res["retrieved_chunks"]) > 0

    # Module 12 Agentic AI
    agent = InventoryAgenticAssistant(dec_df, sus_df)
    ag_res = agent.run("Check which products need urgent restocking.")
    assert len(ag_res["tool_trace"]) == 4

    # Module 13 HITL persistence
    record_hitl_decision("S001", "P0001", 100, 100, "APPROVED", "Unit test sign-off")
    assert len(get_hitl_history()) >= 1
