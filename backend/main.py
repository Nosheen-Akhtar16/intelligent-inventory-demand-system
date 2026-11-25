"""backend/main.py — FastAPI REST Backend for StockPilot AI (Module 13)."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services import ensure_initialized, execute_full_pipeline
from src.database import load_table, load_metadata, record_hitl_decision, get_hitl_history
from src.module7_whatif_simulator import run_whatif_simulation
from src.module10_responsible_ai import explain_sku_prediction
from src.module11_rag_copilot import InventoryRAGCopilot
from src.module12_agentic_assistant import InventoryAgenticAssistant

app = FastAPI(
    title="StockPilot AI API",
    version="1.0.0",
    description="Integrated Inventory Forecasting, Stockout Risk, Sustainability (SDG-12), RAG Copilot & Agentic AI API",
)


class QueryRequest(BaseModel):
    query: str


class HITLApprovalRequest(BaseModel):
    store_id: str
    product_id: str
    recommended_qty: int
    approved_qty: int
    status: str
    note: str = ""


@app.on_event("startup")
def startup_event():
    ensure_initialized()


@app.get("/health")
def health():
    return {"status": "healthy", "service": "StockPilot AI FastAPI Backend"}


@app.get("/overview")
def get_overview():
    ensure_initialized()
    return {
        "dataset_profile": load_metadata("dataset_profile"),
        "validation_report": load_metadata("validation_report"),
        "forecast_metrics": load_metadata("forecast_metrics"),
        "risk_metrics": load_metadata("risk_metrics"),
        "decision_summary": load_metadata("decision_summary"),
        "sustainability_summary": load_metadata("sustainability_summary"),
    }


@app.get("/decisions")
def get_decisions():
    ensure_initialized()
    return load_table("inventory_decisions").to_dict(orient="records")


@app.get("/what-if")
def get_whatif(demand_change_pct: float = 20.0):
    ensure_initialized()
    df = load_table("inventory_decisions")
    sim_df, summary = run_whatif_simulation(df, demand_change_pct=demand_change_pct)
    return {"summary": summary, "items": sim_df.to_dict(orient="records")}


@app.get("/sustainability")
def get_sustainability():
    ensure_initialized()
    return {
        "summary": load_metadata("sustainability_summary"),
        "items": load_table("sustainability_analysis").to_dict(orient="records"),
    }


@app.get("/explain/{store_id}/{product_id}")
def get_explanation(store_id: str, product_id: str):
    ensure_initialized()
    df = load_table("inventory_decisions")
    match = df[(df["Store ID"] == store_id) & (df["Product ID"] == product_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Store-Product pair not found")
    return explain_sku_prediction(
        match.iloc[0],
        load_metadata("model_artifacts"),
        load_metadata("forecast_metrics"),
        load_metadata("risk_metrics"),
    )


@app.post("/hitl/approve")
def post_hitl_approval(req: HITLApprovalRequest):
    record_hitl_decision(req.store_id, req.product_id, req.recommended_qty, req.approved_qty, req.status, req.note)
    return {"status": "recorded", "history": get_hitl_history()}


@app.post("/rag/query")
def query_rag_copilot(req: QueryRequest):
    ensure_initialized()
    copilot = InventoryRAGCopilot()
    copilot.build_index()
    return copilot.answer_query(req.query, load_table("inventory_decisions"))


@app.post("/agent/chat")
def chat_agentic_assistant(req: QueryRequest):
    ensure_initialized()
    agent = InventoryAgenticAssistant(load_table("inventory_decisions"), load_table("sustainability_analysis"))
    return agent.run(req.query)


@app.post("/pipeline/retrain")
def retrain_pipeline():
    return execute_full_pipeline()
