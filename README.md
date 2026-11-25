# StockPilot AI — Autonomous Inventory Intelligence & Decision Support System

StockPilot AI is a full-stack, multi-module retail inventory intelligence platform built with **FastAPI**, **Streamlit**, **SQLite**, **Scikit-Learn**, and **RAG + Agentic Orchestration** (aligned with **UN SDG 12: Responsible Consumption and Production**).

## Project Structure

```text
StockPilot_AI/
├── launch.py                      # One-command launcher (starts FastAPI backend + Streamlit dashboard)
├── run_pipeline.py                # CLI runner for the end-to-end 12-module pipeline
├── requirements.txt               # Python dependencies
├── .env.example                   # Optional environment variables (e.g., OPENAI_API_KEY)
├── config/
│   └── settings.py                # Centralized thresholds, SLA targets, SDG 12 & HITL policies
├── data/
│   ├── retail_store_inventory.csv # Primary retail inventory & demand dataset
│   └── vector_store.json          # Local persistent TF-IDF vector index for RAG Co-Pilot
├── docs_kb/                       # Policy & governance knowledge base indexed by RAG Co-Pilot
│   ├── 01_reorder_policy.md
│   ├── 02_decision_thresholds.md
│   ├── 03_sustainability_sdg12.md
│   └── 04_responsible_ai_hitl.md
├── src/
│   ├── database.py                # SQLite persistence layer (KPIs, decisions, audit trail, HITL queue)
│   ├── services.py                # Unified service façade shared by CLI, API, and Dashboard
│   ├── module1_data_loader.py     # Schema validation & ingestion
│   ├── module2_preprocessing.py   # Feature engineering (lags, rolling stats, seasonality, stock ratios)
│   ├── module3_forecasting.py     # Hybrid ML + Holt-Winters demand forecasting
│   ├── module4_stockout_risk.py   # Multi-factor stockout probability & days-of-cover risk scoring
│   ├── module5_reorder_engine.py  # Dynamic EOQ, safety stock & reorder point optimization
│   ├── module6_decision_engine.py # Rule + ML autonomous recommendation engine
│   ├── module7_whatif_simulator.py# Multi-variable scenario & sensitivity simulator
│   ├── module8_report_generator.py# Executive Markdown & JSON report generator
│   ├── module9_sustainability.py  # SDG 12 spoilage prevention, CO2e & waste reduction metrics
│   ├── module10_responsible_ai.py # Confidence scoring, bias checks, audit logging & HITL gating
│   ├── module11_rag_copilot.py    # Grounded RAG policy & data co-pilot with source citations
│   └── module12_agentic_assistant.py # Multi-step tool-calling agentic assistant
├── backend/
│   └── main.py                    # FastAPI REST API server (with OpenAPI docs at /docs)
├── dashboard/
│   └── app.py                     # Multi-tab interactive Streamlit executive dashboard
├── reports/
│   └── latest_scenario_report.md  # Generated executive briefing report
└── tests/
    └── test_stockpilot.py         # Automated integration & module verification suite
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the full stack (FastAPI Backend + Streamlit Dashboard):**
   ```bash
   python launch.py
   ```
   - **Streamlit Dashboard:** `http://localhost:8501`
   - **FastAPI Backend & Swagger Docs:** `http://localhost:8000/docs`

3. **Run the pipeline via CLI (optional):**
   ```bash
   python run_pipeline.py
   ```

4. **Run the test suite:**
   ```bash
   python -m unittest discover -s tests -v
   ```
