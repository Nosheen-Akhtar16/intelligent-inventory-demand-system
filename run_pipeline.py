"""run_pipeline.py — Execute all 13 StockPilot AI modules, populate SQLite & RAG index, and export report."""
import json
from src.services import execute_full_pipeline

if __name__ == "__main__":
    summary = execute_full_pipeline()
    print(json.dumps(summary, indent=2))
