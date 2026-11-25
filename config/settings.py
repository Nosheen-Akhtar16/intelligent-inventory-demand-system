"""Centralized configuration for StockPilot AI."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_KB_DIR = ROOT_DIR / "docs_kb"
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"

DATASET_PATH = Path(os.getenv("STOCKPILOT_DATASET", DATA_DIR / "retail_store_inventory.csv"))
DB_PATH = Path(os.getenv("STOCKPILOT_DB", DATA_DIR / "stockpilot.db"))
VECTOR_STORE_PATH = Path(os.getenv("STOCKPILOT_VECTOR_STORE", DATA_DIR / "vector_store.json"))

# Part A: Forecasting & Reorder Policy Parameters
HOLDOUT_DAYS = 30
FORECAST_HORIZON_DAYS = 7
SAFETY_FACTOR = 1.2
DEFAULT_DEMAND_CHANGE_PCT = 20.0

# Part A: Module 6 Decision Engine Thresholds
URGENT_PROB_THRESHOLD = 0.70
REORDER_PROB_THRESHOLD = 0.50
MONITOR_PROB_THRESHOLD = 0.30

URGENT_COVERAGE_DAYS = 1.6
REORDER_COVERAGE_DAYS = 2.6
MONITOR_COVERAGE_DAYS = 3.8

# Part B: Module 9 Sustainability Overstock Thresholds (UN SDG-12 Inventory Indicators)
OVERSTOCK_COVERAGE_DAYS = 4.2
OVERSTOCK_TARGET_MULTIPLIER = 1.15

# Optional OpenAI API configuration (System runs 100% offline with built-in synthesis if not set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
API_BASE_URL = os.getenv("STOCKPILOT_API_URL", "http://127.0.0.1:8000")
