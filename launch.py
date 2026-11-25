"""launch.py — One-command launcher for Pipeline + FastAPI Backend + Streamlit Dashboard."""
import subprocess
import sys
import time
from src.services import ensure_initialized

if __name__ == "__main__":
    print("[1/3] Ensuring StockPilot AI pipeline, SQLite DB, and RAG Vector Store are initialized...")
    ensure_initialized()

    print("[2/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    time.sleep(2)

    print("[3/3] Launching Streamlit Interactive Dashboard on http://localhost:8501 ...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], check=True)
    finally:
        api_proc.terminate()
