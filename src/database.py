"""SQLite Persistence & Human-in-the-Loop Approval Queue (Module 13)."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from config.settings import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata_store (key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hitl_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                store_id TEXT,
                product_id TEXT,
                recommended_qty INTEGER,
                approved_qty INTEGER,
                status TEXT,
                reviewer_note TEXT,
                reviewed_at TEXT
            )
        """)


def save_table(df: pd.DataFrame, table_name: str, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def load_table(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def save_metadata(key: str, payload: Dict[str, Any], db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO metadata_store (key, value_json, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload), datetime.now().isoformat()),
        )


def load_metadata(key: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT value_json FROM metadata_store WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value_json"]) if row else {}


def record_hitl_decision(store_id: str, product_id: str, recommended_qty: int, approved_qty: int, status: str, note: str = "", db_path: Path = DB_PATH) -> None:
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO hitl_approvals (store_id, product_id, recommended_qty, approved_qty, status, reviewer_note, reviewed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (store_id, product_id, recommended_qty, approved_qty, status, note, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )


def get_hitl_history(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM hitl_approvals ORDER BY id DESC LIMIT 50").fetchall()
        return [dict(r) for r in rows]
