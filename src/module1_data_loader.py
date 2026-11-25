"""Module 1: Dataset Collection and Understanding (Part A)."""
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from config.settings import DATASET_PATH

EXPECTED_ATTRIBUTES = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
    "Price", "Discount", "Weather Condition", "Holiday/Promotion",
    "Competitor Pricing", "Seasonality",
]


def load_raw_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the retail store inventory dataset (73,100 records, 15 attributes)."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Place retail_store_inventory.csv in data/.")
    return pd.read_csv(dataset_path)


def analyze_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive profiling summary for Module 1."""
    sp = df["Store ID"].astype(str) + "_" + df["Product ID"].astype(str)
    numeric_cols = [
        "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
        "Price", "Discount", "Competitor Pricing",
    ]
    desc_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            desc_stats[col] = {
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "median": round(float(s.median()), 2),
                "max": round(float(s.max()), 2),
            }

    return {
        "total_records": int(len(df)),
        "num_attributes": int(df.shape[1]),
        "columns": list(df.columns),
        "unique_store_product_combinations": int(sp.nunique()),
        "unique_stores": sorted(df["Store ID"].dropna().unique().tolist()),
        "unique_products": sorted(df["Product ID"].dropna().unique().tolist()),
        "categories": df["Category"].value_counts().to_dict(),
        "regions": df["Region"].value_counts().to_dict(),
        "weather_conditions": df["Weather Condition"].value_counts().to_dict(),
        "seasonality_distribution": df["Seasonality"].value_counts().to_dict(),
        "date_start": str(df["Date"].min()),
        "date_end": str(df["Date"].max()),
        "descriptive_stats": desc_stats,
    }
