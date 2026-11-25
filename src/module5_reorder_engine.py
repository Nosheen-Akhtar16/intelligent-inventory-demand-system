"""Module 5: Intelligent Reorder Recommendation (Part A)."""
import math
import pandas as pd
from config.settings import SAFETY_FACTOR


def compute_reorder_recommendations(
    forecast_df: pd.DataFrame, risk_df: pd.DataFrame, safety_factor: float = SAFETY_FACTOR
) -> pd.DataFrame:
    merged = pd.merge(forecast_df, risk_df, on=["Store ID", "Product ID"], how="inner")
    merged["Safety_Factor"] = float(safety_factor)
    merged["Net_Available_Stock"] = merged["Current_Inventory"] + merged["Incoming_Orders"]
    merged["Target_Stock"] = (merged["Forecast_7D_Demand"] * safety_factor).round(2)
    raw_gap = merged["Target_Stock"] - merged["Net_Available_Stock"]
    merged["Recommended_Reorder_Qty"] = raw_gap.apply(lambda x: int(math.ceil(x)) if x > 0 else 0)
    merged["Estimated_Reorder_Cost"] = (merged["Recommended_Reorder_Qty"] * merged["Price"]).round(2)
    merged["Inventory_Coverage_Days"] = (
        merged["Net_Available_Stock"] / merged["Forecast_Daily_Demand"].clip(lower=0.1)
    ).round(2)
    return merged
