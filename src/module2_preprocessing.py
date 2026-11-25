"""Module 2: Data Preprocessing and Validation (Part A)."""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def preprocess_and_validate(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform missing-value detection, duplicate removal, date validation,
    negative inventory/sales/price/forecast checks, business-rule validation,
    and lag/rolling feature engineering.
    """
    df = df_raw.copy()
    initial_rows = len(df)

    missing_by_col = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
    total_missing = int(df.isna().sum().sum())

    duplicates_found = int(df.duplicated(subset=["Store ID", "Product ID", "Date"]).sum())
    if duplicates_found > 0:
        df = df.drop_duplicates(subset=["Store ID", "Product ID", "Date"], keep="last").reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = int(df["Date"].isna().sum())
    if invalid_dates > 0:
        df = df.dropna(subset=["Date"]).reset_index(drop=True)

    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    neg_inventory = int((df["Inventory Level"] < 0).sum())
    neg_sales = int((df["Units Sold"] < 0).sum())
    nonpos_price = int((df["Price"] <= 0).sum())
    neg_ordered = int((df["Units Ordered"] < 0).sum())
    neg_forecast = int((df["Demand Forecast"] < 0).sum())

    df.loc[df["Inventory Level"] < 0, "Inventory Level"] = np.nan
    df.loc[df["Units Sold"] < 0, "Units Sold"] = np.nan
    df.loc[df["Price"] <= 0, "Price"] = np.nan
    df.loc[df["Units Ordered"] < 0, "Units Ordered"] = 0
    df.loc[df["Demand Forecast"] < 0, "Demand Forecast"] = np.nan

    numeric_cols = [
        "Inventory Level", "Units Sold", "Units Ordered", "Demand Forecast",
        "Price", "Discount", "Competitor Pricing",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df.groupby(["Store ID", "Product ID"])[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())

    invalid_discounts = int(((df["Discount"] < 0) | (df["Discount"] > 100)).sum())
    df["Discount"] = df["Discount"].clip(0, 100)
    df["Holiday/Promotion"] = df["Holiday/Promotion"].fillna(0).astype(int).clip(0, 1)
    df["Competitor Pricing"] = np.where(df["Competitor Pricing"] <= 0, df["Price"], df["Competitor Pricing"])

    df["Inventory Level"] = df["Inventory Level"].round().astype(int)
    df["Units Sold"] = df["Units Sold"].round().astype(int)
    df["Units Ordered"] = df["Units Ordered"].round().astype(int)

    df["Store_Product"] = df["Store ID"] + "_" + df["Product ID"]
    grp = df.groupby("Store_Product", group_keys=False)

    df["inv_lag_1"] = grp["Inventory Level"].shift(1).bfill()
    df["inv_lag_3"] = grp["Inventory Level"].shift(3).bfill()
    df["inv_lag_7"] = grp["Inventory Level"].shift(7).bfill()

    df["sales_lag_1"] = grp["Units Sold"].shift(1).bfill()
    df["sales_lag_7"] = grp["Units Sold"].shift(7).bfill()

    df["rolling_mean_7d"] = grp["Units Sold"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["rolling_mean_14d"] = grp["Units Sold"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    df["rolling_std_7d"] = grp["Units Sold"].transform(lambda s: s.rolling(7, min_periods=1).std()).fillna(0.0)

    df["net_available_stock"] = df["Inventory Level"] + df["Units Ordered"]
    df["days_of_supply"] = df["net_available_stock"] / np.maximum(df["rolling_mean_7d"], 0.5)
    df["price_ratio"] = df["Price"] / np.maximum(df["Competitor Pricing"], 0.01)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    future_3d_demand = grp["Units Sold"].transform(
        lambda s: s.shift(-1).rolling(3, min_periods=1).sum().shift(-2)
    )
    future_3d_demand = future_3d_demand.fillna(df["rolling_mean_7d"] * 3.0)
    df["future_3d_demand"] = future_3d_demand

    df["stockout_risk_label"] = (
        (df["net_available_stock"] < df["future_3d_demand"]) | (df["days_of_supply"] < 2.35)
    ).astype(int)

    validation_report = {
        "initial_records": int(initial_rows),
        "validated_records": int(len(df)),
        "total_missing_values_detected": total_missing,
        "missing_by_column": missing_by_col,
        "duplicate_records_removed": duplicates_found,
        "invalid_dates_removed": invalid_dates,
        "negative_inventory_repaired": neg_inventory,
        "negative_sales_repaired": neg_sales,
        "negative_demand_forecast_repaired": neg_forecast,
        "non_positive_price_repaired": nonpos_price,
        "negative_orders_repaired": neg_ordered,
        "out_of_range_discounts_clipped": invalid_discounts,
        "engineered_features": [
            "inv_lag_1", "inv_lag_3", "inv_lag_7",
            "sales_lag_1", "sales_lag_7",
            "rolling_mean_7d", "rolling_mean_14d", "rolling_std_7d",
            "net_available_stock", "days_of_supply", "price_ratio",
            "day_of_week", "month", "is_weekend",
        ],
        "status": "PASSED",
    }
    return df, validation_report
