"""Module 3: Demand Forecasting Model (Holt-Winters Exponential Smoothing — Part A)."""
import warnings
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from config.settings import HOLDOUT_DAYS, FORECAST_HORIZON_DAYS

warnings.filterwarnings("ignore")


def _fit_holt_winters(series: np.ndarray, forecast_steps: int) -> np.ndarray:
    y = np.maximum(np.asarray(series, dtype=float), 0.0)
    try:
        if len(y) >= 28 and np.std(y) > 1e-4:
            model = ExponentialSmoothing(
                y, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            preds = fit.forecast(forecast_steps)
        else:
            raise ValueError("Short series")
    except Exception:
        try:
            model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True)
            preds = fit.forecast(forecast_steps)
        except Exception:
            preds = np.full(forecast_steps, max(0.1, float(np.mean(y[-7:]))))
    return np.maximum(0.1, np.asarray(preds, dtype=float))


def run_demand_forecasting(
    df: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run individual Holt-Winters time-series forecasting across all 100 store-product pairs."""
    summary_rows = []
    holdout_detail_rows = []

    for (store_id, product_id), g in df.groupby(["Store ID", "Product ID"], sort=True):
        g_sorted = g.sort_values("Date")
        dates = g_sorted["Date"].values
        y = g_sorted["Units Sold"].values.astype(float)

        train_y = y[:-holdout_days]
        test_y = y[-holdout_days:]
        test_dates = pd.to_datetime(dates[-holdout_days:]).strftime("%Y-%m-%d").tolist()

        holdout_preds = _fit_holt_winters(train_y, holdout_days)
        mae = float(np.mean(np.abs(test_y - holdout_preds)))
        rmse = float(np.sqrt(np.mean((test_y - holdout_preds) ** 2)))
        smape = float(
            np.mean(2.0 * np.abs(test_y - holdout_preds) / np.maximum(np.abs(test_y) + np.abs(holdout_preds), 1.0)) * 100.0
        )

        for dt_str, act_val, pred_val in zip(test_dates, test_y, holdout_preds):
            holdout_detail_rows.append({
                "Store ID": store_id,
                "Product ID": product_id,
                "Date": dt_str,
                "Actual_Demand": round(float(act_val), 2),
                "Predicted_Demand": round(float(pred_val), 2),
            })

        future_30d_preds = _fit_holt_winters(y, 30)
        forecast_7d = float(np.sum(future_30d_preds[:horizon_days]))
        forecast_30d = float(np.sum(future_30d_preds))
        forecast_daily = float(np.mean(future_30d_preds[:horizon_days]))

        latest_row = g_sorted.iloc[-1]
        modal_category = str(g_sorted["Category"].mode().iloc[0])
        modal_region = str(g_sorted["Region"].mode().iloc[0])

        summary_rows.append({
            "Store ID": store_id,
            "Product ID": product_id,
            "Category": modal_category,
            "Region": modal_region,
            "Latest_Date": pd.to_datetime(latest_row["Date"]).strftime("%Y-%m-%d"),
            "Current_Inventory": int(latest_row["Inventory Level"]),
            "Incoming_Orders": int(latest_row["Units Ordered"]),
            "Price": float(latest_row["Price"]),
            "Forecast_Daily_Demand": round(forecast_daily, 2),
            "Forecast_7D_Demand": round(forecast_7d, 2),
            "Forecast_30D_Demand": round(forecast_30d, 2),
            "Holdout_MAE": round(mae, 3),
            "Holdout_RMSE": round(rmse, 3),
            "Holdout_sMAPE_Pct": round(smape, 2),
        })

    forecast_summary_df = pd.DataFrame(summary_rows)
    holdout_df = pd.DataFrame(holdout_detail_rows)

    overall_mae = float(np.mean(np.abs(holdout_df["Actual_Demand"] - holdout_df["Predicted_Demand"])))
    overall_rmse = float(np.sqrt(np.mean((holdout_df["Actual_Demand"] - holdout_df["Predicted_Demand"]) ** 2)))
    overall_smape = float(forecast_summary_df["Holdout_sMAPE_Pct"].mean())

    metrics = {
        "model_name": "Holt-Winters Exponential Smoothing (Additive Trend + 7-Day Seasonality)",
        "series_count": int(len(forecast_summary_df)),
        "holdout_days": int(holdout_days),
        "forecast_horizon_days": int(horizon_days),
        "overall_mae": round(overall_mae, 3),
        "overall_rmse": round(overall_rmse, 3),
        "overall_smape_pct": round(overall_smape, 2),
        "min_series_mae": round(float(forecast_summary_df["Holdout_MAE"].min()), 3),
        "max_series_mae": round(float(forecast_summary_df["Holdout_MAE"].max()), 3),
    }

    return forecast_summary_df, holdout_df, metrics
