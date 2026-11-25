"""Module 10: Responsible AI and Explainability Module (Part B)."""
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def explain_sku_prediction(
    row: pd.Series,
    model_artifacts: Dict[str, Any],
    forecast_metrics: Dict[str, Any],
    risk_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Provide full prediction transparency, reorder math explanation, model evaluation metrics,
    limitations disclosure, and Human-in-the-Loop governance status.
    """
    features: List[str] = model_artifacts["feature_columns"]
    means = np.array(model_artifacts["scaler_mean"], dtype=float)
    scales = np.array(model_artifacts["scaler_scale"], dtype=float)
    coefs = np.array(model_artifacts["coefficients"], dtype=float)

    x_raw = np.array([float(row[f]) for f in features], dtype=float)
    x_scaled = (x_raw - means) / np.maximum(scales, 1e-8)
    contributions = x_scaled * coefs

    contrib_list = []
    for feat, raw_v, contrib_v in zip(features, x_raw, contributions):
        contrib_list.append({
            "feature": feat,
            "value": round(float(raw_v), 3),
            "log_odds_contribution": round(float(contrib_v), 4),
            "direction": "Increases Stockout Risk" if contrib_v > 0 else "Reduces Stockout Risk",
        })
    contrib_list.sort(key=lambda d: abs(d["log_odds_contribution"]), reverse=True)

    reorder_formula = (
        f"Target Stock = {row['Forecast_7D_Demand']:.1f} (7D Forecast) × {row['Safety_Factor']} (Safety Factor) = {row['Target_Stock']:.1f} units | "
        f"Net Available = {int(row['Current_Inventory'])} (On-Hand) + {int(row['Incoming_Orders'])} (Incoming) = {int(row['Net_Available_Stock'])} units | "
        f"Recommended Reorder = max(0, ceil({row['Target_Stock']:.1f} - {int(row['Net_Available_Stock'])})) = {int(row['Recommended_Reorder_Qty'])} units"
    )

    return {
        "store_id": row["Store ID"],
        "product_id": row["Product ID"],
        "risk_category": row["Risk_Category"],
        "stockout_probability": float(row["Stockout_Probability"]),
        "top_risk_drivers": contrib_list[:6],
        "recommendation_explanation": {
            "decision": row["Decision"],
            "rationale": row["Decision_Rationale"],
            "formula_breakdown": reorder_formula,
        },
        "model_transparency": {
            "forecast_model_mae": forecast_metrics["overall_mae"],
            "forecast_model_rmse": forecast_metrics["overall_rmse"],
            "risk_model_accuracy": risk_metrics["accuracy"],
            "risk_model_precision": risk_metrics["precision"],
            "risk_model_recall": risk_metrics["recall"],
            "risk_model_f1": risk_metrics["f1_score"],
            "limitations": [
                "Holt-Winters Exponential Smoothing projects historical weekly seasonality and cannot anticipate sudden supplier outages.",
                "Logistic Regression relies on lagged inventory and velocity features; unannounced marketing campaigns require manual review.",
                "Sustainability indicators quantify inventory overstock balance rather than direct physical waste or carbon emissions.",
            ],
        },
        "human_in_the_loop": {
            "autonomy_level": "Decision Support Only — Human Manager Approval Required",
            "auto_po_execution": False,
        },
    }
