"""Module 4: Stockout Risk Prediction (Logistic Regression — Part A)."""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

RISK_FEATURE_COLUMNS = [
    "Inventory Level", "Units Ordered", "inv_lag_1", "inv_lag_3", "inv_lag_7",
    "sales_lag_1", "sales_lag_7", "rolling_mean_7d", "rolling_std_7d",
    "days_of_supply", "Holiday/Promotion", "Discount", "price_ratio",
]


def train_and_predict_stockout_risk(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Time-aware Logistic Regression classifier predicting SAFE vs AT_RISK."""
    df_sorted = df.sort_values(["Date", "Store ID", "Product ID"]).reset_index(drop=True)

    unique_dates = np.sort(df_sorted["Date"].unique())
    evaluable_dates = unique_dates[:-3]
    cutoff_idx = int(len(evaluable_dates) * train_ratio)
    cutoff_date = evaluable_dates[cutoff_idx]

    train_mask = df_sorted["Date"] < cutoff_date
    test_mask = (df_sorted["Date"] >= cutoff_date) & (df_sorted["Date"] <= evaluable_dates[-1])

    X_train = df_sorted.loc[train_mask, RISK_FEATURE_COLUMNS].astype(float)
    y_train = df_sorted.loc[train_mask, "stockout_risk_label"].astype(int)

    X_test = df_sorted.loc[test_mask, RISK_FEATURE_COLUMNS].astype(float)
    y_test = df_sorted.loc[test_mask, "stockout_risk_label"].astype(int)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": "Logistic Regression (Time-Aware Split)",
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "cutoff_date": str(pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
        "class_labels": ["SAFE", "AT_RISK"],
    }

    latest_date = df_sorted["Date"].max()
    latest_df = df_sorted[df_sorted["Date"] == latest_date].copy().sort_values(["Store ID", "Product ID"]).reset_index(drop=True)

    X_latest = latest_df[RISK_FEATURE_COLUMNS].astype(float)
    latest_probs = pipeline.predict_proba(X_latest)[:, 1]
    latest_preds = (latest_probs >= 0.50).astype(int)

    latest_df["Stockout_Probability"] = np.round(latest_probs, 4)
    latest_df["Risk_Category"] = np.where(latest_preds == 1, "AT_RISK", "SAFE")

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    model_artifacts = {
        "feature_columns": RISK_FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
    }

    keep_cols = list(dict.fromkeys([
        "Store ID", "Product ID", "Stockout_Probability", "Risk_Category",
        "days_of_supply", "rolling_mean_7d", "rolling_std_7d",
    ] + RISK_FEATURE_COLUMNS))

    return latest_df[keep_cols], metrics, model_artifacts
