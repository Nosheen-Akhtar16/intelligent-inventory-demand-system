# StockPilot AI — Automated Inventory & Scenario Decision Report
**Generated Timestamp:** 2026-11-20 02:04:19  
**Dataset Snapshot Date:** 2024-01-01  

---

## 1. Dataset & Validation Summary (Part A — Modules 1 & 2)
- **Total Records:** 73,100 (15 attributes)
- **Store-Product Combinations:** 100 (5 Stores × 20 Products)
- **Historical Window:** 2022-01-01 to 2024-01-01
- **Data Quality Audit:** PASSED (Negative Demand Forecasts repaired: 673, Missing imputed: 0)

## 2. Demand Forecasting Performance (Part A — Module 3: Holt-Winters)
- **Model:** Holt-Winters Exponential Smoothing (Additive Trend + 7-Day Seasonality)
- **30-Day Holdout Overall MAE:** 87.431 units/day
- **30-Day Holdout Overall RMSE:** 106.590 units/day
- **30-Day Holdout Overall sMAPE:** 71.45%

## 3. Stockout Risk Classification (Part A — Module 4: Logistic Regression)
- **Time-Aware Split Cutoff:** 2023-08-06 (Train: 58,200 | Test: 14,600)
- **Accuracy:** 78.39% | **Precision:** 80.64% | **Recall:** 84.83% | **F1-Score:** 82.68% | **ROC-AUC:** 0.8781
- **Current Snapshot Risk Split:** 68 `AT_RISK` vs 32 `SAFE` store-product series

## 4. Inventory Decision Engine & Reorder Recommendations (Part A — Modules 5 & 6)
- **URGENT REORDER:** 53 items
- **REORDER:** 15 items
- **MONITOR:** 17 items
- **HOLD:** 15 items
- **Total Recommended Reorder Quantity (Safety Factor = 1.2):** 76,537 units (Est. Value: $4,063,963.19)

### Top Priority Replenishment Actions (`URGENT REORDER`)
| Store | Product | Category | Current Inv | Incoming | 7D Forecast | Coverage (Days) | Stockout Prob | Reorder Qty | Decision |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| S002 | P0007 | Clothing | 53 | 40 | 984.9 | 0.66 | 0.99 | **1089** | URGENT REORDER |
| S003 | P0007 | Electronics | 108 | 74 | 979.5 | 1.30 | 0.99 | **994** | URGENT REORDER |
| S005 | P0007 | Toys | 59 | 34 | 930.9 | 0.70 | 0.99 | **1025** | URGENT REORDER |
| S004 | P0012 | Clothing | 85 | 36 | 918.3 | 0.92 | 0.99 | **981** | URGENT REORDER |
| S003 | P0011 | Toys | 137 | 46 | 1014.7 | 1.26 | 0.98 | **1035** | URGENT REORDER |
| S004 | P0011 | Electronics | 155 | 47 | 976.6 | 1.45 | 0.98 | **970** | URGENT REORDER |
| S004 | P0017 | Furniture | 100 | 99 | 965.1 | 1.44 | 0.98 | **960** | URGENT REORDER |
| S005 | P0003 | Toys | 98 | 118 | 1014.4 | 1.49 | 0.98 | **1002** | URGENT REORDER |
| S004 | P0002 | Furniture | 181 | 85 | 843.3 | 2.21 | 0.97 | **746** | URGENT REORDER |
| S004 | P0007 | Groceries | 134 | 97 | 836.5 | 1.93 | 0.97 | **773** | URGENT REORDER |
| S001 | P0003 | Clothing | 69 | 191 | 1019.5 | 1.79 | 0.96 | **964** | URGENT REORDER |
| S002 | P0009 | Toys | 62 | 183 | 988.8 | 1.73 | 0.96 | **942** | URGENT REORDER |
| S004 | P0013 | Groceries | 113 | 67 | 857.4 | 1.47 | 0.96 | **849** | URGENT REORDER |
| S002 | P0006 | Groceries | 160 | 78 | 835.7 | 1.99 | 0.95 | **765** | URGENT REORDER |
| S004 | P0019 | Groceries | 65 | 166 | 960.6 | 1.68 | 0.94 | **922** | URGENT REORDER |

## 5. What-If Demand Variation Scenario (+20.0% Demand Shock — Module 7)
| Metric | Baseline (Normal Demand) | What-If Scenario | Delta |
| :--- | ---: | ---: | ---: |
| `URGENT REORDER` Count | 53 | 57 | +4 |
| `REORDER` Count | 15 | 14 | -1 |
| Total Reorder Units Needed | 76,537 | 99,379 | +22,842 |
| Average Inventory Coverage (Days) | 2.78 | 2.32 | -0.46 |
| Items with Projected 7D Shortage | — | 100 | — |
| Total Projected Shortage Units | — | 76,537 | — |

## 6. Sustainability Intelligence Layer (Part B — Module 9: UN SDG-12 Indicators)
- **Overstocked Store-Product Items Flagged:** 7 / 100
- **High Potential Waste Risk Items:** 0
- **Total Excess Inventory Units:** 0 units
- **Average Inventory Utilization Rate:** 100.0%
- **SDG-12 Inventory Efficiency Score:** 95.1 / 100
- *Disclosure: These metrics are inventory-based operational indicators, not measured physical waste or carbon-emission reductions.*

## 7. Responsible AI Governance Notice (Part B — Module 10)
- All reorder quantities and risk alerts operate strictly as **human-in-the-loop decision support**. No automated purchase orders are executed without human approval.