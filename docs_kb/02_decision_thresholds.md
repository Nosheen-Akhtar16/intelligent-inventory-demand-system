# Stockout Risk & Inventory Decision Engine Rules
- URGENT REORDER: Triggered when Stockout Probability >= 0.70 OR Inventory Coverage < 1.6 days (with positive reorder gap).
- REORDER: Triggered when Stockout Probability >= 0.50 (AT_RISK) OR Inventory Coverage < 2.6 days.
- MONITOR: Triggered when Stockout Probability is between 0.30 and 0.49 OR Inventory Coverage < 3.8 days.
- HOLD: Triggered when product is SAFE with Inventory Coverage >= 3.8 days and low stockout probability.