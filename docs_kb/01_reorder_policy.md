# StockPilot AI Inventory Reorder Policy
1. Reorder Quantity Formula: Recommended Reorder Quantity = max(0, ceil((Forecast_7D_Demand * 1.2) - (Current_Inventory + Incoming_Orders))).
2. Safety Factor: A standard 1.2x (20% buffer) safety factor protects against normal daily demand volatility.
3. Inventory Coverage: Inventory Coverage (in days) equals Net Available Stock divided by Forecast Daily Demand. It measures how many days current and incoming stock will last.