"""dashboard/app.py — Interactive Streamlit Inventory Dashboard (Module 13)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.services import ensure_initialized, execute_full_pipeline
from src.database import load_table, load_metadata, record_hitl_decision, get_hitl_history
from src.module7_whatif_simulator import run_whatif_simulation
from src.module10_responsible_ai import explain_sku_prediction
from src.module11_rag_copilot import InventoryRAGCopilot
from src.module12_agentic_assistant import InventoryAgenticAssistant

st.set_page_config(page_title="StockPilot AI — Inventory Intelligence", page_icon="📦", layout="wide")
ensure_initialized()

decision_df = load_table("inventory_decisions")
holdout_df = load_table("forecast_holdout")
sus_df = load_table("sustainability_analysis")
profile = load_metadata("dataset_profile")
val_rep = load_metadata("validation_report")
fc_metrics = load_metadata("forecast_metrics")
risk_metrics = load_metadata("risk_metrics")
model_art = load_metadata("model_artifacts")
sus_summary = load_metadata("sustainability_summary")
report_md = load_metadata("latest_report_md").get("markdown", "")

st.sidebar.title("📦 StockPilot AI")
st.sidebar.caption("Part A: Forecasting & Risk Engine | Part B: Responsible AI, RAG, Agentic AI & SDG-12")

section = st.sidebar.radio(
    "Dashboard Sections",
    [
        "1. Overview",
        "2. Demand Forecasting",
        "3. Risk Monitoring",
        "4. Reorder Management",
        "5. What-if Analysis",
        "6. Sustainability",
        "7. AI Copilot",
        "8. Decision Assistant",
    ],
)

if st.sidebar.button("🔄 Re-run Full Pipeline"):
    with st.spinner("Running all 13 StockPilot AI modules..."):
        execute_full_pipeline()
    st.rerun()

if section == "1. Overview":
    st.title("📊 StockPilot AI — Inventory Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records Analyzed", f"{profile['total_records']:,}", f"{profile['unique_store_product_combinations']} Store-Products")
    c2.metric("Urgent Reorders", int((decision_df["Decision"] == "URGENT REORDER").sum()))
    c3.metric("At-Risk Items", int((decision_df["Risk_Category"] == "AT_RISK").sum()), f"Acc: {risk_metrics['accuracy']*100:.1f}%")
    c4.metric("Recommended Reorder", f"{int(decision_df['Recommended_Reorder_Qty'].sum()):,} units")
    c5.metric("SDG-12 Efficiency", f"{sus_summary['sdg12_efficiency_score']} / 100")

    col1, col2 = st.columns(2)
    with col1:
        fig_dec = px.pie(decision_df, names="Decision", title="Decision Engine Distribution", hole=0.45)
        st.plotly_chart(fig_dec, use_container_width=True)
    with col2:
        cat_agg = decision_df.groupby("Category", as_index=False)[["Current_Inventory", "Forecast_7D_Demand", "Recommended_Reorder_Qty"]].sum()
        fig_cat = px.bar(cat_agg, x="Category", y=["Current_Inventory", "Forecast_7D_Demand", "Recommended_Reorder_Qty"], barmode="group", title="Inventory vs 7D Forecast by Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.download_button("📥 Download Automated Scenario Report (.md)", data=report_md, file_name="StockPilot_Scenario_Report.md")

elif section == "2. Demand Forecasting":
    st.title("📈 Holt-Winters Demand Forecasting (30-Day Holdout)")
    k1, k2, k3 = st.columns(3)
    k1.metric("Overall 30-Day MAE", f"{fc_metrics['overall_mae']:.3f}")
    k2.metric("Overall RMSE", f"{fc_metrics['overall_rmse']:.3f}")
    k3.metric("Series Modeled", fc_metrics["series_count"])

    s_sel = st.selectbox("Select Store", sorted(decision_df["Store ID"].unique()))
    p_sel = st.selectbox("Select Product", sorted(decision_df["Product ID"].unique()))
    sub_ho = holdout_df[(holdout_df["Store ID"] == s_sel) & (holdout_df["Product ID"] == p_sel)]
    row_info = decision_df[(decision_df["Store ID"] == s_sel) & (decision_df["Product ID"] == p_sel)].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub_ho["Date"], y=sub_ho["Actual_Demand"], mode="lines+markers", name="Actual Demand"))
    fig.add_trace(go.Scatter(x=sub_ho["Date"], y=sub_ho["Predicted_Demand"], mode="lines+markers", name="Holt-Winters Forecast"))
    fig.update_layout(title=f"30-Day Holdout: {s_sel} / {p_sel} (Series MAE: {row_info['Holdout_MAE']})")
    st.plotly_chart(fig, use_container_width=True)

elif section == "3. Risk Monitoring":
    st.title("⚠️ Stockout Risk Monitoring & Responsible AI Explainability")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{risk_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{risk_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{risk_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{risk_metrics['f1_score']*100:.2f}%")

    st.subheader("🔍 Per-Product Prediction & Reorder Explanation")
    c_a, c_b = st.columns(2)
    s_exp = c_a.selectbox("Store ID", sorted(decision_df["Store ID"].unique()), key="exp_s")
    p_exp = c_b.selectbox("Product ID", sorted(decision_df["Product ID"].unique()), key="exp_p")
    target_row = decision_df[(decision_df["Store ID"] == s_exp) & (decision_df["Product ID"] == p_exp)].iloc[0]
    exp_data = explain_sku_prediction(target_row, model_art, fc_metrics, risk_metrics)

    st.info(f"**Risk Category:** `{exp_data['risk_category']}` | **Stockout Probability:** `{exp_data['stockout_probability']:.2%}` | **Decision:** `{exp_data['recommendation_explanation']['decision']}`")
    st.write("**Reorder Calculation Breakdown:**", exp_data["recommendation_explanation"]["formula_breakdown"])
    st.dataframe(exp_data["top_risk_drivers"], use_container_width=True)

elif section == "4. Reorder Management":
    st.title("🛒 Intelligent Reorder Management & Human-in-the-Loop Approval")
    st.caption("Responsible AI Governance: No automatic purchase orders are issued without human manager sign-off.")
    st.dataframe(
        decision_df[["Store ID", "Product ID", "Category", "Decision", "Current_Inventory", "Incoming_Orders", "Forecast_7D_Demand", "Inventory_Coverage_Days", "Stockout_Probability", "Recommended_Reorder_Qty"]],
        use_container_width=True,
    )

    st.subheader("✅ Human-in-the-Loop Purchase Order Approval")
    urgent_sub = decision_df[decision_df["Recommended_Reorder_Qty"] > 0]
    with st.form("hitl_form"):
        col1, col2, col3, col4 = st.columns(4)
        sel_pair = col1.selectbox("Store-Product", [f"{r['Store ID']}_{r['Product ID']}" for _, r in urgent_sub.iterrows()])
        st_id, pr_id = sel_pair.split("_")
        rec_q = int(urgent_sub[(urgent_sub["Store ID"] == st_id) & (urgent_sub["Product ID"] == pr_id)]["Recommended_Reorder_Qty"].iloc[0])
        app_q = col2.number_input("Approved Qty", min_value=0, value=rec_q)
        act_status = col3.selectbox("Action", ["APPROVED", "MODIFIED", "REJECTED"])
        note = col4.text_input("Reviewer Note", placeholder="Approved after supplier check")
        if st.form_submit_button("Submit Decision"):
            record_hitl_decision(st_id, pr_id, rec_q, int(app_q), act_status, note)
            st.success(f"Recorded {act_status} for {st_id}/{pr_id} ({app_q} units).")

    hist = get_hitl_history()
    if hist:
        st.dataframe(hist, use_container_width=True)

elif section == "5. What-if Analysis":
    st.title("🔮 What-if Demand Variation Simulator")
    pct = st.slider("Demand Variation (%)", min_value=-50, max_value=100, value=20, step=5)
    sim_df, sim_sum = run_whatif_simulation(decision_df, demand_change_pct=float(pct))

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Urgent Reorders", sim_sum["scenario_urgent_count"], f"{sim_sum['scenario_urgent_count'] - sim_sum['baseline_urgent_count']:+d} vs Baseline")
    w2.metric("Scenario Reorder Units", f"{sim_sum['scenario_total_reorder_units']:,}", f"{sim_sum['additional_reorder_units_needed']:+,d}")
    w3.metric("Items with 7D Shortage", sim_sum["items_with_projected_shortage"])
    w4.metric("Projected Shortage Units", f"{sim_sum['total_projected_shortage_units']:,}")

    st.dataframe(
        sim_df[["Store ID", "Product ID", "Decision", "Scenario_Decision", "Inventory_Coverage_Days", "Scenario_Coverage_Days", "Recommended_Reorder_Qty", "Scenario_Reorder_Qty", "Scenario_Shortage_Units"]],
        use_container_width=True,
    )

elif section == "6. Sustainability":
    st.title("🌱 Sustainability Intelligence Layer (UN SDG-12 Indicators)")
    st.warning(sus_summary["disclaimer"])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("SDG-12 Efficiency Score", f"{sus_summary['sdg12_efficiency_score']} / 100")
    s2.metric("Overstocked SKUs", sus_summary["overstock_count"])
    s3.metric("High Waste Risk SKUs", sus_summary["high_waste_risk_count"])
    s4.metric("Total Excess Units", f"{sus_summary['total_excess_units']:,}")

    st.dataframe(
        sus_df[sus_df["Is_Overstocked"]][["Store ID", "Product ID", "Category", "Net_Available_Stock", "Forecast_7D_Demand", "Inventory_Coverage_Days", "Excess_Stock_Units", "Potential_Waste_Risk", "Sustainability_Recommendation"]],
        use_container_width=True,
    )

elif section == "7. AI Copilot":
    st.title("📚 RAG-Based Inventory Copilot")
    preset = st.selectbox(
        "Example Queries",
        [
            "What is the reorder policy?",
            "Why is a particular product marked AT_RISK?",
            "Explain the inventory recommendation for Store S001.",
            "What does inventory coverage mean?",
        ],
    )
    user_q = st.text_input("Ask StockPilot Copilot:", value=preset)
    if st.button("Ask Copilot"):
        copilot = InventoryRAGCopilot()
        copilot.build_index()
        res = copilot.answer_query(user_q, decision_df)
        st.markdown(res["answer"])
        with st.expander("View Retrieved Knowledge Base Chunks"):
            st.json(res["retrieved_chunks"])

elif section == "8. Decision Assistant":
    st.title("🤖 Agentic AI Inventory Decision Assistant (Tool Calling)")
    agent_preset = st.selectbox(
        "Example Agent Workflows",
        [
            "Check which products need urgent restocking.",
            "Run a +20% demand surge simulation for Store S001.",
            "Audit overstock sustainability risks and suggest stock adjustments.",
        ],
    )
    agent_q = st.text_input("Instruction for Agent:", value=agent_preset)
    if st.button("Execute Agent"):
        agent = InventoryAgenticAssistant(decision_df, sus_df)
        out = agent.run(agent_q)
        st.markdown(out["response"])
        with st.expander("🔎 Inspect Multi-Step Plan & Executed Tool Trace"):
            st.write("**Plan:**", out["plan"])
            st.json(out["tool_trace"])
