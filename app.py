import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="FreightCast AI — Capacity Decision System",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        margin-bottom: 10px;
    }
    .savings-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }
    .badge-high {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-normal {
        background-color: #00c0f2;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #21c354;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load artifacts
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "models")
    
    with open(os.path.join(models_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    ridge = joblib.load(os.path.join(models_dir, "ridge_model.pkl"))
    rf = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    gbr = joblib.load(os.path.join(models_dir, "gbr_model.pkl"))
    
    sample_path = os.path.join(models_dir, "recent_sample.csv")
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
    
    return {
        "meta": meta,
        "scaler": scaler,
        "models": {
            "Ridge Regression (Champion)": ridge,
            "Random Forest": rf,
            "Gradient Boosting": gbr
        },
        "sample_df": sample_df
    }

def main():
    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure 'train_export.py' has finished running.")
        return

    meta = artifacts["meta"]
    features = meta["features"]
    feature_meta = meta["feature_meta"]
    assumptions = meta["assumptions"]
    band_edges = meta["band_edges"]

    # Sidebar Header
    st.sidebar.image("https://img.icons8.com/clouds/200/cargo-ship.png", width=120)
    st.sidebar.title("FreightCast Controls")
    st.sidebar.markdown("Configure operational conditions & simulate 7-day ahead capacity.")

    # Model selector
    selected_model_name = st.sidebar.selectbox(
        "Select Forecasting Model",
        options=list(artifacts["models"].keys()),
        index=0
    )
    active_model = artifacts["models"][selected_model_name]

    # Scenario Presets
    st.sidebar.subheader("Quick Scenario Presets")
    preset = st.sidebar.radio(
        "Load Preset Conditions:",
        options=["Normal Baseline", "Port Congestion Surge", "Severe Storm / Bad Weather", "Supply Chain Bottleneck"],
        index=0
    )

    # Defaults from feature_meta
    default_vals = {col: feature_meta[col]["recent"] for col in features}

    # Override defaults based on preset
    if preset == "Port Congestion Surge":
        default_vals["port_congestion_level"] = min(1.0, feature_meta["port_congestion_level"]["max"] * 0.95)
        default_vals["loading_unloading_time"] = feature_meta["loading_unloading_time"]["max"] * 0.90
        default_vals["risk_level"] = 2
    elif preset == "Severe Storm / Bad Weather":
        default_vals["weather_condition_severity"] = feature_meta["weather_condition_severity"]["max"] * 0.95
        default_vals["delay_probability"] = 0.85
        default_vals["eta_variation_hours"] = 15.0
        default_vals["risk_level"] = 2
    elif preset == "Supply Chain Bottleneck":
        default_vals["traffic_congestion_level"] = feature_meta["traffic_congestion_level"]["max"] * 0.9
        default_vals["customs_clearance_time"] = feature_meta["customs_clearance_time"]["max"] * 0.9
        default_vals["lead_time_days"] = feature_meta["lead_time_days"]["max"] * 0.85
        default_vals["risk_level"] = 1

    # Sidebar Sliders for Operational Features
    st.sidebar.subheader("Operational Telemetry Inputs")
    user_inputs = {}

    # Core demand status
    st.sidebar.markdown("**Current Demand Baseline**")
    user_inputs["planning_demand"] = st.sidebar.slider(
        "Current 30-Day Planning Demand",
        min_value=float(feature_meta["planning_demand"]["min"] * 0.8),
        max_value=float(feature_meta["planning_demand"]["max"] * 1.2),
        value=float(default_vals["planning_demand"]),
        step=20.0
    )

    # Lags track planning demand proportionally
    ratio = user_inputs["planning_demand"] / max(1.0, default_vals["planning_demand"])
    for lag in [24, 48, 168, 336, 720]:
        user_inputs[f"planning_lag_{lag}"] = default_vals[f"planning_lag_{lag}"] * ratio
    user_inputs["demand_mean_7d"] = default_vals["demand_mean_7d"] * ratio
    user_inputs["demand_std_7d"] = default_vals["demand_std_7d"]

    # Calendar features
    user_inputs["hour_of_day"] = st.sidebar.slider("Hour of Day (0-23)", 0, 23, int(default_vals.get("hour_of_day", 12)))
    user_inputs["day_of_week"] = st.sidebar.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x], index=int(default_vals.get("day_of_week", 0)))
    user_inputs["month"] = st.sidebar.selectbox("Month", options=list(range(1, 13)), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1], index=int(default_vals.get("month", 5))-1)

    # Risk level
    risk_mapping = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}
    rev_risk = {v: k for k, v in risk_mapping.items()}
    chosen_risk_str = st.sidebar.selectbox("Supply Chain Risk Level", list(risk_mapping.keys()), index=int(default_vals.get("risk_level", 0)))
    user_inputs["risk_level"] = risk_mapping[chosen_risk_str]

    with st.sidebar.expander("⚡ Port & Transit Conditions", expanded=True):
        user_inputs["port_congestion_level"] = st.slider(
            "Port Congestion Level",
            float(feature_meta["port_congestion_level"]["min"]),
            float(feature_meta["port_congestion_level"]["max"]),
            float(default_vals["port_congestion_level"])
        )
        user_inputs["traffic_congestion_level"] = st.slider(
            "Traffic Congestion Level",
            float(feature_meta["traffic_congestion_level"]["min"]),
            float(feature_meta["traffic_congestion_level"]["max"]),
            float(default_vals["traffic_congestion_level"])
        )
        user_inputs["weather_condition_severity"] = st.slider(
            "Weather Condition Severity",
            float(feature_meta["weather_condition_severity"]["min"]),
            float(feature_meta["weather_condition_severity"]["max"]),
            float(default_vals["weather_condition_severity"])
        )
        user_inputs["eta_variation_hours"] = st.slider(
            "ETA Variation (Hours)",
            float(feature_meta["eta_variation_hours"]["min"]),
            float(feature_meta["eta_variation_hours"]["max"]),
            float(default_vals["eta_variation_hours"])
        )

    with st.sidebar.expander("📦 Warehouse & Supply Operations", expanded=False):
        user_inputs["warehouse_inventory_level"] = st.slider(
            "Warehouse Inventory Level",
            float(feature_meta["warehouse_inventory_level"]["min"]),
            float(feature_meta["warehouse_inventory_level"]["max"]),
            float(default_vals["warehouse_inventory_level"])
        )
        user_inputs["shipping_costs"] = st.slider(
            "Current Shipping Cost Index",
            float(feature_meta["shipping_costs"]["min"]),
            float(feature_meta["shipping_costs"]["max"]),
            float(default_vals["shipping_costs"])
        )
        user_inputs["lead_time_days"] = st.slider(
            "Lead Time (Days)",
            float(feature_meta["lead_time_days"]["min"]),
            float(feature_meta["lead_time_days"]["max"]),
            float(default_vals["lead_time_days"])
        )
        user_inputs["supplier_reliability_score"] = st.slider(
            "Supplier Reliability Score",
            float(feature_meta["supplier_reliability_score"]["min"]),
            float(feature_meta["supplier_reliability_score"]["max"]),
            float(default_vals["supplier_reliability_score"])
        )
        user_inputs["customs_clearance_time"] = st.slider(
            "Customs Clearance Time",
            float(feature_meta["customs_clearance_time"]["min"]),
            float(feature_meta["customs_clearance_time"]["max"]),
            float(default_vals["customs_clearance_time"])
        )
        user_inputs["loading_unloading_time"] = st.slider(
            "Loading/Unloading Time",
            float(feature_meta["loading_unloading_time"]["min"]),
            float(feature_meta["loading_unloading_time"]["max"]),
            float(default_vals["loading_unloading_time"])
        )
        user_inputs["handling_equipment_availability"] = st.slider(
            "Equipment Availability",
            float(feature_meta["handling_equipment_availability"]["min"]),
            float(feature_meta["handling_equipment_availability"]["max"]),
            float(default_vals["handling_equipment_availability"])
        )
        user_inputs["delay_probability"] = st.slider(
            "Delay Probability",
            float(feature_meta["delay_probability"]["min"]),
            float(feature_meta["delay_probability"]["max"]),
            float(default_vals["delay_probability"])
        )
        user_inputs["order_fulfillment_status"] = default_vals.get("order_fulfillment_status", 1.0)

    # Convert user inputs to aligned feature array
    input_row = pd.DataFrame([[user_inputs[col] for col in features]], columns=features)
    input_scaled = artifacts["scaler"].transform(input_row)

    # Predict
    predicted_demand = float(active_model.predict(input_scaled)[0])
    current_demand = user_inputs["planning_demand"]
    pct_change = ((predicted_demand - current_demand) / current_demand) * 100.0

    # Demand Band
    if predicted_demand < band_edges[0]:
        band_label = "Low Demand"
        band_class = "badge-low"
    elif predicted_demand < band_edges[1]:
        band_label = "Normal Demand"
        band_class = "badge-normal"
    else:
        band_label = "High Demand (Alert)"
        band_class = "badge-high"

    # Capacity Booking Calculator
    ships_needed = int(np.ceil(predicted_demand / assumptions["demand_per_ship"]))
    sqft_needed = int(np.ceil(predicted_demand * assumptions["sqft_per_demand"]))
    workers_needed = int(np.ceil(predicted_demand / assumptions["demand_per_worker"]))

    early_cost = (
        ships_needed * assumptions["ship_early"]
        + sqft_needed * assumptions["warehouse_early"]
        + workers_needed * assumptions["labour_early"]
    )
    rush_cost = (
        ships_needed * assumptions["ship_early"] * (1 + assumptions["ship_rush_premium"])
        + sqft_needed * assumptions["warehouse_early"] * (1 + assumptions["warehouse_rush_premium"])
        + workers_needed * assumptions["labour_early"] * (1 + assumptions["labour_rush_premium"])
    )
    net_savings = rush_cost - early_cost

    # ==========================
    # Main Dashboard Tabs
    # ==========================
    st.title("🚢 FreightCast AI — 7-Day Planning & Booking Engine")
    st.markdown("Dynamic machine learning system for advance logistics resource allocation and cost minimization.")

    tab1, tab2, tab3 = st.tabs(["🎯 7-Day Capacity Planner", "📊 Historical Insights & EDA", "⚙️ Model Benchmarks & Logic"])

    with tab1:
        # Top Savings Announcement Card
        st.markdown(f"""
        <div class="savings-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: white;">💰 Projected Advance Booking Savings: ${net_savings:,.0f}</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">
                        By committing 7 days ahead, you avoid the {int(assumptions['ship_rush_premium']*100)}% ship and {int(assumptions['warehouse_rush_premium']*100)}% warehouse rush surcharges.
                    </p>
                </div>
                <div style="text-align: right;">
                    <span class="{band_class}" style="font-size: 1.1rem;">{band_label}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4 Key Metrics in Columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="7-Day Demand Forecast",
                value=f"{predicted_demand:,.0f} units",
                delta=f"{pct_change:+.2f}% vs Current"
            )
        with col2:
            st.metric(
                label="Ships to Charter",
                value=f"{ships_needed} vessels",
                help=f"Based on {assumptions['demand_per_ship']} units per cargo ship"
            )
        with col3:
            st.metric(
                label="Warehouse Floor Space",
                value=f"{sqft_needed:,.0f} sq ft",
                help=f"Based on {assumptions['sqft_per_demand']} sq ft per unit demand"
            )
        with col4:
            st.metric(
                label="Dock Labour Needed",
                value=f"{workers_needed} staff",
                help=f"Based on {assumptions['demand_per_worker']} units processed per worker"
            )

        st.markdown("---")

        # Two Column Visuals: Gauge Chart & Cost Comparison
        vcol1, vcol2 = st.columns([1, 1])

        with vcol1:
            st.subheader("Forecast vs Historical Capacity Bands")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_demand,
                domain={'x': [0, 1], 'y': [0, 1]},
                delta={'reference': current_demand, 'position': "top"},
                gauge={
                    'axis': {'range': [band_edges[0] * 0.9, band_edges[1] * 1.1]},
                    'bar': {'color': "#1e3c72"},
                    'steps': [
                        {'range': [0, band_edges[0]], 'color': "rgba(33, 195, 84, 0.25)"},
                        {'range': [band_edges[0], band_edges[1]], 'color': "rgba(0, 192, 242, 0.25)"},
                        {'range': [band_edges[1], band_edges[1] * 1.3], 'color': "rgba(255, 75, 75, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': band_edges[1]
                    }
                }
            ))
            fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with vcol2:
            st.subheader("Financial Comparison: Early vs Rush Booking")
            cost_df = pd.DataFrame({
                "Booking Strategy": ["Advance Booking (7 Days)", "Rush Market (Spot Rate)"],
                "Total Cost ($)": [early_cost, rush_cost],
                "Color": ["#21c354", "#ff4b4b"]
            })
            fig_cost = px.bar(
                cost_df,
                x="Booking Strategy",
                y="Total Cost ($)",
                text_auto="$,.0f",
                color="Booking Strategy",
                color_discrete_map={"Advance Booking (7 Days)": "#2a5298", "Rush Market (Spot Rate)": "#e74c3c"}
            )
            fig_cost.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cost, use_container_width=True)

        # Feature Impact Explanation
        st.subheader("💡 Why is the forecast at this level? (Factor Analysis)")
        ridge_weights = meta["ridge_weights"]
        
        # Calculate impact for top operational factors
        impacts = []
        for feat in meta["ops_features"]:
            val_norm = (user_inputs[feat] - feature_meta[feat]["mean"]) / (feature_meta[feat]["max"] - feature_meta[feat]["min"] + 1e-6)
            weight = ridge_weights.get(feat, 0.0)
            impact = val_norm * weight
            impacts.append({
                "Feature": feat.replace("_", " ").title(),
                "Direction": "Pushes Demand Higher" if impact >= 0 else "Pushes Demand Lower",
                "Impact Score": abs(impact),
                "Current Value": f"{user_inputs[feat]:.2f}"
            })
        
        impact_df = pd.DataFrame(impacts).sort_values("Impact Score", ascending=False).head(6)
        
        fig_imp = px.bar(
            impact_df,
            x="Impact Score",
            y="Feature",
            color="Direction",
            orientation="h",
            color_discrete_map={"Pushes Demand Higher": "#e74c3c", "Pushes Demand Lower": "#27ae60"}
        )
        fig_imp.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab2:
        st.subheader("Supply Chain History & Telemetry Samples")
        st.markdown("Explore the 30-day moving average demand pattern compared with recent operational telemetry.")
        if artifacts["sample_df"] is not None:
            st.dataframe(artifacts["sample_df"], use_container_width=True)
            
            # Trend Chart
            fig_trend = px.line(
                artifacts["sample_df"],
                x="timestamp",
                y=["planning_demand", "target"],
                labels={"value": "Demand Units", "timestamp": "Timeline", "variable": "Series"},
                title="Historical Planning Demand vs 7-Day Forward Target",
                color_discrete_sequence=["#1f77b4", "#ff7f0e"]
            )
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Sample telemetry will be displayed once the full dataset is loaded.")

    with tab3:
        st.subheader("Model Benchmarks on Unseen Test Data")
        metrics_df = pd.DataFrame(meta["test_metrics"]).T.reset_index()
        metrics_df.rename(columns={"index": "Model Name"}, inplace=True)
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.dataframe(metrics_df.style.highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
                                    .highlight_max(subset=["R2"], color="#d4edda"), use_container_width=True)
        with col_m2:
            st.markdown(r"""
            **Key Methodological Insights:**
            * **Why raw demand failed:** Raw hourly demand has $|r| \le 0.01$ and autocorrelation $\approx 0$, making single-hour prediction noise.
            * **Why 30-day rolling demand works:** A 30-day moving window isolates operational momentum ($r > 0.85$ correlation with the 7-day future).
            * **Why Ridge is Champion:** Regularized $L_2$ regression avoids overfitting the collinear lag variables while offering exact, transparent coefficient decomposition.
            """)

    # Footer
    st.markdown("---")
    st.caption("Machine Learning Innovative Project — Standalone Freight Forecasting Application · 2026")

if __name__ == "__main__":
    main()
