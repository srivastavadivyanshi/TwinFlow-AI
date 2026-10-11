import streamlit as st
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go

from ai_engine import (
    train_model,
    predict_production,
    optimize_operation
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TwinFlow AI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: white;
}

/* Header */

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #8fa2b9;
    font-size: 16px;
    margin-bottom: 20px;
}

/* Cards */

.metric-card {
    background-color: #0c1b2e;
    border: 1px solid #1c3048;
    border-radius: 14px;
    padding: 20px;
    min-height: 130px;
}

.metric-title {
    color: #8fa2b9;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.metric-status {
    color: #2dd8a4;
    font-size: 13px;
    margin-top: 5px;
}

/* Digital Twin */

.twin-card {
    background-color: #101f33;
    border: 1px solid #28415e;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    min-height: 150px;
}

.twin-icon {
    font-size: 40px;
}

.twin-title {
    color: white;
    font-size: 16px;
    font-weight: 700;
}

.twin-value {
    color: #8fa2b9;
    margin-top: 8px;
}

/* AI Box */

.ai-box {
    background-color: #111d36;
    border: 1px solid #394d75;
    border-radius: 14px;
    padding: 24px;
}

.ai-title {
    color: #a58cff;
    font-size: 22px;
    font-weight: 700;
}

.ai-text {
    color: #aebbd0;
    line-height: 1.6;
}

/* Equipment */

.equipment-card {
    background-color: #0c1b2e;
    border: 1px solid #1c3048;
    border-radius: 12px;
    padding: 18px;
}

/* Status */

.status-online {
    color: #2dd8a4;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return train_model()


model = load_model()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data_path = Path(__file__).resolve().parent / "data" / "well_production.csv"
    if not data_path.exists():
        st.error(f"Dataset not found: {data_path}")
        st.stop()
    return pd.read_csv(data_path)


df = load_data()


# =========================================================
# CURRENT SENSOR VALUES
# =========================================================

pressure = 245
temperature = 78
flow_rate = 126
water_cut = 25

choke = 62
pump_speed = 80


# =========================================================
# CURRENT PRODUCTION
# =========================================================

current_production = predict_production(

    model,

    pressure,
    temperature,
    flow_rate,

    choke,
    pump_speed,

    water_cut
)


# =========================================================
# OPTIMIZATION
# =========================================================

optimization = optimize_operation(

    model,

    pressure,
    temperature,
    flow_rate,

    water_cut
)


optimized_production = optimization[
    "production"
]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🛢️ TwinFlow AI</h2>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Well-to-Surface Digital Twin"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Digital Twin",
            "AI Optimization",
            "Equipment",
            "Dataset"
        ]
    )

    st.divider()

    st.markdown("### System Status")

    st.write("🟢 Sensors Online")

    st.write("🟢 Digital Twin Active")

    st.write("🟢 XGBoost Model Ready")

    st.write("🟢 Optimizer Ready")

    st.divider()

    st.caption(
        "Smart Automation Prototype"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛢️ TwinFlow AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Digital Twin for Well-to-Surface Optimization • Smart Automation'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LIVE STATUS
# =========================================================

col_status1, col_status2 = st.columns([5, 1])

with col_status2:

    st.success("● LIVE")


st.divider()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.subheader("📊 Live Production Metrics")

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            🛢️ Production
            </div>

            <div class="metric-value">
            {current_production:.0f}
            </div>

            <div class="metric-status">
            bbl/day
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            📈 Well Pressure
            </div>

            <div class="metric-value">
            {pressure}
            </div>

            <div class="metric-status">
            bar • Normal
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            💧 Flow Rate
            </div>

            <div class="metric-value">
            {flow_rate}
            </div>

            <div class="metric-status">
            m³/h
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
            🌡️ Temperature
            </div>

            <div class="metric-value">
            {temperature}
            </div>

            <div class="metric-status">
            °C • Stable
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()

    # =====================================================
    # DIGITAL TWIN
    # =====================================================

    st.subheader(
        "🔄 Digital Twin — Well to Surface"
    )

    n1, a1, n2, a2, n3, a3, n4 = st.columns(7)


    with n1:

        st.markdown(
            f"""
            <div class="twin-card">

            <div class="twin-icon">🛢️</div>

            <div class="twin-title">
            Production Well
            </div>

            <div class="twin-value">
            {pressure} bar
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with a1:

        st.markdown(
            "<h1 style='text-align:center'>→</h1>",
            unsafe_allow_html=True
        )


    with n2:

        st.markdown(
            f"""
            <div class="twin-card">

            <div class="twin-icon">🔵</div>

            <div class="twin-title">
            Flowline
            </div>

            <div class="twin-value">
            {flow_rate} m³/h
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with a2:

        st.markdown(
            "<h1 style='text-align:center'>→</h1>",
            unsafe_allow_html=True
        )


    with n3:

        st.markdown(
            """
            <div class="twin-card">

            <div class="twin-icon">⚙️</div>

            <div class="twin-title">
            Separator
            </div>

            <div class="twin-value">
            Normal
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with a3:

        st.markdown(
            "<h1 style='text-align:center'>→</h1>",
            unsafe_allow_html=True
        )


    with n4:

        st.markdown(
            f"""
            <div class="twin-card">

            <div class="twin-icon">🏭</div>

            <div class="twin-title">
            Surface Facility
            </div>

            <div class="twin-value">
            {current_production:.0f} bbl/day
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # =====================================================
    # AI RECOMMENDATION
    # =====================================================

    st.subheader("🤖 AI Optimization")

    ai1, ai2 = st.columns([2, 1])


    with ai1:

        st.markdown(
            f"""
            <div class="ai-box">

            <div class="ai-title">
            ⚡ Smart Recommendation
            </div>

            <p class="ai-text">

            XGBoost predicts production based on
            well and surface operating parameters.

            The optimization engine searches for
            an improved choke and pump operating point
            while applying configured constraints.

            </p>

            <hr>

            <b>Current Choke:</b>
            {choke}%

            <br><br>

            <b>Recommended Choke:</b>
            {optimization["choke"]}%

            <br><br>

            <b>Recommended Pump Speed:</b>
            {optimization["pump_speed"]}%

            </div>
            """,
            unsafe_allow_html=True
        )


    with ai2:

        st.metric(
            "Current Production",
            f"{current_production:.0f} bbl/day"
        )

        st.metric(
            "Optimized Production",
            f"{optimized_production:.0f} bbl/day"
        )

        improvement = (

            (
                optimized_production
                - current_production
            )
            / current_production
            * 100
        )

        st.metric(
            "Predicted Change",
            f"{improvement:.1f}%"
        )


    st.divider()


    # =====================================================
    # PRODUCTION CHART
    # =====================================================

    st.subheader(
        "📈 Production Analysis"
    )

    chart_df = df.tail(50).reset_index(
        drop=True
    )

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            y=chart_df["production"],
            mode="lines",
            name="Production"
        )
    )


    fig.add_hline(
        y=current_production,
        line_dash="dash",
        annotation_text="Current Prediction"
    )


    fig.update_layout(
        height=400,
        xaxis_title="Sample",
        yaxis_title="Production (bbl/day)",
        template="plotly_dark"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DIGITAL TWIN PAGE
# =========================================================

elif page == "Digital Twin":

    st.header("🔄 Digital Twin")

    st.write(
        "Virtual representation of the well-to-surface "
        "production system."
    )

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Pressure",
            f"{pressure} bar"
        )

        st.metric(
            "Temperature",
            f"{temperature} °C"
        )

        st.metric(
            "Flow Rate",
            f"{flow_rate} m³/h"
        )


    with col2:

        st.metric(
            "Choke",
            f"{choke}%"
        )

        st.metric(
            "Pump Speed",
            f"{pump_speed}%"
        )

        st.metric(
            "Water Cut",
            f"{water_cut}%"
        )


    st.info(
        "Digital Twin status: Simulation running"
    )


# =========================================================
# AI OPTIMIZATION PAGE
# =========================================================

elif page == "AI Optimization":

    st.header("🤖 AI Optimization Engine")

    st.write(
        "XGBoost production prediction + "
        "optimization engine"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Current Choke",
            f"{choke}%"
        )


    with c2:

        st.metric(
            "Optimal Choke",
            f"{optimization['choke']}%"
        )


    with c3:

        st.metric(
            "Optimal Pump Speed",
            f"{optimization['pump_speed']}%"
        )


    st.divider()


    st.success(
        f"Predicted optimized production: "
        f"{optimized_production:.0f} bbl/day"
    )


    if st.button(
        "⚙️ Apply Recommendation",
        use_container_width=True
    ):

        st.success(
            "Optimization command simulated successfully."
        )


# =========================================================
# EQUIPMENT PAGE
# =========================================================

elif page == "Equipment":

    st.header("⚙️ Equipment Health")

    equipment = {

        "ESP Pump": 86,

        "Control Valve": 92,

        "Separator": 95,

        "Pipeline": 88
    }


    for name, health in equipment.items():

        st.subheader(name)

        st.progress(
            health / 100
        )

        if health >= 90:

            st.success(
                f"{health}% — Excellent"
            )

        elif health >= 80:

            st.info(
                f"{health}% — Healthy"
            )

        else:

            st.warning(
                f"{health}% — Attention Required"
            )


# =========================================================
# DATASET PAGE
# =========================================================

elif page == "Dataset":

    st.header("📂 Well Production Dataset")

    st.write(
        f"Dataset contains {len(df):,} records."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "TwinFlow AI | Digital Twin • XGBoost • Optimization • Smart Automation"
)

# bash : python -m streamlit run app.py