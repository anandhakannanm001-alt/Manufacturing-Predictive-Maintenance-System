"""
Advanced Streamlit Dashboard for Predictive Maintenance System.
Fixed Version - Works with FastAPI Backend
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-critical {
    background-color: #ffebee;
    border-left: 5px solid #d32f2f;
    padding: 15px;
    border-radius: 5px;
}

.status-high {
    background-color: #fff3e0;
    border-left: 5px solid #f57c00;
    padding: 15px;
    border-radius: 5px;
}

.status-medium {
    background-color: #fffde7;
    border-left: 5px solid #fbc02d;
    padding: 15px;
    border-radius: 5px;
}

.status-low {
    background-color: #e8f5e9;
    border-left: 5px solid #388e3c;
    padding: 15px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# API CONFIG
# =============================================================================

API_URL = "http://127.0.0.1:8000"

# =============================================================================
# REQUEST SESSION
# =============================================================================

session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("http://", adapter)
session.mount("https://", adapter)

# =============================================================================
# SAMPLE DATA
# =============================================================================

@st.cache_data
def load_sample_data():
    np.random.seed(42)

    data = []

    for i in range(50):
        data.append({
            "machine_id": f"M{i+1:03d}",
            "type_encoded": np.random.randint(0, 3),
            "air_temperature_k": round(np.random.normal(298, 2), 2),
            "process_temperature_k": round(np.random.normal(310, 3), 2),
            "rotational_speed_rpm": round(np.random.normal(1500, 120), 2),
            "torque_nm": round(np.random.normal(40, 5), 2),
            "tool_wear_min": round(np.random.uniform(50, 280), 2)
        })

    return pd.DataFrame(data)

# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_api_health():
    try:
        response = session.get(
            f"{API_URL}/health",
            timeout=5
        )

        if response.status_code == 200:
            return True

        return False

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def make_prediction(machine_data):
    try:
        response = session.post(
            f"{API_URL}/predict",
            json=machine_data,
            timeout=15
        )

        if response.status_code == 200:
            return response.json()

        st.error(f"API Error: {response.status_code}")
        return None

    except Exception as e:
        st.error(f"Connection Error: {e}")
        logger.error(f"Prediction failed: {e}")
        return None


def batch_predict(df):
    try:
        payload = {
            "machines": df.to_dict(orient="records")
        }

        response = session.post(
            f"{API_URL}/predict/batch",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()

        st.error(f"Batch API Error: {response.status_code}")
        return None

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return None

# =============================================================================
# HEADER
# =============================================================================

def render_header():
    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("🔧 Predictive Maintenance Dashboard")
        st.caption("AI-Powered Manufacturing Intelligence")

    with col2:
        st.write("")

        if get_api_health():
            st.success("API ONLINE")
        else:
            st.error("API OFFLINE")

# =============================================================================
# METRICS
# =============================================================================

def render_metrics(batch_result):

    st.subheader("📊 System Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    total = batch_result["total_machines"]

    with c1:
        st.metric(
            "Machines",
            total
        )

    with c2:
        st.metric(
            "Critical",
            batch_result["critical_count"]
        )

    with c3:
        st.metric(
            "High Risk",
            batch_result["high_risk_count"]
        )

    with c4:
        st.metric(
            "Medium Risk",
            batch_result["medium_risk_count"]
        )

    with c5:
        st.metric(
            "Avg Failure",
            f"{batch_result['average_failure_probability']:.1%}"
        )

# =============================================================================
# CHARTS
# =============================================================================

def render_charts(batch_result):

    st.subheader("📈 Risk Analytics")

    risk_df = pd.DataFrame({
        "Risk": ["Critical", "High", "Medium", "Low"],
        "Count": [
            batch_result["critical_count"],
            batch_result["high_risk_count"],
            batch_result["medium_risk_count"],
            batch_result["low_risk_count"]
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk_df,
            values="Count",
            names="Risk",
            title="Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig2 = px.bar(
            risk_df,
            x="Risk",
            y="Count",
            color="Risk",
            title="Machines by Risk"
        )

        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# TABLE
# =============================================================================

def render_predictions(batch_result):

    st.subheader("🚀 Machine Predictions")

    rows = []

    for pred in batch_result["predictions"]:

        rows.append({
            "Machine": pred["machine_id"],
            "Risk": pred["risk_level"],
            "Failure Probability": round(pred["failure_probability"] * 100, 2),
            "Health Score": round(pred["health_score"], 2),
            "RUL Hours": round(pred["estimated_hours_to_failure"], 2)
            if pred["estimated_hours_to_failure"]
            else 0,
            "Recommendation": pred["recommendation"]
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# SINGLE MACHINE
# =============================================================================

def single_machine_prediction():

    st.subheader("🔍 Single Machine Prediction")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:
            machine_id = st.text_input("Machine ID", "M001")

            machine_type = st.selectbox(
                "Machine Type",
                ["Low", "Medium", "High"]
            )

        with col2:
            air_temp = st.number_input(
                "Air Temp (K)",
                value=298.0
            )

            process_temp = st.number_input(
                "Process Temp (K)",
                value=310.0
            )

        with col3:
            rpm = st.number_input(
                "RPM",
                value=1500
            )

            torque = st.number_input(
                "Torque",
                value=40.0
            )

        tool_wear = st.slider(
            "Tool Wear",
            0,
            300,
            100
        )

        submit = st.form_submit_button("Predict")

    if submit:

        type_map = {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }

        payload = {
            "machine_id": machine_id,
            "type_encoded": type_map[machine_type],
            "air_temperature_k": float(air_temp),
            "process_temperature_k": float(process_temp),
            "rotational_speed_rpm": float(rpm),
            "torque_nm": float(torque),
            "tool_wear_min": float(tool_wear)
        }

        result = make_prediction(payload)

        if result:

            risk = result["risk_level"]

            css_class = f"status-{risk.lower()}"

            st.markdown(
                f"""
                <div class="{css_class}">
                <h3>{risk} Risk</h3>
                <p><b>Failure Probability:</b> {result['failure_probability']:.1%}</p>
                <p><b>Confidence:</b> {result['confidence']:.1%}</p>
                <p><b>Recommendation:</b> {result['recommendation']}</p>
                <p><b>Remaining Hours:</b> {result['estimated_hours_to_failure']:.1f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Detailed Report"):
                st.text(result["explanation"])

# =============================================================================
# MAIN
# =============================================================================

def main():

    render_header()

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Dashboard",
        "Single Machine",
        "About"
    ])

    with tab1:

        sample_df = load_sample_data()

        batch_result = batch_predict(sample_df)

        if batch_result:

            render_metrics(batch_result)

            st.markdown("---")

            render_charts(batch_result)

            st.markdown("---")

            render_predictions(batch_result)

        else:
            st.error("""
            Cannot connect to API.

            First run:
            python -m uvicorn api.main:app --reload
            """)

    with tab2:
        single_machine_prediction()

    with tab3:

        st.markdown("""
        ## Predictive Maintenance System

        ### Features
        - Real-time machine monitoring
        - AI failure prediction
        - Risk assessment
        - Remaining useful life prediction
        - Cost analysis
        - Maintenance recommendations

        ### Tech Stack
        - FastAPI
        - Streamlit
        - Scikit-learn
        - Plotly
        - Pandas
        """)

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()