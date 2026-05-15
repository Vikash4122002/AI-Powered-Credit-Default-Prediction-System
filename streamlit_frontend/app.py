"""
Streamlit Frontend for Credit Default Prediction API
FastAPI + Docker + AWS EC2 + Streamlit
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# Configuration
# ============================================

API_BASE_URL = "http://3.109.32.46:8000"

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="🏦",
    layout="wide"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}

.status-online {
    background-color: #2ed573;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    color: white;
    font-weight: bold;
}

.status-offline {
    background-color: #ff4757;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    color: white;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# Sidebar
# ============================================

with st.sidebar:

    st.title("🏦 Credit Default Predictor")

    st.markdown("---")

    st.subheader("🔌 API Status")

    try:

        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )

        if response.status_code == 200:

            st.markdown(
                '<span class="status-online">✅ API Online</span>',
                unsafe_allow_html=True
            )

            api_status = True

        else:

            st.markdown(
                '<span class="status-offline">❌ API Error</span>',
                unsafe_allow_html=True
            )

            api_status = False

    except Exception:

        st.markdown(
            '<span class="status-offline">❌ API Offline</span>',
            unsafe_allow_html=True
        )

        api_status = False

    st.markdown("---")

    st.subheader("🧠 Model Details")

    st.markdown("""
    - Stacking Ensemble
    - FastAPI Backend
    - Docker Deployment
    - AWS EC2 Hosting
    - 23 Features
    """)

    st.markdown("---")

    st.subheader("🔗 API Links")

    st.markdown(f"[Swagger Docs]({API_BASE_URL}/docs)")
    st.markdown(f"[Health Check]({API_BASE_URL}/health)")

# ============================================
# Header
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Default Risk Predictor</h1>
    <p>FastAPI + Docker + AWS EC2 + Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# Tabs
# ============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Single Prediction",
    "📁 Batch Prediction",
    "📊 Analytics",
    "📖 API Docs"
])

# ============================================
# TAB 1 - SINGLE PREDICTION
# ============================================

with tab1:

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📝 Customer Information")

        limit_bal = st.number_input(
            "Credit Limit",
            min_value=0,
            value=50000,
            step=10000
        )

        sex = st.selectbox(
            "Gender",
            [1, 2],
            format_func=lambda x: "Male" if x == 1 else "Female"
        )

        education = st.selectbox(
            "Education",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Others"
            }[x]
        )

        marriage = st.selectbox(
            "Marital Status",
            [1, 2, 3],
            format_func=lambda x: {
                1: "Married",
                2: "Single",
                3: "Others"
            }[x]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        pay_0 = st.selectbox(
            "Payment Status",
            [-2, -1, 0, 1, 2, 3, 4],
            format_func=lambda x: {
                -2: "No Consumption",
                -1: "Paid Duly",
                0: "Minimum Payment"
            }.get(x, f"Delay {x} months")
        )

        bill_amt = st.number_input(
            "Last Bill Amount",
            min_value=0,
            value=10000,
            step=1000
        )

        pay_amt = st.number_input(
            "Last Payment Amount",
            min_value=0,
            value=5000,
            step=1000
        )

        threshold = st.slider(
            "Risk Threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.30,
            step=0.05
        )

        predict_btn = st.button(
            "🔮 Predict Default Risk",
            type="primary",
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Prediction Result")

        features = [
            float(limit_bal),
            float(sex),
            float(education),
            float(marriage),
            float(age),
            float(pay_0),
            0, 0, 0, 0, 0,
            float(bill_amt),
            0, 0, 0, 0, 0,
            float(pay_amt),
            0, 0, 0, 0, 0
        ]

        if predict_btn:

            if api_status:

                try:

                    payload = {
                        "features": features,
                        "threshold": threshold
                    }

                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/predict",
                        json=payload,
                        timeout=20
                    )

                    if response.status_code == 200:

                        result = response.json()

                        prediction = result.get("prediction", 0)
                        probability = result.get("probability", 0)
                        risk_level = result.get("risk_level", "LOW")
                        threshold_used = result.get(
                            "threshold_used",
                            threshold
                        )

                        col_a, col_b, col_c = st.columns(3)

                        with col_a:
                            st.metric(
                                "Prediction",
                                "Default" if prediction == 1 else "No Default"
                            )

                        with col_b:
                            st.metric(
                                "Probability",
                                f"{probability:.2%}"
                            )

                        with col_c:
                            st.metric(
                                "Risk Level",
                                risk_level
                            )

                        if prediction == 1:
                            st.error("⚠️ High Risk Customer")
                        else:
                            st.success("✅ Low Risk Customer")

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            title={"text": "Default Probability"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "steps": [
                                    {"range": [0, 30], "color": "lightgreen"},
                                    {"range": [30, 70], "color": "orange"},
                                    {"range": [70, 100], "color": "red"}
                                ]
                            }
                        ))

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.error(
                            f"API Error: {response.status_code}"
                        )

                        st.write(response.text)

                except Exception as e:

                    st.error(f"Error: {e}")

            else:

                st.error("Backend API is Offline")

# ============================================
# TAB 2 - BATCH PREDICTION
# ============================================

with tab2:

    st.subheader("📁 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file, header=None)

        st.success(f"Loaded {len(df)} records")

        st.dataframe(df.head())

        if st.button("🚀 Run Batch Prediction"):

            try:

                customers = df.iloc[:, :23].values.tolist()

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/predict_batch",
                    json={
                        "customers": customers,
                        "threshold": 0.30
                    },
                    timeout=60
                )

                if response.status_code == 200:

                    result = response.json()

                    results_df = pd.DataFrame(
                        result["results"]
                    )

                    st.dataframe(results_df)

                    csv = results_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )

                    fig = px.histogram(
                        results_df,
                        x="probability",
                        title="Risk Score Distribution"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

            except Exception as e:

                st.error(str(e))

# ============================================
# TAB 3 - ANALYTICS
# ============================================

with tab3:

    st.subheader("📊 Model Analytics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Value": [
            "76.2%",
            "46.8%",
            "57.1%",
            "51.4%",
            "76.2%"
        ]
    })

    st.dataframe(metrics_df)

    chart_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            76.2,
            46.8,
            57.1,
            51.4
        ]
    })

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        title="Model Performance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ============================================
# TAB 4 - API DOCUMENTATION
# ============================================

with tab4:

    st.subheader("📖 API Documentation")

    st.markdown(f"""
    ### Base URL

    `{API_BASE_URL}`

    ### Available Endpoints

    #### Health Check
    `{API_BASE_URL}/health`

    #### Swagger Documentation
    `{API_BASE_URL}/docs`

    #### Single Prediction
    `{API_BASE_URL}/api/v1/predict`

    #### Batch Prediction
    `{API_BASE_URL}/api/v1/predict_batch`

    ### Sample Request

    ```json
    {{
        "features": [
            50000,
            1,
            2,
            1,
            35,
            0,
            0,
            0,
            0,
            0,
            0,
            10000,
            0,
            0,
            0,
            0,
            0,
            5000,
            0,
            0,
            0,
            0,
            0
        ],
        "threshold": 0.3
    }}
    ```
    """)

# ============================================
# Footer
# ============================================

st.markdown("---")

st.caption(
    "🚀 Credit Default Prediction System | FastAPI + Docker + AWS EC2 + Streamlit"
)