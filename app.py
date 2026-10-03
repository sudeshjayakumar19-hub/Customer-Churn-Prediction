"""
app.py
------
Streamlit web application for the Customer Churn Prediction project.

This app loads the pre-trained model saved by src/train.py and allows
the user to enter customer information to get:
- Predicted churn probability
- Risk level (Low / Medium / High)
- Potential review reasons (associated characteristics, not causal factors)

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# ── Add src/ to the Python path ─────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from src.predict import load_model_bundle, predict_churn, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD

# ═══════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════
# Load model (cached so it is loaded only once per session)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_model():
    model_path = os.path.join(_PROJECT_ROOT, "models", "churn_model.joblib")
    if not os.path.exists(model_path):
        # Auto-train model on first deployment (needed for Streamlit Cloud)
        with st.spinner("🔄 First run — training model (this takes ~30 seconds)..."):
            import subprocess
            subprocess.run(
                [sys.executable, os.path.join(_PROJECT_ROOT, "setup_model.py")],
                check=True
            )
    return load_model_bundle(model_path)

try:
    bundle = get_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Title and introduction
# ═══════════════════════════════════════════════════════════════════════════

st.title("📊 Customer Churn Prediction")
st.markdown(
    f"""
    **Model in use:** `{bundle['model_name']}`
    &nbsp;|&nbsp;
    **ROC-AUC:** `{bundle['metrics']['roc_auc']:.4f}`
    &nbsp;|&nbsp;
    **Recall:** `{bundle['metrics']['recall']:.4f}`
    &nbsp;|&nbsp;
    **F1-score:** `{bundle['metrics']['f1']:.4f}`

    ---
    Fill in the customer details on the left and click **Predict** to see
    the churn probability and risk assessment.

    > ⚠️ **Disclaimer:** Risk levels use project-defined thresholds
    > (Low < {RISK_LOW_THRESHOLD:.0%}, Medium {RISK_LOW_THRESHOLD:.0%}–{RISK_HIGH_THRESHOLD:.0%}, High > {RISK_HIGH_THRESHOLD:.0%}).
    > These are **not** statistically validated cutoffs.
    > Review reasons are *associated characteristics*, not causal factors.
    """
)

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — Customer input form
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("🧑‍💼 Customer Information")
st.sidebar.markdown("Enter customer details below:")

with st.sidebar.form("customer_form"):

    st.subheader("Demographics")
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents?", ["Yes", "No"])

    st.subheader("Account & Contract")
    tenure         = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    contract       = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    st.subheader("Billing")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    # TotalCharges is derived from tenure × monthly, but we expose it directly
    # for users who know the value. Default to tenure × monthly as suggestion.
    total_charges   = st.number_input(
        "Total Charges ($)",
        min_value=0.0, max_value=10000.0,
        value=round(float(tenure) * float(monthly_charges), 2),
        step=1.0
    )

    st.subheader("Phone Services")
    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    st.subheader("Internet Services")
    internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("🔍 Predict Churn Probability")

# ═══════════════════════════════════════════════════════════════════════════
# Prediction and results display
# ═══════════════════════════════════════════════════════════════════════════

if submitted:
    # Build the customer dict that mirrors the raw CSV columns
    customer_dict = {
        "customerID":       "APP-USER",
        "gender":           gender,
        "SeniorCitizen":    senior_citizen,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           int(tenure),
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   float(monthly_charges),
        "TotalCharges":     float(total_charges),
    }

    result = predict_churn(customer_dict, bundle)
    prob = result["churn_probability"]
    risk = result["risk_level"]
    reasons = result["review_reasons"]

    # ── Results section ──────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{prob*100:.1f}%",
            delta=None,
        )

    with col2:
        risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
        st.metric(
            label="Risk Level",
            value=f"{risk_color.get(risk, '')} {risk}",
        )

    with col3:
        st.metric(
            label="Model Used",
            value=bundle["model_name"],
        )

    # ── Progress bar for probability ─────────────────────────────────────────
    st.markdown("### Churn Probability Gauge")
    st.progress(int(prob * 100))
    if prob < RISK_LOW_THRESHOLD:
        st.success(f"✅ **Low Risk** — Predicted churn probability: {prob*100:.1f}%. "
                   f"This customer does not show strong churn-associated characteristics.")
    elif prob <= RISK_HIGH_THRESHOLD:
        st.warning(f"⚠️ **Medium Risk** — Predicted churn probability: {prob*100:.1f}%. "
                   f"Some churn-associated characteristics are present. Consider monitoring.")
    else:
        st.error(f"🚨 **High Risk** — Predicted churn probability: {prob*100:.1f}%. "
                 f"Multiple churn-associated characteristics identified. "
                 f"Proactive customer retention may be warranted.")

    # ── Review reasons ────────────────────────────────────────────────────────
    st.markdown("### 📋 Potential Review Reasons")
    st.caption(
        "These are associated characteristics observed in the customer's profile. "
        "They are NOT confirmed causal factors of churn."
    )
    for reason in reasons:
        st.markdown(f"- {reason}")

    # ── Customer summary table ────────────────────────────────────────────────
    with st.expander("📄 View Customer Summary"):
        summary = {
            "Attribute": ["Tenure", "Contract", "Monthly Charges", "Internet Service",
                          "Tech Support", "Payment Method", "Senior Citizen", "Partner", "Dependents"],
            "Value":     [f"{tenure} months", contract, f"${monthly_charges:.2f}",
                          internet_service, tech_support, payment_method,
                          senior_citizen, partner, dependents],
        }
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

else:
    # ── Default state before form submission ─────────────────────────────────
    st.info("👈 Enter customer details in the sidebar and click **Predict Churn Probability**.")
    st.markdown(
        """
        ### About This Tool
        This application uses a machine learning model trained on the IBM Telco Customer Churn
        dataset to estimate the probability that a customer will cancel their service.

        #### Risk Level Thresholds (Project-Defined)
        | Risk Level   | Churn Probability |
        |--------------|-------------------|
        | 🟢 Low Risk   | < 30%             |
        | 🟡 Medium Risk | 30% – 70%         |
        | 🔴 High Risk  | > 70%             |

        #### Model Performance (on hold-out test set)
        | Metric    | Value |
        |-----------|-------|
        | ROC-AUC   | `{roc_auc:.4f}` |
        | Recall    | `{recall:.4f}` |
        | F1-score  | `{f1:.4f}` |
        | Precision | `{precision:.4f}` |
        | Accuracy  | `{accuracy:.4f}` |
        """.format(**bundle["metrics"])
    )

# ═══════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption(
    "Customer Churn Prediction | IBM Telco Dataset | "
    "Internship ML Capstone Project | "
    "All metrics computed from actual model training on real data."
)
