import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# LOAD ARTIFACTS
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.markdown("### AI-powered customer retention intelligence dashboard")

# =========================
# TABS (PRO UI STRUCTURE)
# =========================
tab1, tab2 = st.tabs(["🧾 Customer Input", "📊 Prediction Result"])

# =========================
# INPUT TAB
# =========================
with tab1:

    st.subheader("Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    st.divider()

    st.subheader("Monthly Billing")

    col4, col5 = st.columns(2)

    with col4:
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    with col5:
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    st.divider()

    st.subheader("Services")

    col6, col7, col8 = st.columns(3)

    with col6:
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])

    with col7:
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])

    with col8:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

# =========================
# FEATURE ENGINEERING FUNCTION
# =========================
def yes_no(val):
    return 1 if val == "Yes" else 0

def encode_gender(val):
    return 1 if val == "Male" else 0

def safe_div(a, b):
    return a / (b + 1)

# =========================
# BUILD INPUT
# =========================
input_dict = {
    "gender": encode_gender(gender),
    "SeniorCitizen": yes_no(senior),
    "Partner": yes_no(partner),
    "Dependents": yes_no(dependents),
    "tenure": tenure,
    "PhoneService": yes_no(phone),
    "PaperlessBilling": yes_no(paperless),
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,

    "OnlineSecurity_Yes": yes_no(online_security),
    "OnlineBackup_Yes": yes_no(online_backup),
    "DeviceProtection_Yes": yes_no(device_protection),
    "TechSupport_Yes": yes_no(tech_support),
    "StreamingTV_Yes": yes_no(streaming_tv),
    "StreamingMovies_Yes": yes_no(streaming_movies),

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,

    "IsMonthToMonth": 1 if contract == "Month-to-month" else 0,

    "AvgMonthlySpend": safe_div(total_charges, tenure)
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=features, fill_value=0)

# scale numeric features safely
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# =========================
# PREDICTION TAB
# =========================
with tab2:

    if st.button("🚀 Generate Prediction"):

        prob = model.predict_proba(input_df)[:, 1][0]

        st.subheader("Prediction Outcome")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Churn Probability", f"{prob:.2f}")

        with colB:
            if prob < 0.3:
                st.success("Low Risk")
            elif prob < 0.7:
                st.warning("Medium Risk")
            else:
                st.error("High Risk")

        with colC:
            st.metric("Decision",
                      "Retain Customer" if prob < 0.5 else "At Risk")

        st.divider()

        st.info("💡 This prediction is based on trained ML model + engineered features.")