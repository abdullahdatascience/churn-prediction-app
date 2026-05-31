import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# =========================
# LOAD ARTIFACTS
# =========================
model = joblib.load(os.path.join(os.path.dirname(__file__), "churn_model.pkl"))
features = joblib.load(os.path.join(os.path.dirname(__file__), "features.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.pkl"))

# =========================
# APP UI
# =========================
st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details to predict churn risk")

# =========================
# INPUT FIELDS
# =========================
tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
avg_spend = st.number_input("Avg Monthly Spend", 0.0, 200.0, 70.0)

is_month_to_month = st.selectbox("Month-to-Month Contract (1 = Yes, 0 = No)", [0, 1])
fiber_optic = st.selectbox("Fiber Optic Internet (1 = Yes, 0 = No)", [0, 1])
online_security = st.selectbox("Online Security (1 = Yes, 0 = No)", [0, 1])
tech_support = st.selectbox("Tech Support (1 = Yes, 0 = No)", [0, 1])

# =========================
# PREDICTION
# =========================
if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "AvgMonthlySpend": avg_spend,
        "IsMonthToMonth": is_month_to_month,
        "InternetService_Fiber optic": fiber_optic,
        "OnlineSecurity_Yes": online_security,
        "TechSupport_Yes": tech_support
    }])

    # align features exactly
    input_data = input_data.reindex(columns=features, fill_value=0)

    # scale numeric columns (safe check)
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # prediction
    prob = model.predict_proba(input_data)[:, 1][0]

    # risk label
    if prob < 0.3:
        risk = "🟢 LOW RISK (Likely to Stay)"
    elif prob < 0.7:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🔴 HIGH RISK (Likely to Churn)"

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Prediction Result")

    st.write("Churn Probability:", round(prob, 4))
    st.write("Risk Level:", risk)