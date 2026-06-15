import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn AI System",
    layout="wide"
)

st.title("📊 Customer Retention & Churn Prediction System")
st.markdown("AI-powered ML system using a trained pipeline (production ready).")

# =========================
# LOAD PIPELINE
# =========================
pipeline = joblib.load("churn_pipeline.pkl")

# =========================
# BUSINESS LOGIC
# =========================
def risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# =========================
# NAVIGATION
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Batch Prediction"]
)

# =========================
# SINGLE PREDICTION
# =========================
if menu == "Single Prediction":

    st.header("👤 Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", 0, 100, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    # Create input dataframe (RAW FORMAT ONLY)
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "InternetService": InternetService,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    if st.button("Predict Churn"):

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        risk = risk_level(prob)

        st.subheader("📌 Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Probability", f"{prob:.2f}")
        c2.metric("Risk Level", risk)
        c3.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Prediction System")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        try:
            # =========================
            # FIX FEATURE MISMATCH
            # =========================
            expected_cols = pipeline.feature_names_in_
            df = df.reindex(columns=expected_cols, fill_value=0)

            # =========================
            # PREDICTIONS
            # =========================
            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                csv,
                "churn_predictions.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error: {e}")
