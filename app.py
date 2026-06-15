import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Churn Prediction System",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

# =========================
# LOAD PIPELINE
# =========================
pipeline = joblib.load("churn_pipeline.pkl")

# =========================
# RISK FUNCTION
# =========================
def risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

# =========================
# MENU
# =========================
menu = st.sidebar.radio("Menu", ["Single Prediction", "Batch Prediction"])

# =========================
# SINGLE PREDICTION
# =========================
if menu == "Single Prediction":

    st.header("👤 Single Customer Prediction")

    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", 0, 100, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

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

    if st.button("Predict"):

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        risk = risk_level(prob)

        st.subheader("Result")
        st.write("Probability:", round(prob, 2))
        st.write("Risk:", risk)
        st.write("Prediction:", "Churn" if pred == 1 else "No Churn")

# =========================
# BATCH PREDICTION (FIXED)
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        try:
            # =========================
            # FIX 1: CLEAN DIRTY VALUES
            # =========================
            df = df.replace(" ", np.nan)
            df = df.replace("", np.nan)
            df = df.fillna(0)

            # =========================
            # FIX 2: ALIGN FEATURES
            # =========================
            expected_cols = pipeline.feature_names_in_
            df = df.reindex(columns=expected_cols, fill_value=0)

            # =========================
            # FIX 3: PREDICTION
            # =========================
            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "churn_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
