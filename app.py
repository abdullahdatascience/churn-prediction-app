import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Churn Prediction System", layout="wide")
st.title("📊 Customer Churn Prediction System")

# =========================
# LOAD PIPELINE
# =========================
pipeline = joblib.load("churn_pipeline.pkl")
FEATURES = list(pipeline.feature_names_in_)

# numeric columns (important fix)
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# =========================
# RISK FUNCTION
# =========================
def risk_level(prob):
    if prob < 0.3:
        return "Low Risk"
    elif prob < 0.6:
        return "Medium Risk"
    return "High Risk"

# =========================
# MENU
# =========================
menu = st.sidebar.radio("Menu", ["Single Prediction", "Batch Prediction"])

# =========================
# CLEAN FUNCTION (IMPORTANT FIX)
# =========================
def clean_data(df):
    df = df.replace([" ", ""], np.nan)
    df = df.reindex(columns=FEATURES)

    for col in FEATURES:
        if col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)

    # fill NaN safely
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    for col in FEATURES:
        if col not in NUMERIC_COLS:
            df[col] = df[col].fillna("No")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df

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
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])

    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

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
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    input_df = clean_data(input_df)

    if st.button("Predict"):

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        st.subheader("Result")
        st.write("Probability:", round(prob, 2))
        st.write("Risk:", risk_level(prob))
        st.write("Prediction:", "Churn" if pred == 1 else "No Churn")

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        try:
            df = clean_data(df)

            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "churn_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
