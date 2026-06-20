import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn AI Dashboard PRO", layout="wide")
st.title("📊 Customer Churn Prediction System (PRO DASHBOARD)")


# =========================
# LOAD MODEL
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception:
    st.error("⚠️ Model could not be loaded. Please check system setup.")
    st.stop()


# =========================
# SAFE CSV LOADER
# =========================
def safe_read_csv(file):
    try:
        df = pd.read_csv(file)
        if df is None or df.empty:
            return None, "empty"
        return df, None
    except Exception:
        return None, "invalid"


# =========================
# CLEANING
# =========================
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================
# FIXED UNIVERSAL PREPROCESSOR
# =========================
def prepare_input(df, pipeline):
    df = df.copy()

    # normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

    # universal mapping (VERY IMPORTANT FIX)
    mapping = {
        "yes": 1,
        "no": 0,
        "male": 1,
        "female": 0
    }

    # convert simple categorical values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.lower().map(lambda x: mapping.get(x, x))

    # try safe alignment with model
    try:
        model_features = pipeline.feature_names_in_
        aligned = {}

        for f in model_features:
            key = f.lower().replace(" ", "")

            if key in df.columns:
                aligned[f] = df[key]
            else:
                aligned[f] = 0  # safe fallback

        return pd.DataFrame(aligned)

    except Exception:
        # fallback if model doesn't expose feature names
        return df.fillna(0)


# =========================
# RISK LEVEL
# =========================
def risk_level(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.6:
        return "Medium Risk"
    return "High Risk"


# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "👤 Single Prediction", "📂 Batch Prediction"]
)


# =========================================================
# DASHBOARD
# =========================================================
if menu == "📊 Dashboard":

    st.header("Business Insights Dashboard")

    file = st.file_uploader("Upload dataset for insights", type=["csv"])

    if file:
        df, error = safe_read_csv(file)

        if error == "invalid":
            st.error("⚠️ Invalid file format.")
            st.stop()

        if error == "empty":
            st.warning("⚠️ File is empty.")
            st.stop()

        df = clean_data(df)

        st.metric("Total Customers", len(df))

        if "churn" in df.columns:
            churn_rate = (df["churn"].astype(str).str.lower() == "yes").mean()
            st.metric("Churn Rate", f"{churn_rate:.2%}")

        if "monthlycharges" in df.columns:
            st.metric("Avg Monthly Charges", f"{df['monthlycharges'].mean():.2f}")


# =========================================================
# SINGLE PREDICTION
# =========================================================
elif menu == "👤 Single Prediction":

    st.header("Customer Churn Prediction")

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

    if st.button("Predict"):

        try:
            input_df = clean_data(input_df)
            input_df = prepare_input(input_df, pipeline)

            prob = pipeline.predict_proba(input_df)[0][1]
            pred = pipeline.predict(input_df)[0]

            col1, col2, col3 = st.columns(3)

            col1.metric("Churn Probability", f"{prob:.2f}")
            col2.metric("Risk Level", risk_level(prob))
            col3.metric("Decision", "Will Churn" if pred == 1 else "Will Stay")

        except Exception:
            st.error("⚠️ Prediction failed. Please check input values or model compatibility.")


# =========================================================
# BATCH PREDICTION
# =========================================================
elif menu == "📂 Batch Prediction":

    st.header("Batch Customer Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df, error = safe_read_csv(file)

        if error == "invalid":
            st.error("⚠️ Invalid file format.")
            st.stop()

        if error == "empty":
            st.warning("⚠️ File is empty.")
            st.stop()

        if st.button("Run Prediction"):

            try:
                df_clean = clean_data(df)
                df_clean = prepare_input(df_clean, pipeline)

                df["Churn_Probability"] = pipeline.predict_proba(df_clean)[:, 1]
                df["Prediction"] = pipeline.predict(df_clean)
                df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

                st.success("Prediction completed successfully")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results", csv, "churn_results.csv", "text/csv")

            except Exception:
                st.error("⚠️ Prediction failed due to dataset mismatch.")
