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
# SAFE MODEL LOADING
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "churn_pipeline.pkl")

try:
    pipeline = joblib.load(MODEL_PATH)
    EXPECTED_COLS = list(pipeline.feature_names_in_)
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# =========================
# CLEANING FUNCTIONS
# =========================
def clean_data(df):
    df = df.copy()
    df = df.replace([" ", ""], np.nan)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def align_columns(df):
    return df.reindex(columns=EXPECTED_COLS)

def final_preprocess(df):
    df = align_columns(df)

    # fix object columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    df = df.replace(["nan", "NaN", "None"], np.nan)

    # numeric fix
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0)

    # categorical fill
    for col in df.columns:
        if col not in NUMERIC_COLS:
            df[col] = df[col].fillna("No")

    df = df.fillna(0)
    return df

# =========================
# RISK FUNCTION
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

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":

    st.header("Business Insights Dashboard")

    file = st.file_uploader("Upload dataset for insights", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df = clean_data(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Customers", len(df))

        with col2:
            churn_rate = 0
            if "Churn" in df.columns:
                churn_rate = (df["Churn"].astype(str).str.lower() == "yes").mean()
            st.metric("Churn Rate", f"{churn_rate:.2%}")

        with col3:
            if "MonthlyCharges" in df.columns:
                st.metric("Avg Monthly Charges", f"{df['MonthlyCharges'].mean():.2f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly Charges")
            fig, ax = plt.subplots()
            ax.hist(df["MonthlyCharges"], bins=20)
            st.pyplot(fig)

        with col2:
            st.subheader("Tenure Distribution")
            fig, ax = plt.subplots()
            ax.hist(df["tenure"], bins=20)
            st.pyplot(fig)

# =========================
# SINGLE PREDICTION
# =========================
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

        input_df = final_preprocess(input_df)

        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Churn Probability", f"{prob:.2f}")
        col2.metric("Risk Level", risk_level(prob))
        col3.metric("Prediction", "Churn" if pred == 1 else "No Churn")

# =========================
# BATCH PREDICTION
# =========================
elif menu == "📂 Batch Prediction":

    st.header("Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        if st.button("Run Prediction"):

            df_clean = final_preprocess(df)

            df["Churn_Probability"] = pipeline.predict_proba(df_clean)[:, 1]
            df["Prediction"] = pipeline.predict(df_clean)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.success("Prediction completed successfully")
            st.dataframe(df.head())

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Distribution")
                fig, ax = plt.subplots()
                df["Risk_Level"].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

            with col2:
                st.subheader("Probability Distribution")
                fig, ax = plt.subplots()
                ax.hist(df["Churn_Probability"], bins=20)
                st.pyplot(fig)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "results.csv", "text/csv")
