import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Churn AI Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction System (PRO)")

# =========================
# SAFE MODEL LOAD
# =========================
try:
    pipeline = joblib.load("churn_pipeline.pkl")
    FEATURES = list(pipeline.feature_names_in_)
except Exception:
    st.error("❌ Model file not found or corrupted.")
    st.stop()

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# =========================
# SAFE CSV LOADER
# =========================
def safe_read(file):
    encodings = ["utf-8", "latin1", "ISO-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            continue

    return None

# =========================
# DATA CLEANER
# =========================
def clean_data(df):
    df = df.replace([" ", ""], np.nan)
    df = df.reindex(columns=FEATURES)

    for col in FEATURES:
        if col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)

    # fill missing values safely
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)

    for col in FEATURES:
        if col not in NUMERIC_COLS:
            df[col] = df[col].fillna("No")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df

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
menu = st.sidebar.radio("Menu", ["Dashboard", "Single Prediction", "Batch Prediction"])

# =========================
# DASHBOARD
# =========================
if menu == "Dashboard":
    st.header("📊 Business Insights Dashboard")

    st.markdown("Upload dataset to see insights")

    file = st.file_uploader("Upload CSV for Analysis", type=["csv"])

    if file:
        df = safe_read(file)

        if df is None:
            st.error("❌ Cannot read file. Please upload valid CSV.")
            st.stop()

        df = clean_data(df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn Distribution")

            churn_counts = df.get("Churn", pd.Series(["No"] * len(df))).value_counts()

            fig, ax = plt.subplots()
            ax.bar(churn_counts.index.astype(str), churn_counts.values)
            st.pyplot(fig)

        with col2:
            st.subheader("Monthly Charges Distribution")

            fig, ax = plt.subplots()
            ax.hist(df["MonthlyCharges"], bins=20)
            st.pyplot(fig)

# =========================
# SINGLE PREDICTION
# =========================
elif menu == "Single Prediction":

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
        st.metric("Churn Probability", f"{prob:.2f}")
        st.write("Risk:", risk_level(prob))
        st.write("Prediction:", "Churn" if pred == 1 else "No Churn")

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = safe_read(file)

        if df is None:
            st.error("❌ File reading failed. Please convert CSV to UTF-8.")
            st.stop()

        df = clean_data(df)

        if st.button("Run Prediction"):

            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.success("Prediction completed successfully")
            st.dataframe(df.head())

            # =========================
            # CHARTS
            # =========================
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Distribution")
                df["Risk_Level"].value_counts().plot(kind="bar")
                st.pyplot(plt.gcf())

            with col2:
                st.subheader("Churn Probability")
                plt.hist(df["Churn_Probability"], bins=20)
                st.pyplot(plt.gcf())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "results.csv", "text/csv")
