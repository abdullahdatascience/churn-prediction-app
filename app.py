import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
    st.error("⚠️ Model loading failed. Check churn_pipeline.pkl file.")
    st.stop()


# =========================
# GET MODEL FEATURES
# =========================
try:
    MODEL_FEATURES = list(pipeline.feature_names_in_)
except:
    st.error("⚠️ Model does not expose feature names. Re-train with sklearn >= 1.0")
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
    except:
        return None, "invalid"


# =========================
# CLEAN FUNCTION (LIGHT ONLY)
# =========================
def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.fillna("Unknown")
    return df


# =========================
# STRONG SCHEMA ALIGNMENT (KEY FIX)
# =========================
def align_schema(df):
    df = df.copy()

    # drop extra columns safely
    df = df[[col for col in df.columns if col in MODEL_FEATURES]]

    # add missing columns
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    # reorder exactly
    df = df[MODEL_FEATURES]

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
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "👤 Single Prediction", "📂 Batch Prediction"]
)


# =========================================================
# DASHBOARD
# =========================================================
if menu == "📊 Dashboard":

    file = st.file_uploader("Upload dataset", type=["csv"])

    if file:
        df, error = safe_read_csv(file)

        if error:
            st.error("⚠️ Invalid or empty file")
            st.stop()

        df = clean_data(df)

        st.metric("Total Rows", len(df))

        if "Churn" in df.columns:
            churn_rate = (df["Churn"].astype(str).str.lower() == "yes").mean()
            st.metric("Churn Rate", f"{churn_rate:.2%}")


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
            input_df = align_schema(input_df)

            prob = pipeline.predict_proba(input_df)[0][1]
            pred = pipeline.predict(input_df)[0]

            col1, col2, col3 = st.columns(3)

            col1.metric("Churn Probability", f"{prob:.2f}")
            col2.metric("Risk Level", risk_level(prob))
            col3.metric("Decision", "Will Churn" if pred == 1 else "Will Stay")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")


# =========================================================
# BATCH PREDICTION
# =========================================================
elif menu == "📂 Batch Prediction":

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df, error = safe_read_csv(file)

        if error:
            st.error("⚠️ Invalid or empty file")
            st.stop()

        if st.button("Run Prediction"):

            try:
                df_clean = clean_data(df)

                # 🔥 KEY FIX
                df_model = align_schema(df_clean)

                df["Churn_Probability"] = pipeline.predict_proba(df_model)[:, 1]
                df["Prediction"] = pipeline.predict(df_model)
                df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

                st.success("Prediction completed successfully")
                st.dataframe(df.head())

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results", csv, "churn_results.csv", "text/csv")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {str(e)}")
