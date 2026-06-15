import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Retention AI System",
    layout="wide"
)

st.title("📊 Customer Retention & Churn Prediction System")
st.markdown("AI-powered ML system for churn prediction + business intelligence dashboard.")

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
    ["Single Prediction", "Batch Prediction", "Business Dashboard"]
)

# =========================
# SINGLE PREDICTION
# =========================
if menu == "Single Prediction":

    st.header("👤 Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", 0, 100, 12)

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    if st.button("Predict Churn"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        risk = risk_level(prob)

        st.subheader("📌 Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.2f}")
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
            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "churn_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# BUSINESS DASHBOARD
# =========================
elif menu == "Business Dashboard":

    st.header("📊 Business Intelligence Dashboard")

    file = st.file_uploader("Upload Dataset for Analytics", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
        df["Prediction"] = pipeline.predict(df)

        # ================= KPI =================
        total = len(df)
        churn_rate = df["Prediction"].mean()
        retention_rate = 1 - churn_rate
        high_risk = (df["Churn_Probability"] > 0.6).sum()

        st.subheader("📌 KPIs")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Customers", total)
        c2.metric("Churn Rate", f"{churn_rate:.2%}")
        c3.metric("Retention Rate", f"{retention_rate:.2%}")
        c4.metric("High Risk", high_risk)

        # ================= CHART 1 =================
        st.subheader("📊 Churn Distribution")

        fig, ax = plt.subplots()
        df["Prediction"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"], rotation=0)
        st.pyplot(fig)

        # ================= CHART 2 =================
        st.subheader("💰 Monthly Charges vs Churn Risk")

        fig2, ax2 = plt.subplots()
        ax2.scatter(df["MonthlyCharges"], df["Churn_Probability"])
        ax2.set_xlabel("Monthly Charges")
        ax2.set_ylabel("Churn Probability")
        st.pyplot(fig2)

        # ================= CHART 3 =================
        st.subheader("📄 Contract Impact")

        fig3, ax3 = plt.subplots()
        df.groupby("Contract")["Prediction"].mean().plot(kind="bar", ax=ax3)
        st.pyplot(fig3)

        st.success("Dashboard generated successfully 🚀")
