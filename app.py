import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Retention AI System",
    layout="wide"
)

st.title("📊 Customer Retention & Churn Prediction System")
st.markdown("AI-powered system to predict customer churn and support retention decisions.")

# =========================
# LOAD ARTIFACTS (FIXED)
# =========================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

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

def action_plan(risk):
    if risk == "High Risk":
        return "📞 Immediate retention call + discount offer"
    elif risk == "Medium Risk":
        return "📧 Engagement email + usage monitoring"
    else:
        return "✅ No action required"

# =========================
# NAVIGATION
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["Single Customer Prediction", "Batch Prediction", "Business Dashboard"]
)

# =========================
# SINGLE PREDICTION
# =========================
if menu == "Single Customer Prediction":

    st.header("👤 Single Customer Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
        is_month_to_month = st.selectbox("Month-to-Month Contract", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        total_services = st.slider("Total Services Subscribed", 0, 10, 3)
        avg_monthly_spend = st.number_input("Average Monthly Spend", 0.0, 200.0, 70.0)

    # =========================
    # ENCODING
    # =========================
    contract = 1 if is_month_to_month == "Yes" else 0

    internet_map = {"DSL": 0, "Fiber Optic": 1, "No": 2}
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer": 2,
        "Credit card": 3
    }

    internet = internet_map[internet_service]
    payment = payment_map[payment_method]

    # =========================
    # INPUT VECTOR
    # =========================
    input_data = np.array([[
        tenure,
        monthly_charges,
        total_charges,
        contract,
        internet,
        payment,
        total_services,
        avg_monthly_spend
    ]])

    input_scaled = scaler.transform(input_data)

    # =========================
    # PREDICTION
    # =========================
    if st.button("Analyze Risk"):

        prob = model.predict_proba(input_scaled)[0][1]
        pred = model.predict(input_scaled)[0]

        risk = risk_level(prob)
        action = action_plan(risk)

        st.subheader("📌 Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.2f}")
        c2.metric("Risk Level", risk)
        c3.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

        st.success(action)

        st.subheader("🔍 Explainability")
        st.info(
            "Top drivers:\n"
            "- Contract type\n"
            "- Monthly charges\n"
            "- Tenure\n"
            "- Internet service\n\n"
            "👉 (Add SHAP visualization for full production upgrade)"
        )

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Customer Risk Scoring")

    file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        try:
            df = df[features]
        except:
            st.error("Uploaded file does not match required features.")
            st.stop()

        df_scaled = scaler.transform(df)

        probs = model.predict_proba(df_scaled)[:, 1]
        preds = model.predict(df_scaled)

        def risk(p):
            if p < 0.3:
                return "Low Risk"
            elif p < 0.6:
                return "Medium Risk"
            else:
                return "High Risk"

        df["Churn_Probability"] = probs
        df["Prediction"] = preds
        df["Risk_Level"] = df["Churn_Probability"].apply(risk)
        df["Action"] = df["Risk_Level"].apply(action_plan)

        st.subheader("Preview Results")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

# =========================
# BUSINESS DASHBOARD
# =========================
elif menu == "Business Dashboard":

    st.header("📊 Business Insights Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", "7043")
    col2.metric("Churn Rate", "26.5%")
    col3.metric("High Risk Customers", "Approx. 1800")

    st.subheader("Key Insights")

    st.markdown("""
    - Month-to-month contracts increase churn risk  
    - Fiber optic users have higher churn  
    - High monthly charges increase churn probability  
    - Low tenure customers are most likely to churn  
    """)

    st.subheader("Feature Importance (Illustrative)")

    st.bar_chart({
        "Contract": 0.35,
        "Tenure": 0.25,
        "Monthly Charges": 0.20,
        "Internet Service": 0.10,
        "Payment Method": 0.10
    })