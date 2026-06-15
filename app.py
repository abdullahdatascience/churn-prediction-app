import streamlit as st
import pandas as pd
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
# LOAD PIPELINE (ONLY ONE FILE)
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
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure", 0, 100, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    # =========================
    # BUILD INPUT DATAFRAME
    # =========================
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # =========================
    # PREDICTION
    # =========================
    if st.button("Analyze Risk"):
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]

        risk = risk_level(prob)
        action = action_plan(risk)

        st.subheader("📌 Prediction Result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.2f}")
        c2.metric("Risk Level", risk)
        c3.metric("Prediction", "Will Churn" if pred == 1 else "Will Stay")

        st.success(action)

# =========================
# BATCH PREDICTION
# =========================
elif menu == "Batch Prediction":

    st.header("📂 Batch Customer Risk Scoring")

    file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        try:
            df["Churn_Probability"] = pipeline.predict_proba(df)[:, 1]
            df["Prediction"] = pipeline.predict(df)
            df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "churn_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# =========================
# BUSINESS DASHBOARD
# =========================
elif menu == "Business Dashboard":
    st.header("📊 Business Insights Dashboard")
    st.info("Add KPIs, churn trends, and visualizations here.")
