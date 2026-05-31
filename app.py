import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# =========================
# LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn AI System", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("AI-powered churn prediction with explainable AI")

# =========================
# INPUT SECTION
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

st.divider()

# =========================
# FEATURE ENGINEERING
# =========================
def yes_no(x):
    return 1 if x == "Yes" else 0

def gender_encode(x):
    return 1 if x == "Male" else 0

avg_spend = total / (tenure + 1)

input_dict = {
    "gender": gender_encode(gender),
    "SeniorCitizen": yes_no(senior),
    "Partner": yes_no(partner),
    "Dependents": yes_no(dependents),
    "tenure": tenure,
    "PhoneService": 1,
    "PaperlessBilling": yes_no(paperless),
    "MonthlyCharges": monthly,
    "TotalCharges": total,

    "OnlineSecurity_Yes": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_Yes": 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,

    "IsMonthToMonth": 1 if contract == "Month-to-month" else 0,

    "AvgMonthlySpend": avg_spend
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=features, fill_value=0)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict Churn Risk"):

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Churn Probability", f"{prob:.2f}")

    with colB:
        if prob < 0.3:
            st.success("Low Risk")
        elif prob < 0.7:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

    with colC:
        st.metric("Decision", "Retain" if prob < 0.5 else "At Risk")

    # =========================
    # SAFE SHAP EXPLANATION (FIXED)
    # =========================
    st.subheader("🧠 Explainable AI (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)

        # IMPORTANT FIX: use real input instead of fake zeros
        shap_input = input_df.copy()

        shap_values = explainer.shap_values(shap_input)

        # FIX FOR OLD + NEW SHAP VERSIONS
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Ensure correct shape (force matrix)
        shap_values = np.array(shap_values)

        fig, ax = plt.subplots()

        shap.summary_plot(
            shap_values,
            shap_input,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP explanation could not be rendered in cloud environment.")
        st.text(str(e))

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Built with ❤️ for Customer Retention Analytics")