import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# =========================
# LOAD ARTIFACTS
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "churn_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "features.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.set_page_config(page_title="Churn AI Platform", layout="wide")

st.title("🚀 Customer Churn Intelligence Platform (FAANG Level)")
st.markdown("Advanced ML + Explainable AI System")

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📂 Batch Prediction", "🧠 Model Explainability"])

# =========================
# HELPER FUNCTIONS
# =========================
def yes_no(x):
    return 1 if x == "Yes" else 0

def encode_gender(x):
    return 1 if x == "Male" else 0

# =========================
# FEATURE ENGINEERING
# =========================
def build_input(data):
    data["AvgMonthlySpend"] = data["TotalCharges"] / (data["tenure"] + 1)
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]
    df[num_cols] = scaler.transform(df[num_cols])
    return df

# =========================
# TAB 1 - SINGLE PREDICTION
# =========================
with tab1:

    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        tenure = st.slider("Tenure", 0, 72, 12)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

    with col3:
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    input_data = {
        "gender": encode_gender(gender),
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
        "IsMonthToMonth": 1 if contract == "Month-to-month" else 0
    }

    if st.button("Predict Churn Risk"):

        X = build_input(input_data)

        prob = model.predict_proba(X)[0][1]

        st.metric("Churn Probability", f"{prob:.2f}")

        if prob > 0.7:
            st.error("High Risk Customer")
        elif prob > 0.3:
            st.warning("Medium Risk Customer")
        else:
            st.success("Low Risk Customer")

        # =========================
        # SHAP LOCAL EXPLANATION
        # =========================
        st.subheader("🧠 Why this prediction? (SHAP)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots()
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1],
            X,
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)

# =========================
# TAB 2 - BATCH PREDICTION
# =========================
with tab2:

    st.subheader("Upload Customer CSV")

    file = st.file_uploader("Upload dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview:", df.head())

        st.success("File uploaded successfully")

# =========================
# TAB 3 - GLOBAL EXPLANATION
# =========================
with tab3:

    st.subheader("Feature Importance (Global SHAP)")

    explainer = shap.TreeExplainer(model)

    # sample for speed
    sample = pd.DataFrame(np.zeros((100, len(features))), columns=features)

    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], sample, show=False)
    st.pyplot(fig)

    st.info("Top drivers of churn shown using SHAP values")