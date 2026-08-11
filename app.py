import streamlit as st
import pandas as pd
import numpy as np
import pickle


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the company.")


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.header("Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )


with col2:

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# Contract and Payment Information
# --------------------------------------------------

col4, col5, col6 = st.columns(3)

with col4:

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


with col5:

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col6:

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# --------------------------------------------------
# Charges
# --------------------------------------------------

col7, col8 = st.columns(2)

with col7:

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


with col8:

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Create dataframe
    new_customer = pd.DataFrame({

        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })


    # --------------------------------------------------
    # Same encoding used in your notebook
    # --------------------------------------------------

    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]


    # IMPORTANT:
    # This reproduces your notebook's factorize approach.
    #
    # However, factorize() on a single new customer can
    # produce incorrect mappings. See note below.

    for col in categorical_columns:
        new_customer[col] = pd.factorize(new_customer[col])[0]


    # Convert to numeric
    new_customer = new_customer.astype(float)


    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------

    new_customer_scaled = scaler.transform(new_customer)


    # --------------------------------------------------
    # Prediction Probability
    # --------------------------------------------------

    probability = model.predict_proba(new_customer_scaled)[0][1]


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(new_customer_scaled)[0]


    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to CHURN"
        )

    else:

        st.success(
            "✅ Customer is likely to STAY"
        )