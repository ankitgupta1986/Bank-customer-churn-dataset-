# app.py - Customer Churn Prediction Streamlit App

import os
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer will leave the bank or not using ANN Model")

# Load Model and Scaler
@st.cache_resource
def load_files():
    model_path = 'Customer_Churn_model.pkl'
    scaler_path = 'Customer_Churn_scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error('Required files are missing: ann_model.h5 and/or scaler.pkl.')
        return None, None

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f'Failed to load model or scaler: {e}')
        return None, None

model, scaler = load_files()

if model is None or scaler is None:
    st.stop()

# Sidebar
st.sidebar.header("Customer Information")

# User Inputs
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider("Age", 18, 100, 35)

tenure = st.sidebar.slider("Tenure", 0, 10, 5)

balance = st.sidebar.number_input("Balance", value=50000.0)

num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.sidebar.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    value=50000.0
)

# Gender Encoding
if gender == 'Male':
    gender = 1
else:
    gender = 0

# Geography Encoding
geo_germany = 0
geo_spain = 0

if geography == 'Germany':
    geo_germany = 1
elif geography == 'Spain':
    geo_spain = 1

# Create Input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain]
})

# Scale Input
input_scaled = scaler.transform(input_data)

# Prediction Button
if st.button("Predict Churn"):

    prediction = model.predict(input_scaled)
    prediction_probability = prediction[0][0]

    st.subheader("Prediction Result")

    if prediction_probability > 0.5:
        st.error("⚠️ Customer is likely to leave the bank")
    else:
        st.success("✅ Customer is likely to stay with the bank")

    st.write(f"Prediction Probability: {prediction_probability:.2f}")

# Footer
st.markdown("---")
st.markdown("Developed using Streamlit + TensorFlow ANN")
