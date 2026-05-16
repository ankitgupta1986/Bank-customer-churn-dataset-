# app.py - Customer Churn Prediction Streamlit App

import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
        st.error(f'Required files are missing: {model_path} and/or {scaler_path}.')
        return None, None

    try:
        try:
            model = joblib.load(model_path)
        except ModuleNotFoundError as mnf:
            # Some pickled Keras models reference the standalone 'keras' package.
            # Map 'keras' to 'tensorflow.keras' in sys.modules and retry.
            if "keras" in str(mnf).lower():
                try:
                    import sys
                    import tensorflow as tf
                    sys.modules['keras'] = tf.keras
                    model = joblib.load(model_path)
                except Exception as e2:
                    raise e2
            else:
                raise
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

    # Prefer probability output when available
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(input_scaled)[0]
        # assume positive class is at index 1
        try:
            prediction_probability = float(prob[1])
        except Exception:
            prediction_probability = float(prob[0])
        predicted_label = int(prediction_probability > 0.5)
    else:
        # fallback to predict()
        pred = model.predict(input_scaled)
        pred_arr = np.asarray(pred)

        # Helper to extract a single scalar from various shapes
        if pred_arr.size == 1:
            val = pred_arr.item()
            if isinstance(val, (float, np.floating)):
                prediction_probability = float(val)
                predicted_label = int(prediction_probability > 0.5)
            else:
                predicted_label = int(val)
                prediction_probability = None
        else:
            # multiple values: common shapes are (1,) (1,1) or (1,2) for probabilities
            first = pred_arr[0]
            first = np.asarray(first)
            if first.size == 1:
                val = first.ravel()[0]
                try:
                    prediction_probability = float(val)
                    predicted_label = int(prediction_probability > 0.5)
                except Exception:
                    predicted_label = int(val)
                    prediction_probability = None
            else:
                # e.g., probability vector like [prob_class0, prob_class1]
                try:
                    prediction_probability = float(first.ravel()[1])
                except Exception:
                    prediction_probability = float(first.ravel()[0])
                predicted_label = int(prediction_probability > 0.5)

    st.subheader("Prediction Result")

    if prediction_probability is None:
        if predicted_label == 1:
            st.error("⚠️ Customer is likely to leave the bank")
        else:
            st.success("✅ Customer is likely to stay with the bank")
        st.write(f"Predicted Class: {predicted_label}")
    else:
        if prediction_probability > 0.5:
            st.error("⚠️ Customer is likely to leave the bank")
        else:
            st.success("✅ Customer is likely to stay with the bank")
        st.write(f"Prediction Probability: {prediction_probability:.2f}")

