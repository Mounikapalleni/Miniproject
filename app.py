import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Smartphone Addiction Analyzer",
    page_icon="📱",
    layout="centered"
)

# -----------------------------
# Load Model + Scaler + Feature Order
# -----------------------------
@st.cache_resource
def load_artifacts():
    return {
        "Model": pickle.load(open("logistic_regression_model.pkl", "rb")),
        "Scaler": pickle.load(open("scaler.pkl", "rb")),
        "Feature_Order": pickle.load(open("feature_order.pkl", "rb"))
    }

artifacts = load_artifacts()

# -----------------------------
# Title
# -----------------------------
st.title("📱 Smartphone Addiction Detection System")
st.markdown("Data-driven behavioral addiction prediction using Logistic Regression")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Usage Details")

col1, col2 = st.columns(2)

with col1:
    social_time = st.number_input("Social Media (hrs/day)", 0.0, 24.0, 2.0)
    gaming_time = st.number_input("Gaming (hrs/day)", 0.0, 24.0, 1.0)
    streaming_time = st.number_input("Streaming (hrs/day)", 0.0, 24.0, 2.0)

with col2:
    calls_mins = st.number_input("Calls Duration (mins/day)", 0.0, 1440.0, 30.0)
    data_usage = st.number_input("Data Usage (GB/month)", 0.0, 1000.0, 10.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Addiction Level"):

    # Convert calls to hours
    calls_hrs = calls_mins / 60

    # Calculate total screen time
    total_screen_time = social_time + gaming_time + streaming_time + calls_hrs

    # Get saved feature order
    feature_order = artifacts["Feature_Order"]

    # Create input dataframe
    input_df = pd.DataFrame([[
        total_screen_time,
        data_usage,
        calls_hrs,
        social_time,
        streaming_time,
        gaming_time
    ]], columns=feature_order)

    # Scale input
    scaler = artifacts["Scaler"]
    input_scaled = scaler.transform(input_df)

    # Predict using Logistic Regression
    model = artifacts["Model"]
    prediction = model.predict(input_scaled)[0]

    # Probability
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = round(max(probabilities) * 100, 2)

    # -----------------------------
    # Behavioral Alert
    # -----------------------------
    if total_screen_time >= 12:
        st.warning("⚠ Extreme Screen Exposure (12+ hrs/day detected)")

    # -----------------------------
    # Display Prediction
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success("✅ Low Addiction Risk")
        st.progress(30)
    elif prediction == "Moderate":
        st.warning("⚠ Moderate Addiction Risk")
        st.progress(60)
    else:
        st.error("🚨 High Addiction Risk")
        st.progress(90)

    st.write(f"Confidence: {confidence}%")
    st.write(f"Total Screen Time: **{round(total_screen_time,2)} hrs/day**")
