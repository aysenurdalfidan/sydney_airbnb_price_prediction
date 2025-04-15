import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Page config
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="centered")

# Load model
with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Title & intro
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Sydney Airbnb Price Predictor</h1>
    <p style='text-align: center;'>🏡 Estimate the nightly price based on listing features.</p>
    <hr>
""", unsafe_allow_html=True)

# Feature inputs
col1, col2 = st.columns(2)

with col1:
    beds = st.slider("Number of Beds", 1, 10, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 1)
    minimum_nights = st.slider("Minimum Nights Required", 1, 30, 2)

with col2:
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
    avg_sentiment_score = st.slider("Average Review Sentiment", -1.0, 1.0, 0.2)

# Predict button
if st.button("Predict Price 📈"):
    # Prepare feature array
    features = np.array([[beds, bathrooms, minimum_nights, availability_365, avg_sentiment_score]])

    # Predict & reverse log transformation
    log_price = model.predict(features)[0]
    predicted_price = np.expm1(log_price)

    st.markdown(f"""
        <div style='background-color:#e6f5ec; padding: 20px; border-radius: 10px;'>
            <h3 style='color:#388e3c;'>🌟 Estimated Nightly Price: <span style='color:#1b5e20;'>${predicted_price:.2f} AUD</span></h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<small>Note: Predictions are based on statistical modeling and do not guarantee actual prices.</small>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; font-size:13px;'>Developed by <b>Ayşenur D.</b> • Powered by XGBoost</p>
""", unsafe_allow_html=True)
