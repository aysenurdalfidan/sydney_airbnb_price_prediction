import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

# Page configuration
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="centered")

# Load the trained model
with open("best_lgb.pkl", "rb") as f:
    model = pickle.load(f)

# App header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Sydney Airbnb Price Predictor</h1>
    <p style='text-align: center;'>🏡 Estimate the nightly rental price based on listing details.</p>
    <hr>
""", unsafe_allow_html=True)

# Input features
col1, col2 = st.columns(2)

with col1:
    beds = st.slider("Number of Beds", 1, 10, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 1)
    minimum_nights = st.slider("Minimum Nights", 1, 30, 2)

with col2:
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
    avg_sentiment_score = st.slider("Average Review Sentiment", -1.0, 1.0, 0.2)

# Predict button
if st.button("Predict Price 📈"):
    # Ensure the features match the trained model
    feature_names = ["beds", "bathrooms", "minimum_nights", "availability_365", "avg_sentiment_score"]
    features_df = pd.DataFrame([[beds, bathrooms, minimum_nights, availability_365, avg_sentiment_score]],
                               columns=feature_names)

    # Predict and inverse log
    log_price = model.predict(features_df)[0]
    predicted_price = np.expm1(log_price)

    st.markdown(f"""
        <div style='background-color:#e6f5ec; padding: 20px; border-radius: 10px;'>
            <h3 style='color:#388e3c;'>🌟 Estimated Nightly Price: <span style='color:#1b5e20;'>${predicted_price:.2f} AUD</span></h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<small>Note: This prediction is based on a machine learning model and does not guarantee actual booking prices.</small>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; font-size:13px;'>Created by <b>Ayse D.</b> • Powered by LightGBM</p>
""", unsafe_allow_html=True)
