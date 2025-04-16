import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="centered")

model, feature_columns = joblib.load("best_lgb_with_columns.pkl")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Sydney Airbnb Price Predictor</h1>
    <p style='text-align: center;'>🏡 Estimate the nightly rental price based on listing details.</p>
    <hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    beds = st.slider("Number of Beds", 1, 10, 2)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 1)
    minimum_nights = st.slider("Minimum Nights", 1, 30, 2)

with col2:
    availability_365 = st.slider("Availability (days/year)", 0, 365, 180)
    avg_sentiment_score = st.slider("Average Review Sentiment", -1.0, 1.0, 0.2)

if st.button("Predict Price 📈"):

    # Kullanıcıdan alınan ve sabit olarak atanacak tüm feature'ları topla
    input_data = {
        "beds": beds,
        "bathrooms": bathrooms,
        "minimum_nights": minimum_nights,
        "availability_365": availability_365,
        "avg_sentiment_score": avg_sentiment_score,

        # Sabit değerler (örnek olarak verildi – projene göre değiştirilebilir)
        "latitude": -33.86,
        "longitude": 151.21,
        "number_of_reviews": 15,
        "reviews_per_month": 0.5,
        "calculated_host_listings_count": 1,
        "distance_to_suburb_center": 1.2,
        "distance_to_sydney_opera_house": 3.1,
        "distance_to_harbour_bridge": 2.9,
        "distance_to_bondi_beach": 5.2,
        "distance_to_darling_harbour": 2.7,
        "distance_to_taronga_zoo": 4.0,
        "distance_to_the_rocks": 3.0,
        "distance_to_royal_botanic_garden": 2.5,
        "distance_to_manly_beach": 6.2,
        "distance_to_circular_quay": 3.4,
        "distance_to_queen_victoria_building": 2.0,
        "weighted_sentiment": 0.3,
        "bathroom_ratio": bathrooms / beds,

        # OHE kategorik değişkenler (sadece biri 1 olabilir)
        "room_type_Hotel room": 0,
        "room_type_Private room": 0,

        # Sezon (örnek: yaz için)
        "season_spring": 0,
        "season_summer": 1,
        "season_winter": 0,
    }

    # Doğru sırayla DataFrame oluştur
    features_df = pd.DataFrame([input_data])[feature_columns]

    # Tahmin
    log_price = model.predict(features_df)[0]
    predicted_price = np.expm1(log_price)

    # Sonuç
    st.markdown(f"""
        <div style='background-color:#e6f5ec; padding: 20px; border-radius: 10px;'>
            <h3 style='color:#388e3c;'>🌟 Estimated Nightly Price: <span style='color:#1b5e20;'>${predicted_price:.2f} AUD</span></h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<small>Note: This prediction is based on a machine learning model and does not guarantee actual booking prices.</small>", unsafe_allow_html=True)

st.markdown("""
    <hr>
    <p style='text-align:center; font-size:13px;'>Created by <b>Ayse D.</b> • Powered by LightGBM</p>
""", unsafe_allow_html=True)
