import streamlit as st
import numpy as np
import pickle
from PIL import Image

# Stil ayarı: sayfa konfigürasyonu
st.set_page_config(page_title="Airbnb Price Predictor", page_icon="🏡", layout="centered")

# Modeli yükle
with open("best_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Uygulama başlığı
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Sydney Airbnb Fiyat Tahmincisi</h1>
    <p style='text-align: center;'>🏡 Kullanıcı girişlerine göre tahmini gecelik fiyatı hesaplar.</p>
    <hr>
""", unsafe_allow_html=True)

# Özellik girişleri (dilersen genişletebilirsin)
col1, col2 = st.columns(2)

with col1:
    beds = st.slider("Yatak Sayısı", 1, 10, 2)
    bathrooms = st.slider("Banyo Sayısı", 1, 5, 1)
    minimum_nights = st.slider("Minimum Konaklama (gece)", 1, 30, 2)

with col2:
    availability_365 = st.slider("Yıllık Uygunluk (gün)", 0, 365, 180)
    avg_sentiment_score = st.slider("Yorum Skoru (Sentiment)", -1.0, 1.0, 0.2)

# Tahmin butonu
if st.button("Fiyatı Tahmin Et 📈"):
    # Özellik dizisi - modelin beklediği sıraya göre düzenlenmeli
    features = np.array([[beds, bathrooms, minimum_nights, availability_365, avg_sentiment_score]])

    # Tahmin ve log geri dönüşümü
    log_price = model.predict(features)[0]
    predicted_price = np.expm1(log_price)

    st.markdown(f"""
        <div style='background-color:#e6f5ec; padding: 20px; border-radius: 10px;'>
            <h3 style='color:#388e3c;'>🌟 Tahmini Gecelik Fiyat: <span style='color:#1b5e20;'>${predicted_price:.2f} AUD</span></h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<small>Not: Tahminler istatistiksel modele dayalıdır ve kesin fiyat garantisi vermez.</small>", unsafe_allow_html=True)

# Alt bilgi
st.markdown("""
    <hr>
    <p style='text-align:center; font-size:13px;'>Hazırlayan: <b>Ayşenur D.</b> • XGBoost modeli ile geliştirildi</p>
""", unsafe_allow_html=True)
