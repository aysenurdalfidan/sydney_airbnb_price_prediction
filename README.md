# Sydney Airbnb Price Prediction

This project aims to predict nightly Airbnb rental prices in Sydney using the Inside Airbnb open dataset. The goal is to develop a reliable machine learning model that can estimate nightly rental prices based on key property characteristics and review sentiment analysis.

## Overview
Short-term rental platforms such as Airbnb have significant impacts on urban housing markets and tourism patterns. Understanding the price dynamics of these listings can help local governments:
- Monitor housing pressure zones
- Inform tourism-related policy decisions
- Assess the impact of short-term rentals on long-term housing availability

This project combines data preprocessing, natural language processing (NLP), and machine learning to deliver an interpretable and performant prediction model.

## Features Used
The model incorporates features such as:
- Room type
- Number of beds and bathrooms
- Availability throughout the year
- Distance to key Sydney landmarks
- Review sentiment score (TextBlob-based)

## Sentiment Analysis
A separate notebook (`sentiment.ipynb`) processes user reviews and generates a sentiment score for each listing. These scores are then merged with the main dataset to enhance the model's understanding of customer experience.

## Model
An XGBoost model was tuned using `RandomizedSearchCV` and trained on log-transformed price values for better prediction stability. Final model performance:
- RMSE: ~241 AUD
- MAE: ~102 AUD
- R² Score: ~0.52

Alternative models (LightGBM, Random Forest) were evaluated, but XGBoost yielded the most balanced performance.

## Deployment
The model is deployed via a Streamlit application (`app.py`). Users can input listing characteristics to receive a predicted nightly rate in real-time.

## Data Source
[Inside Airbnb - Sydney](http://insideairbnb.com/get-the-data.html)

## License
This project is for educational and non-commercial purposes only.
