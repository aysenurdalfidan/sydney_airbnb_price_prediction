# Sydney Airbnb Price Prediction

This project aims to predict nightly Airbnb rental prices in Sydney using the Inside Airbnb open dataset. The goal is to develop a reliable machine learning model that estimates nightly prices based on key listing features and review sentiment analysis.

## Overview
Short-term rental platforms like Airbnb significantly affect urban housing markets and tourism. Understanding pricing dynamics can help local governments:
- Monitor high-pressure rental zones
- Guide tourism and housing policy
- Assess the impact of short-term rentals on long-term housing availability

This project combines data preprocessing, natural language processing (NLP), and machine learning to deliver an interpretable price prediction model.

## Features Used
The model incorporates features such as:
- Room type
- Number of beds and bathrooms
- Availability throughout the year
- Distance to key Sydney landmarks
- Review sentiment score (TextBlob-based)

## Sentiment Analysis
A separate notebook (`sentiment.ipynb`) processes user reviews and generates a sentiment score for each listing. These scores are merged with the main dataset to enhance the model's understanding of customer experience.

## Model
A tuned LightGBM model was trained on log-transformed price values for improved prediction accuracy.

**Final model performance:**
- RMSE: 240.34 AUD
- MAE: 100.74 AUD
- R² Score: 0.527

## Deployment
While an initial Streamlit app was built (`app.py`), the model is better suited for batch predictions and data analysis use cases rather than real-time user input.

## Data Source
[Inside Airbnb - Sydney](http://insideairbnb.com/get-the-data.html)

## License
This project is for educational and non-commercial purposes only.
