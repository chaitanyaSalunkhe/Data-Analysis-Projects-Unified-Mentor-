import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

# Load saved model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to compute sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity if text else 0

st.set_page_config(page_title="NASA Comment Likes Predictor", layout="centered")
st.title(" Predict Facebook Comment Likes")

# Inputs
text = st.text_area("Enter Facebook Comment")
comments_count = st.slider("Number of Replies", 0, 100, 3)
year = st.selectbox("Year", [2020, 2021, 2022, 2023])
month = st.slider("Month", 1, 12, 6)
hour = st.slider("Hour of Day", 0, 23, 12)

# Process inputs
sentiment = get_sentiment(text)
text_len = len(text)

input_df = pd.DataFrame({
    "commentsCount": [comments_count],
    "Sentiment": [sentiment],
    "TextLength": [text_len],
    "Year": [year],
    "Month": [month],
    "Hour": [hour]
})

# Predict
if st.button("Predict Likes"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f" Predicted Likes: {int(prediction[0])}")
