import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load trained model and feature names
model = joblib.load("logistic_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # Ensure this matches training column order

st.title("Hotel Popularity Prediction")
st.markdown("Enter hotel details to predict its popularity:")


# Input fields for user
score_adjusted = st.number_input("Score Adjusted (e.g., 4.3)", min_value=1.0, max_value=5.0, value=4.5)
bubble_rating = st.selectbox("Bubble Rating", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=7)
price_curr_min = st.number_input("Current Minimum Price (e.g., 100)", value=100.0)
location_grade = st.number_input("Location Grade (e.g., 85)", min_value=0.0, max_value=100.0, value=90.0)
photos = st.number_input("Number of Photos", value=50)
discount = st.selectbox("Has Discount?", [0, 1])
class_4_5 = st.selectbox("Hotel Class 4-5 Star", [0, 1])

# Create user input dictionary
user_input = {
    'score_adjusted': score_adjusted,
    'bubble_rating': bubble_rating,
    'price_curr_min': price_curr_min,
    'location_grade': location_grade,
    'photos': photos,
    'discount': discount,
    'class_4_5': class_4_5
}

# Fill missing values and ensure correct order
input_data = [user_input.get(col, 0) for col in feature_names]
input_array = np.array(input_data).reshape(1, -1)

# Predict button
if st.button("Predict Popularity"):
    prob = model.predict_proba(input_array)[0][1]
    prediction = 1 if prob >= 0.6 else 0  # Use confidence threshold of 0.6 instead of 0.5

    st.markdown("### Prediction Result")
    if prediction == 1:
        st.success(f"✅ This hotel is likely to be **popular** with {prob*100:.2f}% confidence.")
    else:
        st.warning(f"⚠️ This hotel is **less likely to be popular**, confidence: {(1 - prob)*100:.2f}%.")
