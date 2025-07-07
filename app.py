import streamlit as st
import joblib
import numpy as np

# Load trained model and features
model = joblib.load("logistic_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # List of column names used in training

st.title("Hotel Popularity Prediction App")
st.markdown("Fill in hotel details to predict whether it is likely to be popular.")

# Input form
score_adjusted = st.number_input("Score Adjusted", min_value=1.0, max_value=5.0, value=4.3)
bubble_rating = st.selectbox("Bubble Rating", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=7)
price_curr_min = st.number_input("Minimum Price", min_value=0.0, value=100.0)
location_grade = st.number_input("Location Grade", min_value=0.0, max_value=100.0, value=90.0)
photos = st.number_input("Number of Photos", min_value=0, value=50)
discount = st.selectbox("Has Discount?", [0, 1])
class_4_5 = st.selectbox("Hotel Class 4-5 Star?", [0, 1])

# Prepare input
user_input = {
    'score_adjusted': score_adjusted,
    'bubble_rating': bubble_rating,
    'price_curr_min': price_curr_min,
    'location_grade': location_grade,
    'photos': photos,
    'discount': discount,
    'class_4_5': class_4_5
}

# Fill remaining features with 0
input_data = [user_input.get(col, 0) for col in feature_names]
input_array = np.array(input_data).reshape(1, -1)

# Predict
if st.button("Predict Popularity"):
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1] * 100
    if prediction == 1:
        st.success(f"✅ This hotel is likely to be **popular** ({prob:.2f}% confidence).")
    else:
        st.warning(f"⚠️ This hotel is **less likely to be popular** ({100 - prob:.2f}% confidence).")
