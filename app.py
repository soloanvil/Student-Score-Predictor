import streamlit as st
import joblib
import pandas as pd

# Load trained models
score_model = joblib.load("student_score_model.pkl")
pass_model = joblib.load("student_pass_model.pkl")

# Streamlit UI
st.title("📚 Student Exam Performance Predictor")
st.write("Enter study details to predict exam score and pass/fail status.")

# User inputs
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)
previous_score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, step=0.5)

if st.button("Predict"):
    input_data = pd.DataFrame([[study_hours, previous_score]], columns=["Study Hours", "Previous Exam Score"])
    
    # Predictions
    predicted_score = score_model.predict(input_data)[0]
    pass_prediction = pass_model.predict(input_data)[0]
    pass_probability = pass_model.predict_proba(input_data)[0][1] * 100  # Probability of passing
    
    # Display results
    st.success(f"Predicted Exam Score: {predicted_score:.2f}")
    if pass_prediction == 1:
        st.success(f"✅ Student is likely to PASS ({pass_probability:.2f}% confidence)")
    else:
        st.error(f"❌ Student is likely to FAIL ({pass_probability:.2f}% confidence)")