import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🏥 Heart Disease Prediction App")
st.write("Enter patient details to check the risk of heart disease.")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert categorical inputs
sex = 1 if sex == "Male" else 0

# Make prediction
if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    user_input_scaled = scaler.transform(user_input)
    
    prediction = model.predict(user_input_scaled)[0]
    
    if prediction == 1:
        st.error("⚠️ High risk of heart disease. Consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease. Stay healthy!")
