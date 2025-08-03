import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("student_mental_health_model.pkl")
pressure_encoder = joblib.load("pressure_encoder.pkl")
social_encoder = joblib.load("social_encoder.pkl")
activity_encoder = joblib.load("activity_encoder.pkl")
diet_encoder = joblib.load("diet_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title(" Student Mental Health Prediction")
st.write("This app predicts whether a student may face mental health issues based on lifestyle inputs.")

# Input fields
academic_pressure = st.selectbox("Academic Pressure", pressure_encoder.classes_)
social_life = st.selectbox("Social Life", social_encoder.classes_)
physical_activity = st.selectbox("Physical Activity", activity_encoder.classes_)
diet = st.selectbox("Diet", diet_encoder.classes_)

# Predict
if st.button("Predict"):
    # Transform inputs using encoders
    ap_encoded = pressure_encoder.transform([academic_pressure])[0]
    sl_encoded = social_encoder.transform([social_life])[0]
    pa_encoded = activity_encoder.transform([physical_activity])[0]
    d_encoded = diet_encoder.transform([diet])[0]
    
    # Create input array
    input_data = [[ap_encoded, sl_encoded, pa_encoded, d_encoded]]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    # Display result
    st.success(f" Prediction: {result}")
