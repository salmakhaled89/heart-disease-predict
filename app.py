# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_models()

 
st.set_page_config(
    page_title="Heart Disease Predictor",
    layout="wide"
)

 
st.title(" Heart Disease Prediction App")

st.subheader("Patient Health Information")

 
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)

with col2:
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
    restecg = st.selectbox(
        "Resting ECG Results",
        options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy"
    )
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    exang = st.selectbox("Exercise-Induced Angina", options=["No", "Yes"])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=["Upsloping", "Flat", "Downsloping"],
        help="0: Upsloping, 1: Flat, 2: Downsloping"
    )
    num_major_vessels = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, step=1)
    thal = st.selectbox(
        "Thalassemia",
        options=["Normal", "Fixed Defect", "Reversable Defect"],
        help="0: Normal, 1: Fixed Defect, 2: Reversable Defect"
    )

 
sex_encoded = 1 if sex == "Male" else 0
chest_pain_encoded = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain_type)
fasting_encoded = 1 if fasting_blood_sugar == "Yes" else 0
restecg_encoded = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
exang_encoded = 1 if exang == "Yes" else 0
slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal_encoded = ["Normal", "Fixed Defect", "Reversable Defect"].index(thal)

 
input_features = np.array([[
    age,
    sex_encoded,
    chest_pain_encoded,
    resting_bp,
    cholesterol,
    fasting_encoded,
    restecg_encoded,
    max_hr,
    exang_encoded,
    oldpeak,
    slope_encoded,
    num_major_vessels,
    thal_encoded
]])

 

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    predict_button = st.button("Predict", use_container_width=True, type="primary")

if predict_button:
 
    input_scaled = scaler.transform(input_features)
    
   
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]   
    
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        if prediction == 1:
            st.error("**High Risk of Heart Disease**")
            st.markdown(f"""
                <div style='background-color: #fee; padding: 20px; border-radius: 10px; border-left: 5px solid red;'>
                    <p style='color: #666;'>
                        Please consult a healthcare professional for further evaluation.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("**Low Risk of Heart Disease**")
            st.markdown(f"""
                <div style='background-color: #efe; padding: 20px; border-radius: 10px; border-left: 5px solid green;'>
                    <p style='color: #666;'>
                        Keep maintaining a healthy lifestyle! 
                    </p>
                </div>
            """, unsafe_allow_html=True)