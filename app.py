import streamlit as st
import joblib
import pandas as pd

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Prediction Model By Kamran")
st.markdown("Provide the Following Details")

#collecting Data
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', "TA", 'ASY'])
resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Suger > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angine = st.selectbox("Exercise-Induced Angina",["Y", "N"])

Oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        "RestingBP" : resting_bp,
        "Cholesterol" : cholesterol,
        "FastingBS" : fasting_bs,
        "MaxHR" : max_hr,
        "Sex_" + sex : 1,
        "ChestPainType_" + chest_pain : 1,
        "RestingECG_" + resting_ecg : 1,
        "ExcerciseAngine_" + exercise_angine : 1,
        "ST_Slope_" + st_slope : 1 
    }
    
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("⚠️ High Rish of Heart Disease")
    else:
        st.success("✔️ Low Risk of Heart Disease")
    