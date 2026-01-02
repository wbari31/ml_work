# Deploying model
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Insurance cost Predictor", layout="centered")
st.write("""This app predicts insurance costs.""")
BASE_DIR = Path(__file__).resolve().parent
model=joblib.load(BASE_DIR/'best_rf.pkl')
scaler=joblib.load(BASE_DIR/'scaler.pkl')
label_encoders=joblib.load(BASE_DIR/'label_encoders.pkl')

#numerical features
age = st.sidebar.number_input('age', 18, 100, 30)
bmi = st.sidebar.number_input('bmi', 10.0, 60.0, 25.0)
children = st.sidebar.number_input('children', 0, 10, 1)
scaled_num_cols=scaler.transform([[age, bmi, children]])
age=scaled_num_cols[0][0]
bmi=scaled_num_cols[0][1]
children=scaled_num_cols[0][2]
#categorical features
sex = st.sidebar.selectbox('sex', ('female', 'male'))
sex=label_encoders['sex'].transform([sex])

#sex=0
smoker = st.sidebar.selectbox('smoker', ('yes', 'no'))
smoker=label_encoders['smoker'].transform([smoker])
region = st.sidebar.selectbox('region', ('southwest', 'southeast', 'northwest', 'northeast'))
region=label_encoders['region'].transform([region])

#st.write(f'sex: {sex}, smoker: {smoker}, region: {region}, age: {age}, bmi: {bmi}, children: {children}')
def user_input_features():
    data={
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    return data

if st.button('Predict insurance cost'):
    input_data = user_input_features()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.subheader('Predicted Insurance Cost:')
    st.write(f"INR {prediction[0]:.2f}")
