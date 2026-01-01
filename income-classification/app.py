# Deploying model
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Income Predictor", layout="centered")
st.write("""This app predicts income.""")
BASE_DIR = Path(__file__).resolve().parent
model=joblib.load(BASE_DIR+'/xgbclassifier_classifier_model.pkl')
scaler=joblib.load(BASE_DIR+'/scaler.pkl')
te=joblib.load(BASE_DIR+'/target_encoder.pkl')
pt=joblib.load(BASE_DIR+'/power_transformer_skewness_handler.pkl')

#numerical features
age = st.sidebar.number_input('age', 17, 90, 25)
fnlwgt = st.sidebar.number_input('fnlwgt', 10000, 1000000, 200000)
education_num = st.sidebar.number_input('education-num', 1, 16, 10)
capital_gain = st.sidebar.number_input('capital-gain', 0, 99999, 5000)
capital_loss = st.sidebar.number_input('capital-loss', 0, 99999, 0)
hours_per_week = st.sidebar.number_input('hours-per-week', 1, 99, 40)

#categorical features
workclass = st.sidebar.selectbox('workclass', ('Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'))
marital_status = st.sidebar.selectbox('marital-status', ('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'))
occupation = st.sidebar.selectbox('occupation', ('Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'))
relationship = st.sidebar.selectbox('relationship', ('Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'))
race = st.sidebar.selectbox('race', ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'))
sex = st.sidebar.selectbox('sex', ('Male', 'Female'))
native_country = st.sidebar.selectbox('native-country', ('United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Portugal', 'Nicaragua', 'Greece', 'Ecuador', 'Ireland', 'France', 'Trinadad&Tobago', 'Cambodia', 'Thailand', 'Yugoslavia', 'Peru', 'Hungary', 'Holand-Netherlands'))

cols=['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


def user_input_features():
    #categorical features. Maintaining the same order as in training data.
    encoded_cols_df = pd.DataFrame([{'age': age, 
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education-num': education_num,
            'marital-status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': native_country}])
    
    encoded_cols_df=pd.DataFrame(te.transform(encoded_cols_df), columns=cols)

    #handling skewness for capital-gain and capital-loss
    skew_df = pd.DataFrame([{
    "capital-gain": capital_gain,
    "capital-loss": capital_loss
    }])

    #logger.info(f"=======================================skew_df['capital-gain'][0]:{skew_df}")

    skew_df = pd.DataFrame(pt.transform(skew_df),columns=['capital-gain','capital-loss'])
    #scaling numerical features
    scale_num_df=pd.DataFrame([{
    "fnlwgt": fnlwgt,
    "capital-gain": skew_df['capital-gain'][0],
    "capital-loss": skew_df['capital-loss'][0],
    "age": age,
    "hours-per-week": hours_per_week
    }])

    scale_num_df= pd.DataFrame(scaler.transform(scale_num_df), columns=['fnlwgt','capital-gain', 'capital-loss','age','hours-per-week'])

    data = {'age': scale_num_df['age'][0], 
            'workclass': encoded_cols_df['workclass'][0],
            'fnlwgt': scale_num_df['fnlwgt'][0],
            'education-num': education_num,
            'marital-status': encoded_cols_df['marital-status'][0],
            'occupation': encoded_cols_df['occupation'][0],
            'relationship': encoded_cols_df['relationship'][0],
            'race': encoded_cols_df['race'][0],
            'sex': encoded_cols_df['sex'][0],
            'capital-gain': scale_num_df['capital-gain'][0],
            'capital-loss': scale_num_df['capital-loss'][0],
            'hours-per-week': scale_num_df['hours-per-week'][0],
            'native-country': encoded_cols_df['native-country'][0]}
    
    return data



if st.button('Predict Income'):
    input_data = user_input_features()
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    income_label = 'greater than 50K' if prediction[0]==1 else 'less than or equal to 50K'
    st.write(income_label)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
