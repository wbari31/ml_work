# Deploying model
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from pathlib import Path


st.set_page_config(page_title="Bean Classification", layout="centered")
st.write("""This app predicts bean class.""")
BASE_DIR = Path(__file__).resolve().parent
model=joblib.load(BASE_DIR/'model.pkl')
scaler=joblib.load(BASE_DIR/'scaler.pkl')
power_transformer=joblib.load(BASE_DIR/'power_transformer.pkl')
pca=joblib.load(BASE_DIR/'pca.pkl')

Area = st.sidebar.number_input('Area', 0.0, 500000.0,250000.0 ,.1)
Perimeter = st.sidebar.number_input('Perimeter', 0.0, 2000.0, 1000.0,.1)
MajorAxisLength = st.sidebar.number_input('MajorAxisLength', 0.0, 1000.0, 500.0,.1)
MinorAxisLength = st.sidebar.number_input('MinorAxisLength', 0.0, 1000.0, 500.0,.1)
AspectRation = st.sidebar.number_input('AspectRation', 1.00, 5.00, 2.50,.01)
Eccentricity = st.sidebar.number_input('Eccentricity', 0.00, 1.00, .50,.01)
ConvexArea = st.sidebar.number_input('ConvexArea', 0.0, 500000.0,250000.0 ,.1)
EquivDiameter = st.sidebar.number_input('EquivDiameter', 0.0, 1000.0, 500.0,.1)
Extent = st.sidebar.number_input('Extent', 0.00, 1.00, .50,.01)
Solidity = st.sidebar.number_input('Solidity', 0.00, 1.00, .50,.01)
roundness = st.sidebar.number_input('roundness', 0.00, 1.00, .50,.01)
Compactness = st.sidebar.number_input('Compactness', 0.00, 1.00, .50,.01)
ShapeFactor1 = st.sidebar.number_input('ShapeFactor1', 0.00, 1.00, .50,.01)
ShapeFactor2 = st.sidebar.number_input('ShapeFactor2', 0.00, 1.00, .50,.01)
ShapeFactor3 = st.sidebar.number_input('ShapeFactor3', 0.00, 1.00, .50,.01)
ShapeFactor4 = st.sidebar.number_input('ShapeFactor4', 0.00, 1.00, .50,.01)

skewed_cols=['Area','Perimeter','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea',
             'EquivDiameter','Solidity','ShapeFactor4']
all_feature_cols=['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
       'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
       'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
       'ShapeFactor3', 'ShapeFactor4']

def user_input_features():
    input_df = pd.DataFrame([{
            'Area': Area, 
            'Perimeter': Perimeter,
            'MajorAxisLength': MajorAxisLength,
            'MinorAxisLength': MinorAxisLength,
            'AspectRation': AspectRation,
            'Eccentricity': Eccentricity,
            'ConvexArea': ConvexArea,
            'EquivDiameter': EquivDiameter,
            'Extent': Extent,
            'Solidity': Solidity,
            'roundness': roundness,
            'Compactness': Compactness,
            'ShapeFactor1': ShapeFactor1,
            'ShapeFactor2': ShapeFactor2,
            'ShapeFactor3': ShapeFactor3,
            'ShapeFactor4': ShapeFactor4
            }])
    
    input_df_transformed = power_transformer.transform(input_df[skewed_cols])
    input_df[skewed_cols] = pd.DataFrame(input_df_transformed,columns=skewed_cols, index=input_df.index)

    input_df_scaled = scaler.transform(input_df)
    input_df = pd.DataFrame(input_df_scaled,columns=input_df.columns, index=input_df.index)

    input_df_pca = pca.transform(input_df)
    input_df_pca = pd.DataFrame(input_df_pca, 
                                columns=[f"PC{i+1}" for i in range(input_df_pca.shape[1])])
    
    return input_df_pca

if st.button('Predict Bean Class'):
    input_data = user_input_features()
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Prediction Probability')
    st.write(prediction_proba)


    
    
