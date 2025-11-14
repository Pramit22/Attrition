import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

st.set_page_config(page_title='👥 Employee Attrition Prediction', layout='centered')

st.title('👥 Employee Attrition Prediction App')

# Load model
model = joblib.load("model.pkl")

# Load feature list
with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

st.header('Enter Employee Details:')

inputs = {}
for col in feature_cols:
    if st.text_input(col, "").replace('.','',1).replace('-','',1).isdigit():
        inputs[col] = float(st.text_input(col, "0"))
    else:
        inputs[col] = st.text_input(col, "")

# Convert to DataFrame
input_df = pd.DataFrame([inputs])

st.subheader('Input Preview')
st.dataframe(input_df)

if st.button('🔍 Predict Attrition'):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f'⚠️ Employee likely to leave (Probability: {prob:.2f})')
    else:
        st.success(f'✅ Employee likely to stay (Probability: {prob:.2f})')
