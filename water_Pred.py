import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained model
model = load_model('water_quality_tuned_model')

st.title("💧 Water Quality Disease Predictor")
st.markdown("Enter the water quality parameters to predict the possible waterborne disease.")

# Create two columns
col1, col2 = st.columns(2)

# Left Column Inputs
with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
    solids = st.number_input("Solids", min_value=0.0, value=8000.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=8.0)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=320.0)

# Right Column Inputs
with col2:
    conductivity = st.number_input("Conductivity", min_value=0.0, value=420.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=14.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=75.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=3.5)

# Predict Button
if st.button("Predict Disease"):
    user_data = pd.DataFrame([{
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }])
    
    result = predict_model(model, data=user_data)
    predicted_label = result['prediction_label'].iloc[0]

    st.success(f"🚑 Predicted Disease: **{predicted_label}**")
