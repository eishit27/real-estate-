# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="California Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --- Load Model and Columns ---
try:
    with open('linear_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('model_columns.pkl', 'rb') as columns_file:
        model_columns = pickle.load(columns_file)
except FileNotFoundError:
    st.error("Model or column files not found. Please ensure they are in the correct directory.")
    st.stop() # Stop the app if files are not found

# --- App Header ---
st.title('🏠 California Real Estate Price Predictor')
st.markdown("""
This app predicts the median house value for a district in California based on its features.
Please provide the input values in the sidebar.
""")

# --- Sidebar for User Input ---
st.sidebar.header('Input Features')

# Function to create input fields
def user_input_features():
    med_inc = st.sidebar.number_input('Median Income ($10,000s)', min_value=0.0, max_value=20.0, value=3.87, step=0.1)
    house_age = st.sidebar.number_input('Median House Age', min_value=1, max_value=100, value=28)
    ave_rooms = st.sidebar.number_input('Average Rooms', min_value=1.0, max_value=20.0, value=5.4, step=0.1)
    ave_bedrms = st.sidebar.number_input('Average Bedrooms', min_value=0.5, max_value=10.0, value=1.1, step=0.1)
    population = st.sidebar.number_input('Population', min_value=1, max_value=40000, value=1425)
    ave_occup = st.sidebar.number_input('Average Occupancy', min_value=1.0, max_value=20.0, value=3.0, step=0.1)
    latitude = st.sidebar.slider('Latitude', min_value=32.0, max_value=42.0, value=35.6, step=0.1)
    longitude = st.sidebar.slider('Longitude', min_value=-125.0, max_value=-114.0, value=-119.5, step=0.1)
    
    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Panel ---
st.header('Prediction')

# Display user input as a table
st.subheader('Your Input:')
st.dataframe(input_df)

# Prediction button
if st.button('Predict House Value'):
    # Ensure the order of columns matches the model's training data
    input_df = input_df[model_columns]
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # The model predicts in units of $100,000. Convert it to dollars.
    predicted_price = prediction[0] * 100000
    
    st.success(f'**Predicted Median House Value: ${predicted_price:,.2f}**')
    
    st.balloons()

st.markdown("---")
st.write("Built with Streamlit and Scikit-learn.")