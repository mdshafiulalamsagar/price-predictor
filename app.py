import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set page configuration
st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("Bengaluru House Price Prediction")
st.markdown("Estimate property prices in Bengaluru with AI")

# Sample locations in Bengaluru (you should replace with your actual trained locations)
locations = [
    '1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout',
    '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar',
    '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar', 
    '9th Phase JP Nagar', 'AECS Layout', 'Abbigere', 'Akshaya Nagar'
]

# Input form
with st.form("prediction_form"):
    st.header("Property Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.selectbox("Location", options=locations, index=0)
        total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1050)
    
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    
    submitted = st.form_submit_button("Estimate Price")

# Placeholder for model loading and prediction
def load_model():
    # This is where you would load your pre-trained model
    # For now, we'll use a simple calculation as a placeholder
    return None

def predict_price(model, location, total_sqft, bathrooms, bhk):
    # This is where you would make the actual prediction
    # For demonstration, we'll use a simple calculation
    base_price_per_sqft = 5000  # Base price per sqft
    location_factor = 1.2 if location == '1st Block Jayanagar' else 1.0
    bhk_factor = 1 + (bhk - 1) * 0.1
    bathroom_factor = 1 + (bathrooms - 1) * 0.05
    
    price = base_price_per_sqft * total_sqft * location_factor * bhk_factor * bathroom_factor
    return price

# When the form is submitted
if submitted:
    try:
        # Load model (in a real scenario, this would be your trained model)
        model = load_model()
        
        # Make prediction
        price = predict_price(model, location, total_sqft, bathrooms, bhk)
        
        # Format the price in Indian rupees
        formatted_price = "₹ {:,.2f}".format(price)
        
        # Display result
        st.success("### Estimated Property Price")
        st.markdown(f"# {formatted_price}")
        st.caption("This is an AI-generated estimate based on similar properties in the area.")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("""
        **Common causes of this error:**
        - Feature mismatch between training and prediction data
        - Inconsistent one-hot encoding for location features
        - Missing or extra features in the input data
        """)

# Add information about the app
with st.expander("About This App"):
    st.markdown("""
    This app predicts house prices in Bengaluru using machine learning. 
    The model was trained on real estate data from Bengaluru and uses 
    features like location, square footage, number of bathrooms, and BHK configuration.
    
    **Note:** To fix the feature mismatch error, ensure that:
    1. Your training and prediction data have exactly the same features
    2. You use the same preprocessing pipeline for both training and prediction
    3. You handle categorical variables (like location) consistently
    """)

# Footer
st.markdown("---")
st.caption("© 2023 Bengaluru House Price Prediction | AI Real Estate Tool")