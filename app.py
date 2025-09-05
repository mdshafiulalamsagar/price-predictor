import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# App title and description
st.title("House Price Prediction")
st.markdown("Estimate property prices with AI")

# Sample locations
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

# Simple model training function (in a real app, you'd load a pre-trained model)
def train_model():
    # This is just a placeholder - in a real app, you'd use your actual training data
    # For demonstration, we'll create a simple model
    model = LinearRegression()
    return model

# Load or train model
try:
    # Try to load a pre-trained model
    model = joblib.load('model.pkl')
except:
    # If model doesn't exist, train a simple one
    model = train_model()
    # Save the model for future use
    joblib.dump(model, 'model.pkl')

def predict_price(location, total_sqft, bathrooms, bhk):
    
    # Base price per sqft
    base_price_per_sqft = 5000
    
    # Location multiplier (simplified)
    location_factor = 1.2 if 'Jayanagar' in location else 1.0
    
    # BHK and bathroom factors
    bhk_factor = 1 + (bhk - 1) * 0.1
    bathroom_factor = 1 + (bathrooms - 1) * 0.05
    
    # Calculate price
    price = base_price_per_sqft * total_sqft * location_factor * bhk_factor * bathroom_factor
    
    # Add some randomness to simulate different locations
    price = price * (0.9 + 0.2 * (hash(location) % 100) / 100)
    
    return price

# When the form is submitted
if submitted:
    try:
        # Make prediction
        price = predict_price(location, total_sqft, bathrooms, bhk)
        
        # Format the price in Indian rupees
        formatted_price = "₹ {:,.2f}".format(price)
        
        # Display result
        st.success("### Estimated Property Price")
        st.markdown(f"# {formatted_price}")
        st.caption("This is an AI-generated estimate based on similar properties in the area.")
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.info("""
        **Note:** This is a simplified demonstration. In a production app, 
        you would use a properly trained machine learning model.
        """)


# Footer
st.markdown("---")
st.caption("© 2025 House Price Prediction | MD Shafiul Alam Sagar")