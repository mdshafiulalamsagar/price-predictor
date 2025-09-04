import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load the trained model and location data
@st.cache_resource
def load_model():
    model = joblib.load('trained_model.pkl')
    return model

@st.cache_data
def load_location_data():
    with open('location_columns.json', 'r') as f:
        locations = json.load(f)
    return locations

model = load_model()
location_names = load_location_data()

# App title and description
st.title("🏠 House Price Prediction")
st.markdown("""
Predict the price of houses based on features like location, square footage, number of bathrooms, and BHK.
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", options=location_names)
    total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, value=1000.0, step=50.0)
    
with col2:
    bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)

# Prediction function
def predict_price(location, sqft, bath, bhk):    
    loc_index = np.where(np.array(location_names) == location)[0][0]

    x = np.zeros(len(location_names))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Predict button
if st.button("Predict Price", type="primary"):
    try:
        price = predict_price(location, total_sqft, bath, bhk)
        st.success(f"Estimated Price: ₹{price:,.2f} Lakhs")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Add some information about the model
with st.expander("About this model"):
    st.markdown("""
    This model was built using:
    - Linear Regression algorithm
    - Bengaluru House Price Dataset
    - Features like location, square footage, bathrooms, and BHK
    
    The model has been trained on preprocessed data that includes:
    - Handling missing values
    - Converting square footage to numeric values
    - Location-based grouping for better generalization
    - Outlier removal
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn")