import streamlit as st
import json
import pickle
import numpy as np
from PIL import Image

# Load JSON data
with open('columns.json', 'r') as f:
    data = json.load(f)

# Set page configuration
st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .main {background-color: #fafafa;}
    .stTitle, .stHeader {color: #2c3e50; font-family: 'Arial', sans-serif;}
    .stButton>button {background-color: #3498db; color: white; padding: 0.5em 2em; border-radius: 8px; font-size: 1.2em;}
    .stButton>button:hover {background-color: #2980b9; cursor: pointer;}
    .sidebar .sidebar-content {padding-top: 2rem;}
    .input-textbox {padding: 0.5em; font-size: 1.1em; width: 100%; border: 1px solid #bdc3c7; border-radius: 5px;}
    </style>
    """,
    unsafe_allow_html=True
)

# Display a resized header image
image = Image.open("House.jpg")
st.image(image.resize((800, 200)), caption="Find your dream home's value", use_column_width=True)

# Title and description
st.title("🏠 House Price Prediction")
st.subheader("Instantly estimate the price of a house in your desired location")
st.markdown("Simply enter the area, number of bedrooms, and bathrooms below to get an estimated price.")

# Input fields in columns for a sleek layout
col1, col2, col3 = st.columns(3)
with col1:
    selected_location = st.selectbox("Select Desired Location", data["data_columns"][3:])
with col2:
    area = st.text_input("Area (sq. ft.)", value="1000", help="Enter the total area in square feet.")
with col3:
    bedrooms_bhk = st.text_input("BHK / Bedrooms", value="3", help="Enter the number of bedrooms.")

bathrooms = st.text_input("Bathrooms", value="2", help="Include all full and half bathrooms.")

# Load the pre-trained model
with open('home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Define prediction function
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(np.array(data['data_columns']) == location)[0][0]
    x = np.zeros(len(data['data_columns']))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

# Prediction button with enhanced styling
if st.button("💡 Predict Price"):
    try:
        area_val = float(area)
        bedrooms_val = int(bedrooms_bhk)
        bathrooms_val = int(bathrooms)

        predicted_price = predict_price(selected_location, area_val, bathrooms_val, bedrooms_val)
        st.success(f"🏡 Estimated Price: ₹ {round(predicted_price, 2)} Lakh")
    except ValueError:
        st.warning("Please enter valid numerical values for area, BHK, and bathrooms.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Footer message with custom styling
st.markdown(
    """
    <div style="text-align: center; margin-top: 2rem;">
    🔍 Created by <a href="#" style="color: #3498db; text-decoration: none;">Salman Khan</a> | © 2024 House Price Predictor
    </div>
    """,
    unsafe_allow_html=True
)
