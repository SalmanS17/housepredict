import streamlit as st
import json
import pickle
import numpy as np
from PIL import Image

# Load JSON data
with open('columns.json', 'r') as f:
    data = json.load(f)

# Set page configuration with a stylish theme
st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 8px;}
    .stTitle {color: #333333;}
    </style>
    """,
    unsafe_allow_html=True
)

# Display a header image
image = Image.open("House.jpg")
st.image(image, use_column_width=True)

# Title and description
st.title("🏠 House Price Prediction")
st.subheader("Get an instant estimation of house prices in your desired location.")
st.write("Simply provide the area, number of bedrooms, and bathrooms to predict a price.")

# Input fields in an organized layout
col1, col2, col3 = st.columns(3)
with col1:
    selected_location = st.selectbox("Select Desired Location", data["data_columns"][3:])
with col2:
    area = st.text_input("Area of the house (in sq. ft.)", value="1000")
with col3:
    bedrooms_bhk = st.text_input("Number of BHK / Bedrooms", value="3")

bathrooms = st.text_input("Number of Bathrooms", value="2", help="Include all full and half bathrooms.")

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

# Prediction button and result
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

# Footer message
st.markdown(
    """
    <br><br>
    <center>🔍 Created by [Your Name](#) | © 2024 House Price Predictor</center>
    """,
    unsafe_allow_html=True
)
