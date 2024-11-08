import streamlit as st
import json
import pickle
import numpy as np
import os

# Function to load JSON data from a file
with open('columns.json', 'r') as f:
        data= json.load(f)


# Set the title of the page
st.set_page_config(page_title="My App", layout="wide")
st.image("header.jpg", use_column_width=True ) 
st.title("Welcome to My App")

selected_location = st.selectbox(
    "Select Desired Location",
   data["data_columns"][3:],
)

area = st.text_input("Enter the area of the house (in square feet)", value="1000")
bedrooms_bhk = st.text_input("Enter the number of BHK / Bedrooms", value="3")
bathrooms = st.text_input("Enter the number of bathrooms", value="2")


# Load the pre-trained model
with open('home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(np.array(data['data_columns']) == selected_location)[0][0]
    #print(loc_index)

    x = np.zeros( len(data['data_columns']))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    #print(x)

    return model.predict([x])[0]

if st.button("Start Prediction"):
    # Convert the inputs into the required format (e.g., numerical values)
    try:
        area = float(area)
        bedrooms_bhk = int(bedrooms_bhk)
        bathrooms = int(bathrooms)
    except ValueError:
        st.warning("Please enter valid numbers for all inputs.")

    try:
        predicted_price=predict_price(selected_location,area,bathrooms,bedrooms_bhk)
        st.success(str(round(predicted_price,2))+" (IND Lakh)")
    except Exception:
       st.warning("Problem in Model")  
    
    
    
    
    #st.success('House Prices is : *****' )
#else:
#    st.error('There is some problem, try later' );


