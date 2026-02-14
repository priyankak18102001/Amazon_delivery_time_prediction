import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Amazon Delivery Predictor", layout="centered")



# Load model
MODEL_URL = "https://drive.google.com/file/d/1hgJxWa0UD5w7sGhAC3xYJfNV4Qloi7mB/view?usp=drive_link"
MODEL_PATH = "delivery_model.pkl"

FEATURE_URL = "https://drive.google.com/file/d/1hgJxWa0UD5w7sGhAC3xYJfNV4Qloi7mB/view?usp=drive_link"
FEATURE_PATH = "feature_columns.pkl"

feature_importance = pd.read_csv("feature_importance.xls")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Download feature columns if not present
if not os.path.exists(FEATURE_PATH):
    gdown.download(FEATURE_URL, FEATURE_PATH, quiet=False)

# Load files
model = pickle.load(open(MODEL_PATH, "rb"))
feature_columns = pickle.load(open(FEATURE_PATH, "rb"))

st.title("Amazon Delivery Time Prediction")

st.write("Enter delivery details:")

# Numeric input
agent_age = st.number_input("Agent Age", min_value=18, max_value=60)
agent_rating = st.number_input("Agent Rating", min_value=1.0, max_value=5.0)
distance = st.number_input("Distance (km)")

#Dropdown inputs
weather = st.selectbox("weather", ["sunny","stormy","sandstorms","cloudy","fog","windy"])
traffic = st.selectbox("traffic",["low","medium","high","jam"])
vehicle = st.selectbox("vehicle",["motorcycle","scooter","van","bicycle"])
area = st.selectbox("area",["urban","semi-urban","metropolitan","other"])
category = st.selectbox("category",[
    "clothing","electronics","sports","cosmetics","toys","snacks","shoes",
    "jewelry","outdoors","grocery","books","kitchen","home","pet supplies","skincare"
])

if st.button("Predict Delivery Time"):

    # Arrange inputs in correct order
    input_data = pd.DataFrame(columns = feature_columns)
    input_data.loc[0] =0
    
    # Fill numeric features
    input_data["Agent_Age"] = agent_age
    input_data["Agent_Rating"] = agent_rating
    input_data["distance"] = distance

    #Fill categorical features (one-hot encoded)
    weather_col = f"weather_{weather.capitalize()}"
    traffic_col = f"traffic_{traffic.capitalize()}"
    vehicle_col = f"vehicle_{vehicle}"
    area_col = f"Area_{area.capitalize()}"
    category_col = f"Category_{category.capitalize()}"

    for col in [weather_col , traffic_col , vehicle_col , area_col , category_col]:
        if col in input_data.columns:
            input_data[col] =1

    prediction = model.predict(input_data)

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")

# Feature importance section
st.subheader("Top Factors Affecting Delivery Time")

top_features = feature_importance.head(10)

fig = plt.figure()
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importance")

st.pyplot(fig)    
