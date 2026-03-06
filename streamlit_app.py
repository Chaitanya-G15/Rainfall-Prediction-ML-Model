import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Rainfall Prediction App ☔",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
model = joblib.load('rainfall_prediction_model.pkl')

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
    body {
        background-color: #F0F2F6;
    }
    .main {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
        padding: 2rem;
        border-radius: 15px;
    }
    .stButton>button {
        color: white;
        background: #008CBA;
        padding: 0.75em 1.5em;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #005f73;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #004d61;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# App Title and Description
# -----------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414974.png", width=130)
with col2:
    st.title("🌧️ Rainfall Prediction System")
    st.markdown("""
    #### Predict the possibility of rainfall using live weather data 🌦️  
    Enter weather parameters below to get an instant prediction.
    """)

# -----------------------------
# Input Section
# -----------------------------
st.markdown("### 🔧 Input Weather Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    pressure = st.slider('Pressure (hPa)', 900, 1100, 1000)
    dewpoint = st.slider('Dew Point (°C)', -10, 40, 20)
    humidity = st.slider('Humidity (%)', 0, 100, 50)

with col2:
    cloud = st.slider('Cloud Cover (%)', 0, 100, 50)
    sunshine = st.number_input('Sunshine Duration (hours)', 0.0, 24.0, 8.0, step=0.5)
    windspeed = st.slider('Wind Speed (km/h)', 0, 100, 10)

with col3:
    winddirection = st.selectbox(
        'Wind Direction',
        ('North', 'East', 'South', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest')
    )

# Convert direction to degree
direction_mapping = {
    'North': 0, 'Northeast': 45, 'East': 90, 'Southeast': 135,
    'South': 180, 'Southwest': 225, 'West': 270, 'Northwest': 315
}
wind_dir_degree = direction_mapping[winddirection]

# -----------------------------
# Predict Button
# -----------------------------
st.markdown("---")
predict_btn = st.button('🔍 Predict Rainfall')

if predict_btn:
    input_data = np.array([[pressure, dewpoint, humidity, cloud, sunshine, wind_dir_degree, windspeed]])
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    st.markdown("### 🧠 Prediction Result")
    st.progress(100)
    
    if prediction[0] == 1:
        st.error('☔ **Rain is likely to occur! Carry an umbrella!**')
        st.snow()
    else:
        st.success('☀️ **No rain expected today! Enjoy your day!**')
        st.balloons()

    st.metric(label="Probability of Rain", value=f"{proba[0][1]*100:.2f}%")

# -----------------------------
# Sidebar Information
# -----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1116/1116463.png", width=120)
st.sidebar.title("📘 About This App")
st.sidebar.write("""
This intelligent rainfall prediction app uses a **Random Forest Classifier**  
trained on **historical weather data** to predict the chance of rain.
""")

st.sidebar.markdown("### ⚙️ Features Used")
st.sidebar.write("""
- Atmospheric Pressure  
- Dew Point Temperature  
- Humidity Percentage  
- Cloud Cover  
- Sunshine Duration  
- Wind Direction & Speed
""")

st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.info("""
- **Accuracy:** 80%  
- **Precision:** 0.79  
- **Recall:** 0.93  
- **F1 Score:** 0.86
""")

with st.sidebar.expander("ℹ️ How It Works"):
    st.write("""
    The model takes real-time weather parameters,  
    processes them using a Random Forest algorithm,  
    and predicts the likelihood of rainfall.
    """)

with st.sidebar.expander("💡 About Data"):
    st.write("""
    Dataset used for training includes multiple years of meteorological data  
    containing parameters like humidity, pressure, cloud cover, and temperature.
    """)

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Chaitanya** ✨")

