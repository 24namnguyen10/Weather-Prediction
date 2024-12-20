import streamlit as st
import joblib
import numpy as np

# Load the weather model
model_path = 'weather_model.joblib'
model = joblib.load(model_path)

# Streamlit app
def main():
    st.title("Weather Prediction App")
    st.write("Upload data to predict weather conditions based on the trained model.")

    # Input features
    st.sidebar.header("Input Weather Parameters")
    max_temp = st.sidebar.slider("Max Temperature (°C)", 0, 50, 30)
    min_temp = st.sidebar.slider("Min Temperature (°C)", 0, 40, 24)
    wind = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 5)
    rain = st.sidebar.number_input("Rainfall (mm)", 0.0, 100.0, 1.0, step=0.1)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 80)
    cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
    pressure = st.sidebar.slider("Pressure (hPa)", 950, 1050, 1010)

    features = [max_temp, min_temp, wind, rain, humidity, cloud, pressure]

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict([features])
            st.success(f"Predicted Output: {prediction[0]}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
