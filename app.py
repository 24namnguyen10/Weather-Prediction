import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder.appName("Weather Prediction App").getOrCreate()

# Path to the pre-trained model
model_path = 'weather_model'  # Replace with the actual path to your model

# Verify and load the pre-trained model
if os.path.exists(model_path):
    weather_model = PipelineModel.load(model_path)
    st.write("Model loaded successfully from:", model_path)
else:
    st.error(f"Model path does not exist: {model_path}")
    st.stop()

# Streamlit UI
def main():
    st.title("Weather Prediction App")

    # Input features
    st.sidebar.header("Input Weather Parameters")
    max_temp = st.sidebar.slider("Max Temperature (°C)", 0, 50, 30)
    min_temp = st.sidebar.slider("Min Temperature (°C)", 0, 40, 24)
    wind = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 5)
    rain = st.sidebar.number_input("Rainfall (mm)", 0.0, 100.0, 1.0, step=0.1)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 80)
    cloud = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
    pressure = st.sidebar.slider("Pressure (hPa)", 950, 1050, 1010)

    if st.button("Predict"):
        # Create a DataFrame from user inputs
        input_data = pd.DataFrame({
            "max_temp": [max_temp],
            "min_temp": [min_temp],
            "wind": [wind],
            "rain": [rain],
            "humidity": [humidity],
            "cloud": [cloud],
            "pressure": [pressure],
        })

        # Convert to Spark DataFrame
        spark_df = spark.createDataFrame(input_data)

        # Make predictions
        predictions = weather_model.transform(spark_df)

        # Collect and display predictions
        predictions = predictions.select("prediction").collect()
        st.success(f"Prediction: {predictions[0]['prediction']}")

if __name__ == "__main__":
    main()
