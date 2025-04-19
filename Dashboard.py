import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Load model and dataset
model = joblib.load("CO_Prediction_Model.pkl")
df = pd.read_csv("Cleaned_AirQuality_Data.csv")  # replace with actual file path

# 🧠 Main Section: Dashboard Layout
st.set_page_config(layout="wide")  
st.title("📊 Air Quality Prediction Dashboard")
st.write("This dashboard predicts **CO(GT)** (Carbon Monoxide level) using sensor data.")

# Sidebar: Prediction Input
st.sidebar.title("📥 Predict CO(GT)")
st.sidebar.markdown("Enter sensor values below to predict Carbon Monoxide level:")
features = [    
    'CO_rolling_mean_3','CO_lag_1','C6H6(GT)','PT08.S2(NMHC)','NOx(GT)','PT08.S1(CO)','PT08.S5(O3)','NO2(GT)','CO_lag_2','PT08.S4(NO2)','CO_rolling_std_3','Hour']
user_inputs = [st.sidebar.number_input(f"{feature}", step=0.01) for feature in features]

if st.sidebar.button("Predict"):
    pred = model.predict(np.array(user_inputs).reshape(1, -1))
    st.sidebar.success(f"Predicted CO(GT): {pred[0]:.3f} mg/m³")


# Rectangular Plot 1: Feature Importance
st.subheader("🕒 CO(GT) Time Patterns")
st.image("./Plots/Hour_Day_Month.png", caption="Hourly, Daily, and Monthly CO(GT) Trends", use_container_width =True)

# Two Square Plots Side-by-Side
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Actual vs Predicted")
    st.image("./Plots/Actual_vs_Predicted.png", caption="Actual vs Predicted CO(GT)", use_container_width=True)

with col2:
    st.subheader("📊 Histogram")
    st.image("./Plots/Hist_of_All.png", caption="Histogram of All Features", use_container_width=True)


st.subheader("⭐ CO(GT) Important Features")
st.image("./Plots/Feature_Imp.png", caption="Feature Importance for CO(GT)", use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏳️‍🌈 Correlation Heatmap")  
    st.image("./Plots/All_Corr.png", caption="Correlation Heatmap of Features", use_container_width=True)

with col2:
    st.subheader("🏳️‍🌈 Correlation of 'CO' to Others")
    st.image("./Plots/Corr_to_CO.png", caption="Correlation of 'CO' to Others", use_container_width=True)

# Display Cleaned Dataset
st.subheader("🗃️ Cleaned Dataset (Top 10 Rows)")
st.dataframe(df.head(10))

# Display Model Summary
st.subheader("📝 Model Summary")
st.markdown("<h4>This model is a Random Forest Regressor trained on responses of a gas multisensor device deployed on the field in an Italian city.</h4>", unsafe_allow_html=True)
st.write(model)

# Model Metrics
st.subheader("🤖 Model Metrics")
st.write("R²: ", 0.9674055645463262)
st.write("MAE(Mean Absolute Error): ", 0.08281854265588677)
st.write("MSE(Mean Squared Error): ", 0.022659066567831333)
st.write("RMSE(Root Mean Squared Error): ", 0.1505292880732229)


# Resource Links
st.subheader("🔗 Resources")
st.markdown("""
- 📚 [Dataset Link (UCI Repository)](https://archive.ics.uci.edu/dataset/360/air+quality)
- 💻 [GitHub Repository](https://github.com/DhyeyFultariya/AirQuality-CO-Prediction_Model)
""")

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🏁 Dashboard Ended. Thank you for showing Interest. Use the sidebar to predict CO(GT)!")
st.subheader("👨‍💻 Created by: Dhyey Fultariya")
