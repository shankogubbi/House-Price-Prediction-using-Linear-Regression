import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

features = np.array([[
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup
]])
if st.button("Predict Price"):
    features = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup
    ]])

    prediction = model.predict(features)[0]
    st.success(f"Predicted Price: ${prediction * 100000:,.2f}")
