import streamlit as st
import numpy as np
import joblib
# Load model
model = joblib.load("iris_model.pkl")

st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

st.title("Iris Flower Prediction App")
st.write("Enter flower measurements below:")

# Layout (better UI)
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")

with col2:
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

# Predict button
if st.button("Predict"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(sample)

    st.success("Predicted Species: " + str(prediction[0]))