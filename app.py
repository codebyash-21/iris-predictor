import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="Iris Predictor", page_icon="🌸")

st.title("🌸 Simple Flower Classifier")
st.markdown("Enter the flower measurements below to see the AI's prediction.")

# Load the saved model
@st.cache_resource # This keeps the app fast
def load_model():
    return joblib.load('iris_model.pkl')

model = load_model()

# Create 2 columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    s_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.4)
    s_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, 3.4)

with col2:
    p_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.3)
    p_width = st.number_input("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict Species", type="primary"):
    features = np.array([[s_length, s_width, p_length, p_width]])
    prediction = model.predict(features)
    
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[prediction[0]]
    
    st.success(f"The AI predicts this is a **{result}**!")