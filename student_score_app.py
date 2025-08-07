import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Function to train and save the model
def train_model():
    # Synthetic dataset: [Study Hours, Sleep Hours, Attendance %]
    X = np.array([
        [2, 6, 70],
        [4, 7, 80],
        [6, 5, 90],
        [8, 6, 95],
        [10, 7, 100],
        [1, 4, 60],
    ])
    y = np.array([65, 75, 85, 90, 95, 55])  # Corresponding scores

    model = LinearRegression()
    model.fit(X, y)

    # Save the model to a file
    with open("student_score_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model

# Load the model if available, otherwise train it
try:
    with open("student_score_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = train_model()

# Streamlit UI
st.title("Student Score Predictor")

st.write("Enter the following details to predict the student's score:")

study_hours = st.slider("Study Hours per Day", 0, 12, 6)
sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
attendance = st.slider("Attendance Percentage", 0, 100, 90)

if st.button("Predict Score"):
    input_data = np.array([[study_hours, sleep_hours, attendance]])
    predicted_score = model.predict(input_data)[0]
    st.success(f"Predicted Student Score: {predicted_score:.2f}")
