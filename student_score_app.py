
import streamlit as st
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Load or train the model
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