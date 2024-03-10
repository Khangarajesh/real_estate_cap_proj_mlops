import streamlit as st
import pandas as pd
import numpy as np
import mlflow

#model loading 
logged_model = 'runs:/932df9f8ac9b4cff87b3394bb9aee28a/model'
# Load model.
loaded_model = mlflow.sklearn.load_model(logged_model)

def predict(features):
    prediction = loaded_model.predict([features])
    return prediction