import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import joblib 
from joblib import load
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import category_encoders as ce 

file_path = pathlib.Path(__file__).parent.as_posix() + '/f_df.joblib'
df = joblib.load(file_path)
df.drop('Unnamed: 0.1', axis =1, inplace =True)

logged_model = 'runs:/f030090d1f1140608b5bcc1133ae9251/model'
pipeline = mlflow.sklearn.load_model(logged_model)[0]

model = pipeline
print(model.predict(pd.Series(df.iloc[0, :]).to_frame().T))
preprocesser = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), ['built_up_area', 'bedRoom', 'bathroom', 'servant room','price']), #'store room'
        ('cat', OrdinalEncoder(), ['property_type']),
        ('ohe', OneHotEncoder(drop = 'first', sparse_output = False), ['agePossession', 'furnishing_type','luxury_category', 'floor_category', 'balcony']),
        ('ce', ce.TargetEncoder(),['sector'])
    ],remainder = 'passthrough'
        )
preprocesser.fit(df)       
print(preprocesser.named_transformers_['cat'].categories_)