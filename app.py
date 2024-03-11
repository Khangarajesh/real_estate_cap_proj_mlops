import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import joblib 
from joblib import load
import pathlib


def predict(model,features):
    prediction = model.predict(features)
    return prediction

def main():
        
    st.set_page_config(page_title="Price Prediction")
    st.title("Price Prediction")

    #df = pd.read_pickle("df.pkl") 
    file_path = pathlib.Path(__file__).parent.as_posix() + '/f_df.joblib'
    df = joblib.load(file_path)
    
    #df.drop(['Unnamed: 0.1'], axis = 1, inplace = True)
    #st.dataframe(df)
    
    #model loading 
    logged_model = 'runs:/f030090d1f1140608b5bcc1133ae9251/model'
    pipeline = mlflow.sklearn.load_model(logged_model)[0]
    #pipeline_path = pathlib.Path(__file__).parent.as_posix() + '/model/model.joblib'
    #pipeline = load(pipeline_path)

    #property type
    prop_type = st.selectbox('Property Type',['flat', 'house'])
    prop_type_1 = 0 if prop_type == 'flat' else 1
    #built_up_area
    built_up_area = st.number_input('Built Up Area')
    #sector
    sector = st.selectbox('Sector', sorted(np.unique(df['sector'])))
    #Bedrroms
    bedroom = float(st.selectbox('Number of Bedrooms', sorted(np.unique(df['bedRoom']))))
    #Bathrooms
    bathroom = float(st.selectbox('Number of Bathrooms', sorted(np.unique(df['bathroom']))))
    #Balcony
    balcony = st.selectbox('Number of Balconies', sorted(np.unique(df['balcony'])))
    #property_age
    propert_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))
    #servent_room
    servent_room = st.selectbox('Servent Room', np.unique(df['servant room']))
    #store_room
    #store_room = st.selectbox('Store Room', np.unique(df['store room']))
    #furnishing_type
    furnishing_type = st.selectbox('Furnishing Type', sorted(np.unique(df['furnishing_type'])))
    #luxary_category
    luxary_category = st.selectbox('Luxary Category', sorted(np.unique(df['luxury_category'])))
    #floor_category
    floor_category = st.selectbox('Floor Category', sorted(np.unique(df['floor_category'])))
    
    
    if st.button('Predict'):
        data =[[prop_type_1, built_up_area, sector, bedroom, bathroom, balcony, propert_age, servent_room, furnishing_type, luxary_category, floor_category]]
        columns = ['property_type', 'built_up_area', 'sector', 'bedRoom', 'bathroom', 'balcony',
                'agePossession', 'servant room',
                'furnishing_type', 'luxury_category', 'floor_category']
       
        prediction = pipeline.predict(pd.Series({'property_type':prop_type_1, 'built_up_area':built_up_area, 'sector':sector, 'bedRoom':bedroom, 
                         'bathroom':bathroom, 'balcony':balcony,
                'agePossession':propert_age, 'servant room':servent_room,
                'furnishing_type':furnishing_type, 'luxury_category':luxary_category, 'floor_category':floor_category}).to_frame().T)

        #st.dataframe(one_df)
        base_price = np.expm1(prediction[0])
        st.write(f"The price of the {prop_type} wil be between {round(base_price - 0.22, 2)} Cr to {round(base_price + 0.22, 2)} Cr")
        
        
if __name__ == '__main__':
    main()