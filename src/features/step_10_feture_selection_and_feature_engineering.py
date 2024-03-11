import pandas as pd
import numpy as np
import pathlib
import re
from sklearn.preprocessing import OrdinalEncoder
import joblib 

def read_file(input_path='/data/processed/gurgaon_properties_missing_values_imputed.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + input_path
    return pd.read_csv(file_path)

def send_file(file, output_path = '/data/processed/gurgaon_properties_post_feature_selection.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + output_path
    file.to_csv(file_path)
    
def categorize_luxary (score):
    if 0 <= score <50:
      return 'Low'
    elif 50 <= score <150:
      return 'Medium'
    elif 150 <= score <175:
      return 'High'
    else:
      return None
  
def categorize_floor (score):
    if 0 <= score <=2:
      return 'Low Floor'
    elif 3 <= score <=10:
      return 'Medium Floor'
    elif 11 <= score <=60:
      return 'High Floor'
    else:
      return None

def feature_selection_and_engg():
    df = read_file()
    #we will drop society column and price_per_sqft temporarily because we are not going to ask society as an input to our user
    #same for price_per_sqft
    train_df = df.drop(['Unnamed: 0', 'price_per_sqft', 'society'], axis = 1)
    
    #luxary score
    #we can not ask the user for luxary score because he do not know what does luxary score = 58 means. so we will convert this column into category
    train_df['luxury_category'] = train_df['luxary_score'].apply(categorize_luxary)
    
    #floor number
    train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)
    train_df.drop(['floorNum', 'luxary_score'], inplace = True, axis = 1)
    #Applying Ordinal encoder to convert categories to numerical

    
    data_label_encoded = train_df.copy()
    #extracting the name of categorical columns
    categorical_cols = train_df.select_dtypes(include = ['object']).columns

    for col in categorical_cols:
      oe = OrdinalEncoder()
      data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])

    #seperate output column
    x_label = data_label_encoded.drop('price', axis = 1)
    y_label = data_label_encoded['price']
        

    #Now we will do featur selection using different different techniques. Each technique will give some score to each feature. 
    #at the end we will combine all the score from different feature selection techniques and will select the most important feature
    #after doing feature selection we will drop the features whoes score is very low.
    
    x_label.drop(['others', 'pooja room', 'study room'], inplace = True, axis = 1)
    export_df = x_label
    export_df['price'] = y_label
  
    joblib.dump(export_df,'f_df.joblib')
    send_file(export_df)
    
    
if __name__ == '__main__':
    feature_selection_and_engg()