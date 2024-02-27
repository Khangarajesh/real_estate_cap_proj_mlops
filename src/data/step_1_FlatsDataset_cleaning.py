import pandas as pd
import numpy as np
import pathlib
import re

#create function for reading data
def read_file(input_path):
  #function for reading file, data cleaning and storing in data/processed folder
  curr_dir = pathlib.Path(__file__)
  home_dir = curr_dir.parent.parent.parent
  file_path = home_dir.as_posix()+input_path
  file = pd.read_excel(file_path)
  return file

#create a function to keep unit of all values in price column in Crore
def price_in_crore_KHANGAR(x):
  if type(x) == float:
    return x
  elif x[1] == 'Lac':
    return round(float(x[0])/100, 2)
  else:
    return round(float(x[0]), 2)

def save_file(file,output_path):
  #send the clean file
  curr_dir = pathlib.Path(__file__)
  home_dir = curr_dir.parent.parent.parent
  file_path = home_dir.as_posix()+output_path
  pathlib.Path(file_path).mkdir(parents = True, exist_ok = True)
  file.to_csv(file_path+'/flats_cleaned.csv', index = True)

def data_cleaning():
    #read file using function
    flats = read_file("/data/raw/flats.xlsx")
    
    #function for data cleaning
    #removing unnecessary columns
    flats.drop(['link', 'property_id'], axis = 1, inplace = True)
    #renaming column
    flats.rename(columns = {'area' : 'price_per_sqft'}, inplace = True)
    #dealing with society column
    #remove rows containing society value as special char
    pattern = r'\d+\.\d+\ ★'
    flats['society'] = flats['society'].apply(lambda x : re.sub(pattern, '',str(x))).str.strip().str.lower()
    #dealing with price column
    #remove rows where price value is not numeric
    flats = flats[flats['price'] != 'Price on Request']
    flats.drop(286, inplace = True) #row containing value 'price'
    flats['price'] = flats['price'].str.split(' ').apply(lambda x: price_in_crore_KHANGAR(x))
    #dealing with price_per_sqft
    flats['price_per_sqft'] = flats['price_per_sqft'].str.split('/').str[0].str.replace('₹', '').str.replace(',', '').astype(float)
    #dealing with bedroom
    flats['bedRoom'] = flats['bedRoom'].str.replace('Bedrooms', '').str.replace('Bedroom', '').astype(float)
    #dealig with bathroom
    flats['bathroom'] = flats['bathroom'].str.replace('Bathrooms', '').str.replace('Bathroom', '').astype('float')
    #dealing with balcony
    flats['balcony'] = flats['balcony'].str.replace('Balconies', '').str.replace('Balcony', '').str.replace('No', '0')
    #dealing with additional room
    flats['additionalRoom'].fillna('Not Available', inplace = True)
    flats['additionalRoom'] = flats['additionalRoom'].str.strip().str.lower()
    #dealing with floor num
    flats['floorNum'] = flats['floorNum'].str.split(' ').str.get(0).str.replace('Ground', '0').str.replace('Lower', '0').str.replace('Basement', '-1').str.extract(r'(-?\d+)')
    #dealing with dir and facing
    flats['facing'].fillna('NA', inplace = True)
    #Calculating area with the help of price_per_sqft and price column
    flats.insert(4, column = 'area', value = round(flats['price']*10000000/flats['price_per_sqft']))
    flats.insert(5, column = 'property_type', value = 'flat')


    #send the clean file
    save_file(flats, '/data/processed')


if __name__ == '__main__':
    
     data_cleaning()  #this will run the function and create the output of the ste
