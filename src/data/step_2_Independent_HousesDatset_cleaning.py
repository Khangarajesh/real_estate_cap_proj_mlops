import pandas as pd 
import numpy as np
import re
import pathlib

house = pd.read_excel("data/raw/independent-house.xlsx")

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
  home_dir = pathlib.Path(__file__).parent.parent.parent
  file_path = home_dir.as_posix() + output_path
  pathlib.Path(file_path).mkdir(parents = True, exist_ok= True)
  file.to_csv(file_path+'/independent_house_cleaned.csv',index = True)


def independent_house_data_cleaning():
  independent_house = read_file("/data/raw/independent-house.xlsx")
  independent_house.drop(columns = {'link', 'property_id'}, inplace = True)
  
  #dealing with price column
  independent_house = independent_house[independent_house['price'] != 'Price on Request']
  independent_house['price'] = independent_house['price'].str.split(' ').apply(lambda x : price_in_crore_KHANGAR(x))
  
  #dealing with rate column
  independent_house = independent_house.rename(columns = {'rate' : 'price_per_sqft'})
  independent_house['price_per_sqft'] = independent_house['price_per_sqft'].str.replace('₹', '').str.split('/').str.get(0).str.replace(',', '').astype(float)

  #dealing with bedroom column
  independent_house['bedRoom'] = independent_house['bedRoom'].str.split(' ').str.get(0).astype(float)
  
  #dealing with bathroom column
  independent_house['bathroom'] = independent_house['bathroom'].str.split(' ').str.get(0).astype(float)
  
  #dealing with balcony column
  independent_house['balcony'] = independent_house['balcony'].str.replace('Balconies', '').str.replace('Balcony', '').str.replace('No', '0')
  
  independent_house['additionalRoom'].fillna('Not Available',inplace = True)
  
  #dealing with number of floor
  independent_house['noOfFloor'] = independent_house['noOfFloor'].str.split(' ').str.get(0).astype(float)
  independent_house.rename(columns = {'noOfFloor' : 'floorNum'}, inplace = True)
  
  #dealing with facing column
  independent_house['facing'].fillna('NA', inplace = True)
  
  #add area column
  independent_house['area'] = independent_house['price']*10000000/independent_house['price_per_sqft']
  independent_house.insert(5, column = 'property_type', value = 'house')
  
  pattern = r'(\d+\.\d+\ ★)'
  independent_house['society'] = independent_house['society'].apply(lambda x : re.sub(pattern, '', str(x))).str.strip().str.lower()
    
  save_file(independent_house,'/data/processed') 
  
  
if __name__ == '__main__':
    # run function 
    independent_house_data_cleaning()
  

