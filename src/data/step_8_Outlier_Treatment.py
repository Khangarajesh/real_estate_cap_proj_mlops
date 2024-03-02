import pandas as pd
import numpy as np
import pathlib
import re

def read_file(input_path='/data/processed/gurgaon_properties_cleaned_v2.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + input_path
    return pd.read_csv(file_path)

def send_file(file, output_path = '/data/processed/gurgaon_properties_outliers_trated.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + output_path
    file.to_csv(file_path)
    

def extract_built_up (x):
   pattern_1 = r"Built Up area: ([-+]?\d*\.\d+|\d+) sq\.ft\. \((\d+\.\d+) sq\.m\.\)"
   pattern_2 = r"Built Up area: ([-+]?\d*\.\d+|\d+) \((\d+\.\d+) sq\.m\.\)"
   #pattern_3 = r"Built Up area: ([-+]?\d*\.\d+|\d+) sq\.yards\ \((\d+\.\d+) sq\.m\.\)"
   pattern_4 = r"Plot area \d+(\.\d+)?\(\d+\.\d+ sq\.m\.\)"
   pattern_5 = r'Plot area (\b\d+\b)'
   match_1 = re.search(pattern_1, x)
   match_2 = re.search(pattern_2, x)
   #match_3 = re.search(pattern_3, x)
   match_4 = re.search(pattern_4, x)
   match_5 = re.search(pattern_5, x)
   if match_1:
    return float(match_1.group(1))
   elif match_2:
    return float(match_2.group(2))*10.7
   #elif match_3:
    #return float(match_3.group(1))*9.0
   elif match_4:
    patt = r"\((\d+\.\d+)\s*sq\.m\.\)"
    tx = re.search(pattern_4, x).group(0)
    match_4 = re.search(patt, tx)
    return float(match_4.group(1))*10.7
   elif match_5:
    return  float(match_5.group(1))
   else:
    return None
 
def carpet (x):
   x = str(x)
   pattern_1 = r"Carpet area: ([-+]?\d*\.\d+|\d+) sq\.ft\. \((\d+\.\d+) sq\.m\.\)"
   pattern_2 = r"Carpet area: ([-+]?\d*\.\d+|\d+) \((\d+\.\d+) sq\.m\.\)"
   pattern_3 = r'Carpet area: ([-+]?\d*\.\d+|\d+)\s*sq\.yards'
   pattern_4 = r"Carpet area: ([-+]?\d*\.\d+|\d+) sq\.ft\."
   match_1 = re.search(pattern_1, x)
   match_2 = re.search(pattern_2, x)
   match_3 = re.search(pattern_3, x)
   match_4 = re.search(pattern_4, x)
   if match_1:
    return float(match_1.group(1))
   elif match_2:
     return float(match_2.group(2))*10.7
   elif match_3:
    return float(match_3.group(1))*9.0
   elif match_4:
    return  float(match_4.group(1))
   else:
    return None
    
def remove_outliers():
    df = read_file()

    Q1 = df['price_per_sqft'].describe()['25%']
    Q3 = df['price_per_sqft'].describe()['75%']
    iqr = Q3-Q1
    lower = Q1 - 1.5*iqr
    upper = Q3 + 1.5*iqr
    
    outliers = df[(df['price_per_sqft'] > upper)].sort_values(by = 'price_per_sqft', ascending = False)
      
    #we can see from the data the area column is not appropriate (i.e not in sq.ft) and it seems in sq.yards format due to this the price per_sq.ft is going out of range   
    outliers['area'] = outliers['area'].apply(lambda x : x*9 if x < 1000 else x)
    outliers['built_up_area'] = outliers['built_up_area'].apply(lambda x : x*9 if x < 1000 else x)
    #recalculating price_per_sqft
    outliers['price_per_sqft'] = (outliers['price']*10000000) / outliers['area']
    
    #updating original table with re calculated price_per_sqft
    df.update(outliers)
    
    #Area
    Q1 = df['area'].describe()['25%']
    Q3 = df['area'].describe()['75%']
    iqr = Q3-Q1
    lower = Q1 - 1.5*iqr
    upper = Q3 + 1.5*iqr

    df = df[df['area'] < 100000]
    #sice it seems like there is an issue with the entered data.The properties that have price above 100000 have very lesss price also impossibly less price per sqft so we will remove such rows
    
    
    #3405, 3691,3782, 3008 -- 2 bed rooms in very large area
    #440 -- correction
    
    df.drop([3405, 3691,3782, 3008,3661,1831,1410,484,168,3590, 1221], inplace = True)
    
    #Bedroom
    #we will limit the number of bedroos to 12
    df = df[df['bedRoom']<12]
    
    df['built_up_area'] = df['areaWithType'].apply(extract_built_up)
    #Since built_up_area less than 181 is not possible we will remove thise rows
    df.drop(df[df['built_up_area'] <= 181.31272].index, inplace = True)
    
    #there are some cases where number of rooms in given area are very high so we will take a ratio of area to the room and set the threshold of 2perecntile below this threshold we will remove the rows
    x = df[df['price_per_sqft'] <= 20000]
    (x['area']/x['bedRoom']).quantile(0.02)
    
    df = df[(df['area']/df['bedRoom'])>198]
    
    #while doing outlier treatment I found that most of the values in built_up_area was not appropriate due to the function which is populating built_up_area was not working well so i did correction and automaticall all the outliers goes away
    
    df.loc[1481, 'areaWithType'] = 4550*9
    
    #Carpet Area
    #there were areas which was not justified becaue no of bedrooms given in those is practically not possible so we will remove those rows
    #Ther is isssue in our function it is populating carpetr_area column wrongly
    df['carpet_area'] = df['areaWithType'].apply(carpet)
    #droping the rows having area to bedroom ratio less than 212.4
    df.drop(df[(df['carpet_area']/df['bedRoom']) < 212.4].index, inplace = True)
    
    #Price_per_sqft
    x = df[df['price_per_sqft'] <= 20000]
    #adding a column area/bedroom
    df['area_bedroom_ratio'] = round(df['area']/df['bedRoom'])
    outliers = df[(df['area_bedroom_ratio'] < 246.5) & (df['bedRoom'] > 3)]
    outliers['bedRoom'] = round(outliers['bedRoom']/outliers['floorNum'])
    df.update(outliers)
    df.drop(df[(df['area_bedroom_ratio'] < 246.5) & (df['bedRoom'] > 3)].index, inplace = True)
    
    send_file(df)
    
    
if __name__ == '__main__':
    remove_outliers()
    
