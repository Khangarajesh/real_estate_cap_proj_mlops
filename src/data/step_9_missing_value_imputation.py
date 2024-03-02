import pandas as pd
import numpy as np
import pathlib
import re

def read_file(input_path='/data/processed/gurgaon_properties_outliers_trated.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + input_path
    return pd.read_csv(file_path)

def send_file(file, output_path = '/data/processed/gurgaon_properties_missing_values_imputed.csv'):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + output_path
    file.to_csv(file_path)
    
#function for imputing missing values
def carpet (x):
    pattern_1 = r"\b\d+\.\d+\b"
    pattern_2 = r"Carpet area: ((\d+(\.\d+)?)) sq\.m\."
    pattern_3 = r"Carpet area: \d+ \((\d+(\.\d+)?) sq\.m\.\)"
    result_1 = re.search(pattern_1, x)
    result_2 = re.search(pattern_2, x)
    result_3 = re.search(pattern_3, x)
    result_2
    if result_1 :
      return float(result_1.group())*10.7
    elif result_2:
      return float(result_2.group(1))*10.7
    elif result_3:
      return float(result_3.group(1))*10.7
    else:
      return np.nan

def built_up(x):
    pattern_1 = r"Built Up area: ((\d+(\.\d+)?)) sq\.m\."
    pattern_2 = r"Built Up area: ((\d+(\.\d+)?))"
    result_1 = re.search(pattern_1, x)
    result_2 = re.search(pattern_2, x)
    if result_1:
     return float(result_1.group(1))*10.7
    elif result_2:
      return float(result_2.group(1))*10.7
    else :
      return np.nan
    
#based on sector and property type we will replace undefined value

def sec_proptype_imput (df,row):
    if row['agePossession'] == 'undefined':
      mode_value = df[(df['property_type'] == row['property_type']) & (df['sector'] == row['sector'])]['agePossession'].mode()
      if not mode_value.empty:
        return mode_value.iloc[0]
      else :
        return np.nan
    else:
      return row['agePossession']
  

def sec_imput (df,row):
    if row['agePossession'] == 'undefined':
      mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
      if not mode_value.empty:
        return mode_value.iloc[0]
      else :
        return np.nan
    else:
      return row['agePossession']

def proptype_imput (df,row):
    if row['agePossession'] == 'undefined':
      mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
      if not mode_value.empty:
        return mode_value.iloc[0]
      else :
        return np.nan
    else:
      return row['agePossession']
  
def missing_value_imputation():
    df = read_file()
    #since there is linear relation between two parameters we can calculate the average ration which this two parameters hold and then using that ratio
    #we can estimate any one missing parametr from  available parameter

    #calculating ratio between super_built_up to built_up
    super_to_built_up = (df['super_built_up_area']/df['built_up_area']).median()

    #calculating ratio between carpet to built_up
    carpet_to_built_up = (df['carpet_area']/df['built_up_area']).median()

    super_to_built_up, carpet_to_built_up
    #our goal is to impute all the missing values in built_up_area. To do this we have calculated the raio in previous step so that if built_area is missing then we can utilise super_built up or carpet_area to impute built_up
    #lets check is there any row where all the areas are missing
    all_missing = df[df['super_built_up_area'].isna() & df['built_up_area'].isna() & df['carpet_area'].isna()]
    all_missing['carpet_area'] = all_missing['areaWithType'].apply(carpet)
    all_missing['built_up_area'] = all_missing['areaWithType'].apply(built_up)
    df.update(all_missing)
    
    df = df[~(df['super_built_up_area'].isna() & df['built_up_area'].isna() & df['carpet_area'].isna())]
    #imputing built_up using both super and carpet
    sc = df[~(df['super_built_up_area'].isna()) & df['built_up_area'].isna() & ~(df['carpet_area'].isna())]

    sc['built_up_area'].fillna((sc['super_built_up_area']/super_to_built_up + sc['carpet_area']/carpet_to_built_up)/2, inplace = True)
    df.update(sc)
    
    #imputing built_up using super_built_up
    sb = df[~(df['super_built_up_area'].isna()) & (df['built_up_area'].isna())]

    sb['built_up_area'].fillna(sb['super_built_up_area']/super_to_built_up, inplace = True)
    df.update(sb)
    
    #imputing built_up using carpet area

    cb = df[(df['built_up_area'].isna()) & ~(df['carpet_area'].isna())]

    cb['built_up_area'] = round(cb['carpet_area'] / carpet_to_built_up)
    df.update(cb)
    
    #there are somme points above 2.0 cr and behind 2000 sq.ft. This points indicates the very high price for area below 2000 sq.ft which is not possible we will call this points as "anamoly"
    #Investigatting anamoly points
    anamoly_df = df[(df['built_up_area'] <2000) & (df['price'] > 5)]
    
    #replacing built_up_area with area
    anamoly_df['built_up_area'] = anamoly_df['area']

    #updating original df
    df.update(anamoly_df)
    
    #Droping out unenecessary columns
    df.drop(['area', 'super_built_up_area', 'carpet_area', 'area_bedroom_ratio', 'areaWithType'], inplace = True, axis = 1)
    
    #floor num
    #we can see there are 15 mising values out of which 3 are house so we will replace missing values in floorNum with median values of floorNum of house
    floor_median = df[df['property_type'] == 'house']['floorNum'].median()

    df['floorNum'].fillna(floor_median, inplace = True)
    
    #facing 
    # we will drop the facing column because there are 1030 missing values and there is possible way ti impute them
    df.drop('facing', inplace = True, axis = 1)
    
    #agePossession
    df.drop(['level_0', 'index'], axis = 1, inplace = True)
    
    #there are total 297 undefined values which we need to replace with appropriate values
    #df['agePossession'] = df.apply(lambda x: sec_proptype_imput(df, x['property_type'], x['sector']) , axis = 1)
    df['agePossession'] = df.apply(lambda x: sec_proptype_imput(df, x) , axis = 1)
    #still there are some rows where agePosession is undefined. This happened may be because of after applying the function we are getting mode as 'undefined'
    #so now we will replace the "undefined" only based on sector
    df['agePossession'] = df.apply(lambda x: sec_imput(df,x), axis = 1)
    
    #replace the "undefined" only based on property_type
    df['agePossession'] = df.apply(lambda x: proptype_imput(df,x), axis = 1)
    send_file(df)
    
if __name__ == '__main__':
    missing_value_imputation()