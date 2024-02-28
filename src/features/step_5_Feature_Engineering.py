import pandas as pd
import numpy as np
import pathlib
import re
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
import ast

def read_file(input_path):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + input_path
    return pd.read_csv(file_path)


def save_file(file, output_path):
    home_dir = pathlib.Path(__file__).parent.parent.parent
    file_path = home_dir.as_posix() + output_path
    file.to_csv(file_path,index = False)    
    
 
# Extract the super built up area from areaWithType column
def extract_super_built_up (x):
    pattern = r"Super Built up area ([-+]?\d*\.\d+|\d+)([-+]?\d*\.\d+|\d+)sq\.m\."
    match = re.search(pattern, x)
    if match:
       return float(match.group(1))
    else:
       return None
 
# Extract the built up area from areaWithType column
def extract_built_up (x):
    pattern_1 = r"Built Up area: ([-+]?\d*\.\d+|\d+) sq\.ft\. \((\d+\.\d+) sq\.m\.\)"
    pattern_2 = r"Built Up area: ([-+]?\d*\.\d+|\d+) \((\d+\.\d+) sq\.m\.\)"
    pattern_3 = r"Built Up area: ([-+]?\d*\.\d+|\d+) sq\.yards\ \((\d+\.\d+) sq\.m\.\)"
    match_1 = re.search(pattern_1, x)
    match_2 = re.search(pattern_2, x)
    match_3 = re.search(pattern_3, x)
    if match_1:
     return float(match_1.group(1))
    elif match_2:
     return float(match_2.group(1))
    elif match_3:
     return float(match_3.group(1))*9.0
    else:
     return None  
 
# Extract the Carpet area from areaWithType column
def carpet (x):
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
     return float(match_2.group(1))
    elif match_3:
     return float(match_3.group(1))*9.0
    elif match_4:
     return  float(match_4.group(1))
    else:
     return None 
 
#convert from sq.m. to sq.ft.
def convert_to_sqft(text, area_value):
    if np.isnan(area_value):
        return np.NaN
    match = re.search(r'{}\((\d+\.?\d*) sq.m.\)'.format(int(area_value)), text)
    if match:
        sq_m_value = float(match.group(1))
        return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
    return area_value

#extract plot area in case of house and store it as a built up area
def extract_plot(text):
    pattern = r'Plot area ([-+]?\d*\.\d+|\d+)(([-+]?\d*\.\d+|\d+) sq.m.)'
    match = re.search(pattern, text)
    match_1 = re.search(r'\b\d+\b', text)
    if match:
      return float(match.group(1))
    elif match_1:
      return  float(match_1.group())
    else:
      return np.nan

def convert_scale(row):
   if np.isnan(row['area']) or np.isnan(row['built_up_area']):
     return row['built_up_area']
   elif round(row['area']/row['built_up_area']) == 9.0:
     return row['built_up_area']*9.0
   elif round(row['area']/row['built_up_area']) == 11.0:
     return row['built_up_area']*10.7
   else:
     return row['built_up_area'] 
 
def date_time(row):
   if pd.isna(row['agePossession']):
    return np.nan
   elif row['agePossession'] not in ["1 to 5 Year Old", "5 to 10 Year Old", "0 to 1 Year Old", "undefined", "10+ Year Old", "Under Construction", "Within 6 months",
                 "Within 3 months", "By 2023", "By 2024", "By 2025", "By 2027"]:
      parsed_date = datetime.datetime.strptime(row['agePossession'], "%Y-%m-%d %H:%M:%S")
      return str(parsed_date.month) + " " +  str(parsed_date.year)
   else:
      return row['agePossession']

def categorize_age_possession(val):
    if pd.isna(val):
      return "undefined"
    if val == "Within 3 months" or val == 'Within 6 months' or val == "0 to 1 Year Old":
      return "new Property"
    if val == "1 to 5 Year Old":
      return "relatively new"
    if val == "5 to 10 Year Old":
      return "moderately old"
    if val == "10+ Year Old":
      return "old property"
    if val == "Under Construction" or val.split(" ")[0] == ("By"):
      return "under construction"
    try :
      int(val.split(" ")[-1])
      if int(val.split(" ")[-1]) > datetime.date.today().year:
        return "under construction"
      elif int(val.split(" ")[-1]) == datetime.date.today().year and int(val.split(" ")[0])<= datetime.date.today().month:
        return "new Property"
      elif datetime.date.today().year - int(val.split(" ")[-1]) >=1 and datetime.date.today().year - int(val.split(" ")[-1]) <5:
        return "relatively new"
      elif datetime.date.today().year - int(val.split(" ")[-1]) >=5 and datetime.date.today().year - int(val.split(" ")[-1]) <10:
        return "moderately old"
      else :
        return "old property"
    except:
        return "undefined"
  
  
# function to extract the count of furnishing from furnishing detail
def extract_count(details, furnish):
    if isinstance(details, str):
      if f"No {furnish}" in details:
        return 0
      pattern = re.compile(f"(\d+) {furnish}")
      match = pattern.search(details)
      if match:
        return int(match.group(1))
      elif furnish in details:
        return 1
    return 0
  
def feature_eng():
    df = read_file("/data/processed/gurgaon_properties_cleaned_v1.csv")
    
    #dealing with area and area with type column
    
    #Remove trailing spaces
    df['areaWithType'] = df['areaWithType'].str.strip()
    
    #create column super_built_up_area
    df.insert(7,column = 'super_built_up_area',value = df['areaWithType'].apply(lambda x: extract_super_built_up(str(x))))
    df['super_built_up_area'] = df.apply(lambda x : convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis = 1)
    
    #create column built_up_area
    df.insert(8,column = 'built_up_area',value = df['areaWithType'].apply(lambda x : extract_built_up(str(x))))
    df['built_up_area'] = df.apply(lambda x : convert_to_sqft(x['areaWithType'], x['built_up_area']), axis = 1)
    
    #create column carpet_area
    df.insert(9,column = 'carpet_area',value = df['areaWithType'].apply(lambda x : carpet(str(x))))
    df['carpet_area'] = df.apply(lambda x : convert_to_sqft(x['areaWithType'], x['carpet_area']), axis = 1)
    
    all_nan = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]

    all_nan_index = all_nan.index
    all_nan['built_up_area'] = all_nan.apply(convert_scale, axis = 1)
    df.loc[all_nan_index, 'built_up_area'] = all_nan['built_up_area']
    
    df = df[~(np.isnan(df['area']) & np.isnan(df['super_built_up_area']) & np.isnan(df['carpet_area']) & np.isnan(df['floorNum']))]
    
    #dealing with additional room column
    df['additionalRoom'] = df['additionalRoom'].str.strip().str.lower()
    
    #create list of type of additional room column
    columns = ['servant room', 'pooja room', 'study room', 'others', 'store room']
    
    for col in columns:
      df[col] = df['additionalRoom'].str.contains(col).astype(int)
      
    #dealing with agepossesion column
    
    current_date = datetime.date.today()
    current_year = current_date.year
    current_month = current_date.month
    
    df = df[~df['agePossession'].isnull()]
    
    df['agePossession'] = df.apply(date_time, axis = 1)
    df['agePossession'] = df['agePossession'].apply(lambda x :categorize_age_possession(x))
    
    #dealing with furnishdetails
    #store all the rows in furnishDetails in one list
    furnish = []
    for i in df['furnishDetails'].dropna():
      furnish.extend(i.replace('[', '').replace(']', '').replace("'", '').split(", "))
      
    #extract unique furnishDetails
    unique_furnishings = list(set(furnish))
    
    #remove "No" and numbers
    columns_to_include = list(set([re.sub(r'\d+|No', '', fur).strip() for fur in unique_furnishings]))
    
    columns_to_include = columns_to_include[1:]
    
    #create column for each furnishing and store its count
    for i in columns_to_include:
      df[i] = df['furnishDetails'].apply(lambda x : extract_count(x, i))
      
    furnish_columns = df[[
    'AC',
    'Fridge',
    'Modular Kitchen',
    'TV',
    'Curtains',
    'Dining Table',
    'Stove',
    'Water Purifier',
    'Microwave',
    'Fan',
    'Washing Machine',
    'Chimney',
    'Sofa',
    'Exhaust Fan',
    'Bed',
    'Light',
    'Geyser',
    'Wardrobe']]
    
    scale = StandardScaler()
    scaled_data = scale.fit_transform(furnish_columns)
    
    #lets consider 2 clusters
    kmeans = KMeans(n_clusters = 3, random_state = 42)
    kmeans.fit(scaled_data)
    cluster_assignment = kmeans.predict(scaled_data)
    
    
        
    #2 = furnished
    #0 = semi furnished
    #1 = unfurnished
    
    df=df.iloc[:, :-19]
    df['furnishing_type'] = cluster_assignment
    
    #dealing with features (ther are 638 missing values in feature column)
    #we will identify rows containing missing values and then we will use another dataset to fill this values 
    
    apartment = read_file('/data/raw/appartments (1).csv')
    
    temp_df = df[df['features'].isna()][['society', 'features']]
    
    apartment['PropertyName'] = apartment['PropertyName'].str.strip().str.lower()
    x = temp_df.merge(apartment, how = 'left', left_on = 'society', right_on = 'PropertyName')[['TopFacilities']]
    
    df.loc[temp_df.index,'features'] = x.values  #we have reduced null values by 154
    df['features'] = df['features'].str.strip()
    
    #like furnishing details we will create multiple columns using feature column and finally we will categorize rows using KMeans clustering
    
    #convert string representaion of list to actual list in features column
    df['features'] = df['features'].apply(lambda x : ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])
    
    
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(df['features'])
    
    
    features_df = pd.DataFrame(mlb.fit_transform(df['features']), columns  = mlb.classes_)
    #lets consider 2 clusters
    kmeans = KMeans(n_clusters = 2, random_state = 42)
    kmeans.fit(features_df)
    cluster_assignment = kmeans.predict(features_df)
    
    features_df['features_category'] = cluster_assignment
     
     
    # we can see it making only 2 clusters
    #0=for multiple features available
    #1=for less numbers of features available
    
    # Define the weights for each feature as provided
# Assigning weights based on perceived luxury contribution
    weights = {
    '24/7 Power Backup': 8,
    '24/7 Water Supply': 4,
    '24x7 Security': 7,
    'ATM': 4,
    'Aerobics Centre': 6,
    'Airy Rooms': 8,
    'Amphitheatre': 7,
    'Badminton Court': 7,
    'Banquet Hall': 8,
    'Bar/Chill-Out Lounge': 9,
    'Barbecue': 7,
    'Basketball Court': 7,
    'Billiards': 7,
    'Bowling Alley': 8,
    'Business Lounge': 9,
    'CCTV Camera Security': 8,
    'Cafeteria': 6,
    'Car Parking': 6,
    'Card Room': 6,
    'Centrally Air Conditioned': 9,
    'Changing Area': 6,
    "Children's Play Area": 7,
    'Cigar Lounge': 9,
    'Clinic': 5,
    'Club House': 9,
    'Concierge Service': 9,
    'Conference room': 8,
    'Creche/Day care': 7,
    'Cricket Pitch': 7,
    'Doctor on Call': 6,
    'Earthquake Resistant': 5,
    'Entrance Lobby': 7,
    'False Ceiling Lighting': 6,
    'Feng Shui / Vaastu Compliant': 5,
    'Fire Fighting Systems': 8,
    'Fitness Centre / GYM': 8,
    'Flower Garden': 7,
    'Food Court': 6,
    'Foosball': 5,
    'Football': 7,
    'Fountain': 7,
    'Gated Community': 7,
    'Golf Course': 10,
    'Grocery Shop': 6,
    'Gymnasium': 8,
    'High Ceiling Height': 8,
    'High Speed Elevators': 8,
    'Infinity Pool': 9,
    'Intercom Facility': 7,
    'Internal Street Lights': 6,
    'Internet/wi-fi connectivity': 7,
    'Jacuzzi': 9,
    'Jogging Track': 7,
    'Landscape Garden': 8,
    'Laundry': 6,
    'Lawn Tennis Court': 8,
    'Library': 8,
    'Lounge': 8,
    'Low Density Society': 7,
    'Maintenance Staff': 6,
    'Manicured Garden': 7,
    'Medical Centre': 5,
    'Milk Booth': 4,
    'Mini Theatre': 9,
    'Multipurpose Court': 7,
    'Multipurpose Hall': 7,
    'Natural Light': 8,
    'Natural Pond': 7,
    'Park': 8,
    'Party Lawn': 8,
    'Piped Gas': 7,
    'Pool Table': 7,
    'Power Back up Lift': 8,
    'Private Garden / Terrace': 9,
    'Property Staff': 7,
    'RO System': 7,
    'Rain Water Harvesting': 7,
    'Reading Lounge': 8,
    'Restaurant': 8,
    'Salon': 8,
    'Sauna': 9,
    'Security / Fire Alarm': 9,
    'Security Personnel': 9,
    'Separate entry for servant room': 8,
    'Sewage Treatment Plant': 6,
    'Shopping Centre': 7,
    'Skating Rink': 7,
    'Solar Lighting': 6,
    'Solar Water Heating': 7,
    'Spa': 9,
    'Spacious Interiors': 9,
    'Squash Court': 8,
    'Steam Room': 9,
    'Sun Deck': 8,
    'Swimming Pool': 8,
    'Temple': 5,
    'Theatre': 9,
    'Toddler Pool': 7,
    'Valet Parking': 9,
    'Video Door Security': 9,
    'Visitor Parking': 7,
    'Water Softener Plant': 7,
    'Water Storage': 7,
    'Water purifier': 7,
    'Yoga/Meditation Area': 7
    }
    
    features_df = features_df[list(weights.keys())]*weights.values()
    features_df['luxary_score'] = features_df.sum(axis = 1)
    
    
    df.reset_index(inplace = True)
    df['luxary_score'] = features_df['luxary_score']
    
    df.reset_index(inplace = True)
    df.drop(columns = ['nearbyLocations','furnishDetails','features','additionalRoom'],inplace=True)
    
    save_file(df,'/data/processed/gurgaon_properties_cleaned_v2.csv')