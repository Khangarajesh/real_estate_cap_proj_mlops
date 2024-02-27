import pandas as pd 
import numpy as np
import re 
import pathlib


def read_data(file_path):
  curr_dir = pathlib.Path(__file__)
  home_dir = curr_dir.parent.parent.parent
  processed_f_path = home_dir.as_posix() + file_path
  return pd.read_csv(processed_f_path)

def save_data(file, output_path):
  curr_dir = pathlib.Path(__file__)
  home_dir = curr_dir.parent.parent.parent
  processed_f_path = home_dir.as_posix() + output_path
  file.to_csv(processed_f_path)

def merge():
  flats = read_data('/data/processed/flats_cleaned.csv')
  independent_house = read_data('/data/processed/independent_house_cleaned.csv')
  df = pd.concat([flats, independent_house], ignore_index = True)
  df.drop(columns = {'Unnamed: 0'}, inplace = True)
  df = df.sample(df.shape[0], ignore_index = True)
  
  save_data(df,'/data/processed/gurgaon_properties.csv')
  
  
if __name__ == '__main__':
    merge()