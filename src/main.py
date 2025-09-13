# Python Library imports
import logging
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

# Local Python Module Imports
from src.utils.config import load_config, read_from_yaml, update_yaml_keys # Handles loading and reading the configuration (.yaml) file
from src.data_ingestion.data_ingest import extract_from_db, convert_to_df # Ingest the dataset and saves as .csv file
from src.data_preparation.data_preparation import DataPreparation # Creates an DataPreparation Object to handle data preparation
from src.model_training.model_training import ModelTraining # Creates an ModelTraining Object to handle model training

def main():
  # Create an config object to get configuration settings from config.yaml
  config = load_config()

  # Check if the .db/.csv file exist in data/raw/
  if not Path(config['file_path']).exists():
    logging.info("raw_data.csv not found! Performing Data Ingestion...")
    # Ingest Data and saves as .csv 
    rows, column_names = extract_from_db(config['db_url'])
    convert_to_df(rows, column_names)

  # Get .csv from 'data/raw' folder
  df = pd.read_csv(config["file_path"])

  # Initialize a DataPreparation Object and process the 'df' DataFrame
  # Remember your preprocesser object is created internally here 
  data_prep = DataPreparation(config)

  # Call .clean_data function to clean data
  cleaned_df = data_prep.clean_data(df)

  # Call .correct_datatype function to correct feature datatype 
  corrected_df = data_prep.correct_datatype(cleaned_df)

  # Initialize a ModelTraining Object with the created preprocessor (from data_prep)
  model_training = ModelTraining(config, data_prep.preprocessor)
  

if __name__ == "__main__":
    main()
