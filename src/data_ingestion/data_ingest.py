# Python Library imports
import logging, os, requests 
import sqlite3
import pandas as pd

# Local Python Module Imports
from src.utils.config import update_yaml_keys, write_to_yaml

def extract_from_db(db_url):
    """
    Open a connection to the database file to extract records and column headers
    Combine the records and column headers into a pandas DataFrame

    Args:
        db_url (str): url to the db file

    Returns:
        rows (List of List): query result from the .db file
        column_names (List): list of column headers
    """

    # Get the .db file from cloud
    response = requests.get(db_url)
    response.raise_for_status() 

    # Download the file
    db_name = db_url.split("/")[-1]
    with open(f'data/raw/{db_name}', "wb") as f:
        f.write(response.content)
    
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(f'data/raw/{db_name}')
    cursor = conn.cursor()

    # Execute a SELECT query to retrieve all data from the table
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table';")

    # Fetch all the rows returned by the query
    table_name_list = cursor.fetchall()

    # Extract the first name
    table_name = table_name_list[0][0]

    print(f"Found {table_name} Table")

    # Try to get column headers from the Database
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    if columns_info:
        column_names = [info[1] for info in columns_info]
        print("Column headers found")
    else:   
        print("No Column headers found")

    # Execute a SELECT query to retrieve all data from the table
    cursor.execute(f"SELECT * FROM {table_name}")

    # Fetch all the rows returned by the query
    rows = cursor.fetchall()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    # Delete the .db file
    os.remove(f'data/raw/{db_name}')
    
    return rows, column_names

def convert_to_df(rows, column_names):
    """
    Convert the query result and column headers into a pandas DataFrame

    Args:
        rows (List of List): query result from the .db file
        column_names (List): list of column headers

    Returns:
        df (pandas DataFrame): DataFrame consisting of records from the .db file
    """
    # Combine the column headers and records into a DataFrame
    data_df = pd.DataFrame(rows, columns = column_names)

    # Replace ' ' to '_' for consistency
    data_df.columns = [col.replace(' ', '_') for col in data_df.columns]
    
    # Reset index for good measure
    data_df.reset_index(inplace = True, drop = True)

    # Save the DataFrame as a .csv in the 'data' folder
    data_df.to_csv('data/raw/raw_dataset.csv', index = False)

