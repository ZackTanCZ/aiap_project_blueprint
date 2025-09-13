import logging
import os, yaml, json, joblib, pickle
from typing import Dict, Any, List, Union

# Handles configuration(.yaml) files
def read_from_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_config(path: str = "src/utils/config.yaml") -> Dict[str, Any]:
    return read_from_yaml(path)

def write_to_yaml(path: str, content: Dict[str, Any]):
    with open(path, 'w') as f:
        yaml.dump(content, f, default_flow_style=False) 
        
def update_yaml_keys(path: str, updates: Dict[str, Any]) -> None:
    """
    Update one or more key-value pairs in YAML file.
    
    Usage:
        # Single key-value pair
        update_yaml_keys('src/utils/config.yaml', {'cv': 10})
        
        # Multiple key-value pairs
        update_yaml_keys('src/utils/config.yaml', {'cv': 10, 'scoring': 'r2', 'val_size': 0.3})
    """
    config = read_from_yaml(path)
    config.update(updates)
    write_to_yaml(path, config)
    return read_from_yaml(path)

# Handles Joblib Objects - use for Scikit Objects(Models, Transformers, Pipeline)
def save_as_joblib(object, filepath):
    joblib.dump(object, filepath)

def load_joblib(object_name: str):
    return joblib.load(object_name)

# Handles pickle objects - Everything else that is not Scikit Learn
def save_as_pkl(object, filepath):
    pickle.dump(object, filepath)

def load_pkl(object_name: str):
    return pickle.load(object_name)

# Handles json files
def save_as_json(object, pathfile:str):
    with open(f'{pathfile}.json', 'w') as f:
        json.dump(object, f, indent=4)

# Handles Scikit ColumnTransformer Object
def save_column_transformer(column_transformer, column_transformer_name: str):
    """
    Saves the sklearn ColumnTransformer object to the 'artifact' directory 

    Args:
        column_transformer: ColumnTransformer object
        column_transformer_name: filename of the ColumnTransformer object 
    """
    col_transformer_dir = 'artifact'
    dir_path = os.path.join(col_transformer_dir, f'{column_transformer_name}.joblib')
    save_as_joblib(column_transformer, dir_path)
    
def load_column_transformer(column_transformer_name: str):
    """
    Loads the sklearn ColumnTransformer object from the 'artifact' directory

    Args: 
        column_transformer_name: filename of .joblib file to load the ColumnTransformer Object
    """
    col_transformer_dir = 'artifact'
    dir_path = os.path.join(col_transformer_dir, f'{column_transformer_name}.joblib')
    column_transformer = load_joblib(dir_path)
    return column_transformer

# Handles Scikit Learn Models
def save_model(model, model_name: str):
    """
    Saves the sklearn Estimator model to the 'model' directory

    Args:
        model: Estimator object
        model_name: filename of the estimator object
    """
    model_dir = 'model'
    dir_path = os.path.join(model_dir, f'{model_name}.joblib')
    save_as_joblib(model, dir_path)

def load_model(model_name:str):
    """
    Loads the sklearn Estimator object from the 'model' directory

    Args: 
        model_name: filename of .joblib file to load the estimator object
    """
    model_dir = 'model'
    dir_path = os.path.join(model_dir, f'{model_name}.joblib')
    model = load_joblib(dir_path)
    return model

                          
