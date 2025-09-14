# Python Library Imports
import logging, re
from typing import Any, Dict
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# Local Python Module Imports
from src.utils.config import save_column_transformer

class DataPreparation:
    """
    
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.
        
        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.
        
        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.
        
        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        pass

    def correct_datatype(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrects the datatype for each feature found in the input DataFrame
        (e.g. Ordinal, Nomial, Numerical, Bool)
        Args:
        ----
        df (pd.DataFrame): The input DataFrame 

        Returns:
        --------
        pd.DataFrame: The corrected DataFrame.
        
        """
        pass

  # private method
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        # Instantiate the Pipeline objects each for different transformer
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown = "ignore"))])
        ordinal_transformer = Pipeline(steps=[("ordinal", OrdinalEncoder(categories = [self.config["flat_type_categories"]],
                                                                         handle_unknown = "use_encoded_value",
                                                                         unknown_value = -1,),)])
        
        # Instantiate the ColumnTransformer Object to execute the transformer process
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder = "passthrough", n_jobs = -1,)
        return preprocessor
