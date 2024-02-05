# Imports
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def load_data(path):
    """function to load a dataset

    Args:
        path (str): path to the dataset 
    
    Returns:
        DataFrame: a dataframe of the dataset
    """
    try:    
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        return None
    
