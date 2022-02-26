from io import BytesIO
from pathlib import Path
import pkgutil
import pandas as pd
from .parsers import get_sep
import os
def list_datasets(path=''):
    """
    Lists the available datasets to be used with the load_data function.
    arguments:
        path:  path relative to phosphodisco data folder. pass 'demo' to list demo datasets.
               pass nothing to get list contents of data folder.
    returns:
        list of files and directories within the specified path.
    """
    data_folder = Path(__file__).parent / 'data' / path
    return os.listdir(data_folder)

def load_data(
        dataset:str, 
        parser=pd.read_csv, 
        **parser_kwargs
        ):
    f"""
    Loads available demo datasets and returns it as a DataFrame.
    arguments:
        dataset: one of: {list_datasets()}
    returns:
        data: DataFrame with loaded dataset.
    """
    data_path = Path(__file__).parent / 'data' / dataset
    if not data_path.exists():
        raise ValueError(f'{dataset} is not one of the available datasets in the provided path. {list_datasets(str(data_path.parent))}')
    data = parser(str(data_path), **parser_kwargs)
    return data

