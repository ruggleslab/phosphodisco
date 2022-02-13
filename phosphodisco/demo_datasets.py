from io import BytesIO
from pathlib import Path
import pkgutil
import pandas as pd
from .parsers import get_sep
import os
def list_datasets():
    """
    Lists the available datasets to be used with the load_data function.
    """
    demo_folder = Path(__file__).parent / 'data' / 'demo'
    return os.listdir(demo_folder)

def load_data(dataset:str, **read_csv_kwargs):
    f"""
    Loads available demo datasets and returns it as a DataFrame.
    arguments:
        dataset: one of: {list_datasets()}
    returns:
        data: DataFrame with loaded dataset.
    """
    try:
        data_handle = BytesIO(pkgutil.get_data('phosphodisco', f'data/demo/{dataset}'))
    except:
        raise ValueError(f'{dataset} is not one of the available demo datasets. {list_datasets()}')
    data = pd.read_csv(data_handle, sep=get_sep(dataset), **read_csv_kwargs)
    return data

