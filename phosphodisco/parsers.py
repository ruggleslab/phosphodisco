import pandas as pd
import numpy as np
from pandas import DataFrame
import logging
from typing import Optional, Iterable
from .classes import ProteomicsData


def get_sep(file_path: str) -> str:
    if file_path[-4:] == '.tsv':
        return '\t'
    elif file_path[-4:] == '.csv':
        return ','
    raise ValueError('Input file is not a .csv or .tsv')


def read_protein(file_path: str) -> DataFrame:
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=0).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    ).astype(float)


def read_annotation(file_path: str) -> DataFrame:
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=0).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    )


def read_phospho(file_path: str) -> Optional[DataFrame]:
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=[0, 1]).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    ).astype(float)


def read_list(file_path: str):
    with open(file_path, 'r') as fh:
        return [s.strip() for s in fh.readlines()]
    

def column_normalize(df: DataFrame, method: str) -> DataFrame:
    if method == "median_of_ratios":
        return df.divide(df.divide(df.mean(axis=1), axis=0).median())

    if method == "median":
        return df.divide(df.median())

    if method == "upper_quartile":
        return df.divide(np.nanquantile(df, 0.75))

    if method == "twocomp_median":
        pass
        #TODO make two comp

    raise ValueError(
        'Passed method not valid. Must be one of: median_of_ratios, median, upper_quartile, '
        'twocomp_median.'
    )


def prepare_data(
        ph_file: str,
        prot_file: str,
        normalize_method: Optional[str] = None,
        min_common_values: int = 5,
        normed_phospho: Optional[DataFrame] = None,
        modules: Optional[Iterable] = None,
        clustering_parameters_for_modules: Optional[dict] = None,
        putative_regulator_list: Optional[list] = None,
) -> ProteomicsData:

    phospho = read_phospho(ph_file)
    protein = read_protein(prot_file)
    if normalize_method:
        phospho = column_normalize(phospho, normalize_method)
        protein = column_normalize(protein, normalize_method)

    return ProteomicsData(
        phospho=phospho,
        protein=protein,
        min_common_values=min_common_values,
        normed_phospho=normed_phospho,
        modules=modules,
        clustering_parameters_for_modules=clustering_parameters_for_modules,
        possible_regulator_list=putative_regulator_list
    )


def read_fasta(fasta_file) -> dict:
    with open(fasta_file, 'r') as fh:
        aa_seqs = {
            seq.split()[0]: seq.split(']')[-1].replace('\s', '').replace('\n', '')
            for seq in fh.read().strip().split('>') if seq != ''
        }
    return aa_seqs
