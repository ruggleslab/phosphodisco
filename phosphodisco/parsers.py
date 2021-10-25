import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Optional, Iterable, Tuple, List, Union


def get_sep(file_path: str) -> str:
    """Figure out the sep based on file name. Only helps with tsv and csv.

    Args:
        file_path: Path of file.

    Returns: sep

    """
    if file_path[-4:] == '.tsv':
        return '\t'
    elif file_path[-4:] == '.csv':
        return ','
    raise ValueError('Input file is not a .csv or .tsv')


def read_protein(file_path: str) -> DataFrame:
    """Reads in protein abundance values. Proteins as rows, samples as columns.
    First column must be protein identifier.

    Args:
        file_path: Path to protein csv or tsv.

    Returns: DataFrame with proteins as rows, samples as columns.

    """
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=0).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    ).astype(float)

def read_gct(path: str, 
             index_cols: Optional[List[str]]=['geneSymbol', 'variableSites'], 
             regex: Optional[Union[str, None]]=None, 
             sample_cols: Optional[Union[List, None]]=None, 
             annotation_rows: Optional[Union[List, None]]=None
            ) -> Tuple[DataFrame, DataFrame]:
    """
    Reads in a gct file and formats the dataframe so it's ready for phospho disco
    path: path to file.gct
    index_cols:  columns which to use as an index. For phospho/acetyl, etc this should 
                 be two columns e.g. ['geneSymbol', 'variableSites'] whereas for protein it's one column, e.g. ['geneSymbol']
    regex:       [optional] regular expression to quickly subselect sample columns e.g.  
                 to get only sample columns that end in a digit: r'.*\d$'
    sample_cols: [optional] to select sample columns using exact names; not used if you provided a regex
    
    returns: pd.DataFrame with sample columns and index_cols
    """
    with open(path, 'r') as handle:
        next(handle)
        #the 2nd row of a gct file gives us the dimensions
        nrows, ncols, nrowmeta, ncolsmeta = [int(i) for i in next(handle).split()] 
    df = pd.read_csv(
        path, sep='\t', skiprows=2, low_memory=False
    ).replace(
         ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    )
    # the metadatatable is transposed in the gct file, hence we are indexing everything but 
    # the first ncolsmeta rows, and everything but the first nrowsmeta columns
    sample_df = df.set_index(index_cols).iloc[ncolsmeta:, nrowmeta-len(index_cols)+1:].copy()
    annots_df = df.set_index(df.columns[0]).iloc[:ncolsmeta, nrowmeta:].copy()
    if regex is not None:
        sample_df = sample_df.loc[:,sample_df.columns.str.match(regex)]
        annots_df = annots_df.loc[:,annots_df.columns.str.match(regex)]
    elif sample_cols is not None:
        try:
            sample_df = sample_df.loc[:, sample_cols]
            annots_df = annots_df.loc[:, sample_cols]
        except KeyError:
            non_matched_cols = set(sample_cols).difference(sample_df.columns)
            raise IndexError(
                f"The following columns were not found in the sample columns of the provided gct file \npath:\n{path}\
                \nmismatched columns:\n{non_matched_cols}"
            )
    if annotation_rows is not None:
        try:
            annots_df = annots_df.loc[annotation_rows, :]
        except KeyError:
            non_matched_cols = set(annotation_rows).difference(annots_df.index)
            raise IndexError(
                f"The following columns were not found in the annotation rows columns of the provided gct file \npath:\
                \n{path}\nmismatched columns:\n{non_matched_cols}"
            )

    return sample_df, annots_df

def filter_dups(group:pd.DataFrame):
    """
    Meant to be used in apply after a groupby call.
    For a set of rows (group) find the one with the lowest number of NaNs or tied for the lowest number.
    group: pd.DataFrame
    """
    nan_counts = group.apply(lambda r: r.isnull().sum(), axis=1)
    min_nan_counts = nan_counts.min()
    return group.loc[nan_counts == min_nan_counts].head(1)

# def deduplicate_rows(
#     df: pd.DataFrame,
#     samples,
#     sequence_col = 'sequence',
#     maximize_cols = ['bestScore', 'Best_scoreVML', 'bestDeltaForwardReverseScore']
#     ) -> pd.DataFrame:

#     groupby = df.index.names
#     df = df.reset_index()

#     df[maximize_cols] = df[maximize_cols].astype(float)
#     df['numNAs'] = df[samples].isnull().sum(axis=1)
#     if sequence_col:
#         df['len'] = df[sequence_col].str.len()
#         return df.groupby(groupby).apply(lambda row: row.nsmallest(1, columns=['numNAs','len'], keep='all').nlargest(1, columns=maximize_cols, keep='first')).reset_index(level=-1, drop=True)
#     return df.groupby(groupby).apply(lambda row: row.nsmallest(1, columns=['numNAs'], keep='all').nlargest(1, columns=maximize_cols, keep='first')).reset_index(level=-1, drop=True)


def read_annotation(file_path: str) -> DataFrame:
    """Reads in sample annotation file. Sample as rows, annotations as columns.

    Args:
        file_path: Path to protein csv or tsv. First column must be sample identifier.

    Returns: DataFrame with samples as rows, annotations as columns.

    """
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=0).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    )


def read_phospho(file_path: str) -> Optional[DataFrame]:
    """Reads in protein abundance values. Proteins as rows, samples as columns. First two columns
    must be protein, variable stie identifiers, respectively. Can use this for raw or normalized
    phospho data tables.

    Args:
        file_path: Path to protein csv or tsv.

    Returns: DataFrame with phosphosites as rows, samples as columns.

    """
    sep = get_sep(file_path)
    return pd.read_csv(file_path, sep=sep, index_col=[0, 1]).replace(
        ['na', 'NA', 'NAN', 'nan', 'NaN', 'Na'], np.nan
    ).astype(float)


def read_list(file_path: str):
    """Reads in a \n separated file of things into a list.

    Args:
        file_path: Path to file.

    Returns: List

    """
    with open(file_path, 'r') as fh:
        return [s.strip() for s in fh.readlines()]
    

def column_normalize(df: DataFrame, method: str) -> DataFrame:
    """Normalizes samples for coverage.

    Args:
        df: DataFrame to column normalize.
        method: Which method to use: 'median_of_ratios', 'median', 'upper_quartile' currently
        accepted.

    Returns: Normalized DataFrame.

    """
    if method == "median_of_ratios":
        return df.divide(df.divide(df.mean(axis=1), axis=0).median())

    if method == "median":
        return df.divide(df.median())

    if method == "upper_quartile":
        return df.divide(np.nanquantile(df, 0.75))

    # if method == "quantile":
    #     pass
        #TODO add two comp

    raise ValueError(
        'Passed method not valid. Must be one of: median_of_ratios, median, upper_quartile, '
        'twocomp_median.'
    )


def read_fasta(fasta_file) -> dict:
    """Parse fasta into a dictionary.

    Args:
        fasta_file: path to fasta file.

    Returns: dictionary of genes: seq.

    """
    with open(fasta_file, 'r') as fh:
        aa_seqs = {
            seq.split()[0]: seq.split(']')[-1].replace('\s', '').replace('\n', '')
            for seq in fh.read().strip().split('>') if seq != ''
        }
    return aa_seqs


def read_gmt(gmt_file: str) -> dict:
    """Parser for gmt files, specifically from ptm-ssGSEA

    Args:
        gmt_file: Name of gmt file.

    Returns:
        Dictionary of ptm sets. Keys are labels for each set. Values are dictionaries with
        structure: {aa sequence: site name}

    """
    result = {}
    with open(gmt_file, 'r') as fh:
        for line in fh.readlines():
            line = line.strip().split('\t')
            name = line[0]
            site_labels = line[1]
            seqs = line[2:]
            seq_labels = {seqs[i].split('-')[0]: label for i, label in enumerate(site_labels.split('|')[1:])}
            result.update({name: seq_labels})

    return result
