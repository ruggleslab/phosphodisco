from collections import Counter
from itertools import product
from pathlib import Path
from typing import Union, Optional
from pandas import DataFrame
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
from .constants import var_site_delimiter, protein_id_col, variable_site_col, seq_col, gene_symbol_col, variable_site_aa_col

def find_aa_seqs(
        aa_seq: str,
        var_sites: str,
        n_flanking: int = 7
):
    """Grabs the flanking AA sequence around a given location in a protein sequence string.

    Args:
        aa_seq: Protein sequence string.
        var_sites: Integer location of the site of interest (1-indexed, not 0-indexed).
        n_flanking: The number of flanking AAs to grab around the site of interest.

    Returns: AA sequence centered around var_site.

    """
    sites = [max(int(v.strip())-1, 0) for v in var_sites.split(var_site_delimiter)]
    seqs = []
    for var_site in sites:
        n = int(var_site)
        if len(aa_seq) < n:
            return '_'*(1+(n_flanking*2))

        left_ = '_'*max((n_flanking - n), 0)
        right_ = '_'*max(((n+n_flanking+1) - len(aa_seq)), 0)
        aas = aa_seq[max((n-n_flanking), 0):min(len(aa_seq), (n+n_flanking+1))]
        seqs.append(left_ + aas + right_)
    return var_site_delimiter.join(seqs)


def df_to_aa_seqs(
        IDs_df: DataFrame,
        fasta_dict: dict,
        n_flanking: int = 7
):
    """Takes a specifically structured DataFrame specifying variable site locations and defines
    all amino acids sequences around those sites in a new column.

    Args:
        IDs_df: DataFrame of variable site locations for each phosphosite. The
        Structure of this df is very strict. It must have a column called 'protein_id' which
        will match the name of proteins in the fasta file. It must have a column called
        variable_site_col which contained a ',' separated list of variable sites integers.
        NB: these must be integers, so variable sites like 'S225s T227t' must be converted to
        '225,227'. Multiple sites will be returned as ',' separated strings
        fasta_dict: Dictionary structured like {"protein id": "protein sequence"}
        n_flanking: The number of flanking amino acids to pull out around each variable site.

    Returns: Series of amino acids seqs for each row.

    """
    aas = IDs_df.apply(
        lambda row: find_aa_seqs(
            fasta_dict.get(row[protein_id_col], ''),
            row[variable_site_col],
            n_flanking
        ), axis=1
    )
    return aas


def make_module_sequence_dict(
        IDs_df: DataFrame,
        fasta_dict: dict,
        module_col: str,
        n_flanking: int = 7
):
    """Creates a dictionary with all of the amino acid seqs per module.

    Args:
        IDs_df: DataFrame of variable site locations for each phosphosite. The
        Structure of this df is very strict. It must have a column called 'protein_id' which
        will match the name of proteins in the fasta file. It must have a column called
        variable_site_col which contained a ',' separated list of variable sites integers.
        NB: these must be integers, so variable sites like 'S225s T227t' must be converted to
        '225,227'. Multiple sites will be returned as ',' separated strings
        fasta_dict: Dictionary structured like {"protein id": "protein sequence"}
        module_col: Name of column with module labels per site.
        n_flanking: The number of flanking amino acids to pull out around each variable site.

    Returns: Dictionary like {"module A": ['seq1', 'seq2']}

    """
    IDs_df = IDs_df.copy()
    IDs_df[seq_col] = df_to_aa_seqs(IDs_df, fasta_dict, n_flanking).copy()
    d = IDs_df.groupby(module_col)[seq_col].agg(lambda col: var_site_delimiter.join(col)).to_dict()
    return {k: v.split(var_site_delimiter) for k, v in d.items()}


def calculate_motif_enrichment(
        module_aas: dict,
        background_aas: list,
) -> dict:
    """Calculates statistical enrichment of each amino acid at each site surrounding
    modifications per module.

    Args:
        module_aas: Dictionary like {"module A": ['seq1', 'seq2']}, output of make_module_sequence_dict
        background_aas: List of all AA sequences that were possible to get in the modules.

    Returns: Dictionary of dataframes. Keys are module labels. Values are dataframes with -log10
    pvalues of enrichment of every amino acid in every position.

    """

    module_freqs = {
        module: pd.DataFrame([Counter(tup) for tup in list(zip(*aas))])
        for module, aas in module_aas.items()
    }
    background_freqs = pd.DataFrame([Counter(tup) for tup in list(zip(*background_aas))]).fillna(0)
    n_seqs_background = len(background_aas)

    module_ps = {}
    for module, freqs in module_freqs.items():
        n_seqs_in_module = freqs.sum(axis=1)[0]
        freqs = freqs.reindex(background_freqs.columns, axis=1).fillna(0)

        fe = freqs.combine(
            background_freqs, lambda mod_col, back_col: pd.Series([
                fisher_exact(
                    [
                        [mod_col[i], n_seqs_in_module-mod_col[i]],
                        [back_col[i], n_seqs_background-back_col[i]]
                    ]
                ) for i in range(len(mod_col))])
        )
        odds = fe.apply(lambda row: pd.Series([i[0] for i in row]))
        odds = (odds > 1) + -1*(odds <= 1)
        ps = fe.apply(lambda row: pd.Series([i[1] for i in row]))
        ps = odds.combine(-np.log10(ps), lambda col1, col2: col1.multiply(col2))

        module_ps[module] = ps

    return module_ps

def aa_overlap(
        seq1, seq2
        ):
    """
    Calculates the amount of positions that are the same between two iterables.
    If the iterables do not have the same length, only compares up to the length of the shorter iterable.
    Args:
        seq1: iterable
        seq2: iterable

    Returns:
        overlap: int 
    """
    overlap = sum(
            i[0]==i[1] for i in zip(seq1, seq2)
            )
    return overlap

def aa_overlap_from_df(
        seq_df: DataFrame,
        module_col: str
        ):
    f"""
    Calculates inverse Hamming distance for all pairwise combinations of phospho sites.
    
    Args:
        seq_df:     DataFrame where each row is a phosphosite. 
                    Created by classes.ProteomicsData.collect_aa_sequences --> module_seq_df attribute
                    Contains phosphosite, module and sequence information in each row.
                    Column names: '{gene_symbol_col}', '{variable_site_aa_col}', module_col, '{seq_col}'
                    '{gene_symbol_col}' contains the gene symbol in each row.
                    '{variable_site_aa_col}' contains variable site indeces in comma-separated format, 
                    e.g. "S203s,T208t' - these will be used for labeling in plots
                    '{seq_col}' column with comma-separated peptide sequences
        module_col: column in seq_df that contains module labels.
    Returns: Dictionary of DataFrames. Keys are module labels. Values are dataframes with inverse
             aa_overlap for pairwise comparisons between phosphosites.
    """
    ### Uses ProteomicsData.module_seq_df for seq_df
    ### format of module_seq_df (col order not deterministic):
    ### site_name_col, variable_site_col, variable_site_aa_col,
    ### protein_id_col, module_col, seq_col, gene_symbol_col
    seq_df = seq_df.copy()
    # Need to split variable sites with multiple potential phosphorylations into separate sites, i.e.
    # 'S204s,T208t' turns into two sites. We need to do this for both variable_site_aa_col and seq_col
    seq_df[variable_site_aa_col] = seq_df[variable_site_aa_col].str.split(',')
    seq_df[seq_col] = seq_df[seq_col].str.split(',')
    seq_df['seq_var_site_col'] = seq_df.apply(
            lambda row: list(zip(row[variable_site_aa_col], row[seq_col])),
            axis=1
            )
    seq_df = seq_df.explode('seq_var_site_col')
    seq_df[variable_site_aa_col] = seq_df['seq_var_site_col'].str.get(0)
    seq_df[seq_col] = seq_df['seq_var_site_col'].str.get(1)
    seq_df = seq_df.drop_duplicates(subset=[variable_site_aa_col, gene_symbol_col, module_col])
    # potentially add filtering against seqs with too many '_' 
    # which can happen if the given protein_id was not correct. 
    module_aa_sim_dfs_dict = {}
    relevant_cols = [variable_site_aa_col, gene_symbol_col, seq_col]
    dup_col_rename = {col:(col + '_1') for col in relevant_cols}
    for module, chunk in seq_df.groupby(module_col):
        aas = chunk[relevant_cols]
        #constructing a df that has all pairwise combinations of rows in chunk
        aas = pd.concat(
                [
                    pd.concat(row) for row in product(
                        (i[1] for i in aas.iterrows()),
                        (i[1] for i in aas.rename(columns=dup_col_rename).iterrows())
                        )
                    ],
                axis=1
                ).T.reset_index()
        # calculating overlap for each pair, and consolidating gene symbol and variable site into a single col
        aas['aa_overlap'] = aas.apply(
                lambda row: aa_overlap(row[seq_col], row[dup_col_rename[seq_col]]), 
                axis=1
                )
        aas['aa_name1'] = aas[gene_symbol_col] + '-' + aas[variable_site_aa_col]
        aas['aa_name2'] = aas[dup_col_rename[gene_symbol_col]] + '-' + aas[dup_col_rename[variable_site_aa_col]]
        aas = aas.pivot(index='aa_name1', columns='aa_name2', values='aa_overlap')
        module_aa_sim_dfs_dict.update({module:aas})
    return module_aa_sim_dfs_dict 

def plot_aa_overlap(
        module_overlap_df_dict: dict,
        save_path: Optional[str]=None
        ):
    """
    Plots aa_overlap heatmap for each module.
    Args:
        module_overlap_df_dict: dictionary, output of aa_overlap_from_df
                                 keys are modules, values are DataFrames of aa_overlap scores for each
                                 pair of phosphosites within the module
        save_fig:               path to folder where pdfs of plots should be saved.
                                saves no plots if None.
    Returns:
        None
    """
    for module, module_df in module_overlap_df_dict.items():  
        fig_len = 0.5*module_df.shape[0]
        fig_width = 0.4*module_df.shape[1]

        fig = plt.figure(figsize = (fig_len, fig_width))
        sns.heatmap(module_df, xticklabels = module_df.columns, yticklabels = module_df.index)
        plt.title(f'Cluster {module}')
        if save_path is not None:
            plt.savefig(Path(save_path) / Path(f'heatmap.aa_overlap.module{module}.pdf'))
    plt.show()
    plt.close()
