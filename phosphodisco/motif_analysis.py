from collections import Counter
from pandas import DataFrame
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
from .constants import var_site_delimiter, protein_id_col, variable_site_col, seq_col


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
    IDs_df[seq_col] = df_to_aa_seqs(IDs_df, fasta_dict, n_flanking)
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
