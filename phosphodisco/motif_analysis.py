from collections import Counter
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
from .constants import var_site_delimiter, protein_id_col, variable_site_col, seq_col


def find_aa_seqs(
        aa_seq,
        var_sites,
        n_flanking = 7
):
    sites = [int(v.strip()) for v in var_sites.split(var_site_delimiter)]
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
        IDs_df,
        fasta_dict,
        n_flanking = 7
):
    aas = IDs_df.apply(
        lambda row: find_aa_seqs(
            fasta_dict.get(row[protein_id_col], ''),
            row[variable_site_col],
            n_flanking
        ), axis=1
    )
    return aas


def make_module_sequence_dict(IDs_df, fasta_dict, module_col, n_flanking=7):
    IDs_df[seq_col] = df_to_aa_seqs(IDs_df, fasta_dict, n_flanking)
    d = IDs_df.groupby(module_col)[seq_col].agg(lambda col: var_site_delimiter.join(col)).to_dict()
    return {k: v.split(var_site_delimiter) for k, v in d.items()}


def calculate_motif_enrichment(
        module_aas,
        background_aas,
):

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
