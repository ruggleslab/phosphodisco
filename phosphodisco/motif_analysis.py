import re


def find_aa_seq(acc, varsite):
    seqs = []
    for site in varsite.strip().split():
        n = int(site[1:-1])-1
        aa_seq = aa_seqs.get(acc, '')
        left_ = '_'*max((n_flanking - n), 0)
        right_ = '_'*max(((n+n_flanking+1) - len(aa_seq)), 0)
        aas = aa_seq[max((n-n_flanking), 0):min(len(aa_seq), (n+n_flanking+1))]
        seqs.append(left_+aas+right_)
    return ','.join(seqs)


def get_aa_sequence(module_IDs_df, fasta_dict, n_flanking = 7):
    pass


def collect_all_background_aas(site_vector, fasta_dict):
    pass


def calculate_motif_enrichment(aas, background_aas):
    pass

