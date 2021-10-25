import pandas as pd
from pandas import Series, DataFrame
from gseapy import enrichr
import scipy.stats
from .parsers import read_gmt
from .utils import multiple_tests_na


def enrichr_per_module(
        modules: Series,
        background_gene_list,
        gene_sets: str = 'GO_Biological_Process_2018',
        **enrichr_kws
):
    """Runs gseapy.enrichr on genes in each module.

    Args:
        modules: Module Series with sites as the index.
        background_gene_list: List of all unique genes that could have ended up in modules.
        gene_sets: Which gene sets to use. See for options http://amp.pharm.mssm.edu/Enrichr/#stats
        **enrichr_kws: Additional keyword args for gseapy.enrichr()

    Returns: Dictionary of DataFrames with module names as keys and enrichr results as values.

    """
    results = {}
    for module, genes in modules.groupby(modules).groups.items():
        genes = list(set([i[0] for i in genes]))
        res = enrichr(
            gene_list=genes,
            gene_sets=gene_sets,
            background=background_gene_list,
            **enrichr_kws
        )
        res = res.results[[
            'Gene_set',
            'Genes',
            'Overlap',
            'Odds Ratio',
            'P-value',
            'Adjusted P-value',
            'Term'
        ]].set_index('Term')
        results[module] = res
    return results


def ptm_per_module(
        module_seq_dict,
        background_seqs,
        ptm_set_gmt: str = 'data/ptm.sig.db.all.flanking.human.v1.9.0.gmt'
):
    """Calculates enrichment for each PTM-ssGSEA set per module via hypergeometric test.

    Args:
        module_seq_dict: Dictionary of amino acid sequences, module names as keys, lists of
        peptide seqs as values.
        background_seqs: List of all peptide seqs that could have ended up in modules, i.e. sites
        that went into the clustering algorithm.
        ptm_set_gmt: Path to gmt file with PTM-ssGSEA sets to compare against.

    Returns: Dictionary with keys are module names, values are DataFrames with set enrichment
    results per module.

    """
    ptm_set_gmt = read_gmt(ptm_set_gmt)
    ptm_set_gmt = {
        k: {item for item in v.items() if item[0] in background_seqs}
        for k, v in ptm_set_gmt.items()
    }
    ptm_set_gmt = {k: v for k, v in ptm_set_gmt.items() if len(v) >= 2}

    if len(list(module_seq_dict.values())[0]) < 15:
        raise ValueError('Module sequences must be at least 15 AAs long')
    if len(list(module_seq_dict.values())[0]) > 15:
        module_seq_dict = {
            k: [seq[int((len(seq)/2-0.5)-7): int((len(seq)/2-0.5)+8)]
                for seq in v] for k, v in module_seq_dict.items()
        }
    background_seqs = set(background_seqs)
    M = len(background_seqs)

    results = {}
    for module, seqs in module_seq_dict.items():
        seqs = set(seqs)
        N = len(seqs)
        module_results = pd.DataFrame(
            columns=[
                'Site_set',
                'Sites',
                'Overlap',
                'P-value',
                'Term'
            ]
        )
        for term, sites in ptm_set_gmt.items():
            sites = dict(sites)
            n = len(sites)
            overlap = seqs.intersection(set(sites.keys()))
            x = len(overlap)
            overlap = [sites[seq] for seq in overlap]
            pval = scipy.stats.hypergeom.sf(x, M, n, N, loc=1)
            line = pd.Series({
                'Site_set':','.join(sites.values()),
                'Sites': ','.join(overlap),
                'Overlap': x,
                'P-value': pval,
                'Term':term
                },
                name=term
            )
            module_results = module_results.append(line)
        module_results['Adjusted P-value'] = multiple_tests_na(
            module_results['P-value'], method='fdr_bh'
        )
        results[module] = module_results.set_index('Term')

    return results
