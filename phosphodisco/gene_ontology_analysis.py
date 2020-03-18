import pandas as pd
from gseapy import enrichr
import scipy.stats
from .parsers import read_gmt
from .utils import multiple_tests_na


def enrichr_per_module(
        modules,
        background_gene_list,
        gene_sets: str = 'GO_Biological_Process_2018',
        **enrichr_kws
):
    results = {}
    for module, genes in modules.groupby(modules).groups:
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
    if len(list(module_seq_dict.values())[0]) < 15:
        raise ValueError('Module sequences must be at least 15 AAs long')
    if len(list(module_seq_dict.values())[0]) > 15:
        module_seq_dict = {
            k: [seq[(len(seq)/2-0.5)-7: (len(seq)/2-0.5)+8]
                for seq in v] for k, v in module_seq_dict.items()
        }
    background_seqs = set(background_seqs)
    M = len(background_seqs)

    ptm_set_gmt = read_gmt(ptm_set_gmt)
    ptm_set_gmt = {
        k: {item for item in v.items() if item[0] in background_seqs}
        for k, v in ptm_set_gmt.items()
    }
    ptm_set_gmt = {k: v for k, v in ptm_set_gmt.items() if len(v) >= 2}

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
            n = len(sites)
            overlap = seqs.intersection(set(sites.keys()))
            x = len(overlap)
            overlap = [sites[seq] for seq in overlap]
            pval = scipy.stats.hypergeom.sf(x, M, n, N, loc=1)
            line = pd.Series({
                'Genes': ','.join(overlap),
                'Overlap': x,
                'P-value': pval},
                name=term
            )
            module_results = module_results.append(line)
        module_results['Adjusted P-value'] = multiple_tests_na(
            module_results['P-value'], method='fdr_bh'
        )
        results[module] = module_results.set_index('Term')

    return results
