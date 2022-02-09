import phosphodisco as phdc
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
# Invocation
# snakemake --snakefile phdc.smk --cores 3 -n --configfile config-custom.yml --cluster-config cluster.json


# rule all:
#     input:
#         corr_phospho="corr_phospho/lscc-v3.2-prot-normalized-phospho.75frac-no-na.top50frac_stdev.corr.csv"

## Config vars:
protein_prefix = Path(config['input_protein']).stem
phospho_prefix = Path(config['input_phospho']).stem
na_frac_threshold = config.get('na_frac_threshold', 0.25)
std_quantile_threshold = config.get('std_quantile_threshold', 0.5)


# We are including the hypercluster Snakefile with the modification that we removed the rule all from that file because we can only have one of those
include: "hypercluster.smk"

rule all:
    input:
        chosen_clusters=f"clustering/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.corr/clustering/chosen_clusters.txt"
        
rule read_phospho_and_protein:
    """
    Takes in a matching protein file and phospho file, in either .gct or csv format, and removes duplicates
    """
    input:
#         phospho=Path(config['input_protein_phospho_data_folder']) / Path(config['input_phospho']),
#         protein=Path(config['input_protein_phospho_data_folder']) / Path(config['input_protein'])
        phospho=Path(config['input_phospho']),
        protein=Path(config['input_protein'])

    output:
        clean_protein=f"clean_phospho/{protein_prefix}.clean.csv",
        clean_phospho=f"clean_phospho/{phospho_prefix}.clean.csv"
    params:
        regex=config.get('phospho_regex', None)
    run:
        #reading in a .gct, .csv or .tsv protein file
        if Path(input.protein).suffix == '.gct':
            protein = phdc.parsers.read_gct(
                input.protein,
                index_cols=['geneSymbol'],
                regex=params.regex
            )[0]
        elif Path(input.protein).suffix == '.csv' or Path(input.protein).suffix == '.tsv':
            protein = phdc.parsers.read_protein(input.protein)
        else:
            raise ValueError(f'Provided protein file {input.protein} is neither a .gct nor a .csv or .tsv file')
        #reading in a .gct, .csv or .tsv phospho file
        if Path(input.phospho).suffix == '.gct':
            phospho = phdc.parsers.read_gct(
                input.phospho, 
                regex=params.regex
            )[0]
        elif Path(input.phospho).suffix == '.csv' or Path(input.phospho).suffix == '.tsv':
            phospho = phdc.parsers.read_phospho(input.phospho)
        else:
            raise ValueError(f'Provided phospho file {input.phospho} is neither a .gct nor a .csv or .tsv file')
        # Next let's filter duplicates by always keeping the dup with the lowest number of NaNs        
        # The following looks more complicated than it is - we
        # 1. check if there are duplicate proteins/ phosphosites
        # 2. if there are, we subselect duplicate rows, and for each 
        #    set of dups pick the row with the least amount of missing values
        # 3. we combine the deduplicated rows with the rest of the rows that had no dups to begin with
        # We do this splitting and combining for efficiency reasons - groupby.apply 
        # operations are expensive when there are lots of groups
        if 0 < protein.index.duplicated(keep=False).sum():
            protein = pd.concat([
                protein.loc[
                    protein.index.duplicated(keep=False)
                ].groupby(
                    protein.index.name
                ).apply(
                    phdc.parsers.filter_dups
                ).reset_index(level=0, drop=True),
                protein.loc[
                    ~protein.index.duplicated(keep=False)
                ]], 
                axis=0
            )
        if 0 < phospho.index.duplicated(keep=False).sum():
            phospho = pd.concat([
                phospho.loc[
                    phospho.index.duplicated(keep=False)
                ].groupby(
                    phospho.index.names
                ).apply(
                    phdc.parsers.filter_dups
                ).reset_index(level=[0,1], drop=True),
                phospho.loc[
                    ~phospho.index.duplicated(keep=False)
                ]], 
                axis=0
            )
        #Let's write to file
        protein.to_csv(output.clean_protein)
        phospho.to_csv(output.clean_phospho)

rule normalize_phospho:
    input:
        clean_protein=f"clean_phospho/{protein_prefix}.clean.csv",
        clean_phospho=f"clean_phospho/{phospho_prefix}.clean.csv"
    output:
        normalized_phospho=f"normalized_phospho/{phospho_prefix}.normed.csv"
    run:
        proteomics_obj = phdc.ProteomicsData(
        phospho = pd.read_csv(input.clean_phospho, index_col=[0,1]),
        protein = pd.read_csv(input.clean_protein, index_col=[0]),
        min_common_values=config['min_common_vals']
        )
        proteomics_obj.normalize_phospho_by_protein(
        cv=3, 
        ridge_cv_alphas = [5 ** i for i in range(-5, 5)]
        )
        proteomics_obj.normed_phospho.to_csv(output.normalized_phospho)

rule correlate_phospho:
    input:
        normalized_phospho=f"normalized_phospho/{phospho_prefix}.normed.csv"
#         normalized_phospho="normalized_phospho/lscc-v3.2-phosphoproteome-ratio-norm-NArm_clean_prot_normed.csv" 
    output:
        filt_phospho=f"clean_phospho/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.csv",
        corr_phospho=f"corr_phospho/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.corr.csv"
#          filt_phospho="clean_phospho/lscc-v3.2-phosphoproteome-ratio-norm-NArm_0.75frac-no-na.top50frac_stdev.csv",
#         corr_phospho="corr_phospho/lscc-v3.2-phosphoproteome-ratio-norm-NArm_0.75frac-no-na.top50frac_stdev.corr.csv"
#          filt_phospho="clean_phospho/lscc-v3.2-prot-normalized-phospho.75frac-no-na.top50frac_stdev.csv",
#         corr_phospho="corr_phospho/lscc-v3.2-prot-normalized-phospho.75frac-no-na.top50frac_stdev.corr.csv"
    run:
        normed_phospho = pd.read_csv(input.normalized_phospho, index_col=[0,1])
        normed_phospho = normed_phospho.loc[
            normed_phospho.isnull().sum(axis=1)<
            normed_phospho.shape[1]*na_frac_threshold
        ]
        normed_phospho = normed_phospho.loc[
            normed_phospho.std(axis=1)>
            np.quantile(normed_phospho.std(axis=1), std_quantile_threshold)
        ]
        normed_phospho.to_csv(output.filt_phospho)
        
        phospho_corr = normed_phospho.transpose().corr()
        phospho_corr = phospho_corr.fillna(0)
        phospho_corr.columns = ['%s-%s' % (a, b) for a, b in phospho_corr.columns]
        phospho_corr.to_csv(output.corr_phospho)

rule choose_cluster:
    input:
        rand_scores=f"clustering/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.corr/clustering/adjusted_rand_score_label_comparison.txt",
        module_label_files=expand(
        "%s/{input_file}/%s/{labs}_labels.txt" % (output_folder, intermediates_folder),
        input_file=input_files,
        labs=config["param_sets_labels"],
    )
#         normalized_phospho="normalized_phospho/lscc-v3.2-phosphoproteome-ratio-norm-NArm_clean_prot_normed.csv" 
    output:
        chosen_clusters=f"clustering/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.corr/clustering/chosen_clusters.txt"
    run:
        df = pd.read_csv(input.rand_scores, index_col=[0])
        rand_best_cluster =  df.median().sort_values().tail(1).index[0]
        best_cluster_file = next(
            filter(
                lambda fname: rand_best_cluster in fname, 
                input.module_label_files
            )
        )
        shutil.copyfile(best_cluster_file, output.chosen_clusters)

# rule correlate_annotations:
#     input:
#         phospho=,
#         protein=,
#         normed_phospho=,
#         modules=,
#     output:
#     run:        
#         data = phdc.ProteomicsData(
#             phospho=phospho,
#             protein=protein,
#             normed_phospho=normed,
#             modules=modules
#         )

#         data.calculate_module_scores()
#         data.impute_missing_values()