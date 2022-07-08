import phosphodisco as phdc
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
# Invocation
# snakemake --snakefile phdc.smk --cores 3 -n --configfile config-custom.yml --cluster-config cluster.json

def get_group_indices(annotations, columns=None,  threshold=10):
    """
    Takes in an annotations DataFrame, columns and a minimum threshold, and returns a dictionary
    of unique groups (based on annotations) and sample IDs. I.e. if the columns are cancer type and
    sample type, a group could be ('LSCC', 'tumors').
    Args:
        annotations: DataFrame where rows are sample IDs, and columns are sample annotations
        columns:     columns from annotations to consider
        threshold:   minimum number of samples a group needs to have in order to be returned
    returns:
        groups_dict: dict where groups are keys, and sample IDs are values
    """
    #if (not columns) and (annotations is None):
    #    return None
#    print(f"##########################\nannotations\n{annotations}\ncolumns\n{columns}")
    if columns is None:
        columns = annotations.columns.to_list()
    per_group_counts = annotations.loc[:, columns].value_counts()
    groups_dict = dict()
    for group in per_group_counts.index:
        if per_group_counts.loc[group] < threshold:
            continue
        group_inds = annotations.loc[:,columns].reset_index()

        # print(f"##########################\nannotations\n{annotations}\ncolumns\n{columns}\ngroup_inds\n{group_inds}\n##########################")
        group_inds = group_inds.set_index(columns).loc[group].iloc[:,0].values
        groups_dict.update({group:group_inds})
    return groups_dict

def group_prefixes_from_inds(groups_dict):
    """
    Takes in a groups_dict created by get_group_indices, and returns a list of sanitized group prefixes for file names.
    """
    if not groups_dict:
        return ['all']
    return [
        '_'.join(
            "".join(x for x in annot if x.isalnum()) 
            for annot in group
        ) 
        for group in groups_dict.keys()
    ]    
def mock_annots():
    """
    In case the user does not provide annots, this function will instead generate a
    mock annotation DataFrame with one column.
    """
    prot_cols = pd.read_csv(config['input_protein'], index_col=[0,1], nrows=3).columns
    phospho_cols = pd.read_csv(config['input_phospho'], index_col=[0], nrows=3).columns
    common_cols = prot_cols.intersection(phospho_cols)
    mock_annots = pd.DataFrame(index=common_cols).assign(all='all')
    return mock_annots

## Make two lists:
##    - list of group-prefixes for normalizing
##    - list of group-prefixes for filtering

## Making list of group-prefixes for normalizing phospho
annots = pd.read_csv(config.get('sample_annotations_csv'), index_col=[0]) if config.get('sample_annotations_csv') else mock_annots()
annots_given = config.get('sample_annotations_csv') is not None
norm_cols = config.get('sample_annot_cols_for_normalization')
norm_group_inds_dict = get_group_indices(annotations=annots, columns=norm_cols)
norm_group_prefixes = group_prefixes_from_inds(norm_group_inds_dict) 
#print('norm_group_prefixes', norm_group_prefixes)

## Making list of group-prefixes for filtering normalized phospho
filter_cols = config.get('sample_annot_cols_for_filtering')
all_samples_dict = {('all',):annots.index} if (annots is not None) else None
if (filter_cols is not None) and (annots is not None):
    filter_annots = annots.loc[:,filter_cols]
    filter_col_combinations = [filter_cols[:i] for i in range(1,len(filter_cols)+1)]
    filter_group_inds_dict = dict(
        chain.from_iterable(
        d.items() for d in [
                get_group_indices(annotations=filter_annots, columns=cols) for cols in filter_col_combinations
                ]
        )
    )
    filter_group_inds_dict.update(all_samples_dict)
else:
    #print('lmao here')
    filter_group_inds_dict = all_samples_dict
    #print('big oof')
filter_group_prefixes = group_prefixes_from_inds(filter_group_inds_dict) 

#print('filter_group_prefixes', filter_group_prefixes)

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
#         split_phospho_done = f'split_phospho/{phospho_prefix}/.split_phospho_and_prot_{phospho_prefix}.done'
#        combined_phospho = f'combined/combined_{phospho_prefix}_phospho.csv',
#        combined_protein = f'combined/combined_{phospho_prefix}_protein.csv'

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
####################################################
# New rules
####################################################

rule split_phospho_and_prot:
    input:
        clean_protein=f"clean_phospho/{protein_prefix}.clean.csv",
        clean_phospho=f"clean_phospho/{phospho_prefix}.clean.csv"
    output:
        touch(f'split_phospho/{phospho_prefix}/.split_phospho_and_prot_{phospho_prefix}.done'),
        clean_protein=expand(
        f'split_phospho/{phospho_prefix}' + '/{group_prefix}_protein.csv',
        group_prefix=norm_group_prefixes
        ),
        clean_phospho=expand(
        f'split_phospho/{phospho_prefix}' + '/{group_prefix}_phospho.csv',
        group_prefix=norm_group_prefixes
        )
    run:
# read in phospho
# read in sample annotation table
# 
# split data set into the datasets to be normalized. The multi-level filtering stuff will come after normalizing
        phospho = pd.read_csv(input.clean_phospho, index_col=[0,1])
        protein = pd.read_csv(input.clean_protein, index_col=[0])
        if annots_given:
            annots = pd.read_csv(config['sample_annotations_csv'], index_col=[0])
        else:
            annots = mock_annots()
        norm_cols = config.get('sample_annot_cols_for_normalization')
        group_inds = get_group_indices(annotations=annots, columns=norm_cols)
#creating the path for the directory first, bc pd.DataFrame.to_csv won't do it
        splits_folder=f'split_phospho/{phospho_prefix}' # maybe add sample_annots as a prefix as well?
        Path(splits_folder).mkdir(parents=True, exist_ok=True)
        for group, indices in group_inds.items():
            temp_phospho = phospho.loc[:, indices]
            temp_protein = protein.loc[:, indices]
# the purpose of the group_prefix is to uniquely name the split files. 
# the group_prefix is generated from the annotations, i.e. LSCC_Normal if the group was 'LSCC', 'Normal')
# the group_prefix is also sanitized by only allowing alpha-numeric characters to be used, so we actually create valid file names with the prefix
            group_prefix = '_'.join("".join(x for x in annot if x.isalnum()) for annot in group)
            temp_phospho.to_csv(f'split_phospho/{phospho_prefix}/{group_prefix}_phospho.csv')
            temp_protein.to_csv(f'split_phospho/{phospho_prefix}/{group_prefix}_protein.csv')


rule normalize_phospho_split:
    input:
# note that we are mixing f-strings with snakemake wild card string notation, hence the + sign
        clean_protein=f'split_phospho/{phospho_prefix}' + '/{group_prefix}_protein.csv',
        clean_phospho=f'split_phospho/{phospho_prefix}' + '/{group_prefix}_phospho.csv'
    output:
        normalized_phospho = f"split_phospho/{phospho_prefix}/{phospho_prefix}" + "_{group_prefix}.normed.csv"
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

# In order to combine all the outputs from normalize_phospho_split, we have to
# make a list called group_prefixes of all the possible group_prefixes outside of a snakemake rule.  
# Then we will use expand to expand them.
# The problem with my initial thinking was that snakemake works the opposite direction of nextflow, i.e. from the outputs back to the inputs (snakemake), rather than from the inputs to the outputs (nextflow)
# As such, snakemake cannot possibly work the way I wanted it to,
# i.e. it cannot do a step where it just waits for all the previous jobs to finish
# because that necessitates for it to know when to stop.
# Hence, I think we just have to chart out from the get-go which group_prefixes
# are possible.
## Actually there is experimental support for dynamic files, which is explained here https://snakemake.readthedocs.io/en/v6.0.0/snakefiles/rules.html#dynamic-files 
## Actually we may need to use the checkpoint feature: https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#data-dependent-conditional-execution
## That should enable us to do it the nextflow way
rule combine_normed_phospho:
    input:
        normalized_phospho= expand(
        f"split_phospho/{phospho_prefix}/{phospho_prefix}" + "_{group_prefix}.normed.csv", 
        group_prefix=norm_group_prefixes
        ),
        clean_protein=expand(
        f'split_phospho/{phospho_prefix}' + '/{group_prefix}_protein.csv',
        group_prefix=norm_group_prefixes
        )
    output:
        combined_phospho = f'combined/combined_{phospho_prefix}_phospho.csv',
        combined_protein = f'combined/combined_{phospho_prefix}_protein.csv'
    run:
        combined_phospho = pd.concat(
        [
        pd.read_csv(phospho, index_col=[0,1]) for phospho in input.normalized_phospho
        ],
        axis=1)
        
        combined_protein = pd.concat(
        [
        pd.read_csv(protein, index_col=[0]) for protein in input.clean_protein
        ],
        axis=1)
        
        Path('combined/').mkdir(parents=True, exist_ok=True)
        combined_phospho.to_csv(output.combined_phospho)
        combined_protein.to_csv(output.combined_protein)


#rule normalize_phospho:
#    input:
#        clean_protein=f"clean_phospho/{protein_prefix}.clean.csv",
#        clean_phospho=f"clean_phospho/{phospho_prefix}.clean.csv"
#    output:
#        normalized_phospho=f"normalized_phospho/{phospho_prefix}.normed.csv"
#    run:
#        proteomics_obj = phdc.ProteomicsData(
#        phospho = pd.read_csv(input.clean_phospho, index_col=[0,1]),
#        protein = pd.read_csv(input.clean_protein, index_col=[0]),
#        min_common_values=config['min_common_vals']
#        )
#        proteomics_obj.normalize_phospho_by_protein(
#        cv=3, 
#        ridge_cv_alphas = [5 ** i for i in range(-5, 5)]
#        )
#        proteomics_obj.normed_phospho.to_csv(output.normalized_phospho)

rule correlate_phospho:
    input:
        combined_phospho = f'combined/combined_{phospho_prefix}_phospho.csv',
        #combined_protein = f'combined/combined_{phospho_prefix}_protein.csv'
        #normalized_phospho=f"normalized_phospho/{phospho_prefix}.normed.csv"
    output:
        filt_phospho=f"clean_phospho/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.csv",
        corr_phospho=f"corr_phospho/{phospho_prefix}_{1-na_frac_threshold:.2f}frac-no-na.top{std_quantile_threshold*100:.0f}frac_stdev.corr.csv"
    run:
        normed_phospho = pd.read_csv(input.combined_phospho, index_col=[0,1])
        #print(normed_phospho.shape)
        normed_phospho = normed_phospho.loc[
            normed_phospho.isnull().sum(axis=1)<
            normed_phospho.shape[1]*na_frac_threshold
        ]
        #print(normed_phospho.shape)
        normed_phospho = normed_phospho.loc[
            normed_phospho.std(axis=1)>
            np.quantile(normed_phospho.std(axis=1), std_quantile_threshold)
        ]
        normed_phospho.to_csv(output.filt_phospho)
        #print(normed_phospho.shape)
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
