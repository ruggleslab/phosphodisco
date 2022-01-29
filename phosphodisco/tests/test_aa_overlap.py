mport pandas as pd, numpy as np, seaborn as sns, phosphodisco as phdc

modules=pd.read_csv(
'/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/clustering/lssc_vs_luad/matched_tumors_and_normals_prot-normed_indiv_by_nat_tum_and_lscc-v3.2_vs_luad-v3.0-phospho_0.75frac-no-na.top50frac_stdev_filt_3_levels_0.75frac-no-na.top50frac_stdev.corr/clustering_intermediates/HDBSCAN;min_cluster_size-9_labels.txt',
index_col=[0,1]
    )
modules = modules.loc[modules.iloc[:,0] != -1] #drop phosphosites that did not fit into any module.
modules.head(2)

normed = pd.read_csv(
'/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/lssc_vs_luad/matched_tumors_and_normals_prot-normed_indiv_by_nat_tum_and_lscc-v3.2_vs_luad-v3.0-phospho_0.75frac-no-na.top50frac_stdev_filt_3_levels.normed.csv',
index_col=[0,1]
)
normed.head(2)

phospho = pd.read_csv('/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/lssc_vs_luad/matched_normals_lscc-v3.2_vs_luad-v3.0-phospho.csv', index_col=[0,1])
 
prot = pd.read_csv('/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/lssc_vs_luad/matched_tumors_and_normal_lscc-v3.2_vs_luad-v3.0-prot.clean.csv', index_col=[0]
).loc[:, normed.columns]
prot.head(2)

proteomics_obj = phdc.ProteomicsData(
 phospho=phospho,
 protein=prot,
 normed_phospho=normed,
 modules=modules,
)

with open('/gpfs/data/ruggleslab/phosphodisco/datasets/utilities/GRCh38_latest_protein.faa','r') as fh:
    aa_seqs = {seq.split()[0]: seq.split(']')[-1].replace('\s', '').replace('\n', '') for seq in fh.read().strip().split('>') if seq != ''}

lscc_raw_phospho = pd.read_csv('/gpfs/data/ruggleslab//pancan/LSCC/lscc-v3.2-data-freeze/lscc-v3.2-phosphoproteome-ratio-norm-NArm.gct', skiprows=2, sep='\t')\
.replace('na', np.nan)\
.dropna(subset=['geneSymbol'])\
.set_index(['geneSymbol', 'variableSites'])
lscc_peptide_ids = lscc_raw_phospho.loc[normed.index.intersection(lscc_raw_phospho.index), 'id'].str.split('.')
lscc_peptide_ids = lscc_peptide_ids.str.get(0) + '.' + lscc_peptide_ids.str.get(1).str.split('_').str.get(0)
lscc_peptide_ids.head(2)

luad_raw_phospho = pd.read_csv('/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/luad/luad-v3.0-phospho-isoform-id-dedup.csv').set_index(['geneSymbol', 'variableSites'])
luad_peptide_ids = luad_raw_phospho.loc[normed.index.intersection(luad_raw_phospho.index), 'accession_number'].str.split('.')
luad_peptide_ids = luad_peptide_ids.str.get(0) + '.' + luad_peptide_ids.str.get(1).str.split('_').str.get(0)
luad_peptide_ids.head(2)

combined_peptide_ids = pd.concat([luad_peptide_ids, lscc_peptide_ids]).reset_index().rename(columns={0:'accession_number'})\
.drop_duplicates().set_index(['geneSymbol', 'variableSites'])
combined_dup_peptide_ids = combined_peptide_ids.loc[combined_peptide_ids.index.duplicated(keep=False)].sort_index()
combined_dedup_peptide_ids = combined_peptide_ids.loc[combined_peptide_ids.index.duplicated(keep='first')+(~combined_peptide_ids.index.duplicated(keep=False)) ]

raw_modules=pd.read_csv(
'/gpfs/data/ruggleslab/phosphodisco/datasets/harmonized_sets/clustering/lssc_vs_luad/matched_tumors_and_normals_prot-normed_indiv_by_nat_tum_and_lscc-v3.2_vs_luad-v3.0-phospho_0.75frac-no-na.top50frac_stdev_filt_3_levels_0.75frac-no-na.top50frac_stdev.corr/clustering_intermediates/HDBSCAN;min_cluster_size-9_labels.txt',
index_col=[0,1]
)
raw_modules.head(2)

import re
seq_df = combined_dedup_peptide_ids.loc[raw_modules.index].copy()
seq_df[raw_modules.iloc[:,0].name] = raw_modules.iloc[:,0]
seq_df = seq_df.reset_index()
seq_df['variable_sites'] = seq_df['variableSites'].apply(lambda x: re.sub("[^0-9\s]", "", x).strip().replace(' ', ','))
seq_df = seq_df.rename(columns = {'accession_number':'protein_id', 'variableSites':'variable_sites_names', 'geneSymbol':'gene_symbol'})
seq_df

proteomics_obj.collect_aa_sequences(seq_df, aa_seqs, 'HDBSCAN;min_cluster_size-9', n_flanking=7, var_sites_aa_col='variable_sites_names')
print('Calculating aa_overlap')
module_overlap_df_dict = phdc.motif_analysis.aa_overlap_from_df(proteomics_obj.module_seq_df, module_col='HDBSCAN;min_cluster_size-9')
print('Plotting now')
phdc.motif_analysis.plot_aa_overlap(module_overlap_df_dict=module_overlap_df_dict, save_path='lmao')
