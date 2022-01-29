import pkgutil
from io import BytesIO
from warnings import warn
from pandas import DataFrame, Series
from collections import Counter
import numpy as np
import pandas as pd
import logging
import pkgutil
from typing import Iterable, Optional, Union
from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
import hypercluster
from hypercluster.constants import param_delim, val_delim
import sklearn.impute
from io import BytesIO
from .utils import norm_line_to_residuals, zscore
from .constants import var_site_delimiter, protein_id_col, variable_site_col, seq_col, variable_site_aa_col, gene_symbol_col
from .constants import annotation_column_map, datatype_label
from .parsers import read_fasta, read_phospho, read_protein, read_annotation, column_normalize
from .annotation_association import (
    binarize_categorical, continuous_score_association, categorical_score_association
)
from .nominate_regulators import (
    collapse_possible_regulators, calculate_regulator_coefficients,calculate_regulator_corr
)
from .motif_analysis import make_module_sequence_dict, calculate_motif_enrichment, df_to_aa_seqs
from .gene_ontology_analysis import enrichr_per_module, ptm_per_module


class ProteomicsData:
    def __init__(
            self,
            phospho: DataFrame,
            protein: DataFrame,
            min_common_values: Optional[int] = None,
            normed_phospho: Optional[DataFrame] = None,
            modules: Optional[Iterable] = None,
            clustering_parameters_for_modules: Optional[dict] = None,
            possible_regulator_list: Optional[Iterable] = None,
            annotations: Optional[DataFrame] = None,
            column_types: Optional[Iterable] = None
    ):
        """Main data structure for holding all data from one cohort.

        Args:
            phospho: Dataframe of phosphopeptide abundance data. Each row is a phosphopeptide.
            Each column in a sample. Assumed to be column normalized. Multi-index is required,
            with level 0 is the gene identifier (must be consistend with protein index) and level 1
            is a PTM-site identifier.
            protein: Dataframe of protein abundance data. Each row is a protein. Each column in a
            sample. Assumed to be column normalized.
            min_common_values: Minimum number of common non-na samples between a phoshphopeptide
            and its parent protein for protein normalization step.
            normed_phospho: Dataframe of protein-normalized phosphopeptide abundance data. Each
            row is a phosphopeptide. Each column in a sample. Multi-index is required,
            with level 0 is the gene identifier and level 1 is a PTM-site identifier. Can be
            generated with normalize_phospho_by_protein method.
            modules: Dataframe of module labels per phospho site. MultiIndex values must be a
            subset (or all) of the index of normed_phospho. Can be generated with assign_modules,
            but it is recommended to generate these on pairwise correlation of the
            normed_phospho data, then use hypercluster to paralellize the caculations.
            clustering_parameters_for_modules: Dictionary of all parameters used in hypercluster
            to generate clusters/modules. Not used anywhere, just for reproducibility reasons.
            Also generated with the assign_modules method. E.g. clusterer: KMeans, n_clusters: 3
            possible_regulator_list: List of gene names of possible regulators. e.g. kinases.
            annotations: Dataframe of sample annotations used for annotation associations. Rows
            are samples, columns are annotations. Columns designated as categorical will be
            broken up into binary groups, i.e. one column per category.
            column_types: List or Series where each entry corresponds to the the data type in
            each annotations column. Acceptable type are 'categorical' or 'continuous', or 0 and
            1, which mean categorical and continuous respectively. If there are values other than
            these, those annotation columns will be ignored.
        """
        
        if min_common_values is None:
            min_common_values = 6
        self.min_values_in_common = min_common_values

        common_prots = list(set(phospho.index.get_level_values(0).intersection(protein.index)))
        common_samples = phospho.columns.intersection(protein.columns)
        self.phospho = phospho.reindex(common_samples, axis=1)
        self.protein = protein.reindex(common_samples, axis=1)
        self.common_prots = common_prots
        self.common_samples = common_samples
        self.clustering_parameters_for_modules = clustering_parameters_for_modules
        self.possible_regulator_list = possible_regulator_list

        logging.info('Phospho and protein data have %s proteins in common' % len(common_prots))
        logging.info(
            'Phospho and protein data have %s samples in common, re-indexed to only common '
            'samples.' % len(common_samples)
        )

        common_phospho = self.phospho.loc[common_prots,:]
        common_prot = self.protein.reindex(common_phospho.index.get_level_values(0))
        common_prot.index = common_phospho.index

        normalizable_rows = common_phospho.index[
            ((np.logical_and(common_phospho.notnull(), common_prot.notnull())).sum(axis=1) >= min_common_values)
            ]

        self.normalizable_rows = normalizable_rows
        logging.info(
            'There are %s rows with at least %s non-null values in both phospho and protein' % (
                len(normalizable_rows), min_common_values
            )
        )
        self.normed_phospho = normed_phospho

        if modules is not None:
            self.assign_modules(modules)

        if (annotations is not None) and (column_types is not None):
            self.add_annotations(annotations, column_types)
        elif annotations is not None:
            self.annotations = annotations

    def normalize_phospho_by_protein(
            self,
            prevent_negative_coefficients: bool = True,
            ridge_cv_alphas: Optional[Iterable] = None,
            **ridgecv_kwargs
    ):
        """Normalize phosphopeptide abundance by parent protein abundance.

        Args:
            prevent_negative_coefficients: Whether to allow negative linear regression
            coefficients. Set to False to prevent normalizing for the probably non-biologically
            relevant case where the correlation between a phosphosite and its parent protein is
            negative. These cases will be replaced by missing values.
            ridge_cv_alphas: Regularization parameters to try.
            **ridgecv_kwargs: Additional parameters to pass to sklearn.linear_model.RidgeCV

        Returns: self with normed_phospho assigned

        """

        target = self.phospho.loc[self.normalizable_rows]
        features = self.protein.reindex(target.index.get_level_values(0))
        features.index = target.index
        target = target.transpose()
        features = features.transpose()

        target[datatype_label] = 0
        features[datatype_label] = 1

        data = target.append(features)
        data = data.set_index(datatype_label, append=True)
        data.index = data.index.swaplevel(0, 1)
        data = data.transpose()

        residuals = data.apply(
            lambda row: pd.Series(norm_line_to_residuals(
                ph_line=row[0], prot_line=row[1],
                regularization_values=ridge_cv_alphas,
                prevent_negative_coefficients=prevent_negative_coefficients,
                **ridgecv_kwargs
            )), axis=1
        )
        residuals.columns = target.index
        self.normed_phospho = residuals
        return self

    def impute_missing_values(
            self,
            imputation_method: Optional[str] = None,
            **imputer_kwargs
    ):
        """Imputes missing values in the normed_phospho data. Needed if you are going to
        use this raw table for machine learning purposes, or for visualizing with
        visualize_modules, since that uses hierarchical clustering to order the heatmaps.

        Args:
            imputation_method: Name of imputation method from sklearn to use, default is
            KNNImputer.
            **imputer_kwargs: Additional kwargs to pass to the imputer.

        Returns: self with normed_phospho imputed, no missing values.

        """
        if self.normed_phospho.isnull().sum().sum() == 0:
            return self
        if imputation_method is None:
            imputation_method = 'KNNImputer'
        transformed_data = eval(
            'sklearn.impute.%s(**imputer_kwargs).fit_transform(self.normed_phospho.transpose())'
             % imputation_method
        )
        self.normed_phospho = pd.DataFrame(
            transformed_data.transpose(),
            index=self.normed_phospho.index,
            columns=self.normed_phospho.columns
        )
        return self

    def assign_modules(
            self,
            modules: Optional[DataFrame] = None,
            data_for_clustering: Optional[DataFrame] = None,
            method_to_pick_best_labels: Optional[str] = None,
            min_or_max: Optional[str] = None,
            force_choice: bool = False,
            **multiautocluster_kwargs
    ):
        """Uses either user-provided labels or the MultiAutoClusterer from hypercluster to assign
        the modules attribute. Honestly, I do not suggest using this to de novo assign modules. For
        large datasets (with lots of phosphosites in particular, not just lots of samples) it
        suggested to do this using the hypercluster snakemake pipeline to parallelize and speed
        up this process.

        Args:
            modules: If the module labels are already established, pass a DataFrame with one
            column, containing labels per phosphosite. The index must match the index of the
            normed_phospho attribute.
            data_for_clustering: If using this method to run hypercluster, provide a DataFrame of
            data to cluster. If not provided, by default this method will calculate the pairwise
            correlation of all phosphosites from the normed_phospho data. This will be a large
            drain on memory if it isn't filtered. therefore it is recommended to filter and
            provide a table via this parameter.
            method_to_pick_best_labels: If using hypercluster to assign labels, which metric to
            maximize or minimize to identify the labels to use. This is not recommended because
            it is rare that a single metric can be used to identify the best labels.
            min_or_max: If assigning modules with this method, whether to minimize or maximum the
            method for best labels.
            force_choice: If multiple columns of labels are provided, or there is a tie for the
            metric that is being used to choose best labels, whether to randomly pick a set of
            labels.
            **multiautocluster_kwargs: Keyword args to pass to the hypercluster.MultiAutoClusterer

        Returns: self with modules assigned.

        """
        if modules is not None:
            self.modules = modules
            if self.modules.shape[1] > 1:
                if force_choice is False:
                    raise ValueError(
                        'Too many sets of labels in ProteomicsData.modules, please reassign '
                        'ProteomicsData.modules with a DataFrame with 1 column of labels.'
                    )
                else:
                    self.modules = self.modules.sample(1, axis=1)
            self.modules = self.modules.iloc[:, 0]
            try:
                parameters = self.modules.name.split(param_delim)
                clss = parameters.pop(0)
                parameters.append('clusterer%s%s' % (val_delim, clss))
                parameters = {s.split(val_delim, 1)[0]: s.split(val_delim, 1)[1] for s in parameters}
                self.clustering_parameters_for_modules = parameters
            except AttributeError:
                logging.error(
                    "Modules names not in hypercluster structure. Cannot assign "
                    "ProteomicsData.clustering_parameters_for_module"
                )
            return self

        if data_for_clustering is None:
            logging.warning(
                'Careful: calculating the pairwise correlation between %s sites.'
                % len(self.normed_phospho)
            )
            data_for_clustering = self.normed_phospho.transpose().corr()
            no_na_cols = data_for_clustering.columns[~data_for_clustering.isnull().any()]
            data_for_clustering = data_for_clustering.loc[no_na_cols, no_na_cols]

        mac = hypercluster.MultiAutoClusterer(
            **multiautocluster_kwargs
        ).fit(data_for_clustering).evaluate([method_to_pick_best_labels])
        modules = mac.pick_best_labels(method_to_pick_best_labels, min_or_max)
        self.modules = modules
        self.multiautoclusterer = mac

        if self.modules.shape[1] > 1:
            if force_choice is False:
                raise ValueError(
                    'Too many sets of labels in ProteomicsData.modules, please reassign '
                    'ProteomicsData.modules with a DataFrame with 1 column of labels.'
                )
            else:
                self.modules = self.modules.sample(1, axis=1)
        self.modules = self.modules.iloc[:, 0]

        try:
            parameters = self.modules.name.split(param_delim)
            clss = parameters.pop(0)
            parameters.append('clusterer%s%s' % (val_delim, clss))
            parameters = {s.split(val_delim, 1)[0]: s.split(val_delim, 1)[1] for s in parameters}
            self.clustering_parameters_for_modules = parameters
        except AttributeError:
            logging.error(
                "Modules names not in hypercluster structure. Cannot assign "
                "ProteomicsData.clustering_parameters_for_module"
            )

        return self

    def calculate_module_scores(
            self,
            zscore_first: bool = False,
            **knn_imputer_kwargs
    ):
        """Assigns the module_scores attribute by averaging expression of sites per module.

        Args:
            zscore_first: Whether to zscore the data before calculating the scores.
            **knn_imputer_kwargs: If there are missing values after calculating scores,
            keyword args to impute the rest of the values.

        Returns: self with module_scores assigned.

        """
        modules = self.modules[self.modules != -1]
        abundances = self.normed_phospho.reindex(modules.index)
        if zscore_first:
            abundances = zscore(abundances)
        self.module_scores = abundances.groupby(self.modules).agg('mean')
        if self.module_scores.isnull().sum().sum() > 0:
            module_scores = sklearn.impute.KNNImputer(
                **knn_imputer_kwargs
            ).fit_transform(self.module_scores.transpose())
            self.module_scores = pd.DataFrame(
                module_scores.transpose(),
                index=self.module_scores.index,
                columns=self.module_scores.columns
            )
        return self

    def collect_possible_regulators(
            self,
            possible_regulator_list: Optional[Iterable] = None,
            corr_threshold: float = 0.95,
            imputation_method: Optional[str] = None,
            **imputer_kwargs
    ):
        """Collects protein and phosphoprotein data on all genes in a given list into a single
        DataFrame. Since regulators sometimes self-regulate, confounding any
        effort to normalize for protein abundance, raw phospho data is taken, NOT normed_phospho.

        Args:
            possible_regulator_list: List of gene names of possible regulators. Needs to match
            index of protein DataFrame, and 1st level index of phospho DataFrame.
            corr_threshold: To prevent issues of co-linearity in downstream analysis, features
            with higher than corr_threshold pearson correlation are collapsed by averaging. This
            process is somewhat stochastic, as features are averaged in a pairwise iterative
            manner. To prevent the process, set this value to > 1.
            imputation_method: Pick the method by which missing values in possible regulator data
            are imputed for downstream analysis. Default is KNNImputer.
            **imputer_kwargs: Additional keyword arguments for the imputer.

        Returns: self with possible_regulator_data attribute.

        """
        if self.possible_regulator_list is None and possible_regulator_list is None:
            raise ValueError('Must provide possible_regulator_list')
        if possible_regulator_list is None:
            possible_regulator_list = self.possible_regulator_list
        self.possible_regulator_list = possible_regulator_list

        subset = self.protein.loc[self.protein.index.intersection(possible_regulator_list), :]
        if self.phospho.index.name is None:
            ind_name = 'variableSites'
        else:
            ind_name = self.phospho.index.name[1]

        subset[ind_name] = ''
        subset = subset.set_index(ind_name, append=True)
        possible_regulator_data = subset.append(
            self.phospho.loc[
                self.phospho.index.get_level_values(0).intersection(possible_regulator_list), :
            ]
        )
        if corr_threshold <= 1:
            possible_regulator_data = collapse_possible_regulators(
                possible_regulator_data, corr_threshold
            )
        possible_regulator_data = possible_regulator_data.dropna(how='all')

        if imputation_method is None:
            imputation_method = 'KNNImputer'

        transformed_data = eval(
            'sklearn.impute.%s(**imputer_kwargs).fit_transform(possible_regulator_data.transpose())'
            % imputation_method
        )

        self.possible_regulator_data = pd.DataFrame(
            transformed_data.transpose(),
            index=possible_regulator_data.index,
            columns=possible_regulator_data.columns
        )
        return self

    def calculate_regulator_association(
            self,
            method: str = 'correlation',
            **model_kwargs
    ):
        """Calculate the association between each module score and each feature associated with
        possible regulators.

        Args:
            method: Which method to use for calculating the association. Choices are correlation
            or linear_model.
            **model_kwargs: Keyword arguments to pass to the method to use. If using correlation,
            you can specify 'corr_method' as pearsonr or spearmanr. If using linear_model,
            you can specify whether to model the relationship between kinase and substrate as
            'linear' or 'sigmoid'.

        Returns: self with regulator_coefficients and regulator_association_scores assigned.

        """
        if method == 'linear_model':
            self.regulator_coefficients, self.regulator_association_scores = calculate_regulator_coefficients(
                self.possible_regulator_data,
                self.module_scores,
                **model_kwargs
            )
        elif method == 'correlation':
            self.regulator_coefficients, self.regulator_association_scores = calculate_regulator_corr(
                self.possible_regulator_data,
                self.module_scores,
                **model_kwargs
            )
        else:
            raise ValueError(
                'Method must be in: %s. %s not valid'
                %(', '.join(['correlation', 'linear_model']), method)
            )
        return self

    def add_annotations(self, annotations: DataFrame, column_types: Union[list, Series]):
        """Adds sample annotations.

        Args:
            annotations: DataFrame of sample annotations. Columns are different annotations,
            rows are samples.
            column_types: list or Series specifying whether each column is a categorical (
            'categorical' or 0) or continuous ('continuous' or 1) annotation. Missing values mean
            the column will be ignored. Categorical columns will be binarized and stored in
            categorical_annotations, continuous columns will be stored in continuous_annotations
            and the whole table is stored in an annotations attribute.

        Returns: self with annotations attributes.

        """
        if 'categorical_annotations' in self.__dict__ or 'continuous_annotations' in self.__dict__:
            logging.warning('Overwriting annotation data')
        self.annotations = annotations

        if isinstance(column_types, list):
            column_types = pd.Series(column_types, index=annotations.columns)

        common_samples = annotations.index.intersection(self.normed_phospho.columns)
        ncommon = len(common_samples)
        if ncommon <= 1:
            raise ValueError(
                'Only %s samples in common between annotations and normed_phospho. Must be more '
                'than 1 sample in common. ' % len(ncommon)
            )
        logging.info('Annotations have %s samples in common with normed_phospho' % ncommon)
        annotations = annotations.reindex(common_samples)

        column_types = column_types.astype(str).replace(annotation_column_map)
        not_1_or_0 = np.logical_and((column_types != 1), (column_types != 0))
        if any(not_1_or_0):
            logging.warning(
                'These columns will be ignored, invalid column labels: %s'
                % column_types[not_1_or_0]
            )

        self.categorical_annotations = binarize_categorical(
            annotations,
            annotations.columns[column_types == 0]
        )

        self.continuous_annotations = annotations[
            annotations.columns[column_types == 1]
        ].astype(float)
        return self

    def calculate_annotation_association(
            self,
            cat_method: Optional[str] = None,
            cont_method: Optional[str] = None,
            **multitest_kwargs
    ):
        """Calculates the statistical association between module scores and sample annotations.
        Useful for prioritizing modules to study in depth.

        Args:
            cat_method: Which method to use for statistical associated for categorical variables.
            Options are 'RRA', 'ttest' or 'ranksum'.
            cont_method: Which method to use for statistical associated for continuous variables.
            Options are 'pearsonr' or 'spearmanr'.
            **multitest_kwargs: Additional keyword arguments to pass to the multipletests
            function. Default method is benjamini hochberg procedure.

        Returns: self with annotation_association and annotation_association_FDR attributes

        """

        cont = continuous_score_association(
            self.continuous_annotations,
            self.module_scores,
            cont_method
        )
        cat_annots = self.categorical_annotations

        cat = categorical_score_association(
            cat_annots,
            self.module_scores,
            cat_method
        )
        annotation_association = pd.concat([cont, cat], join='outer', axis=1)
        self.annotation_association = annotation_association

        multitest_kwargs['method'] = multitest_kwargs.get('method', 'fdr_bh')
        fdr = annotation_association.apply(
            lambda col: pd.Series(multipletests(col.values, **multitest_kwargs)[1])
        )

        fdr.index = annotation_association.index
        self.annotation_association_FDR = fdr
        return self

    def collect_aa_sequences(
            self,
            all_sites_modules_df,
            fasta: Union[dict, str],
            module_col,
            var_sites_aa_col: Union[str, None] = None,
            n_flanking: int = 7
    ):
        f"""Method for collecting flanking amino acid sequences around modification sites. Used
        for motif analysis, and rigorously comparing modules across datasets or to gold standard
        sets such as with PTM-ssGSEA

        Args:
            all_sites_modules_df: DataFrame where each row is a phosphosite. Structure of this df is very strict.
                                  columns: ['{protein_id_col}', '{variable_site_col}', '{gene_symbol_col}', module_col]
                                  '{protein_id_col}' will match the name of proteins in the fasta file. 
                                  '{variable_site_col}' contains a ',' separated list of variable sites integers.
                                  NB: these must be integers, so variable sites like 'S225s T227t' must be converted to
                                  '225,227'.
                                  '{gene_symbol_col}' contains the gene symbol 
                                  In addition this must contain a column with the module labels for each
                                  site, the name of which you specify with module_col.
            
            fasta:                Fasta file with protein sequences that match with isoform specifier in the
                                  all_sites_modules_df. 
                                  Can also pass a dictionary in format {'{isoform specifier:protein sequence}'}.
            
            module_col:           The name of the column in all_sites_modules_df which contains module
                                  labels per site.
            var_sites_aa_col:     The name of the column in all_sites_modules_df which contains the 
                                  amino acid(s) and position of the variable site(s), e.g. 'S225s,T227t'.
                                  These names are used for plotting purposes, so are optional.
                                  '{variable_site_col}' will be used instead if not provided.
                                  
            n_flanking:           Number of flanking amino acids to collect. Minimum is 7, so that it can
                                  be used for PTM-ssGSEA.

        Returns: self with module_sequences, module_seq_df and background_sequences attributes.

        """
        if isinstance(fasta, str):
            fasta = read_fasta(fasta)
        #Checking columns of module_df:
        required_cols = {protein_id_col, variable_site_col, variable_site_aa_col, module_col}
        given_cols = set(all_sites_modules_df.columns)
        if not required_cols <= given_cols:
            logging.warning(f'The following columns were expected but not found in all_sites_module_df:\n{required_cols.difference(given_cols)}')

        n_flanking = max(7, n_flanking)
        module_df = all_sites_modules_df.loc[all_sites_modules_df[module_col] != -1, :]
        module_aas = make_module_sequence_dict(module_df, fasta, module_col, n_flanking)
        module_seq_df = module_df.copy() 
        module_seq_df[seq_col] = df_to_aa_seqs(module_df, fasta, n_flanking)
        if var_sites_aa_col is not None:
            module_seq_df[variable_site_aa_col] = module_seq_df[var_sites_aa_col].copy()
        else:
            module_seq_df[variable_site_aa_col] = module_seq_df[variable_site_col].copy()
        self.module_sequences = module_aas # module:sequences dict
        self.module_seq_df = module_seq_df # df for aa similarity calcs
### format of module_seq_df (col order not deterministic):
### variable_site_col, var_sites_aa_col, protein_id_col, module_col, seq_col
        self.background_sequences = var_site_delimiter.join( # list of background sequences
            df_to_aa_seqs(all_sites_modules_df, fasta, n_flanking)
        ).split(var_site_delimiter)
        return self

    def analyze_aa_sequences(
            self,
    ):
        """Counts the aa frequencies at each position, as well as the enrichment over background
        for each module.

        Returns: self with module_freqs and module_aa_enrichment attributes.

        """
        self.module_aa_freqs = {
            module: pd.DataFrame([Counter(tup) for tup in list(zip(*aas))])
            for module, aas in self.module_sequences.items()
        }

        ps = calculate_motif_enrichment(
            module_aas=self.module_sequences,
            background_aas=self.background_sequences,
        )
        self.module_aa_enrichment = ps
        return self

    def calculate_go_set_enrichment(
            self,
            background_gene_list,
            gene_sets: str = 'GO_Biological_Process_2018',
            **enrichr_kws
    ):
        """Uses enrichr from the gseapy package to calculate gene set enrichments for genes in
        each module.

        Args:
            background_gene_list: List of all genes from sites used to identify modules.
            gene_sets: Name of gene sets to use, see enrichr for possibilities:
            http://amp.pharm.mssm.edu/Enrichr/#stats
            **enrichr_kws: Additional keyword args to pass to gseapy.enrichr.

        Returns: self with go_enrichment attribute.

        """
        self.go_enrichment = enrichr_per_module(
            self.modules,
            background_gene_list=background_gene_list,
            gene_sets=gene_sets,
            **enrichr_kws
        )
        return self

    def calculate_ptm_set_enrichment(
            self,
            ptm_set_gmt: str = 'data/ptm.sig.db.all.flanking.human.v1.9.0.gmt'
    ):
        """Uses a hypergeometric test to calculate enrichment for known ptm sets from PTM-ssGSEA
        gmt files. Must have the module_sequences and background_sequences attributes to run.

        Args:
            ptm_set_gmt: Path to gmt file with ptm sets.

        Returns: self with ptm_enrichment attribute.

        """
        self.ptm_enrichment = ptm_per_module(
            self.module_sequences,
            background_seqs=self.background_sequences, ptm_set_gmt=ptm_set_gmt
        )
        return self
    
    def extract_kinase_activation_loop_phosphosites(self, kin_act_phosphosites=None):
        """
        Extracts common phosphosites between ProteomicsData object and the list of known activation loop sites found by Schmidlin et. al 2019. Also allows for partial matches, i.e. S12s S143s will be matched with S12s or S143s and vice versa.
        kin_act_phosphosites: Optional phosphosite dataframe with a MultiIndex, where 
                              level 0 is the gene identifier and level 1 is a PTM-site identifier
        """
        if kin_act_phosphosites is None: #read in dataframe if none is given 
            kin_act_data = BytesIO(pkgutil.get_data('phosphodisco', 'data/kin_act_loops.csv'))
            kin_act_phosphosites = pd.read_csv(kin_act_data,
                            index_col=[0,1])
        #Extract indices 
        phospho_inds = self.phospho.index.to_frame().copy()
        phospho_inds.iloc[:,1] = phospho_inds.iloc[:,1].str.split() #get  first column 
        phospho_inds = phospho_inds.explode(phospho_inds.columns[1]) #get column name

        # Rename and swap index and columns, so we can overlap the new index with the kinase activation loop index
        phospho_inds = phospho_inds.rename(columns={phospho_inds.columns[0]:'geneSymbol_exploded',phospho_inds.columns[1]:'variableSites_exploded'})
        phospho_inds['variableSites_numerical'] = phospho_inds['variableSites_exploded'].str.extract(r'(\d+)').astype(int)
        phospho_inds = phospho_inds.reset_index()
        phospho_inds = phospho_inds.set_index(['geneSymbol_exploded','variableSites_numerical'])

        #Fetch phosphodata from overlapping indices 
        kin_act_index_overlap = kin_act_phosphosites.index.intersection(phospho_inds.index)
        #phospho_inds.columns
        phospho_inds_overlap = phospho_inds.loc[kin_act_index_overlap].iloc[:,0:2].set_index(list(phospho_inds.columns[[0,1]])).index.drop_duplicates()
        kin_act_loop_phospho_data = self.phospho.loc[phospho_inds_overlap]
        self.kin_act_loop_phospho_data = kin_act_loop_phospho_data
        if self.kin_act_loop_phospho_data.shape[0] == 0:
            warn("No phosphosites overlapped with the set of kinase activation loop phosphosites")
        return self
    def correlate_kinase_activation_loop_phosphosites_with_module_scores(self, na_frac=None):
        """
        Correlates kinase activation loop phosphosite data with module scores using the Spearman correlation coefficient.
        Will try to run the necessary functions in case module_scores or kin_act_loop_phospho_data haven't been set by calculate_module_scores and extract_kinase_activation_loop_phosphosites yet. P-values are multiple-testing corrected with FDR (Benjamini Hochberg method).
        
        Args:
            na_frac: float to set which fraction of a row in kin_act_loop_phospho_data is allowed to be NaN and still get included, i.e. 0.7 would mean 70% of values could be NaN in a row. By default and at minimum, 3 non-NaN values are allowed.
        
        Returns:
            self with kin_act_phospho_module_score_correlations and kin_act_phospho_module_score_pvals attributes set.
        """
        if not hasattr(self,"kin_act_loop_phospho_data"):
            logging.info("Kinase activation loop data has not been extracted from ProteomicsData.phospho yet.\nExtracting kinase activation loop data using defaults.")
            self.extract_kinase_activation_loop_phosphosites()
            if self.kin_act_loop_phospho_data.shape[0] == 0:
                raise ValueError("Need kinase activation loop phosphosites that overlap with provided phospho data in order to correlate them with module scores")
        if not hasattr(self, "module_scores"):
            if not hasattr(self, "modules"):
                raise ValueError("Add modules using assign_modules first")
            else:
                self.calculate_module_scores()
        # dropping all rows with less than 3 non-NaN values bc three are required for a p-value
        # should I add another optional NaN filtering step here? Rows with tons of missing values will tend to have terrible p-values, and add to the number of tests we have to correct for.
        if na_frac is not None:
            na_thresh = round(self.kin_act_loop_phospho_data.shape[1] * (1-na_frac))
        else:
            na_thresh = 3
        spearmanr_df = self.kin_act_loop_phospho_data.dropna(thresh=na_thresh).dropna(thresh=3).apply(
            lambda phos_sites_row: self.module_scores.apply(
                lambda scores_row: spearmanr(phos_sites_row, scores_row, nan_policy='omit'),
                axis=1
            ),
            axis=1
        )
        self.kin_act_phospho_module_score_correlations = spearmanr_df.apply(lambda col: col.str.get(0))
        pvals_melted = spearmanr_df.apply(lambda col: col.str.get(1)).melt(ignore_index=False, var_name="module", value_name='p-value').reset_index()
        pvals_melted['adj_pval'] = multipletests(pvals_melted['p-value'], method='fdr_bh')[1]
        self.kin_act_phospho_module_score_pvals = pvals_melted.pivot_table(columns='module', values='adj_pval', index=pvals_melted.columns[[0,1]].to_list())
        
        return self


def prepare_data(
        ph_file: str,
        prot_file: str,
        normalize_method: Optional[str] = None,
        min_common_values: int = 5,
        normed_phospho: Optional[DataFrame] = None,
        modules: Optional[Iterable] = None,
        clustering_parameters_for_modules: Optional[dict] = None,
        putative_regulator_list: Optional[list] = None,
) -> ProteomicsData:
    """Helper function for loading a bunch of files into a ProteomicsData object.

    Args:
        ph_file: Path to phospho data file. File must be tsv or csv with first two column as
        gene name, phosphosite respectively.
        prot_file: Path to protein data file. csv or tsv with first column specifying the gene
        name.
        normalize_method: Optional, if given, the method to use to column normalize for coverage.
        See parsers.column_normalize for options.
        min_common_values: The minimum common values between a protein and its phosphosite to
        consider protein-normalization.
        normed_phospho: Path to protein-normalized phospho data, if pre calculated.
        modules: Path to module labels table, if pre calculated.
        clustering_parameters_for_modules: Dictionary of parameters used in clusterer to get
        module labels, if pre-calculated.
        putative_regulator_list: List of gene names of putative regulators.

    Returns: ProteomicsData object loaded with all provided data.

    """

    phospho = read_phospho(ph_file)
    protein = read_protein(prot_file)
    if normalize_method:
        phospho = column_normalize(phospho, normalize_method)
        protein = column_normalize(protein, normalize_method)

    return ProteomicsData(
        phospho=phospho,
        protein=protein,
        min_common_values=min_common_values,
        normed_phospho=normed_phospho,
        modules=modules,
        clustering_parameters_for_modules=clustering_parameters_for_modules,
        possible_regulator_list=putative_regulator_list
    )


def druggability(self,module_num=None,interactions=None):
    
    """
    Finds druggable genes in a given module
    Args: 
        module_num = Cluster numbers, must be in list form. 
        interactions = gene-drug interactions in .tsv or .csv format. If input is empty, the default is  a file from the DGidb database"
    """
    
    #Read in list of interactions
    if interactions is None: #read in list of interactions taken from DGidb database
        interactions = BytesIO(pkgutil.get_data('phosphodisco', 'data/interactions-Jan2021-dgidb.tsv'))
    else:
        interactions = read_annotation(interactions)
    genes = interactions.iloc[:,0]

    #define a list of druggable genes as a set 
    druggable_genes = set(genes)
    
    #throw error if no modules
    if hasattr(self, 'modules') == False:
        raise AttributeError("No modules assigned, run .assign_modules.")
    
   #define module genes
    module_genes = self.modules.index.get_level_values(0)
    self.druggable_module_genes = druggable_genes.intersection(module_genes)
    self.druggable_module_genes_df = pd.DataFrame(self.modules.loc[self.druggable_module_genes])

    if module_num is not None:
        if isinstance(module_num, list):
            dataframe = self.druggable_module_genes_df.reset_index()
            self.druggable_module_genes_df = dataframe.loc[dataframe.iloc[:,2].isin(module_num)]
            return self
        else:
            return self


def find_druggable_regulators(self,module_num=None,top_num=None, only_druggable=True):
    
    """
    Description: 
        This function helps find druggable regulators and is able to filter them by module. 
        Can also find the top N (where N is a number over zero) regulators associated with each module
    
    Args: 
        module_num = Module numbers of interest. Must be entered in list form. Uses all modules if None.
        top_num = The number of regulators closely associated with the module, must be greater than 0. 
        only_druggable = True or False value, specifies whether you want only druggable  regulators returned.
        
    Modified attributes: 
        self.druggable_regulators_df = DataFrame of only druggable regulators. 
        self.filtered_reg_df =  DataFrame of regulators filtered by druggability, modules of interest and top_num
    
    Returns: 
        self 
        
    """
    if hasattr(self, 'possible_regulator_data')==False:
        raise AttributeError("No regulators nominated, run .collect_possible_regulators")

    possible_regulator_list = self.possible_regulator_data.index.get_level_values(0)
    druggable_genes = self.druggable_module_genes_df.iloc[:,0]
    self.druggable_genes = druggable_genes
    self.possible_reg_list = possible_regulator_list

    if hasattr(self, 'regulator_coefficients')==False:
        raise AttributeError("Run .calculate_regulator_association()")
        
    reg_coeff = proteomics_obj.regulator_coefficients.reset_index()

    
    #return a  list of only druggable regulators 
    druggable_regulators_names = self.possible_regulator_data.loc[possible_regulator_list.intersection(druggable_genes)]
    druggable_regulator_list = set(druggable_regulators_names.reset_index().iloc[:,0])
    self.druggable_regulator_list = druggable_regulator_list
    
    
    #find druggable regulators in regulator_coefficient dataframe
    self.druggable_regulators_df =  reg_coeff[reg_coeff.iloc[:,0].isin(druggable_regulator_list)]
   
    #select druggable or not 
    if only_druggable:
        regulator_df = self.druggable_regulators_df.set_index(druggable_regulators_df.columns[:2]).copy()
    else:
        regulator_df = reg_coeff.copy().set_index(reg_coeff.columns[:2])
                                              
    #select modules
    if module_num is not None: 
        regulator_df = regulator_df.loc[:,module_num]
        
    column_list = regulator_df.columns
    
    #select top_n regulators for each module
    if top_num is not None: 
        combined_top_regs  = set(
                chain.from_iterable(
                    set(regulator_df[x].nlargest(top_num).index) for x in column_list
                    )
                )
        regulator_df = regulator_df.loc[combined_top_regs, column_list]
        
        
    self.filtered_reg_df = regulator_df

    return self


def druggable_regulator_heatmap(self, module_num=None, top_num=None, only_druggable=True):
    
    """
    Description: 
     This function creates a heatmap displaying the association coefficients between nominated, druggable regulators and modules.
     It also calls on the find_druggable_regulators function and uses its output to create the heatmap
    
    Args:
        module_num= Module numbers of interest. Must be entered in list form 
        top_num = The number of regulators closely associated with the module, must be greater than 0. 
        only_druggable = True or False value, specifies whether you want only druggable regulators returned.
    
    Modified attributes:
        self.druggable_bool = returns series containing gene names, associated phosphosites and whether these genes are druggable are not.
                              Druggable genes are "True", non-druggable genes are "False". 
                              This boolean series is used to construct the accompanying color bar for the heatmap. 
    Returns:
        druggability_map = seaborn.matrix.ClusterGrid
    """
    #Error statements
    if hasattr(self, 'regulator_coefficients')==False:
        raise AttributeError("Run .calculate_regulator_association()")

    #call other function
    self.find_druggable_regulators(module_num=module_num,top_num=top_num, only_druggable=only_druggable)
    
    regdf = self.filtered_reg_df
    regdf_copy = self.filtered_reg_df.copy()
    
    #annotation for druggability 
    druggable_or_not = []
    
    
    #Druggability
    druggable_genes = self.druggable_genes
    #use druggable regulators not druggable genes in modules
    for x in regdf.index.get_level_values(0):
        if x in druggable_genes:
            druggable_or_not.append(1)
        else:
            druggable_or_not.append(0)
        
        
    self.druggable_or_not = druggable_or_not
    regdf_copy['druggability'] = druggable_or_not
    
    
   
    druggable_bool = pd.Series(
            data = regdf_copy.index.get_level_values(0).map(
                lambda row: (len(set(row.split('-')).intersection(druggable_genes)) > 0)
                ), #get rows with at least one druggable gene
            index = regdf.index,
            name = 'druggability'
            )
    
    self.druggable_bool = druggable_bool
    
    druggable_colors = druggable_bool.map(dict(zip([False,True], sns.color_palette('husl',2))))
    cmap = dict(zip(['not druggable','druggable'], sns.color_palette('husl',2)))
    
    #Heatmap
    legend_TN  = [mpatches.Patch(color=c,label=l) for l,c in cmap.items()]  
  
    #center heatmap at 
    druggability_map = sns.clustermap(regdf, row_colors=druggable_colors, cmap="coolwarm")
    l2=druggability_map.ax_heatmap.legend(loc='center',bbox_to_anchor=(1.8,1.2),handles=legend_TN,frameon=True, prop={'size':15})
    return druggability_map
