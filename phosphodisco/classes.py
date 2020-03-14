from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import logging
from typing import Iterable, Optional
from statsmodels.stats.multitest import multipletests
import hypercluster
from hypercluster.constants import param_delim, val_delim
import sklearn.impute

from .utils import norm_line_to_residuals
from .constants import annotation_column_map, datatype_label
from .annotation_association import (
    binarize_categorical, continuous_score_association, categorical_score_association
)
from .nominate_regulators import collapse_possible_regulators, calculate_regulator_coefficients, calculate_regulator_corr


class ProteomicsData:
    """

    Args:
        phospho:
        protein:
        min_common_values:
        normed_phospho:
        modules:
        clustering_parameters_for_modules:
        possible_regulator_list:
    """
    def __init__(
            self,
            phospho: DataFrame,
            protein: DataFrame,
            min_common_values: Optional[int] = None,
            normed_phospho: Optional[DataFrame] = None,
            modules: Optional[Iterable] = None,
            clustering_parameters_for_modules: Optional[dict] = None,
            possible_regulator_list: Optional[Iterable] = None,
            annotations: Optional[DataFrame] = None
    ):
        #TODO add annotation cols in this
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
        
        if annotations is not None:
            self.add_annotations(annotations)

    def normalize_phospho_by_protein(
            self,
            ridge_cv_alphas: Optional[Iterable] = None,
            **ridgecv_kwargs
    ):
        """

        Args:
            ridge_cv_alphas:
            **ridgecv_kwargs:

        Returns:

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
                row[0], row[1],
                ridge_cv_alphas,
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
        """

        Args:
            modules:
            method_to_pick_best_labels:
            min_or_max:
            force_choice:
            **multiautocluster_kwargs:

        Returns:

        """
        if data_for_clustering is None:
            data_for_clustering = self.normed_phospho.transpose().corr()
            na_cols = data_for_clustering.columns[~data_for_clustering.isnull().any()]
            data_for_clustering = data_for_clustering.loc[na_cols, na_cols]

        if modules is not None:
            self.modules = modules
        else:
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
            self
    ):
        modules = self.modules[self.modules != -1]
        abundances = self.normed_phospho.reindex(modules.index)
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
        """

        Args:
            possible_regulator_list:
            corr_threshold:

        Returns:

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
        if method == 'linear_model':
            self.regulator_coefficients, self.module_prediction_scores = calculate_regulator_coefficients(
                self.possible_regulator_data,
                self.module_scores,
                **model_kwargs
            )
        elif method == 'correlation':
            self.regulator_coefficients, self.module_prediction_scores = calculate_regulator_corr(
                self.possible_regulator_data,
                self.module_scores,
                **model_kwargs
            )
        else:
            raise ValueError('Method must be in: %s. %s not valid' %(', '.join(['correlation', 'linear_model']), method))
        return self

    def add_annotations(self, annotations: DataFrame, column_types: Series):
        """

        Args:
            annotations:
            column_types:

        Returns:

        """
        if 'categorical_annotations' in self.__dict__ or 'continuous_annotations' in self.__dict__:
            logging.warning('Overwriting annotation data')
        #TODO add check for list or series for col types
        common_samples = annotations.index.intersection(self.normed_phospho.columns)
        ncommon = len(common_samples)
        if ncommon <= 1:
            raise ValueError(
                'Only %s samples in common between annotations and normed_phospho. Must be more '
                'than 1 sample in common. ' % len(ncommon)
            )
        logging.info('Annotations have %sß samples in common with normed_phospho' % ncommon)
        annotations = annotations.reindex(common_samples)

        column_types = column_types.replace(annotation_column_map) #TODO omg fix this whole problem

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
        """

        Args:
            cat_method:
            cont_method:
            **multitest_kwargs:

        Returns:

        """
        if self.categorical_annotations is None or self.continuous_annotations is None:
            raise ValueError(
                'Annotations are not defined. Provide annotation table to add_annotation method.'
            )
        cont = continuous_score_association(
            self.continuous_annotations,
            self.module_scores,
            cont_method
        )
        cat = categorical_score_association(
            self.categorical_annotations,
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
