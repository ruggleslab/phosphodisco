from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import logging
from typing import Iterable, Optional
from itertools import product
from statsmodels.stats.multitest import multipletests
import hypercluster
from hypercluster.constants import param_delim, val_delim
from .utils import norm_line_to_residuals
from .constants import module_combiner_delim, annotation_column_map, datatype_label
from .annotation_association import (
    corr_na, binarize_categorical, continuous_score_association, categorical_score_association
)
from .nominate_regulators import collapse_putative_regulators, calculate_regulator_coefficients


class ProteomicsData:
    """

    Args:
        phospho:
        protein:
        min_common_values:
        normed_phospho:
        modules:
        clustering_parameters_for_modules:
        putative_regulator_list:
    """
    def __init__(
            self,
            phospho: DataFrame,
            protein: DataFrame,
            min_common_values: Optional[int] = None,
            normed_phospho: Optional[DataFrame] = None,
            modules: Optional[Iterable] = None,
            clustering_parameters_for_modules: Optional[dict] = None,
            putative_regulator_list: Optional[Iterable] = None,
            annotations: Optional[DataFrame] = None
    ):
        if min_common_values is None:
            min_common_values = 5
        self.min_values_in_common = min_common_values

        common_prots = list(set(phospho.index.get_level_values(0).intersection(protein.index)))
        common_samples = phospho.columns.intersection(protein.columns)
        self.phospho = phospho.reindex(common_samples, axis=1)
        self.protein = protein.reindex(common_samples, axis=1)
        self.common_prots = common_prots
        self.common_samples = common_samples
        self.clustering_parameters_for_modules = clustering_parameters_for_modules
        self.putative_regulator_list = putative_regulator_list

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
            lambda row: norm_line_to_residuals(
                row[0], row[1],
                ridge_cv_alphas,
                **ridgecv_kwargs
            ), axis=1
        )
        
        self.normed_phospho = residuals.reindex(self.common_samples, axis=1)
        return self

    def assign_modules(
            self,
            modules: Optional[DataFrame] = None,
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
        if modules is not None:
            self.modules = modules
        if 'modules' not in self.__dict__:
            modules = hypercluster.MultiAutoClusterer(
                **multiautocluster_kwargs
            ).fit(self.normed_phospho).pick_best_labels(method_to_pick_best_labels, min_or_max)
            self.modules = modules[modules.columns[0]]

        if self.modules.shape[1] > 1:
            if force_choice is False:
                raise ValueError(
                    'Too many sets of labels in ProteomicsData.modules, please reassign '
                    'ProteomicsData.modules with a DataFrame with 1 column of labels.'
                )
            else:
                self.modules = modules.sample(1, axis=1)

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
            combine_anti_regulated: bool = True,
            anti_corr_threshold: float = 0.9
    ):
        """

        Args:
            combine_anti_regulated:
            anti_corr_threshold:

        Returns:

        """
        abundances = self.normed_phospho.reindex(self.modules.index)
        scores = abundances.groupby(self.modules).agg('mean')

        if combine_anti_regulated:
            corr = {
                (ind1, ind2): corr_na(scores.loc[ind1, :], scores.loc[ind2, :])[0]
                for ind1, ind2 in product(scores.index, scores.index)
                if -corr_na(scores.loc[ind1, :], scores.loc[ind2, :])[0] > anti_corr_threshold
            }
            if corr:
                for clusters_labs in corr.keys():
                    nmems = {k: self.nmembers_per_cluster[k] for k in clusters_labs}
                    major_cluster = max(nmems, key=lambda key: nmems[key])
                    minor_cluster = set(clusters_labs).difference({major_cluster})
                    line = (scores.loc[major_cluster, :] * nmems[major_cluster]).subtract(
                        (scores.loc[minor_cluster, :] * nmems[minor_cluster])
                    ).divide((nmems[major_cluster] + nmems[minor_cluster]))
                    line.name = module_combiner_delim.join(clusters_labs)
                    scores = scores.drop(clusters_labs, axis=0).append(line)

        self.anticorrelated_collapsed = combine_anti_regulated
        self.module_scores = scores.transpose()
        return self

    def collect_putative_regulators(self, possible_regulator_list: Optional[Iterable] = None, corr_threshold: float = 0.9):
        """

        Args:
            possible_regulator_list:
            corr_threshold:

        Returns:

        """
        if self.putative_regulator_list is None and putative_regulator_list is None:
            raise ValueError('Must provide putative_regulator_list')
        if putative_regulator_list is None:
            putative_regulator_list = self.putative_regulator_list
        self.putative_regulator_list = possible_regulator_list
        
        subset = self.protein.loc[possible_regulator_list, :]
        subset[1] = np.nan
        subset = subset.set_index(1, append=True)
        putative_regulator_data = subset.append(self.phospho.loc[possible_regulator_list, :])
        putative_regulator_data = collapse_putative_regulators(
            putative_regulator_data, corr_threshold
        )
        self.putative_regulator_data = putative_regulator_data
        return self

    def calculate_regulator_coefficients(
            self,
            **kwargs
    ):
        """

        Args:
            **kwargs:

        Returns:

        """
        self.regulator_coefficients = calculate_regulator_coefficients(
            self.putative_regulator_data,
            self.module_scores,
            **kwargs
        )
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
        common_samples = annotations.index.intersection(self.normed_phospho.columns)
        ncommon = len(common_samples)
        if ncommon <= 1:
            raise ValueError(
                'Only %s samples in common between annotations and normed_phospho. Must be more '
                'than 1 sample in common. ' % len(ncommon)
            )
        logging.info('Annotations have %sß samples in common with normed_phospho' % ncommon)
        annotations = annotations.reindex(common_samples)

        column_types = column_types.replace(annotation_column_map)

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
        fdr = pd.DataFrame(index=annotation_association.index)
        for col in annotation_association.columns:
            fdr.loc[:, col] = multipletests(annotation_association[col], **multitest_kwargs)[1]
        self.annotation_association_FDR = fdr
        return self
