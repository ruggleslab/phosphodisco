from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Iterable, Optional
from sklearn.linear_model import RidgeCV
from itertools import product
from .constants import continuous_methods


def not_na(array):
    if isinstance(array, pd.Series):
        return ~array.isna()
    return ~np.isnan(array)


def corr_na(array1, array2, corr_method: str = 'pearsonr'):
    if corr_method not in ['pearsonr', 'spearmanr']:
        raise ValueError(
            'Method %s is a valid correlation method, must be: %s'
            % (corr_method, ','.join(continuous_methods.keys()))
        )
    nonull = not_na(array1) & not_na(array2)
    return continuous_methods[corr_method](array1[nonull], array2[nonull])


def norm_line_to_residuals(
        ph_line: Iterable,
        prot_line: Iterable,
        regularization_values: Optional[Iterable] = None,
        cv: Optional[int] = 5
) -> Series:
    if regularization_values is None:
        regularization_values = [2 ** i for i in range(-10, 10, 1)]

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    nonull = (~np.isnan(ph_line) & ~np.isnan(prot_line))
    if sum(nonull) < cv:
        return np.empty(len(ph_line))

    features = prot_line[nonull].values.reshape(-1, 1)
    labels = ph_line[nonull].values
    model = RidgeCV(alphas=regularization_values, cv=cv).fit(features, labels)
    prediction = model.predict(features)
    residuals = labels - prediction

    return pd.Series(residuals, index=ph_line[nonull].index)


class ProteomicsData:

    def __init__(
            self, phospho: DataFrame,
            protein: DataFrame,
            min_common_values: Optional[int] = 5
    ):
        self.min_values_in_common = min_common_values

        common_prots = list(set(phospho.index.get_level_values(0).intersection(protein.index)))
        common_samples = phospho.columns.intersection(protein.columns)
        self.phospho = phospho[common_samples]
        self.protein = protein[common_samples]
        self.common_prots = common_prots
        self.common_samples = common_samples

        logging.info('Phospho and protein data have %s proteins in common' % len(common_prots))
        logging.info('Phospho and protein data have %s samples in common, re-indexed to only '
                     'common samples.' %
                     len(common_samples))

        common_phospho = self.phospho.loc[common_prots,:]
        common_prot = self.protein.reindex(common_phospho.index.get_level_values(0))
        common_prot.index = common_phospho.index
        normalizable_rows = common_phospho.index[
            ((common_phospho.notnull() & common_prot.notnull()).sum(axis=1) >= min_common_values)
            ]

        self.normalizable_rows = normalizable_rows
        logging.info('There are %s rows with at least %s non-null values in both phospho and '
                     'protein' % (len(normalizable_rows), min_common_values))
        self.normed_phospho = None

    def normalize_phospho_by_protein(self, ridge_cv_alphas: Optional[Iterable] = None):
        # TODO test this, make sure datatype_label goes away.
        target = self.phospho.loc[self.normalizable_rows]
        features = self.protein.reindex(target.index.get_level_values(0))
        features.index = target.index
        target = target.transpose()
        features = features.transpose()

        datatype_label = 'datatype_label'
        target[datatype_label] = 0
        features[datatype_label] = 1

        data = target.append(features)
        data = data.set_index(datatype_label, append=True)
        data.index = data.index.swaplevel(0, 1)
        data = data.transpose()

        residuals = data.apply(
            lambda row: norm_line_to_residuals(
                row[0], row[1], ridge_cv_alphas, self.min_values_in_common
            ), axis=1
        )

        self.normed_phospho = residuals


class Clusters:
    def __init__(
            self,
            cluster_labels: Series,
            abundances: DataFrame,
            parameters: Optional[dict]  = None
    ):
        self.cluster_labels = cluster_labels
        self.parameters = parameters
        self.abundances = abundances
        self.nmembers_per_cluster = cluster_labels.value_counts()
        self.cluster_scores: Optional[DataFrame] = None
        self.anticorrelated_collapsed: Optional[bool] = None
        self.anticorrelated_collapsed = None

    def calculate_cluster_scores(
            self,
            combine_anti_regulated: bool = True,
            anti_corr_threshold: float = 0.9
    ):
        """

        Args:
            combine_anti_regulated:
            anti_corr_threshold:

        Returns: Samples x scores dataframe

        """
        abundances = self.abundances.reindex(self.cluster_labels.index)
        scores = abundances.groupby(self.cluster_labels).agg(mean)

        if combine_anti_regulated:
            self.anticorrelated_collapsed = True
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
                        (scores.loc[minor, :] * nmems[minor_cluster])
                    ).divide((nmems[major_cluster] + nmems[minor_cluster]))
                    line.name = '-'.join(clusters_labs)
                    scores = scores.drop(clusters_labs, axis=0).append(line)
        else:
            self.anticorrelated_collapsed = False
        self.cluster_scores = scores.transpose()
