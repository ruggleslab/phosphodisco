from .annotation_association import corr_na
import pandas as pd
import numpy as np
from pandas import DataFrame
from itertools import product
from typing import Iterable, Optional
from sklearn import linear_model, preprocessing
from .utils import SigmoidCV


reg_models = {
    'linear': linear_model.RidgeCV,
    'sigmoid': SigmoidCV
}


def collapse_possible_regulators(reg_data: DataFrame, corr_threshold: float = 0.9) -> DataFrame:

    corr = {
        (ind1, ind2): corr_na(reg_data.loc[ind1, :], reg_data.loc[ind2, :])[0]
        for ind1, ind2 in product(reg_data.index, reg_data.index)
        if (corr_na(reg_data.loc[ind1, :], reg_data.loc[ind2, :])[0] > corr_threshold) and
           (ind1 != ind2)
    }
    high_corr_inds = list(set([i[0] for i in corr.keys()]+[i[1] for i in corr.keys()]))
    low_corr_inds = reg_data.index.difference(high_corr_inds)
    grouped = reg_data.loc[low_corr_inds, :]
    high_corr_data = reg_data.loc[high_corr_inds, :]
    while corr:
        for inds, corr in corr.items():
            name = tuple(['%s-%s' % (bit[0], bit[1]) for bit in zip(*inds)])
            high_corr_data.loc[name, :] = high_corr_data.loc[list(inds), :].mean()
            high_corr_data = high_corr_data.drop(inds, axis=0)

        corr = {
            (ind1, ind2): corr_na(grouped.loc[ind1, :], grouped.loc[ind2, :])[0]
            for ind1, ind2 in product(grouped.index, grouped.index)
            if (corr_na(grouped.loc[ind1, :], grouped.loc[ind2, :])[0] > corr_threshold) and
            (ind1 != ind2)
        }
        high_corr_inds = list(set([i[0] for i in corr.keys()] + [i[1] for i in corr.keys()]))
        low_corr_inds = reg_data.index.difference(high_corr_inds)
        grouped = grouped.append(grouped.loc[low_corr_inds, :])
        print(grouped.shape, type(grouped))
    return grouped


def calculate_regulator_coefficients(
        reg_data: DataFrame,
        cluster_scores: DataFrame,
        scale_data: bool = True,
        model: str = 'sigmoid',
        regularization_values: Optional[Iterable] = None,
        cv_fold: int = 5,
        **model_kwargs
) -> DataFrame:

    if regularization_values is None:
        [2 ** i for i in range(-10, 10, 1)]

    if model not in reg_models.keys():
        raise ValueError('Model %s not in accepted models: %s' % (model, ','.join(reg_models)))

    weights = pd.DataFrame()
    model = reg_models[model]
    model_kwargs['cv'] = cv_fold
    model = model(alphas=regularization_values, **model_kwargs)
    features = reg_data.transpose().values
    print(reg_data.head(), cluster_scores.head())
    if scale_data:
        features = preprocessing.scale(features)
        targets = preprocessing.scale(cluster_scores.transpose())
    print(features, targets)
    for i, y in enumerate(np.nditer(targets)):
        model.fit(features, y)
        weights[cluster_scores.columns[i]] = model.coef_
    return weights
