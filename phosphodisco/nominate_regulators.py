import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Iterable, Optional
from sklearn import linear_model, preprocessing


def collapse_possible_regulators(reg_data: DataFrame, corr_threshold: float = 0.9) -> DataFrame:
    corr = reg_data.transpose().corr()
    high_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) > 1)]
    low_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) <= 1)]
    high_corr_data = reg_data.loc[high_corr_inds, :]
    low_corr_data = reg_data.loc[low_corr_inds, :]
    if len(high_corr_inds) == 0:
        return low_corr_data

    corr = corr.loc[high_corr_inds, high_corr_inds].stack(level=[0, 1])
    corr = corr.loc[[(a, b) != (c, d) for a, b, c, d in corr.index]]
    corr = corr[corr > corr_threshold]

    while corr.shape[0] > 0:
        for i in corr.index:
            a, b, c, d = i
            if (a, b) in high_corr_data.index and (c, d) in high_corr_data.index:
                name = ('%s-%s' %(a, c), '%s-%s' % (b, d))
                high_corr_data = high_corr_data.append(
                    pd.Series(high_corr_data.loc[[(a, b), (c, d)], :].mean(), name=name)
                )
                high_corr_data = high_corr_data.drop([(a, b), (c, d)], axis=0)
        corr = high_corr_data.transpose().corr()
        high_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) > 1)]
        low_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) <= 1)]

        low_corr_data = low_corr_data.append(high_corr_data.loc[low_corr_inds, :])
        if len(high_corr_inds) == 0:
            return low_corr_data
        high_corr_data = high_corr_data.loc[high_corr_inds, :]

        corr = corr.loc[high_corr_inds, high_corr_inds].stack(level=[0, 1])
        corr = corr.loc[[(a, b) != (c, d) for a, b, c, d in corr.index]]
        corr = corr[corr > corr_threshold]

    return low_corr_data


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
        regularization_values = [10 ** i for i in range(-5, 4, 1)]

    if model not in ['linear', 'sigmoid']:
        raise ValueError(
            'Model %s not in accepted models: %s' % (model, ','.join(['linear', 'sigmoid']))
        )

    model = linear_model.RidgeCV

    model_kwargs['cv'] = cv_fold
    model = model(alphas=regularization_values, **model_kwargs)
    features = reg_data.transpose().values
    targets = cluster_scores.transpose().values
    if model == 'sigmoid':
        targets = (1/(1+np.exp(-targets)))
    if scale_data:
        features = preprocessing.scale(features)
        targets = preprocessing.scale(targets)

    model.fit(features, targets)
    weights = pd.DataFrame(
        model.coef_,
        index=cluster_scores.index,
        columns=reg_data.index
    ).transpose()
    return weights
