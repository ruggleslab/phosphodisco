import warnings
from typing import Iterable, Optional
from pandas import Series
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, spearmanr


def norm_line_to_residuals(
        ph_line: Iterable,
        prot_line: Iterable,
        regularization_values: Optional[Iterable] = None,
        cv: Optional[int] = None,
        **ridgecv_kwargs
) -> Series:
    """

    Args:
        ph_line:
        prot_line:
        regularization_values:
        cv:
        **ridgecv_kwargs:

    Returns:

    """
    if regularization_values is None:
        regularization_values = [5 ** i for i in range(-5, 5)]
    if cv is None:
        cv = 3

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    nonull = np.logical_and(~np.isnan(ph_line), ~np.isnan(prot_line))
    if sum(nonull) < cv:
        return np.empty(len(ph_line))

    features = prot_line[nonull].values.reshape(-1, 1)
    labels = ph_line[nonull].values

    ridgecv_kwargs['alphas'] = regularization_values
    ridgecv_kwargs['cv'] = cv
    model = RidgeCV(**ridgecv_kwargs).fit(features, labels)
    prediction = model.predict(features)
    residuals = labels - prediction

    return pd.Series(residuals, index=ph_line[nonull].index)


def multipletests_na(pvals: np.array, **multitest_kwargs):
    mask = np.isfinite(pvals)
    pval_corrected = np.full(pvals.shape, np.nan)
    pval_corrected[mask] = multipletests(pvals[mask], **multitest_kwargs)[1]

    return pval_corrected


def not_na(array):
    if isinstance(array, Series):
        return ~array.isna()
    return ~np.isnan(array)


def corr_na(array1, array2, corr_method: str = 'spearmanr'):
    if corr_method not in ['pearsonr', 'spearmanr']:
        raise ValueError(
            'Method %s is a valid correlation method, must be: %s'
            % (corr_method, ','.join(['pearsonr', 'spearmanr']))
        )
    nonull = np.logical_and(not_na(array1), not_na(array2))
    if sum(nonull) > 2:
        return eval(corr_method)(array1[nonull], array2[nonull])
    return np.nan, np.nan

