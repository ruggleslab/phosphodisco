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
        prevent_negative_coefficients: bool = True,
        **ridgecv_kwargs
) -> Series:
    """Uses CV and regularized linear regression to calculate residuals, representing
    protein-normalized phospho values for one line of data each.

    Args:
        ph_line: Vector of phosphorylation data.
        prot_line: Vector of protein data.
        regularization_values: Which regularization values should be tried during CV to define
        the coefficients.
        cv: The number of cross validation folds to try for calculating the regularization
        value.
        prevent_negative_coefficients: If the linear coefficient between protein and phospho
        values is negative, something complicated is going on in that relationship. Set this to
        True to just return missing values in that case.
        **ridgecv_kwargs: Additional keywork args for sklearn.linear_model.RidgeCV

    Returns: Series of residuals, representing protein abundance-normalized phospho data.

    """
    if regularization_values is None:
        regularization_values = [5 ** i for i in range(-5, 5)]
    if cv is None:
        cv = 3

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    nonull = np.logical_and(~np.isnan(ph_line), ~np.isnan(prot_line))
    if sum(nonull) < cv:
        return pd.Series(np.empty(len(ph_line)), ph_line.index)

    features = prot_line[nonull].values.reshape(-1, 1)
    labels = ph_line[nonull].values

    ridgecv_kwargs['alphas'] = regularization_values
    ridgecv_kwargs['cv'] = cv
    model = RidgeCV(**ridgecv_kwargs).fit(features, labels)
    if prevent_negative_coefficients and (model.coef_[0] <= 0):
        return pd.Series(np.empty(len(ph_line)), ph_line.index)

    prediction = model.predict(features)
    residuals = labels - prediction

    return pd.Series(residuals, index=ph_line[nonull].index)


def multiple_tests_na(pvals: np.array, **multitest_kwargs):
    """Performs statsmodels.stats.multitest.multipletests with tolerance for np.nans

    Args:
        pvals: Vector of p-values to correct
        **multitest_kwargs: Additional keyword args for statsmodels.stats.multitest.multipletests

    Returns: Vector of corrected p-values.

    """
    mask = np.isfinite(pvals)
    pval_corrected = np.full(pvals.shape, np.nan)
    pval_corrected[mask] = multipletests(pvals[mask], **multitest_kwargs)[1]
    return pval_corrected


def not_na(array):
    """Identifies non-null values for pd.Series or np.array

    Args:
        array: Vector of values

    Returns: Vector of which values are non-null.

    """
    if isinstance(array, Series):
        return ~array.isna()
    return ~np.isnan(array)


def corr_na(array1, array2, corr_method: str = 'spearmanr', **addl_kws):
    """Correlation method that tolerates missing values. Can take pearsonr or spearmanr.

    Args:
        array1: Vector of values
        array2: Vector of values
        corr_method: Which method to use, pearsonr or spearmanr.
        **addl_kws: Additional keyword args to pass to scipy.stats corr methods.

    Returns: R and p-value from correlation of 2 vectors.

    """
    if corr_method not in ['pearsonr', 'spearmanr']:
        raise ValueError(
            'Method %s is a valid correlation method, must be: %s'
            % (corr_method, ','.join(['pearsonr', 'spearmanr']))
        )
    nonull = np.logical_and(not_na(array1), not_na(array2))
    if sum(nonull) > 2:
        return eval(corr_method)(array1[nonull], array2[nonull], **addl_kws)
    return np.nan, np.nan


def zscore(df):
    """Row zscores a DataFrame, ignores np.nan

    Args:
        df (DataFrame): DataFrame to z-score

    Returns (DataFrame):
        Row-zscored DataFrame.
    """
    return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)
