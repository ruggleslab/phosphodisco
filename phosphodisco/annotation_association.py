from typing import Iterable, Tuple, Optional
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, binom, ranksums, f_oneway
from .utils import corr_na


def rho_p(rank_vector):
    """Compares each element in the vector to its corresponding value in the null distribution vector,
    using the probability mass function of the binomial distribution.
    Assigns a p-value to each element in the vector, creating the betaScore vector.
    Uses minimum betaScore as rho

    Args:
        rank_vector:

    Returns:

    """
    rank_vector = rank_vector[~np.isnan(rank_vector)]
    n = len(rank_vector)

    betaScores = rank_vector.copy()
    betaScores[0:n] = np.nan
    sorted_ranks = rank_vector.sort_values().index

    for i, k in enumerate(sorted_ranks):
        x = rank_vector[k]
        betaScore = binom.sf(i, n, x, loc=1)
        betaScores[k] = betaScore
    rho = min(betaScores)
    p = min([rho*n, 1])
    return rho, p


def RRA(a: Iterable, b: Iterable) -> Tuple[float]:
    vec = a.append(b).rank(ascending=False, pct=True)
    return rho_p(vec[0:len(a)])


def one_sided_ttest(a: Iterable, b: Iterable, **test_kws) -> Tuple[float]:
    test_kws['nan_policy'] = 'omit'
    stat, p = ttest_ind(a, b, **test_kws)
    p = p*2
    if stat <= 0:
        p = 1 - p
    return stat, p


def one_sided_rank_sum(a: Iterable, b: Iterable) -> Tuple[float]:
    stat, p = ranksums(a, b)
    p = p * 2
    if stat <= 0:
        p = 1 - p
    return stat, p


categorial_methods = {
    'RRA': RRA,
    'ttest': one_sided_ttest,
    'ranksum': one_sided_rank_sum,
}


def binarize_categorical(annotations: DataFrame, columns: Iterable) -> DataFrame:

    binarized = pd.DataFrame(index=annotations.index)
    for col in columns:
        options = set(annotations[col].dropna().unique())
        for opt in options:
            new_col = '%s.%s' % (col, opt)
            others = options.difference(set([opt]))
            binarized.loc[annotations[col] == opt, new_col] = True
            binarized.loc[annotations[col].isin(others), new_col] = False
    return binarized


def categorical_score_association(
        annotations: DataFrame,
        module_scores: DataFrame,
        cat_method: Optional[str] = None,
        **test_kws
) -> DataFrame:
    if cat_method is None:
        cat_method = 'RRA'
    scores = module_scores.copy()
    results = pd.DataFrame(index=scores.index)

    indname = annotations.index.name
    if indname is None:
        indname = 'index'

    compare_fn = lambda row: categorial_methods[cat_method](
        row[temp[True]], row[temp[False]], **test_kws
    )[1]
    for col in annotations.columns:
        temp = annotations[col].reset_index()
        temp = temp.groupby(col)[indname].apply(list)
        results[col] = scores.apply(
            compare_fn,
            axis=1
        )
    return results


def continuous_score_association(
        annotations: DataFrame,
        module_scores: DataFrame,
        cont_method: Optional[str] = None
):
    if cont_method is None:
        cont_method = 'pearsonr'

    scores = module_scores.reindex(annotations.index, axis=1)
    results = pd.DataFrame(index=scores.index)
    for col in annotations.columns:
        results[col] = scores.apply(
            lambda row: corr_na(annotations[col], row, corr_method=cont_method)[1],
            axis=1
        )
    return results



