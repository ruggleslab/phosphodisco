from .classes import Clusters
from .nominate_regulators import corrNA
from typing import Iterable, Tuple
from pandas import DataFrame
import pandas as pd
import scipy.stats


def rho_p(rank_vector):
    """
    Compares each element in the vector to its corresponding value in the null distribution vector,
    using the probability mass function of the binomial distribution.
    Assigns a p-value to each element in the vector, creating the betaScore vector.
    Uses minimum betaScore as rho
    """
    rank_vector = rank_vector.dropna()
    n = len(rank_vector)

    betaScores = rank_vector.copy(deep=True)
    betaScores[0:n] = np.nan
    sorted_ranks = rank_vector.dropna().sort_values().index

    for i, k in enumerate(sorted_ranks):
        x = rank_vector[k]
        betaScore = stats.binom.sf(i, n, x, loc=1)
        betaScores[k] = betaScore
    rho = min(betaScores)
    p = min([rho*n, 1])

    return (rho, p)


def RRA(a: Iterable, b: Iterable) -> Tuple[float]:
    vec = pd.Series(list(a) + list(b)).rank(pct=True)
    return rho_p(vec)


def binarize_categorical(annotations: DataFrame, columns: Iterable) -> DataFrame:

    binarized = pd.DataFrame(index=annotations.index)
    for col in columns:
        options = set(annotations[col].value_counts().keys())
        for opt in options:
            new_col = '%s.%s' % (col, opt)
            others = options.difference(set([opt]))
            binarized.loc[annotations[col] == opt, new_col] = True
            binarized.loc[annotations[col].isin(others), new_col] = False
    return binarized


categorial_methods = {
    'RRA': RRA,
    'ttest': scipy.stats.ttest_ind,
    'ranksum': scipy.stats.ranksums
}


def categorial_score_association(
        annotations: DataFrame,
        clusters: Clusters,
        cat_method: str = 'RRA'
) -> DataFrame:

    scores = clusters.cluster_scores.transpose()
    results = pd.DataFrame(index=scores.index)

    indname = annotations.index.name
    temp = annotations.reset_index()
    for col in annotations.columns:
        temp.groupby(col)[indname].apply(list).to_dict()
        results[col] = scores.apply(
            lambda row: categorial_methods[cat_method](row[True], row[False])[1]
        )
    return results


def continuous_score_association(
        annotations: DataFrame,
        clusters: Clusters,
        corr_method: str = 'pearsonr'
):
    scores = clusters.cluster_scores.transpose()
    results = pd.DataFrame(index=scores.index)
    for col in annotations.columns:
        results[col] = scores.apply(
            lambda row: corrNA(annotations[col], row, corr_method=corr_method)[1]
        )
    return results



