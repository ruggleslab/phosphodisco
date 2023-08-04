import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Iterable, Optional
from sklearn import linear_model, preprocessing
from .utils import corr_na, zscore


def collapse_possible_regulators(
    reg_data: DataFrame, corr_threshold: float = 0.95
) -> DataFrame:
    """Uses mean to collapse rows of possible regulator data that are highly correlated. Slightly
    chaotic, since it just averages two at a time based on iterating through a dictionary. Use
    with caution.

    Args:
        reg_data: DataFrame with possible regulator features as rows, samples as columns.
        corr_threshold: Rows with pearson correlation higher than this value will be averaged
        iteratively until there are no more rows with more than this correlation.

    Returns: DataFrame with rows that do not have pairwise correlation above the corr_threshold.

    """
    reg_data = zscore(reg_data)
    corr = reg_data.transpose().corr()

    high_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) > 1)]
    low_corr_inds = corr.index.difference(high_corr_inds)
    high_corr_data = reg_data.loc[high_corr_inds, :]
    low_corr_data = reg_data.loc[low_corr_inds, :]
    if len(high_corr_inds) == 0:
        return low_corr_data
    corr = (
        corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool))
        .mask(~(corr > corr_threshold))
        .dropna(how="all")
        .dropna(how="all", axis=1)
    )
    corr = corr.stack(level=[0, 1])

    while corr.shape[0] > 0:
        for i in corr.index:
            a, b, c, d = i
            if (a, b) in high_corr_data.index and (c, d) in high_corr_data.index:
                inds_to_mean = [(a, b), (c, d)]

                others_ab = [
                    (d, e, f, g)
                    for d, e, f, g in corr.index
                    if (d, e) is (a, b) or (f, g) is (a, b)
                ]
                others_ab = [
                    (f, g) if (d, e) == (a, b) else (d, e) for d, e, f, g in others_ab
                ]
                inds_to_mean.extend(
                    [
                        (e, f)
                        for e, f in others_ab
                        if ((e, f, c, d) in high_corr_data.index)
                        or ((c, d, e, f) in high_corr_data.index)
                    ]
                )

                name = ("%s-%s" % (a, c), "%s-%s" % (b, d))
                high_corr_data = high_corr_data.append(
                    pd.Series(high_corr_data.loc[inds_to_mean, :].mean(), name=name)
                )
                high_corr_data = high_corr_data.drop(inds_to_mean, axis=0)
        corr = high_corr_data.transpose().corr()
        high_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) > 1)]
        low_corr_inds = corr.index[((corr > corr_threshold).sum(axis=1) <= 1)]

        low_corr_data = low_corr_data.append(high_corr_data.loc[low_corr_inds, :])
        if len(high_corr_inds) == 0:
            return low_corr_data
        high_corr_data = high_corr_data.loc[high_corr_inds, :]
        corr = (
            corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool))
            .mask(~(corr > corr_threshold))
            .dropna(how="all")
            .dropna(how="all", axis=1)
        )
        corr = corr.stack(level=[0, 1])

    return low_corr_data


def calculate_regulator_coefficients(
    reg_data: DataFrame,
    module_scores: DataFrame,
    scale_data: bool = True,
    model: str = "linear",
    regularization_values: Optional[Iterable] = None,
    cv_fold: int = 5,
    **model_kwargs,
) -> DataFrame:
    """Calculates linear model coefficients between regulator data and module scores.

    Args:
        reg_data: DataFrame with possible regulator features as rows, samples as columns.
        module_scores: DataFrame with module scores as rows, samples as columns.
        scale_data: Whether to scale the regulator data and module scores with
        sklearn.preprocess.scale before training the linear model.
        model: Whether the relationship between log2(regulator_data) and module scores should be
        modeled as linear or sigmoid.
        regularization_values: Which regularization values should be tried during CV to define
        the coefficients.
        cv_fold: The number of cross validation folds to try for calculating the regularization
        value.
        **model_kwargs: Additional keyword args for sklearn.linear_model.RidgeCV

    Returns: The first object is a DataFrame with module scores as rows and regulators as columns,
    and model coefficients as values. The second is a Series with module scores as rows and the
    over all model quality score as values.

    """
    if regularization_values is None:
        regularization_values = [5**i for i in range(-5, 5)]

    if model not in ["linear", "sigmoid"]:
        raise ValueError(
            "Model %s not in accepted models: %s"
            % (model, ",".join(["linear", "sigmoid"]))
        )

    features = reg_data.transpose().values
    targets = module_scores.transpose().values
    if model == "sigmoid":
        targets = -np.log2(1 + (2**-targets))
    if scale_data:
        features = preprocessing.scale(features, copy=True)
        targets = preprocessing.scale(targets, copy=True)

    model_kwargs.update({"cv": cv_fold, "alphas": regularization_values})
    model = linear_model.RidgeCV(**model_kwargs)
    model.fit(features, targets)
    weights = pd.DataFrame(
        model.coef_, index=module_scores.index, columns=reg_data.index
    ).transpose()
    scores = pd.Series(
        model.score(features, targets),
        index=module_scores.index,
    )
    return weights, scores


def calculate_regulator_corr(
    reg_data: DataFrame, module_scores: DataFrame, **model_kwargs
):
    """Calculates the correlation between possible regulators and module scores.

    Args:
        reg_data: DataFrame with possible regulator features as rows, samples as columns.
        module_scores: DataFrame with module scores as rows, samples as columns.
        **model_kwargs: Additional keyword args for corr_na, including method which can take
        either pearsonr or spearmanr.

    Returns: Two DataFrames, both with module scores as rows and possible regulators as columns.
    The first one has correlation R values, the second has correlation p values.

    """
    rs = pd.DataFrame(index=reg_data.index)
    ps = pd.DataFrame(index=reg_data.index)
    for i, row in module_scores.iterrows():
        res = reg_data.apply(
            lambda r: pd.Series(corr_na(row, r, **model_kwargs)), axis=1
        )
        rs[i] = res.iloc[:, 0]
        ps[i] = res.iloc[:, 1]
    return rs, ps
