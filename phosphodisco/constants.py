import scipy.stats
from sklearn import linear_model

__doc__="Methods"


reg_models = {
    'linear':linear_model.RidgeCV,
    'sigmoid':linear_model.LogisticRegressionCV
}


continuous_methods = {
    'pearsonr':scipy.stats.pearsonr,
    'spearman':scipy.stats.spearmanr
}
