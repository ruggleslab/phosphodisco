import warnings
from typing import Iterable, Optional
from pandas import Series
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


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


# Adapted from https://github.com/alexmill/alexmill.github.io/tree/91be34d6fafa90cf5d78e7f934328a060c8a70c0/_site/posts/linear-model-custom-loss-function-regularization-python
class SigmoidRegression:
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization
    """

    def __init__(self, loss_function=mean_squared_error,
                 X=None, Y=None, sample_weight=None, coef_init=None,
                 regularization=0.00012, cv=5, alphas=None):
        if alphas is None:
            alphas = [2 ** i for i in range(-10, 10, 1)]
        self.alphas = alphas
        self.cv = cv
        self.regularization = regularization
        self.coef_ = None
        self.loss_function = loss_function
        self.sample_weight = sample_weight
        self.coef_init = coef_init

        self.X = X
        self.Y = Y

    def predict(self, X):
        prediction = np.matmul((1/np.exp(1+X)), self.coef_)
        return (prediction)

    def model_error(self):
        error = self.loss_function(
            self.Y, self.predict(self.X), sample_weight=self.sample_weight
        )
        return error

    def l2_regularized_loss(self, coef_):
        self.coef_ = coef_
        return self.model_error() + sum(self.regularization * np.array(self.coef_) ** 2)

    def fit_no_cv(self, maxiter=250):
        # Initialize coef estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.coef_init) == type(None):
            # set coef_init = 1 for every feature
            self.coef_init = np.array([1] * self.X.shape[1])
        else:
            # Use provided initial values
            pass

        if self.coef_ is not None and all(self.coef_init == self.coef_):
            print("Model already fit once; continuing fit with more itrations.")

        res = minimize(self.l2_regularized_loss, self.coef_init,
                       method='BFGS', options={'maxiter': maxiter})
        self.coef_ = res.x
        self.coef_init = self.coef_


class CustomCrossValidator:
    """
    Cross validates arbitrary model using MAPE criterion on
    list of alphas.
    """

    def __init__(self, X, Y, model_class,
                 sample_weight=None,
                 loss_function=mean_squared_error):

        self.X = X
        self.Y = Y
        self.model_class = model_class
        self.loss_function = loss_function
        self.sample_weight = sample_weight

    def cross_validate(self, alphas, num_folds=10):
        """
        alphas: set of regularization parameters to try
        num_folds: number of folds to cross-validate against
        """

        self.alphas = alphas
        self.cv_scores = []
        X = self.X
        Y = self.Y

        # coef_ values are not likely to differ dramatically
        # between differnt folds. Keeping track of the estimated
        # coef_ coefficients and passing them as starting values
        # to the .fit() operator on our model class can significantly
        # lower the time it takes for the minimize() function to run
        coef_init = None

        for lam in self.alphas:

            # Split data into training/holdout sets
            kf = KFold(n_splits=num_folds, shuffle=True)
            kf.get_n_splits(X)

            # Keep track of the error for each holdout fold
            k_fold_scores = []

            # Iterate over folds, using k-1 folds for training
            # and the k-th fold for validation
            f = 1
            for train_index, test_index in kf.split(X):
                # Training data
                CV_X = X[train_index, :]
                CV_Y = Y[train_index]
                CV_weights = None
                if type(self.sample_weight) != type(None):
                    CV_weights = self.sample_weight[train_index]

                # Holdout data
                holdout_X = X[test_index, :]
                holdout_Y = Y[test_index]
                holdout_weights = None
                if type(self.sample_weight) != type(None):
                    holdout_weights = self.sample_weight[test_index]

                # Fit model to training sample
                alpha_fold_model = self.model_class(
                    regularization=lam,
                    X=CV_X,
                    Y=CV_Y,
                    sample_weight=CV_weights,
                    coef_init=coef_init,
                    loss_function=self.loss_function
                )
                alpha_fold_model.fit_no_cv()

                # Extract coef values to pass as coef_init
                # to speed up estimation of the next fold
                coef_init = alpha_fold_model.coef_

                # Calculate holdout error
                fold_preds = alpha_fold_model.predict(holdout_X)
                fold_mape = self.loss_function(
                    holdout_Y, fold_preds, sample_weight=holdout_weights
                )
                k_fold_scores.append(fold_mape)
                f += 1

            # Error associated with each alpha is the average
            # of the errors across the k folds
            alpha_scores = np.mean(k_fold_scores)
            self.cv_scores.append(alpha_scores)

        # Optimal alpha is that which minimizes the cross-validation error
        self.alpha_star_index = np.argmin(self.cv_scores)
        self.alpha_star = self.alphas[self.alpha_star_index]
        return self


class SigmoidCV (SigmoidRegression):
    def __init__(self, loss_function=mean_squared_error, sample_weight=None,
                 coef_init=None, regularization=0.00012, cv=5, alphas=None):
        super().__init__(loss_function, sample_weight, coef_init, regularization, cv, alphas)
        if alphas is None:
            alphas = [2 ** i for i in range(-10, 10, 1)]
        self.alphas = alphas
        self.cv = cv
        self.regularization = regularization
        self.coef_ = None
        self.loss_function = loss_function
        self.sample_weight = sample_weight
        self.coef_init = coef_init

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        cross_validator = CustomCrossValidator(
            X, Y, SigmoidRegression,
            loss_function=self.loss_function
        )
        cross_validator.cross_validate(self.alphas, num_folds=self.cv)
        alpha_star = cross_validator.alpha_star

        self.regularization = alpha_star
        self.fit_no_cv()
