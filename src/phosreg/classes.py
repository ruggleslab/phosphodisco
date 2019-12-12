from pandas import DataFrame, Series
import logging
import warnings
from typing import Iterable, Optional
from sklearn.linear_model import RidgeCV

datatype_label = 'datatype_label'


def norm_line_to_residuals(
        ph_line: Iterable,
        prot_line: Iterable,
        alphas: Optional[Iterable] = [2 ** i for i in range(-10, 10, 1)],
        cv: Optional[int] = 5
) -> Series:

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    nonull = (ph_line.notnull() & prot_line.not_null())
    if sum(nonull) < cv:
        return np.empty(len(ph_line))

    features = prot_line[nonull].values.reshape(-1, 1)
    labels = ph_line[nonull].values
    model = RidgeCV(alphas=alphas, cv=cv).fit(features, labels)
    prediction = model.predict(features)
    residuals = labels - prediction

    return pd.Series(residuals, index=ph_line[nonull].index)


class ProteomicsData:

    def __init__(
            self, phospho: DataFrame, protein: DataFrame, min_common_values: Optional[int] = 5
    ):
        self.min_values_in_common = min_common_values

        common_prots = phospho.index.get_level_values(0).intersection(protein.index)
        common_samples = phospho.columns.intersection(protein.columns)
        self.phospho = phospho[common_samples]
        self.protein = protein[common_samples]

        logging.INFO('Phospho and protein data have %s proteins in common' % len(common_prots))
        logging.INFO('Phospho and protein data have %s samples in common, reindexed to only '
                     'common samples.' %
                     len(common_samples))

        normalizable_rows = ((phospho.loc[common_prots, common_samples].notnull() &
                             protein.loc[common_prots, common_samples].notnull()).sum(axis=1)
                             > min_common_values)
        self.normalizable_rows = normalizable_rows
        logging.INFO('There are %s rows with at least %s non-null values in both phospho and '
                     'protein' % (normalizable_rows, len(normalizable_rows)))


    def normalize_phospho_by_protein(self, ridge_cv_alphas: Optional[Iterable]):
        #TODO test this, make sure index 0 and 1 goes away.
        target = self.phopsho.loc[self.normalizable_rows]
        features = self.prot.loc[target.index.get_index_values(0)]

        target = target.transpose()
        features = features.transpose()

        target[datatype_label] = 0
        features[datatype_label] = 1
        data = target.append(features).transpose()

        residuals = data.apply(
            lambda row: norm_line_to_residuals(row[0], row[1], ridge_cv_alphas, min_values_in_common)
        )

        self.normed_phospho = residuals


class Clusters:
    def __init__(self, cluster_labels: Series, abundances: DataFrame, parameters: dict):
        self.cluster_labels = cluster_labels
        self.parameters = parameters
        self.abundances = abundances
        self.nmembers_per_cluster = cluster_labels.value_counts()

    def calculate_cluster_scores(
            self,
            combine_anti_regulated: bool=True,
            anti_corr_threshold: float=0.9
    ):
        abundances = self.abundances.reindex(self.cluster_labels.index)
        scores = abundances.groupby(self.cluster_labels).agg(mean)
        scores.corr()

        self.cluster_scores = scores

# class MyClass:
#     # You can optionally declare instance variables in the class body
#     attr: int
#     # This is an instance variable with a default value
#     charge_percent: int = 100
#
#     # The "__init__" method doesn't return anything, so it gets return
#     # type "None" just like any other method that doesn't return anything
#     def __init__(self) -> None:
#         ...
#
#     # For instance methods, omit type for "self"
#     def my_method(self, num: int, str1: str) -> str:
#         return num * str1
#
# # User-defined classes are valid as types in annotations
# x: MyClass = MyClass()
#
# # You can use the ClassVar annotation to declare a class variable
# class Car:
#     seats: ClassVar[int] = 4
#     passengers: ClassVar[List[str]]
#
# # You can also declare the type of an attribute in "__init__"
# class Box:
#     def __init__(self) -> None:
#         self.items: List[str] = []
