import phosphodisco as phdc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

seed = 5
np.random.seed(seed)
prot = pd.util.testing.makeDataFrame()
phospho = pd.util.testing.makeDataFrame()
phospho.index = pd.MultiIndex.from_tuples(
    [(prot.index[np.random.randint(0, 15)], ind) for ind in phospho.index]
)


def test_classes_regulators():
    proteomics = phdc.ProteomicsData(
        phospho, prot, min_common_values=2
    ).normalize_phospho_by_protein()
    proteomics.assign_modules(
        pd.DataFrame(
            {'test;param-1': [np.random.randint(0, 4) for i in range(30)]},
            index=proteomics.normed_phospho.index
        )
    )

    proteomics.calculate_module_scores()
    proteomics.collect_putative_regulators(list(set(phospho.sample(3).index.get_level_values(
        0))), corr_threshold=0.98)
    proteomics.calculate_regulator_coefficients(cv_fold=2)
    return proteomics


def test_classes_annotations():
    proteomics = phdc.ProteomicsData(
        phospho, prot, min_common_values=2
    ).normalize_phospho_by_protein()
    proteomics.assign_modules(
        pd.DataFrame(
            {'test;param-1': [np.random.randint(0, 4) for i in range(30)]},
            index=proteomics.normed_phospho.index
        )
    )
    annotations = pd.DataFrame(
            {
                'cat1': ['A', 'B', 'A', 'B'],
                'cat2': ['A', 'B', 'B', 'C'],
                'cont1': [0.115, 0.01, 0.3, 0.9],
                'cont2': [-1, -2.5, np.nan, 1]
            },
            index=proteomics.protein.columns
    )
    proteomics.calculate_module_scores()
    proteomics.add_annotations(annotations, pd.Series(['categorical', 0, 'continuous', 1]))
    proteomics.annotation_association()
    print(proteomics.annotation_association_FDR)
    assert False
    return proteomics
