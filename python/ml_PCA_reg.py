from sklearn.preprocessing import LabelBinarizer
from DataSets import PCADataSet
from utilities.data_transformers import column_filter
from sklearn.linear_model import LinearRegression
from ml_core import run_batch

numbers = list(range(1, 100))

dataset = PCADataSet()
data_config_dict = {
    'target_cnames': ['TOTAL_FAT', 'TOTAL_LEAN'],
    'feature_options': {
        "Principal Components": {f'{n}': [f'PC{i}' for i in range(1, n + 1)] for n in numbers}
    },
    'transform_options': {
        'SEX': {
            "M": [column_filter("SEX", "M")],
            "F": [column_filter("SEX", "F")]
        }
    },
    'scalar_config': {
    }
}

regressor = LinearRegression()
regressor_parameter_grid = dict(
)

results = run_batch(dataset,
                    data_config_dict,
                    regressor,
                    show_best_runs=3,
                    cv_params=dict(n_splits=5)
)
print(results)