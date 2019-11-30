import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from datasets import DataSet
from utilities.paramutils import combine_options
from utilities.data_transformers import column_filter, standardize_subject_ids
from sklearn.linear_model import LinearRegression, Ridge, LassoLars, Lasso
from runner import execute
from datetime import date

from joblib import dump, load


class PCAAutoDataSet(DataSet):
    def __init__(self, male_file, female_file):
        self.male_file = male_file
        self.female_file = female_file
        super().__init__()

    def load_data(self):
        print('Loading PCAAutoDataSet')
        pca_male = pd.read_csv(self.male_file)  # Meant to be ran from PCA_App folder
        pca_male['SubjectID'] = pca_male['Unnamed: 0']
        pca_female = pd.read_csv(self.female_file)
        pca_female['SubjectID'] = pca_female['Unnamed: 0']

        combined_df = super().common_dataframes().merge(pd.concat([pca_male, pca_female], sort=False), on='SubjectID', how='inner')

        combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])

        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")
        return combined_df


class PCA_ml:
    def __init__(self, male_file, female_file):
        dataset = PCAAutoDataSet(male_file, female_file)
        PCs = [i for i in dataset.load_data().columns if i.startswith('PC')]
        n = len(PCs)
        #numbers = list(range(1, n + 1))


        loader_params = dict(
            scaler_config={"SEX": LabelBinarizer, "default": MinMaxScaler}
        )
        loader_param_grid = dict(
            data_transformers={
                "__cols": "SEX",
                "M": [column_filter("SEX", "M")],
                "F": [column_filter("SEX", "F")],
                "M/F": [],
            },
            feature_cnames=combine_options(additional_options=["SEX"], input={
                "Principal Components": {
                    f'{n}': [f'PC{i}' for i in range(1, n + 1)]
                }
            })
        )

        # regressor = LinearRegression(fit_intercept=True, normalize=True)
        regressor = MLPRegressor(batch_size=10,
                     solver="lbfgs",
                     activation="identity",
                     max_iter=800,
                     hidden_layer_sizes=(1,))
        regressor_parameter_grid = dict(
        )

        save_name = f"RegTrial_{type(dataset).__name__}_{type(regressor).__name__}"
        self.results = execute(dataset,
                               loader_params,
                               loader_param_grid,
                               MLPRegressor(batch_size=10,
                                             solver="lbfgs",
                                             activation="identity",
                                             max_iter=800,
                                             hidden_layer_sizes=(1,)),
                               regressor_parameter_grid,
                               targets="TOTAL_FAT",
                               cv=3,
                               eval_type='regressor',
                               multicore=False
                               )
