import pandas as pd, numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from DataSets import DataSet
from utilities.paramutils import combine_options
from utilities.data_transformers import column_filter, standardize_subject_ids, cut_subject_ids
from utilities.archive.runner import execute
from datetime import date


class PCAAutoDataSet(DataSet):
    def __init__(self, male_file, female_file):
        self.male_file = male_file
        self.female_file = female_file
        super().__init__()

    def load_data(self, process_dir=None):
        # print('Loading PCAAutoDataSet')
        print(self.male_file)
        pca_male = pd.read_csv(self.male_file)  # Meant to be ran from PCA_App folder
        pca_male['SubjectID'] = cut_subject_ids(pca_male['Unnamed: 0'])
        pca_male['SEX0'] = 'M'
        pca_female = pd.read_csv(self.female_file)
        pca_female['SubjectID'] = cut_subject_ids(pca_female['Unnamed: 0'])
        pca_female['SEX0'] = 'F'

        combined_df = DataSet().common_dataframes().merge(pd.concat([pca_male, pca_female], sort=False), on=['SubjectID'], how='outer')
        combined_df.dropna(subset=['PC1'], inplace=True)
        # combined_df['SEX'] = combined_df.apply(lambda x: x['SEX'] if x['SEX'] else x['SEX0'], axis=1)
        combined_df['SEX'] = combined_df['SEX0']

        combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])

        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")

        for col in combined_df.columns:
            if col.startswith('PC'):
                combined_df[col] = combined_df[col].fillna(0)
        # combined_df = combined_df[pd.notnull(combined_df['SEX'])]
        if process_dir is not None:
            combined_df.to_csv(process_dir + 'pca_ds.csv')
        return combined_df


class PCA_ml:
    def __init__(self, male_file, female_file, max_PCs = 10):
        dataset = PCAAutoDataSet(male_file, female_file)

        PCs = [i for i in dataset.load_data().columns if i.startswith('PC')]
        n = np.min([len(PCs), max_PCs])
        #numbers = list(range(1, n + 1))


        loader_params = dict(
            scaler_config={"SEX": LabelBinarizer, "default": MinMaxScaler}
        )
        loader_param_grid = dict(
            data_transformers={
                "__cols": "SEX",
                "M": [column_filter("SEX", "M")],
                "F": [column_filter("SEX", "F")],
                # "M/F": [],
            },
            feature_cnames=combine_options(additional_options=["SEX"], input={
                "PCs": {f'{m}': [f'PC{i}' for i in range(1, m + 1)] for m in range(1, n + 1)}
            })
        )

        # regressor = LinearRegression()
        regressor = MLPRegressor(batch_size=5,
                     solver="lbfgs",
                     activation="relu",
                     max_iter=35,
                     hidden_layer_sizes=(10, 5))
        regressor_parameter_grid = dict(
        )

        save_name = f"RegTrial_{type(dataset).__name__}_{type(regressor).__name__}"
        self.results = execute(dataset,
                               loader_params,
                               loader_param_grid,
                               regressor,
                               regressor_parameter_grid,
                               targets=["TOTAL_FAT", "TOTAL_LEAN"],
                               cv=5,
                               eval_type='regressor',
                               multicore=False,
                               export_best=1
                               )
