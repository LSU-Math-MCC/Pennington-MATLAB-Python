import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from DataSets import DataSet
from utilities.paramutils import combine_options
from utilities.data_transformers import column_filter, standardize_subject_ids
from sklearn.linear_model import LinearRegression, Ridge, LassoLars, Lasso
from runner import execute
from datetime import date

from joblib import dump, load


class PCADataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        pca_male = pd.read_excel('data/Shapeup_Q2_Data_190415.xlsx', sheet_name='PC_Weights_Male')
        pca_female = pd.read_excel('data/Shapeup_Q2_Data_190415.xlsx', sheet_name='PC_Weights_Female')

        combined_df = super().common_dataframes().merge(pd.concat([pca_male, pca_female], sort=False), on='SubjectID', how='inner')

        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")
        combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])
        return combined_df


#numbers = [5, 10, 25, 100, 150]
numbers = list(range(1, 100))

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
        #"bmi": {"N": [], "Y": ["BMI1"]},
        "Principal Components": {f'{n}': [f'PC{i}' for i in range(1, n + 1)] for n in numbers}
    })
)

#regressor = LinearRegression(fit_intercept=True, normalize=True)
regressor = LinearRegression()
regressor_parameter_grid = dict(
)

dataset = PCADataSet()
save_name = f"RegTrial_{type(dataset).__name__}_{type(regressor).__name__}"
results = execute(dataset,
                  loader_params,
                  loader_param_grid,
                  [Ridge(alpha=0.1),
                   MLPRegressor(batch_size=10,
                                solver="lbfgs",
                                activation="identity",
                                max_iter=800,
                                hidden_layer_sizes=(1,))],
                  regressor_parameter_grid,
                  targets=['TOTAL_LEAN', 'TOTAL_PFAT', 'TOTAL_PLEAN'],
                  save_df_report=False,
                  cv=4,
                  eval_type='regressor',
                  export_best=1
                  )
print(f"[REPORT] Writing output to ./reports/{save_name}.xlsx")
results.to_excel(f"reports/{save_name}.xlsx", index=False)

print(results)

#def export_reg(reg, training_set, target):


'''
dataset = PCADataSet()
targets = ["TOTAL_FAT", "TOTAL_LEAN", "GLU"]
df = pd.DataFrame()
for target in targets:
    loader_params['label_cname'] = target
    results = execute(dataset,
                      loader_params,
                      loader_param_grid,
                      regressor,
                      regressor_parameter_grid,
                      save_df_report=False,
                      cv=4,
                      eval_type='regressor'
                      )
    results.insert(0, 'target', target)
    results['regressor'] = type(regressor).__name__
    results['regressor full'] = regressor
    #results['regressor params'] = regressor.__dict__
    df = pd.concat([df, results])
print(f"[REPORT] Writing output to ./reports/{save_name}.xlsx")
df.to_excel(f"reports/{save_name}.xlsx", index=False)

print(df)

def update_excel(df_dir, results):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')[2:].astype('int')
    print(timestamp)
    df = pd.read_excel(df_dir)
    df = pd.concat([df, results])

update_excel()
'''