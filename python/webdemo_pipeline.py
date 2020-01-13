import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

from utilities.data_transformers import cut_subject_ids, column_filter
from DataSets import SS20DataSet, to_DataSet
from ml_core import run_batch


'''
Create DataSet from CSV
    Here we import a ShapeUp excel file into a pandas DataFrame and use to_DataSet to combine the digital anthropometric
data with DXA and blood work target data in an instance of the DataSet class. Typically DataSets are imported from 
DataSets.py, but in this file we have chosen to illustrate the pipeline starting directly from the output of the MATLAB 
code (an 'ObjOrganizer' xlsx spreadsheet).
'''
df = pd.read_excel('data/ShapeUp/ObjOrganizerStyku_v14.xlsx')
df = df.drop(df.columns[[0, 1, 3, 4]], axis=1)  # remove unnecessary label columns
df.rename(columns={col: f"{col.replace('/', '_')}" for col in df.columns}, inplace=True)  # stardardize styku column names
StykuDataset = to_DataSet(df, combine_common=True, subject_cname='Name')
# print(StykuDataset.df)  # print the full DataSet as a pandas DataFrame


'''
Specify Data Extraction Options
    Each instance of the DataSet class has an extract_data() method that is used to finalize the stages of 
preprocessing. This method converts the raw data found in DataSet.df to pruned and standardized input, ready to be 
passed to our ML model. The options for extract_data are given in the data_config_dict.
'''
data_config_dict = {
    # dictionary containing information on how our DataSet should be extracted and scaled.
    'target_cnames': ['TOTAL_FAT', 'TOTAL_LEAN'],
    'feature_options': {
        # collections of features to preform a grid search over
        "bmi": {
            "N": [],
            "Y": ["BMI1"]
        },
        "age": {
            # "N": [],
            "Y": ["age"]
        },
        "sex": {
            # "N": [],
            "Y": ["SEX"],
        },
        "volumes": {
            # "N": [],
            "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
                  "trunkVolume"]
        },
        "measurements": {
            "none": [],
            "common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
            # "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
            #         "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
            #         "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
            #         "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength",
            #         "crotchHeight"],
            # "rAnkle" and "Lankle" are ankle girths
        },
        "a_b": {
            "none": [],
            # "four": ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"],
            "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
                    "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
                    "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
                    "rWristGirth A_B", "Lankle A_B"]
        }
    },
    'transform_options': {
        # categorical features columns to split our dataset over for seperate runs
        'SEX': {
            "M": [column_filter("SEX", "M")],
            "F": [column_filter("SEX", "F")],
            "M/F": [],
        }
    },
    'scalar_config': {
        # column names and their corresponding scalars/encoders
        "SEX": LabelBinarizer,
        "age": MinMaxScaler,
        "default": StandardScaler
    }
}


'''
Configure your local pandas options to ensure proper display of results
'''
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


'''
Example Run 1: Using a list of regressors
    By default, only one example run in this file should be uncommented. Tip: to (un)comment lines in bulk using 
PyCharm, highlight the desired lines of code an press ctrl + '/'.
'''
print('EXAMPLE RUN 1: Using a list of regressors')

regressors = [
    LinearRegression(),
    Lasso(alpha=.001),
    MLPRegressor(solver="lbfgs",
        activation="identity",
        max_iter=800,
        hidden_layer_sizes=(1,))
]

results = run_batch(
    # pass required arguments
    StykuDataset,
    data_config_dict,
    regressors,
    # configure internal parameters for n-fold cross-validation and scoring
    cv_params=dict(
        n_splits=5,
        scorers=['r2', 'explained_variance']  # options: https://scikit-learn.org/stable/modules/model_evaluation.html
    ),
    # print the best 3 runs (sorted by test_r2) for each target and sex option
    show_best_runs=3
)
print('[RESULTS]\n', results)


'''
Example Run 2: Using a single regressor with a hyper-parameter grid search
'''
# print('EXAMPLE RUN 2: Using a single regressor with a hyper-parameter grid search')
#
# regressor = MLPRegressor
# regressor_param_dict = dict(
#     solver=["adam", "lbfgs"],
#     activation=["identity", "tanh"],
#     max_iter=[800],
#     hidden_layer_sizes=[(1,), (15, 5, 3)]
# )
#
# results = run_batch(
#     StykuDataset,
#     data_config_dict,
#     regressor,
#     regressor_param_dict,
#     show_best_runs=3
# )
# print('[RESULTS]\n', results)