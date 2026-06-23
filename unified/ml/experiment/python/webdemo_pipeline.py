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
data with DXA and blood work target data in an instance of the DataSet class. 
Note: We have cleaned up the data and typically DataSets are imported from DataSets.py, So you DON'T need to do this 
step for OUR datasets! However, in this file we have illustrated the pipeline starting directly from the output of the 
MATLAB code (an 'ObjOrganizer' xlsx spreadsheet) so the developer learns how to create Datasets if a researcher uploads 
a new set of data.
'''
df = pd.read_excel('data/ShapeUp/AllStyku_v6.xlsx')
# df = df.loc[[not(sub in ['02ADL0178','02ADL0129','02ADL0205']) for sub in df['SubjectID']]]  #'02ADL0176'


'''
Specify Data Extraction Options
    Each instance of the DataSet class has an extract_data() method that is used to finalize the stages of 
preprocessing. This method converts the raw data found in DataSet.df to pruned and standardized input, ready to be 
passed to our ML model. The options for extract_data are given in the data_config_dict.
'''
eth = ['DEM_RACE']
bmi = ['CA_BMI']
sex = ['DEM_SEX']
age = ['DEM_AGE']
volumesLR = ['VOL_3DO3_TOT', 'VOL_3DO3_Arm_R', 'VOL_3DO3_Arm_L', 'VOL_3DO3_Leg_R', 'VOL_3DO3_Leg_L', 'VOL_3DO3_Trunk']
m_commonLR = ['DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th_R', 'DA_3DO3_CIRC_B_R']
m_allLR = ['DA_3DO3_CIRC_Ch', 'DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th_R', 'DA_3DO3_CIRC_Th_L', 'DA_3DO3_CIRC_C_R', 'DA_3DO3_CIRC_C_L', 'DA_3DO3_CIRC_Wr_R', 'DA_3DO3_CIRC_Wr_L', 'DA_3DO3_CIRC_F_R', 'DA_3DO3_CIRC_F_L', 'DA_3DO3_CIRC_B_R', 'DA_3DO3_CIRC_B_L', 'DA_3DO3_CIRC_A_R', 'DA_3DO3_CIRC_A_L', 'DA_3DO3_LEN_Arm_L', 'DA_3DO3_LEN_Arm_R', 'DA_3DO3_LEN_Leg_L', 'DA_3DO3_LEN_Leg_R']
# ER_common = ['DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_B']
ER_allLR = ['DA_3DO3_ER_Ch', 'DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th_R', 'DA_3DO3_ER_Th_L', 'DA_3DO3_ER_C_R', 'DA_3DO3_ER_C_L', 'DA_3DO3_ER_Wr_L', 'DA_3DO3_ER_F_R', 'DA_3DO3_ER_F_L', 'DA_3DO3_ER_B_R', 'DA_3DO3_ER_B_L', 'DA_3DO3_ER_A_R', 'DA_3DO3_ER_Wr_R', 'DA_3DO3_ER_A_L']

volumes = ['VOL_3DO3_TOT', 'VOL_3DO3_Arm', 'VOL_3DO3_Leg', 'VOL_3DO3_Trunk']
m_common = ['DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th', 'DA_3DO3_CIRC_B']
m_all = ['DA_3DO3_CIRC_Ch', 'DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th', 'DA_3DO3_CIRC_C', 'DA_3DO3_CIRC_Wr', 'DA_3DO3_CIRC_F', 'DA_3DO3_CIRC_B', 'DA_3DO3_CIRC_A', 'DA_3DO3_LEN_Arm', 'DA_3DO3_LEN_Leg']
ER_common = ['DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_B']
ER_all = ['DA_3DO3_ER_Ch', 'DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_C', 'DA_3DO3_ER_Wr', 'DA_3DO3_ER_F', 'DA_3DO3_ER_B', 'DA_3DO3_ER_A']
SA_all = ['DA_3DO3_SA_TOT', 'DA_3DO3_SA_Trunk', 'DA_3DO3_SA_Arm', 'DA_3DO3_SA_Leg']

data_config_dict = {
    # dictionary containing information on how our DataSet should be extracted and scaled.
    'target_cnames': ['BC_DXA_FAT_TOT', 'BC_DXA_LST_TOT', #'BC_DXA_BMC_TOT', 'BC_DXA_BMD_TOT', # this is an example for two targets, you can pick any "cname" to be your target
                      # 'BC_DXA_FAT_Trunk', 'BC_DXA_LST_Trunk', 'BC_DXA_BMC_Trunk',
                      # 'BC_DXA_FAT_Arm', 'BC_DXA_FAT_Leg', 'BC_DXA_LST_Arm','BC_DXA_LST_Leg', 'BC_DXA_BMC_Arm', 'BC_DXA_BMC_Leg', 'BC_DXA_BMD_Arm', 'BC_DXA_BMD_Leg',
                      ],
    'feature_options': {
        # collections of features to preform a grid search over
        "bmi": {
            "N": [],
            "Y": bmi
        },
        "age": {
            "N": [],
            "Y": age
        },
        "sex": {
            # "N": [],
            "Y": sex
        },
        "volumes": {
            "N": [],
            # "Y": volumes
        },
        # "SA": {
        #     "N": [],
        #     "Y": SA_all
        # },
        "measurements": {
            # "none": [],
            "common": m_common,
            "all": m_all
        },
        "a_b": {
            # "none": [],
            "four": ER_common,
            "all": ER_all

        }
    },
    'transform_options': {
        # categorical features columns to split our dataset over for seperate runs
        'DEM_SEX': {
            "M": [column_filter("DEM_SEX", "M")],
            "F": [column_filter("DEM_SEX", "F")],
            # "M/F": [], # always
        }
    },
    'scalar_config': {
        # column names and their corresponding scalars/encoders
        "DEM_SEX": LabelBinarizer,
        # "DEM_AGE": MinMaxScaler,
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
Example Run 1: experimenting with a list of regressors to find best regressor for the data
    By default, only one example run in this file should be uncommented. Tip: to (un)comment lines in bulk using 
PyCharm, highlight the desired lines of code an press ctrl + '/'.
'''
print('EXAMPLE RUN 1: Using a list of regressors')

regressors = [
    # In this example we are trying out three different regressors: Linear, Lasso, MLP.
    LinearRegression(),
    Lasso(alpha=.001),
    MLPRegressor(solver="lbfgs",
        activation="identity",
        max_iter=800,
        hidden_layer_sizes=(1,))
]

results = run_batch(
    # pass required arguments
    df, # dataset created using MyDataset
    data_config_dict, # Option for data extraction and grouping based on dataset created by MyDataset
    regressors, # regression method and hyperparameter
    # configure internal parameters for n-fold cross-validation and scoring
    cv_params=dict(
        n_splits=5,
        scorers=['r2', #'explained_variance'
        ]  # options for different scores can be found at https://scikit-learn.org/stable/modules/model_evaluation.html
    ),
    # print the best 3 runs (sorted by test_r2) for each target and sex option
    show_best_runs=3,
    n_cores=1
)
del results['estimator']
print('[RESULTS]\n', results)
results.to_csv('results.csv')


'''
Example Run 2: Using a single regressor (the best regressor selected from "Example Run 1") 
and experimenting with different hyper-parameter values by a grid search to do hyperparameter tuning
'''
# print('EXAMPLE RUN 2: Using a single regressor with a hyper-parameter grid search')
#
# regressor = MLPRegressor # assuming MLP was the best regressor based on EXample Run 1 outcomes
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
#     regressor_param_dict, # in this example in addition to the regressor type,
#                           #  we give a regressor hyperparameter dictionary as
#                           #  we are doing a grid search to find best parameter.
#     show_best_runs=3
# )
# print('[RESULTS]\n', results)