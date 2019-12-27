import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LassoLars, BayesianRidge, SGDRegressor,ElasticNet, RANSACRegressor
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, TheilSenRegressor, ARDRegression, Perceptron
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import NuSVR, SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from datasets import BIADataset, StykuDataSet, NhanesDataSet
from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2, TrimmedStykuDataSet
from utilities.data_transformers import column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from runner_modular import run_batch, threshold_scan

import time
start = time.time()

dataset = NhanesDataSet()

essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
data_config_dict = {
    'target_cnames': ['TOTAL_FAT', 'TOTAL_LEAN'],
    'feature_options': {
        "bmi": {
            # "N": [],
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
        # "volumes": {
        #     # "N": [],
        #     "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
        #           "trunkVolume"],
        # },
        "measurements": {
            # "none": [],
            "common": ["waist circ", #"hip circ",
                       "rThighGirth", "rbicepGirth"],
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
            # "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            #         "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
            #         "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
            #         "rWristGirth A_B", "Lankle A_B"]
        },
        # "BIA": {
        #     "None": [],
        #     "tbw": ["_ECW_TBW", "_ECW_TBW_of_Right_Arm", "_ECW_TBW_of_Left_Arm", "_ECW_TBW_of_Trunk",
        #             "_ECW_TBW_of_Right_Leg","_ECW_TBW_of_Left_Leg"],
        #     "Impedence": ["_1Mhz_RA_Impedance", "_1Mhz_LA_Impedance", "_1Mhz_TR_Impedance", "_1Mhz_RL_Impedance",
        #                   "_1Mhz_LL_Impedance"],
        #     "Reactance": ["_250khz_RA_Reactance","_250khz_LA_Reactance", "_250khz_TR_Reactance", "_250khz_RL_Reactance",
        #                   "_250khz_LL_Reactance"],
        #     # "PhaseAngle": ["_250khz_RA_Phase_Angle","_250khz_LA_Phase_Angle", "_250khz_TR_Phase_Angle", "_250khz_RL_Phase_Angle",
        #     #                "_250khz_LL_Phase_Angle"]
        # },
    },
    'transform_options': {
        'SEX': {
            # "M": essential_transformers + [column_filter("SEX", "M")],
            # "F": essential_transformers + [column_filter("SEX", "F")],
            "M/F": essential_transformers,
        }
    },
    # 'feature_selection': {
    #
    # },
    'scalar_config': {
        "SEX": LabelBinarizer,
        "age": MinMaxScaler,
        "default": StandardScaler
    }
}

regressor = MLPRegressor(solver="lbfgs",
                        activation="identity",
                        max_iter=800,
                        hidden_layer_sizes=(1,))
regressor_parameter_grid = dict(
)

results = run_batch(dataset,
                    data_config_dict,
                    regressor,
                    regressor_parameter_grid,
                    show_best_runs=3,
                    ext_dataset=StykuDataSet()
)
print(results)

end = time.time()
times = (end - start)/60
print("Run time=%d mins" %times)

