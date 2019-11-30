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
from datasets import BIADataset
from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2, TrimmedStykuDataSet
from utilities.data_transformers import column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from runner import execute

import time
start = time.time()

essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
loader_params = dict(
    data_transformers=essential_transformers,
    scaler_config={"SEX": LabelBinarizer,
                   "age": MinMaxScaler,
                   "default": StandardScaler
                   },
)
loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        #"M": essential_transformers + [column_filter("SEX", "M")],
        #"F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
    feature_cnames=combine_options(additional_options=["SEX"], input={
    "BIA": {
        "None" : [],
        "tbw": ["_ECW_TBW", "_ECW_TBW_of_Right_Arm","_ECW_TBW_of_Left_Arm","_ECW_TBW_of_Trunk","_ECW_TBW_of_Right_Leg",
                "_ECW_TBW_of_Left_Leg"],
        "Impedence": ["_1khz_RA_Impedance","_1khz_LA_Impedance","_1khz_TR_Impedance","_1khz_RL_Impedance","_1khz_LL_Impedance",
                       "_5khz_RA_Impedance","_5khz_LA_Impedance","_5khz_TR_Impedance","_5khz_RL_Impedance","_5khz_LL_Impedance",
                       "_1Mhz_RA_Impedance","_1Mhz_LA_Impedance","_1Mhz_TR_Impedance","_1Mhz_RL_Impedance","_1Mhz_LL_Impedance"],
        "Reactance": ["_5khz_RA_Reactance","_5khz_LA_Reactance","_5khz_TR_Reactance",
                      "_5khz_RL_Reactance","_5khz_LL_Reactance","_50khz_RA_Reactance","_50khz_LA_Reactance",
                      "_50khz_TR_Reactance","_50khz_RL_Reactance","_50khz_LL_Reactance","_250khz_RA_Reactance",
                      "_250khz_LA_Reactance","_250khz_TR_Reactance","_250khz_RL_Reactance","_250khz_LL_Reactance"],
        "PhaseAngle": ["_5khz_RA_Phase_Angle","_5khz_LA_Phase_Angle","_5khz_TR_Phase_Angle",
                       "_5khz_RL_Phase_Angle","_5khz_LL_Phase_Angle","_50khz_RA_Phase_Angle","_50khz_LA_Phase_Angle",
                       "_50khz_TR_Phase_Angle","_50khz_RL_Phase_Angle","_50khz_LL_Phase_Angle","_250khz_RA_Phase_Angle",
                       "_250khz_LA_Phase_Angle","_250khz_TR_Phase_Angle","_250khz_RL_Phase_Angle","_250khz_LL_Phase_Angle"]
    },
    # "bmi": {
    #     "N": [],
    #     "Y": ["BMI1"]
    # },
    # "age": {
    #     "N": [],
    #     "Y": ["age"]
    # },
    # "volumes": {
    #     "N": [],
    #     "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"],
    # },
    # "measurements": {
    #     "none": [],
    #     "common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
    #     "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
    #             "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
    #             "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
    #             "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"],
    #             # "rAnkle" and "Lankle" are ankle girths
    # },
    # "a_b": {
    #     "none": [],
    #     "four": ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"],
    #     "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
    #             "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
    #             "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
    #             "rWristGirth A_B", "Lankle A_B"]
    #     }
})
)

regressor_parameter_grid = dict(
)

#dataset = StykuDataSet().load_data(), BIADataset().load_data()
dataset = BIADataset()
save_name = f"Styku_regressor_runs_trimmed_{type(dataset).__name__}"
results = execute(dataset,
                  loader_params,
                  loader_param_grid,
                  #KernelRidge(alpha=3),
                   #KernelRidge(alpha=2.5),
                   #KernelRidge(alpha=2),
                   #KernelRidge(alpha=1.5),
                   #LassoLars(),
                   #LinearRegression(),
                   MLPRegressor(solver="lbfgs",
                                activation="identity",
                                max_iter=800,
                                hidden_layer_sizes=(1,)),
                   #],
                  regressor_parameter_grid,
                  targets=['TOTAL_FAT', 'TOTAL_LEAN'],
                  cv=4,
                  report=save_name,
                  export_best=3
                  )

print(results)
results.to_excel("data/Styku_regressor_runs_trimmed.xlsx")

