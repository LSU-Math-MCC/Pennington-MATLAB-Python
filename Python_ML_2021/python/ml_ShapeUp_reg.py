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
from DataSets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2, TrimmedStykuDataSet
from utilities.data_transformers import column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from ml_core import run_batch, threshold_scan

import time
start = time.time()

# dataset = StykuDataSet()
# dataset = SS20DataSet()
dataset = CombinedDataSet()

eth = ['race']
bmi = ['BMI1']
sex = ['SEX']
age = ['age']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth", "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm", "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle", "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B", "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]

data_config_dict = {
    'target_cnames': ['TOTAL_FAT', 'TOTAL_LEAN'],
    'feature_options': {
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
            "N": [],
            "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
                  "trunkVolume"]
        },
        "measurements": {
            # "none": [],
            "common": ["waist circ", "hip circ",
                       "rThighGirth", "rbicepGirth"],
            "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength",
                    "crotchHeight"],
            # "rAnkle" and "Lankle" are ankle girths
        },
        "a_b": {
            "none": [],
            "four": ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"],
            "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
                    "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
                    "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
                    "rWristGirth A_B", "Lankle A_B"]
        },
        "BIA": {
            "none": [],
            # "tbw": ["_ECW_TBW", "_ECW_TBW_of_Right_Arm", "_ECW_TBW_of_Left_Arm", "_ECW_TBW_of_Trunk",
            #         "_ECW_TBW_of_Right_Leg","_ECW_TBW_of_Left_Leg"],
            "Impedence": ["_1Mhz_RA_Impedance", "_1Mhz_LA_Impedance", "_1Mhz_TR_Impedance", "_1Mhz_RL_Impedance",
                          "_1Mhz_LL_Impedance"],
            "Reactance": ["_250khz_RA_Reactance","_250khz_LA_Reactance", "_250khz_TR_Reactance", "_250khz_RL_Reactance",
                          "_250khz_LL_Reactance"],
            # "PhaseAngle": ["_250khz_RA_Phase_Angle","_250khz_LA_Phase_Angle", "_250khz_TR_Phase_Angle", "_250khz_RL_Phase_Angle",
            #                "_250khz_LL_Phase_Angle"]
        },
        # "features": {
        #     # Publication-style parameter grid (use w/out other feature options)
        #     "BMI": bmi,
        #     #"Age": age,
        #     #"Measurements": m_common,
        #     #"BMI, Age": bmi+ age,
        #     #"BMI, Measurements": bmi + m_common,
        #     #"Age Meas": age+m_common,
        #     #"BMI, age, meas": bmi + m_common + age,
        #     "Vol": volumes,
        #     "Vol, BMI": volumes + bmi,
        #     "CMeas": m_common,
        #     "Meas": m_all,
        #     "Four Meas, Four ER": m_common + a_b_four,
        #     "Meas, ER": m_all + a_b_all,
        #     "CMeas, Vol": m_common + volumes,
        #     "Meas, Vol": m_all + volumes,
        #     "CMeas, CER, Vol": m_common + a_b_four + volumes,
        #     "Meas, ER, Vol": m_all + a_b_all + volumes,
        #     "CMeas, CER, BMI": m_common + a_b_four + bmi,
        #     "Meas, ER, BMI": m_all + a_b_all + bmi,
        #     "CMeas, CER, Vol, BMI": m_common + a_b_four + volumes + bmi,
        #     "Meas, ER, Vol, BMI": m_all + a_b_all + volumes + bmi,
        #     #"Meas, ER, Vol, Sex": m_all + a_b_all + volumes + sex,
        #     #"Four Meas, Four ER, Sex": m_common + a_b_four + sex,
        # }
    },
    'transform_options': {
        'SEX': {
            # "M": [column_filter("SEX", "M")],
            # "F": [column_filter("SEX", "F")],
            "M/F": [],
        }
    },
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
                    cv_params=dict(n_splits=5)
                    # ext_dataset=StykuDataSet()
)
print(results)

end = time.time()
times = (end - start)/60
print("Run time=%d mins" %times)

