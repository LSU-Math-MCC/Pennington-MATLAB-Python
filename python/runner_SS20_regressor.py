import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LassoLars
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2,TrimmedStykuDataSet, StykuDataSet_3
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
        "M": essential_transformers + [column_filter("SEX", "M")],
        "F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
    feature_cnames=combine_options(additional_options=["SEX"], input={
    "bmi": {
        #"N": [],
        "Y": ["BMI1"]
    },
    "age": {
        #"N": [],
        "Y": ["age"]
    },
    "volumes": {
        #"N": [],
        "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"],
    },
    "measurements": {
        #"none": [],
        #"common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
        "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"],
                # "rAnkle" and "Lankle" are ankle girths
    },
    "a_b": {
        "none": [],
        "common": ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"],
        "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
                "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
                "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
                "rWristGirth A_B", "Lankle A_B"]
        }
})
)

regressor_parameter_grid = dict(

)

dataset = SS20DataSet()
save_name = f"TargetRegTrials_SS20vsSTYKUvsCOMB"
results = execute([SS20DataSet()],
                  loader_params,
                  loader_param_grid,
                  [
                   LinearRegression(),
                  ],
                  regressor_parameter_grid,
                  targets=['TOTAL_FAT', 'TOTAL_LEAN'],
                  cv=4,
                  report=save_name,
                  save_png=True
                  )

print(results)
results.to_excel("data/Styku_regressor_multiple_dataset_2.xlsx")