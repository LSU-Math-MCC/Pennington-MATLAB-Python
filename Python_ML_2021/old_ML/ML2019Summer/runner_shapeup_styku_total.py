from copy import deepcopy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import datetime

import shapeup_common
from datasets_clint import ShapeUpDataSet
from paramutils import combine_options
from runner import execute, save_df
from data_transformers import column_filter, mean_body_part_transformer, age_transformer, bmi_transformer, average_transformer

# Use to store only total output (ran on all parameters)
now = datetime.datetime.now()
styku="styku_v6"
save_name=f"{styku}_total-{now.strftime('%Y_%m_%d-%H_%M')}"
print(f"Running total report on {styku}")
print(f"LOCATION: reports/Conference Paper/{save_name}.xlsx")

essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
loader_param_grid_all = dict(
    data_transformers={
        "__cols": "SEX",
        "M": essential_transformers + [column_filter("SEX", "M")],
        "F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
)
loader_param_grid_all["feature_cnames"] = combine_options(additional_options=["SEX"], input={
    "bmi": {
        #"N": [],
        "Y": ["BMI1"]
    },
    "age": {
        #"N": [],
        "Y": ["age"]
    },
    "lengths": {
        #"N": [],
        "Y": ["LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"],
    },
    "volumes": {
        #"none": [],
        #"total": ["TotalVolume"],
        "all": ["TotalVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "headVolume", "trunkVolume"],
    },
    "measurements": {
        #"none": [],
        #"common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
        "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle"]
                # "rAnkle" and "Lankle" are ankle girths
    },
    "a_b": {
        #"none": [],
        #"four": ["waist_a_over_b", "hip_a_over_b", "rThigh_a_over_b", "r_bicep_a_over_b"],
        "all": ["chest_a_over_b", "waist_a_over_b", "hip_a_over_b", "rThigh_a_over_b", "lThigh_a_over_b",
                "rCalf_a_over_b", "lCalf_a_over_b", "r_wrist_a_over_b", "l_wrist_a_over_b", "r_forearm_a_over_b",
                "l_forearm_a_over_b", "r_bicep_a_over_b", "l_bicep_a_over_b", "r_ankle_a_over_b",
                "l_ankle_a_over_b"]
        }
})

results = execute(ShapeUpDataSet(),
                  shapeup_common.loader_params,
                  loader_param_grid_all,                  shapeup_common.regressor,
                  shapeup_common.regressor_parameter_grid,
                  save_name=save_name,
                  save_df_report=True)
results["set_name"] = styku

print(results)

results.to_excel(f"reports/Conference Paper/{save_name}.xlsx", index=False) # only use of datetime
