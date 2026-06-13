from copy import deepcopy

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import shapeup_common
from datasets_pujan import ShapeUpDataSet
from paramutils import combine_options
from runner import execute, save_df

import time
start = time.time()

loader_param_grid = deepcopy(shapeup_common.loader_param_grid)
loader_param_grid["feature_cnames"] = combine_options(additional_options=["SEX"], input={
    "bmi": {
       # "N": [],
       "Y": ["BMI1"]
    },
    "age": {
       # "N": [],
        "Y": ["age"]
    },
    "volumes": {
      "N": [],
      #"TV only": ["TotalVolume"],
      #"SegV only": ["rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "headVolume", "trunkVolume"],
      "Y": ["TotalVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "headVolume", "trunkVolume"],
    },
    "measurements": {
        #"none": [],
        "common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
        #"lengths": ["LarmLength", "RArmLength", "CollarScalp", "TrunkLength",
         #           "lLegLength", "rLegLength", "crotchHeight"],
        "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                "LarmLength", "RArmLength", "CollarScalp", "TrunkLength",
                "lLegLength", "rLegLength", "crotchHeight"]
                # "rAnkle" and "Lankle" are ankle girths
    },
    "a_b": {
        "none": [],
        # These don't work with new styku file
        #"four": ["waist_a_over_b", "hip_a_over_b", "rThigh_a_over_b", "r_bicep_a_over_b"],
         #"all": ["chest_a_over_b", "waist_a_over_b", "hip_a_over_b", "rThigh_a_over_b", "lThigh_a_over_b",
          #      "rCalf_a_over_b", "lCalf_a_over_b", "r_wrist_a_over_b", "l_wrist_a_over_b", "r_forearm_a_over_b",
           #      "l_forearm_a_over_b", "r_bicep_a_over_b", "l_bicep_a_over_b", "r_ankle_a_over_b",
            #     "l_ankle_a_over_b"]
        }
})

save_name = "styku-overhaul"
results = execute(ShapeUpDataSet(),
                  shapeup_common.loader_params,
                  loader_param_grid,
                  shapeup_common.regressor,
                  shapeup_common.regressor_parameter_grid,
                  save_name=save_name,
                  save_df_report=False)
results["set_name"] = "styku"
save_df(results, save_name)
print(results)
end = time.time()
print("Run time=%d s" %(end - start))
