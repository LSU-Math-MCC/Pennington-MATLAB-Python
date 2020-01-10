import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.linear_model import LassoLars, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from DataSets import NhanesDataSet
from utilities.paramutils import combine_options
from utilities.data_transformers import mean_body_part_transformer, average_transformer
from utilities.archive.runner import execute

import time
start= time.time()

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 199
pd.options.display.width = None

'''
tot_fat="DXDTOFAT"
tot_lean_wo_bmc="DXDTOLE"
tot_pfat="DXDTOPF"

age = "RIDAGEYR"
bmi = "bmxbmi"
gender = "RIAGENDR"
height = "bmxht"
upper_arm_length = "BMXARML"
upper_leg_length = "BMXLEG"

waist_circumference = "bmxwaist"
arm_circumference = "BMXARMC"
calf_circumference = "BMXCALF"
thigh_circumference = "BMXTHICR"
'''

tot_fat = "TOTAL_FAT"
tot_lean_wo_bmc = "TOTAL_LEAN"
tot_pfat = "TOTAL_PFAT"


#Measurements
upper_leg_length = "rLegLength"
upper_arm_length = "RArmLength"
waist_circumference = "waist circ"
arm_circumference = "rbicepGirth"
calf_circumference = "rCalfCirc"
thigh_circumference = "rThighGirth"
gender = ["SEX"]
age = ["age"]
bmi = ["BMI1"]
measurements = [upper_leg_length, upper_arm_length, waist_circumference, arm_circumference, calf_circumference, thigh_circumference]
surface_area = ["DXDRATOT", "DXDRLTOT", "DXDLLTOT", "DXDLATOT", "DXXHEA", "DXDTOA", "DXXPEA",
                "DXXTSA","DXXRRA", "DXXLRA"]
skin_fold = ["BMXTRI","BMXSUB"]

essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
loader_params = dict(
    scaler_config={"SEX": LabelBinarizer,
                   "age": MinMaxScaler,
                   "default": StandardScaler
                   }
)
loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        #"M": [column_filter(gender, 1)],
        #"F": [column_filter(gender, 2)],
        "M/F": essential_transformers,
    },
    # only use combine options for feature_cnames
    feature_cnames=combine_options(additional_options=["SEX"], input={
        "features": {
            "BMI": bmi,
            #"Age": age,
            "SF": skin_fold,
            #"SA": surface_area,
            "Meas": measurements,
            #"BMI,SA": bmi + surface_area,
            #"Meas,SA": measurements + surface_area,
            "BMI,Meas": bmi + measurements,
            "BMI,Meas,SF": bmi + measurements + skin_fold
            #"BMI,Meas,SA": bmi + measurements + surface_area
            #"Age and meas": age + m_common,
            #"BMI, age, meas": bmi + m_common + age,
        }
    }))



    #label_cname={
    #    "SystolicBP": "MEAN_SBP",
    #    "DiastolicBP": "MEAN_DBP",
    #}
regressor_param_grid = {}
regressor = [#TheilSenRegressor(),
             HuberRegressor(),
            LassoLars(),
            KernelRidge(),
             BayesianRidge(),
             #Lasso(),
             #SGDRegressor(),
             LinearRegression(),
             #MLPRegressor(solver="lbfgs",
             #             activation="identity",
             #             max_iter=800,
             #             hidden_layer_sizes=(1,))
                          ]

# shorten # of runs
# loader_param_grid["feature_cnames"] = loader_param_grid["feature_cnames"].iloc[-1:]
# del loader_param_grid["data_transformers"]["M/F"]
# del loader_param_grid["data_transformers"]["F"]
#NhanesDataSet().load_data().to_excel("data/nhanes_loaded.xlsx")

save_name="nhanes-blood-pressure"
results = execute(NhanesDataSet(),
                  loader_params,
                  loader_param_grid,
                  regressor,
                  regressor_param_grid,
                  report='nhanes-blood-pressure',
                  cv=5,
                  targets=[tot_fat, tot_pfat, tot_lean_wo_bmc],
                  export_best=3
                  )
print(results)
#save_df(results, save_name)
results.to_excel("data/N_Hanes_Extensive_Results.xlsx")
end= time.time()
print((start-end)/60)