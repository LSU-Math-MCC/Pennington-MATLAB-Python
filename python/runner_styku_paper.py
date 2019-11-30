import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LassoLars, BayesianRidge, SGDRegressor,ElasticNet, RANSACRegressor
from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, TheilSenRegressor, ARDRegression, Perceptron
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import NuSVR, SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, StykuDataSet_2, TrimmedStykuDataSet
from utilities.data_transformers import column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from runner import execute, open_in_excel

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

eth = ['race']
bmi = ['BMI1']
sex = ['SEX']
age = ['age']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
            "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]

loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        #"M": essential_transformers + [column_filter("SEX", "M")],
        #"F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
    feature_cnames=combine_options(additional_options=["SEX"], input={
    "features": {
        "BMI": bmi,
        #"Age": age,
        #"Measurements": m_common,
        #"BMI, Age": bmi+ age,
        #"BMI, Measurements": bmi + m_common,
        #"Age Meas": age+m_common,
        #"BMI, age, meas": bmi + m_common + age,
        "Vol": volumes,
        "Vol, BMI": volumes + bmi,
        "CMeas": m_common,
        "Meas": m_all,
        "Four Meas, Four ER": m_common + a_b_four,
        "Meas, ER": m_all + a_b_all,
        "CMeas, Vol": m_common + volumes,
        "Meas, Vol": m_all + volumes,
        "CMeas, CER, Vol": m_common + a_b_four + volumes,
        "Meas, ER, Vol": m_all + a_b_all + volumes,
        "CMeas, CER, BMI": m_common + a_b_four + bmi,
        "Meas, ER, BMI": m_all + a_b_all + bmi,
        "CMeas, CER, Vol, BMI": m_common + a_b_four + volumes + bmi,
        "Meas, ER, Vol, BMI": m_all + a_b_all + volumes + bmi,
        #"Meas, ER, Vol, Sex": m_all + a_b_all + volumes + sex,
        #"Four Meas, Four ER, Sex": m_common + a_b_four + sex,
    }
})
)

regressor_parameter_grid = dict(
    #alpha=list(np.linspace(0 , 2, 100)),
    #l1_ratio=list(np.linspace(0, 1, 50))
)

dataset = TrimmedStykuDataSet()
save_name = f"Styku_regressor_runs_{type(dataset).__name__}"

#f = open(f'reports/reg_data.csv', 'w')
#f.write('Estimator, Test, Train\n')
#f.close()

for c in range(1):
    print(f'[STATUS] Run # {c + 1}')
    results = execute(dataset,
                  loader_params,
                  loader_param_grid,
                  [
                    #LinearRegression(),
                    #KernelRidge(alpha=3),
                    MLPRegressor(solver="lbfgs",
                                activation="identity",
                                max_iter=800,
                                hidden_layer_sizes=(1,))
                    ],
                  regressor_parameter_grid,
                  targets=['TOTAL_FAT','TOTAL_LEAN','TOTAL_PFAT'], #, , 'TOTAL_PFAT''TOTAL_FAT','TOTAL_LEAN','TOTAL_PFAT'
                  cv=5,
                  report=save_name
                  )
print(results)
results.to_excel("reports/Styku_Paper_Extensive_Results.xlsx")