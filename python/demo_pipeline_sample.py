import pandas as pd
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

from utilities.data_transformers import cut_subject_ids, column_filter
from datasets import DataSet, to_Dataset
from runner_modular import run_batch

df = pd.read_excel('data/ObjOrganizerStyku_v14.xlsx')
df = df.drop(df.columns[[0, 1, 3, 4]], axis=1)  # remove unnecessary label columns
df.rename(columns={col: f"{col.replace('/', '_')}" for col in df.columns}, inplace=True)  # stardardize styku column names
Styku_Dataset = to_Dataset(df, subject_cname='Name')

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
            # "N": [],
            "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
                  "trunkVolume"]
        },
        "measurements": {
            "none": [],
            "common": ["waist circ", "hip circ",
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
            "all": ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
                    "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B",
                    "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
                    "rWristGirth A_B", "Lankle A_B"]
        }
    },
    'transform_options': {
        'SEX': {
            "M": [column_filter("SEX", "M")],
            "F": [column_filter("SEX", "F")],
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
                         hidden_layer_sizes=(1,)
                         )

results = run_batch(Styku_Dataset,
                    data_config_dict,
                    regressor
                    )
print(results)