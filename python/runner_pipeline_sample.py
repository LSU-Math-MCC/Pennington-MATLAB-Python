import pandas as pd
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

from utilities.data_transformers import cut_subject_ids, column_filter
from datasets import DataSet
from runner_modular import run_batch


def to_Dataset(df, subject_cname='SubjectID'):
    class auto_dataset(DataSet):
        def load_data(self):
            try:
                # find subject IDs and combine with DXA, Blood, and Questionaire data
                df['SubjectID'] = cut_subject_ids(df[subject_cname])  # remove scan suffixes
                if subject_cname != 'SubjectID':
                    del df[subject_cname]  # The next step expects numeric columns

                # TODO: Combine label columns differently than numeric columns
                clean_df = df.replace([0, "[]", "", "nan"], None).dropna()
                clean_df = clean_df.drop_duplicates('SubjectID')
                # clean_df = clean_df.groupby(clean_df['SubjectID'], as_index=False).aggregate('mean')  # take average of duplicates

                combined_df = super().common_dataframes()
                combined_df = combined_df.merge(clean_df, on='SubjectID', how='outer')
                return combined_df
            except KeyError:
                # if there are no subject indentifiers, do not combine with DXA
                warnings.warn(f'Could not find subject identifier column \"{subject_cname}\", loading without DXA.',
                              stacklevel=4)
                return df
    return auto_dataset()


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