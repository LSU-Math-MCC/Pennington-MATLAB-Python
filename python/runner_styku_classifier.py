import pandas as pd
import numpy as np
from datetime import date
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from datasets import StykuDataSet, SS20DataSet, CombinedDataSet, DataSet, StykuDataSet_2, NhanesDataSet
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids, discrete_class, column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from utilities.folder_searcher import Searcher
from utilities.DirectoryGrab import DirGrab
from utilities.data_merger import ListMaker, ListStandardizer, CBDrowMaker, MergeMan
from runner_modular import run_batch
import datetime
now = datetime.datetime.now()

class StykuClassDataSet_2(DataSet):
    def __init__(self):
        super().__init__()

    # looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def search_update():
        path = "data\\"
        strange_path = "data\\ObjOrganizerStyku*.xlsx"
        dir = DirGrab(path)
        files = dir._glob_grabber(strange_path)
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        #print(file_list[1])
        styku_location = file_list[1]
        styku = pd.read_excel(styku_location, na_values=["[]", 0])
        return styku

    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if type(x) is float else x)
        return df

    def load_data(self):
        Styku_df = StykuDataSet.search_update()  # loads DataFrame
        Styku_df = self.standardize_units(Styku_df)
        Styku_df['SubjectID'] = ListMaker(Styku_df.Name)  # creates Subject ID column from Name column
        Styku_df = Styku_df.drop(Styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        Styku_df.rename(columns={col: f"{col.replace('/', '_')}" for col in Styku_df.columns},
                        inplace=True)
        modData = CBDrowMaker(Styku_df)  # creates modified DataFrame
        modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # standardizes subject ID column

        Combo_df= super().common_dataframes_2()  # loads the Combined DataFrame
        modData_2 = CBDrowMaker(Combo_df)  # creates a modified Dataframe
        modData_2['SubjectID'] = ListStandardizer(modData_2.SubjectID)  #

        combined_df = MergeMan(modData, modData_2)

        #print(list(combined_df['GLU_risk']))
        # Remove people with family histories for HBA1C
        combined_df['SubjectID'] = cut_subject_ids(combined_df['SubjectID'])
        combined_df = combined_df.merge(
            pd.read_excel('data/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx', sheet_name='History'),
            on='SubjectID', how='outer')
        combined_df = combined_df.loc[combined_df['Fam_Diabetes'] != 'Yes']

        bia_df = pd.read_excel("data/Shapeup_Adults_Q2_181101.xlsx", sheet_name="BIA")
        bia_df['AGE'] = bia_df['_AGE'].astype(float)
        combined_df = combined_df.merge(bia_df, on='SubjectID', how='outer')


        combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])


        # Remove subjects that did not fast
        combined_df = combined_df.loc[(combined_df['TRIG'] <= 180) | (combined_df['BMI1'] >= 45)] # Remove subjects w/ TRIG > 180 & BMI < 45

        # age column
        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')

        # M/F risk columns
        '''
        def ldl_class(df):
            if df['sex']=='M':
                    if df['LDL'] < 3:
                        return 0
                    elif ...
        '''
        # Diabetes risks
        combined_df['GLU_risk'] = discrete_class(combined_df, 'GLU', [100, 125]) # 0 is healthy
        combined_df['HBA1C_risk'] = discrete_class(combined_df, '_HBA1C', [5.6, 6.4]) # 0 is healthy

        # Heart risks
        combined_df['LDL_risk'] = discrete_class(combined_df, 'LDL', [130, 160]) # 0 is healthy
        combined_df['HDL_risk'] = discrete_class(combined_df, 'HDL', [40, 60]) # 2 (>60) is healthy

        #print(list(combined_df['GLU_risk']))
        return combined_df


# Classes for custom sklearn estimator
class Estimator:
    def __init__(self, **kwargs):
        self.params = {**kwargs}

    def get_params(self, **kwargs):
        return self.params


def apply_threshold(proba_list, cnd, threshold):
    if proba_list[cnd] >= threshold:
        return cnd
    else:
        proba_list[cnd] = 0
        return np.argmax(proba_list)


class threshold_clf(Estimator):
    def __init__(self, clf, cnd, threshold=.5):
        super().__init__(clf=clf, cnd=cnd, threshold=threshold)
        self.clf = clf
        self.cnd = cnd
        self.threshold = threshold

    def fit(self, X, y, **kwargs):
        self.clf.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        probas = self.clf.predict_proba(X, **kwargs)
        preds = [apply_threshold(proba, self.cnd, self.threshold) for proba in probas]
        return preds


def main():
    pass

if __name__ == '__main__':
    dataset = StykuClassDataSet_2()
    #ext_dataset = SS20DataSet()

    essential_transformers = []
    data_config_dict = {
        'target_cnames': ['GLU_risk', 'LDL_risk'],
        'feature_options': {
            "bmi": {
                # "N": [],
                "Y": ["BMI1"]
            },
            "age": {
                # "N": [],
                "Y": ["AGE"]
            },
            "sex": {
                # "N": [],
                "Y": ["SEX"],
            },
            "volumes": {
                # "N": [],
                "Y": ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
                      "trunkVolume"],
            },
            "measurements": {
                # "none": [],
                "common": ["waist circ", "hip circ", "rThighGirth", "rbicepGirth"],
                "all": ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                        "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                        "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                        "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength",
                        "crotchHeight"],
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
                #"M": essential_transformers + [column_filter("SEX", "M")],
                #"F": essential_transformers + [column_filter("SEX", "F")],
                "M/F": essential_transformers,
            }
        },
        'scalar_config': {
            "SEX": LabelBinarizer,
            "age": MinMaxScaler,
            "default": StandardScaler
        }
    }


    regressor = [GaussianNB(),
                 #BayesianGaussianMixture(),
                 #LinearDiscriminantAnalysis(),
                 #QuadraticDiscriminantAnalysis(),
                 # MLPClassifier(hidden_layer_sizes=(15,10),
                 #               solver='lbfgs'),
                 # RandomForestClassifier(n_estimators=100),
                 # ExtraTreesClassifier()
                 ]
    regressor_parameter_grid = dict(
    )

    results = run_batch(
        dataset,
        data_config_dict,
        regressor,
        regressor_parameter_grid,
        eval_type='classifier',
        cnd=0,
        # show_best_runs=1
    )
    print(results)
    results.to_csv('reports/clf_BIA.csv')
    main()