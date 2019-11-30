import numpy as np
import pandas as pd
from datetime import date
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedShuffleSplit,StratifiedKFold

from sklearn.svm import SVC, NuSVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids, discrete_class, column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import iter_params
#from runner_styku_classifier import StykuClassDataSet_2
from datasets import NhanesDataSet
from runner_modular import stratifiedCV


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


def changeClassTypeName(theclass, thename):
    #SE https://stackoverflow.com/questions/1850289/python-change-class-type-name
  theclass.__class__ = type(thename, (type,), {})

class threshold_clf(Estimator):
    def __init__(self, clf, cnd, threshold=.5):
        super().__init__(clf=clf, cnd=cnd, threshold=threshold)
        self.clf = clf
        self.fit_clf = None
        self.cnd = cnd
        self.threshold = threshold
        # changeClassTypeName(threshold_clf,
        #                    f'{type(clf).__name__}_t({threshold})')

    def fit(self, X, y, **kwargs):
        if self.fit_clf is None:
            self.fit_clf = self.clf.fit(X, y, **kwargs) # Needs revision
        else:
            self.fit_clf = self.clf.fit(X, y, **kwargs) # Needs revision

    def predict(self, X, **kwargs):
        if self.fit_clf is None:
            raise Exception('Estimator not fit')
        probas = self.fit_clf.predict_proba(X, **kwargs)
        preds = [apply_threshold(proba, self.cnd, self.threshold) for proba in probas]
        return preds

    def update_threshold(self, threshold):
        self.threshold = threshold


bmi = ['BMI1']
sex = ['SEX']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", #"hip circ",
            "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
            "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]
BIA_PhaseAngle = ["_250khz_RA_Phase_Angle","_250khz_LA_Phase_Angle", "_250khz_TR_Phase_Angle", "_250khz_RL_Phase_Angle",
                               "_250khz_LL_Phase_Angle"]
BIA_tbw = ["_ECW_TBW", "_ECW_TBW_of_Right_Arm", "_ECW_TBW_of_Left_Arm", "_ECW_TBW_of_Trunk",
                        "_ECW_TBW_of_Right_Leg","_ECW_TBW_of_Left_Leg"]
BIA_Reactance = ["_250khz_RA_Reactance","_250khz_LA_Reactance", "_250khz_TR_Reactance", "_250khz_RL_Reactance",
                              "_250khz_LL_Reactance"]
BIA_Impedence = ["_1Mhz_RA_Impedance", "_1Mhz_LA_Impedance", "_1Mhz_TR_Impedance", "_1Mhz_RL_Impedance",
                              "_1Mhz_LL_Impedance"]
race_nh = ['RACETH']
SA_nh = ["DXDRATOT", "DXDRLTOT", "DXDLLTOT", "DXDLATOT", "DXXHEA", "DXDTOA", "DXXPEA",
            "DXXTSA", "DXXRRA", "DXXLRA"]


features = {'BMI, SEX, MEAS': bmi + sex + m_common,
            # 'BMI, SEX, MEAS, ETH': bmi + sex + m_common + race_nh,
            # 'BMI, SEX, MEAS, DXA': bmi + sex + m_common + ['TOTAL_FAT', 'TOTAL_LEAN'],
            # 'BMI, SEX, MEAS, SA': bmi + sex + m_common + SA_nh,
            # 'BMI, SEX, MEAS, DXA, SA': bmi + sex + m_common + SA_nh + ['TOTAL_FAT', 'TOTAL_LEAN'],

            #'BMI, SEX': bmi + sex,
            #'BMI, SEX, ETH': bmi + sex + race_nh,
            }
targets = ['HDL_risk', 'LDL_risk'] #, 'GLU_risk'
regressors = [#GaussianNB(),
                 KNeighborsClassifier(n_neighbors=50),
                 #GaussianMixture(),
                 AdaBoostClassifier(),
                 #BaggingClassifier(n_jobs=-1),
                 #LabelPropagation(),
                 #CalibratedClassifierCV(),
                 SGDClassifier(loss='modified_huber'),
                 #LabelSpreading(),
                 #SVC(probability=True),
                 RandomForestClassifier(),
                 DecisionTreeClassifier(),
                 LogisticRegression(),
                 ExtraTreesClassifier(n_estimators=100, n_jobs=-1),
                 ExtraTreeClassifier(),
                 GradientBoostingClassifier(),
                 #GaussianProcessClassifier(),
                 #DummyClassifier(),
                 #LinearDiscriminantAnalysis(),
                 QuadraticDiscriminantAnalysis(),
                 MLPClassifier(hidden_layer_sizes=(10, 5), activation="logistic")
                 ]

report_df = pd.DataFrame()
for param_label, cnames in features.items():
    print(f'⛷ Params: {param_label}')
    for target in targets:
        cnd = 0  # 'Healthy' on all targets except HDL_risk
        if target == 'HDL_risk':
            cnd = 2
        # Data extraction and train/test split
        print(f'⛇ Target: {target} on cnd {cnd}')
        essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname
                                                                    in ["ArmR", "ThighR", "Waist", "Hip"]]
        loader_params = dict(
            data_transformers=essential_transformers,
            scaler_config={"SEX": LabelBinarizer,
                           "age": MinMaxScaler,
                           "default": StandardScaler,
                           },
        )
        data = NhanesDataSet().extract_data(cnames, target, **loader_params)
        print('⛐ ⚞Extracted⚟ ⛴')

        # Run the classifier at various thresholds
        for est in regressors:
            est0 = threshold_clf(est, cnd)
            print(f'♨ Regressor: {type(est).__name__}')
            for threshold in np.linspace(0, 1, 50):  # list(map(lambda x: pow(x, .25), np.linspace(0, 1, 100)))
                run_dict = dict()  # Can't initially make DataFrame because we're storing list (matrices) as entries
                run_dict['Params'] = param_label
                run_dict['target'] = target
                run_dict['Estimator'] = type(est).__name__
                run_dict[f'condition {cnd} threshold'] = threshold

                est0.update_threshold(threshold)
                run_dict = {**run_dict, **stratifiedCV(data, est0, cnd=cnd, cv=5)}
                run_df = pd.DataFrame(data={key: {0: value} for key, value in run_dict.items()})
                report_df = pd.concat([report_df, run_df], sort=False)
print(report_df)
report_df.fillna(0).to_excel('reports/ROC Analysis/ROC_Claire_3GLU_ParamBatch.xlsx', index=False)