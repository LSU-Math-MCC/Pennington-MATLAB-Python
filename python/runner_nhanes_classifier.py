import pandas as pd
import numpy as np
from datetime import date

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

from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from datasets import SS20DataSet, CombinedDataSet, DataSet, StykuDataSet_2, NhanesDataSet
from runner_styku_classifier import StykuClassDataSet_2
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids, discrete_class, column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from utilities.folder_searcher import Searcher
from utilities.DirectoryGrab import DirGrab
from utilities.data_merger import ListMaker, ListStandardizer, CBDrowMaker, MergeMan
from runner_modular import run_batch, threshold_scan
import datetime
now = datetime.datetime.now()


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
        self.fit_clf = None
        self.cnd = cnd
        self.threshold = threshold
        # changeClassTypeName(threshold_clf,
        #                    f'{type(clf).__name__}_t({threshold})')

    def fit(self, X, y, **kwargs):
        if self.fit_clf is None:
            self.fit_clf = self.clf.fit(X, y, **kwargs) # Needs revision
        else:
            self.fit_clf = self.clf.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self.fit_clf is None:
            raise Exception('Estimator not fit')
        probas = self.fit_clf.predict_proba(X, **kwargs)
        preds = [apply_threshold(proba, self.cnd, self.threshold) for proba in probas]
        return preds

    def update_threshold(self, threshold):
        self.threshold = threshold


dataset = NhanesDataSet()
#ext_dataset = StykuClassDataSet_2()

bmi = ['BMI1']
sex = ['SEX']
m_common = ["waist circ", #"hip circ",
            "rThighGirth", "rbicepGirth"]
race_nh = ['RACETH']
SA_nh = ["DXDRATOT", "DXDRLTOT", "DXDLLTOT", "DXDLATOT", "DXXHEA", "DXDTOA", "DXXPEA",
            "DXXTSA", "DXXRRA", "DXXLRA"]

essential_transformers = []
data_config_dict = {
    'target_cnames': ['GLU_risk', 'LDL_risk', 'HDL_risk'],
    'feature_options': {
        "Params": {
            'BMI, MEAS': bmi + sex + m_common + race_nh,
            'BMI, MEAS, DXA': bmi + sex + m_common + ['TOTAL_FAT', 'TOTAL_LEAN'] + race_nh,
            'BMI, MEAS, SA': bmi + sex + m_common + SA_nh + race_nh,
            'BMI, MEAS, DXA, SA': bmi + sex + m_common + SA_nh + ['TOTAL_FAT', 'TOTAL_LEAN'] + race_nh,

            #'BMI, SEX': bmi + sex,
            #'BMI, SEX, ETH': bmi + sex + race_nh,
            }
        # "bmi": {
        #     # "N": [],
        #     "Y": ["BMI1"]
        # },
        # "age": {
        #     "N": [],
        #     "Y": ["age"]
        # },
        # "sex": {
        #     #"N": [],
        #     "Y": ["SEX"]
        # },
        # "surface_area": {
        #     "none": [],
        #     "all": ["DXDRATOT", "DXDRLTOT", "DXDLLTOT", "DXDLATOT", "DXXHEA", "DXDTOA", "DXXPEA",
        #     "DXXTSA", "DXXRRA", "DXXLRA"]},
        # # "skin_fold": {
        # #     "N":[],
        # #     "Y":["BMXTRI","BMXSUB"]
        # # },
        # "measurements": {
        #     # "none": [],
        #     "common": ["waist circ", "rThighGirth", "rbicepGirth"],
        #     "all": ["rLegLength", "RArmLength", "waist circ", "rbicepGirth", "rCalfCirc", "rThighGirth"],
        #
        # }
    },
    'transform_options': {
        'RACETH': {
            "1": essential_transformers + [column_filter("RACETH", 1)],
            "2": essential_transformers + [column_filter("RACETH", 2)],
            "3": essential_transformers + [column_filter("RACETH", 3)],
            "4": essential_transformers + [column_filter("RACETH", 4)],
            "All": essential_transformers,
        },
        'SEX': {
            "F": essential_transformers + [column_filter("SEX", 1)],
            "M": essential_transformers + [column_filter("SEX", 2)],
            "M/F": essential_transformers,
        }
    },
    'scalar_config': {
        "SEX": LabelBinarizer,
        "age": MinMaxScaler,
        "default": StandardScaler
    }
}

regressor = [#GaussianNB(),
                 KNeighborsClassifier(n_neighbors=50),
                 #GaussianMixture(),
                 #AdaBoostClassifier(),
                 #BaggingClassifier(n_jobs=-1),
                 #LabelPropagation(),
                 #CalibratedClassifierCV(),
                 #SGDClassifier(loss='modified_huber'),
                 #LabelSpreading(),
                 #SVC(probability=True),
                 #RandomForestClassifier(),
                 #DecisionTreeClassifier(),
                 #LogisticRegression(),
                 ExtraTreesClassifier(n_estimators=10),
                 #ExtraTreeClassifier(),
                 #GradientBoostingClassifier(),
                 #GaussianProcessClassifier(),
                 #DummyClassifier(),
                 #LinearDiscriminantAnalysis(),
                 #QuadraticDiscriminantAnalysis(),
                 MLPClassifier(hidden_layer_sizes=(10, 5), activation="logistic", max_iter=800)
                 ]
regressor_parameter_grid = dict(
)

results = threshold_scan(
    50,
    dataset,
    data_config_dict,
    regressor,
    regressor_parameter_grid,
    cnd=0,
    #show_best_runs=5,
    #ext_dataset=ext_dataset
)
print(results)
results.fillna(0).to_excel('reports/ROC Analysis/CutNhanes_Scan.xlsx', index=False)