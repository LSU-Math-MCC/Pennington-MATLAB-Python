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

from DataSets import SS20DataSet, CombinedDataSet, DataSet, StykuDataSet_2, NhanesDataSet
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids, discrete_class, column_filter, mean_body_part_transformer, average_transformer
from utilities.paramutils import combine_options
from utilities.folder_searcher import Searcher
from utilities.DirectoryGrab import DirGrab
from utilities.data_merger import ListMaker, ListStandardizer, CBDrowMaker, MergeMan
from ml_core import run_batch, threshold_scan
import datetime
now = datetime.datetime.now()

dataset = NhanesDataSet()

bmi = ['BMI1']
sex = ['SEX']
m_common = ["waist circ", "rThighGirth", "rbicepGirth"]
race_nh = ['RACETH']
SA_nh = ["DXDRATOT", "DXDRLTOT", "DXDLLTOT", "DXDLATOT", "DXXHEA", "DXDTOA", "DXXPEA", "DXXTSA", "DXXRRA", "DXXLRA"]

data_config_dict = {
    'target_cnames': ['GLU_risk', 'LDL_risk', 'HDL_risk'],
    'feature_options': {
        "bmi": {
            # "N": [],
            "Y": ["BMI1"]
        },
        "age": {
            "N": [],
            "Y": ["age"]
        },
        "sex": {
            #"N": [],
            "Y": ["SEX"]
        },
        "surface_area": {
            "none": [],
            "all": SA_nh
        },
        # "skin_fold": {
        #     "N":[],
        #     "Y":["BMXTRI","BMXSUB"]
        # },
        "measurements": {
            # "none": [],
            "common": ["waist circ", "rThighGirth", "rbicepGirth"],
            "all": ["rLegLength", "RArmLength", "waist circ", "rbicepGirth", "rCalfCirc", "rThighGirth"],
        }
        # "features": {
        #     # Publication-style parameter grid (use w/out other feature options)
        #     'BMI, MEAS': bmi + sex + m_common + race_nh,
        #     'BMI, MEAS, DXA': bmi + sex + m_common + ['TOTAL_FAT', 'TOTAL_LEAN'] + race_nh,
        #     'BMI, MEAS, SA': bmi + sex + m_common + SA_nh + race_nh,
        #     'BMI, MEAS, DXA, SA': bmi + sex + m_common + SA_nh + ['TOTAL_FAT', 'TOTAL_LEAN'] + race_nh,
        #
        #     #'BMI, SEX': bmi + sex,
        #     #'BMI, SEX, ETH': bmi + sex + race_nh,
        # }
    },
    'transform_options': {
        # 'RACETH': {
        #     "1": essential_transformers + [column_filter("RACETH", 1)],
        #     "2": essential_transformers + [column_filter("RACETH", 2)],
        #     "3": essential_transformers + [column_filter("RACETH", 3)],
        #     "4": essential_transformers + [column_filter("RACETH", 4)],
        #     "All": essential_transformers,
        # },
        'SEX': {
            "F": [column_filter("SEX", 1)],
            "M": [column_filter("SEX", 2)],
            "M/F": [],
        }
    },
    'scalar_config': {
        "SEX": LabelBinarizer,
        "age": MinMaxScaler,
        "default": StandardScaler
    }
}

classifiers = [#GaussianNB(),
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

results = threshold_scan(
    [.5, .8],
    dataset,
    data_config_dict,
    classifiers,
    cv_params=dict(
        cnd=0,
        external_validation=0.15  # use 15% of dataset for external validation
    ),
    #show_best_runs=5,
    #ext_dataset=ext_dataset,
    n_cores=-1
)
print(results)