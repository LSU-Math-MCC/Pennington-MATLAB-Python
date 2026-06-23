import numpy as np
import pandas as pd
from datetime import date
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
from sklearn.base import clone
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids, discrete_class, column_filter, mean_body_part_transformer, average_transformer
from runner_styku_classifier import StykuClassDataSet_2

# Good SE post: https://stats.stackexchange.com/questions/139042/ensemble-of-different-kinds-of-regressors-using-scikit-learn-or-any-other-pytho
# Classes for custom sklearn estimator
class derivedEstimator:
    def __init__(self, est, **kwargs):
        self.est = est
        self.params = {**kwargs}

    def fit(self, X, y, **kwargs):
        self.est.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.est.predict(X, **kwargs)

    def get_params(self, **kwargs):
        return self.params


class weightedVote_clf(derivedEstimator):
    def __init__(self, clf, target_weights, **kwargs):
        '''
        This class implements weighted target-based voting within a scikit-learn estimator.
        Input:  clf: SKLearn classifier
                target_weights: List of weights to apply target-wise during voting process
        '''
        super().__init__(est=clf, target_weights=target_weights, **kwargs)
        self.clf = clf
        self.weights = target_weights # TODO: Make a function that derives weights from some scoring metric

    def fit(self, X, Y, **kwargs):
        # Y is a matrix with target values as row vectors. eg. Y = [*y1, ..., *yn] w/ shape: (ntargets, nsamples)
        # We will use the ith list of target values in Y as a target vector y fit with the ith estimator
        self.est_vect = []
        for y in Y:
            clf0 = clone(self.clf) # clone the classifier to remove any previously fit parameters
            self.est_vect += [clf0.fit(X, y)]

    def predict(self, X, **kwargs):
        Pr = []
        for est in self.est_vect:
            Pr += [est.predict_proba(X, **kwargs)] # predict_proba returns a matrix w/ shape: (nsamples, nconditions)
        Pr = np.array(Pr).transpose([1, 0, 2]) # scalar index permutation: ijk -> jik
        # Pr is now a 3-tensor w/ shape: (nsamples, ntargets, nconditions (probas))
        print(np.shape(Pr))

        predictions = []
        for pred_mtrx in Pr:
            # for each subject, the pred_mtrx is a matrix w/ shape: (ntargets, nconditions)
            print(pred_mtrx)
            weighted_probas = np.matmul(np.transpose(self.weights), pred_mtrx) # (P^T * w)^T = w^T * P (think of P^T * w by col-row)
            print(weighted_probas)
            predictions += [np.argmax(weighted_probas)]
        return predictions


bmi = ['BMI1']
sex = ['SEX']
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

features = bmi + sex
report_df = pd.DataFrame()

# Data extraction and train/test split
loader_params = dict(
    scaler_config={"SEX": LabelBinarizer,
                   "age": MinMaxScaler,
                   "default": StandardScaler,
                   },
)

# Create matrix Y
targets = ['GLU_risk', 'HBA1C_risk']

X, Y = None, None
for target in targets:
    data = StykuClassDataSet_2().extract_data(features, target, **loader_params)
    X = data.x_scaled.values if X is None else X
    Y = np.empty(shape=(0, len(X))) if Y is None else Y # to append np array, all dimensions except for the concatenation axis must match
    y = list(data.y.values.astype('int64').ravel())
    Y = np.append(Y, [y], axis=0) # Add each entry as row vector #Y = np.vstack((Y, list(data.y.values.astype('int64').ravel())))

print(np.shape(X), np.shape(Y))

clf = weightedVote_clf(GaussianNB(), target_weights=[.96, .87])
clf.fit(X, Y)
clf.predict(X)
exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Run the classifier at various thresholds
cnd = 0  # 'Healthy' on all targets except HDL_risk
cv = 4

for threshold in np.linspace(0, 1, 1000):
    run_dict = dict()  # Can't initially make DataFrame because we're storing list (matrices) as entries
    run_dict['target'] = target
    run_dict['n_samples'] = len(y)
    run_dict[f'condition {cnd} threshold'] = threshold

    clf = threshold_clf(GaussianNB(), 0, threshold)

    clf_cv = cross_validate(clf, X_train, y_train, scoring='accuracy', cv=cv,
                            return_estimator=True, return_train_score=True)
    run_dict['accuracy_train'] = sum(clf_cv['train_score']) / cv

    internal_test_scores = clf_cv['test_score']
    external_test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clf_cv['estimator']]
    run_dict['inAvg_accuracy_test'] = sum(internal_test_scores) / cv
    run_dict['exAvg_accuracy_test'] = sum(external_test_scores) / cv

    best = clf_cv['estimator'][np.argmax(internal_test_scores)]
    y_pred = best.predict(X_test)

    mtrx = confusion_matrix(y_test, y_pred)
    run_dict['Confusion Matrix'] = mtrx

    # METRICS: https://en.wikipedia.org/wiki/Confusion_matrix (2 x 2 Reduction)
    tp = mtrx[cnd][cnd]
    fn = np.sum(mtrx[cnd, :]) - tp
    fp = np.sum(mtrx[:, cnd]) - tp
    tn = np.sum(np.diagonal(mtrx)) - tp

    run_dict[f'Condition {cnd} Matrix'] = [[tn, fp], [fn, tp]]
    if (fp + tp) != 0:
        run_dict['TPR'] = tp / (fn + tp)  # AKA precision
        run_dict['FNR'] = fn / (fn + tp)  # When cnd is 'healthy', this is the MISSED DETECTION rate (detecting health)
    else:
        run_dict['TPR'] = np.nan  # 'No Positives'
        run_dict['FNR'] = np.nan  # '-'
    if (fn + tn) != 0:
        run_dict['FPR'] = fp / (fp + tn)  # When cnd is 'healthy', this is the FALSE ALARM rate
        run_dict['TNR'] = tn / (fp + tn)
    else:
        run_dict['FPR'] = np.nan  # 'No Negatives'
        run_dict['TNR'] = np.nan  # '-'
    run_dict['recall'] = tp / (tp + fn)
    if run_dict['TPR'] != np.nan:  # 'No Positives'
        run_dict['F1'] = 2 * run_dict['TPR'] * run_dict['recall'] / (run_dict['TPR'] + run_dict['recall'])

    run_dict['estimator'] = best

    run_df = pd.DataFrame(data={key: {0: value} for key, value in run_dict.items()})
    report_df = pd.concat([report_df, run_df], sort=False)

print(report_df)
report_df.fillna(0).to_excel('reports/curvyROC.xlsx', index=False)
