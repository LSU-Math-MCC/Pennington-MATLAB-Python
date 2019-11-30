import numpy as np
import glob
import os

from joblib import load
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

from runner_modular import ext_train_save
from datasets import StykuDataSet, NhanesDataSet

bmi = ['BMI1']
sex = ['SEX']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", #"hip circ",
            "rThighGirth", "rbicepGirth"]

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


# SAVING THE MODEL
model = threshold_clf(MLPClassifier(hidden_layer_sizes=(10,5,3)), cnd=0, threshold=.95)
dataset = NhanesDataSet()
cnames = bmi + sex + m_common
n_features = len(cnames)
target_cname = 'LDL_risk'

print(f'Training and storing model for {target_cname} prediction with {n_features} features from {type(dataset).__name__}')
ext_train_save(model, dataset, cnames, target_cname, eval_type='classifier', cnd=0)

all_models = glob.glob('models/*.joblib')
model_rel_path = max(all_models, key=os.path.getctime)
print('Model location: ', model_rel_path, '\n')

# LOADING THE MODEL
random_input = np.random.rand(n_features)
random_input = random_input.reshape(1, -1)  # reshape recommended for single-sample predictions
print('Random Input:', random_input)

pretrained_model = load(model_rel_path)
prediction = pretrained_model.predict(random_input)
print('Generated output: ', prediction)