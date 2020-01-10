import numpy as np
import glob
import os

from joblib import load
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier

from ml_core import ext_train_save, threshold_clf
from DataSets import StykuDataSet, NhanesDataSet

bmi = ['BMI1']
sex = ['SEX']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", #"hip circ",
            "rThighGirth", "rbicepGirth"]


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

# LOADING THE MODEL AND USING IT TO PREDICT
random_input = np.random.rand(n_features)
random_input = random_input.reshape(1, -1)  # reshape recommended for single-sample predictions
print('Random Input:', random_input)

pretrained_model = load(model_rel_path)
prediction = pretrained_model.predict(random_input)
print('Generated output: ', prediction)