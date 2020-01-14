import numpy as np, pandas as pd
import glob
import os

from joblib import load
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

from ml_core import ext_train_save
from DataSets import StykuDataSet

# SAVING THE MODEL
model = MLPRegressor(solver="lbfgs", activation="identity", max_iter=800, hidden_layer_sizes=(1,))
dataset = StykuDataSet()
bmi = ['BMI1']
sex = ['SEX']
age =['age']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", "hip circ",
            "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
            "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]
scaler_config = {
    "SEX": LabelBinarizer,
    "age": MinMaxScaler,
    "default": StandardScaler
}
cnames = m_all + bmi + age + a_b_all + volumes + sex
n_features = len(cnames)
target_cname = 'TOTAL_LEAN'

print(f'Training and storing model for {target_cname} prediction with {n_features} features from {type(dataset).__name__}')
ext_train_save(model, dataset, cnames, target_cname, scaler_config, eval_type='regressor')

# LOADING THE MODEL AND USING IT TO PREDICT
all_models = glob.glob('models/*_mdl.joblib')
model_rel_path = max(all_models, key=os.path.getctime)
print('Model location: ', model_rel_path)

all_scalars = glob.glob('models/*_sclrs.joblib')
sclr_rel_path = max(all_scalars, key=os.path.getctime)
print('Scalar location: ', sclr_rel_path, '\n')

random_input = np.array([np.random.rand(n_features - 1).tolist() + ['M']], dtype='O')
# random_input = random_input.reshape(1, -1)  # reshape recommended for single-sample predictions
print('Random Input:', random_input)

pretrained_model = load(model_rel_path)
prefit_scalars = load(sclr_rel_path)
input_df = pd.DataFrame(data=random_input, columns=cnames)
input_scaled = prefit_scalars.transform(input_df)
prediction = pretrained_model.predict(input_scaled)
prediction_df = pd.DataFrame(data=[prediction], columns=[target_cname])
prediction_scaled = prefit_scalars.transform(prediction_df)
print('Generated output: ', '\n', prediction_scaled)