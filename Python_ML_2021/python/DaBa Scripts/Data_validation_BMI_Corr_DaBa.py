import pandas as pd
import numpy as np
import pickle
path = '/home/nariman/1-Drive-Nariman/Penington/body-shape-ml/python/experiments/Nariman/data1.csv'

model_coef = [-4.00033766, 0.15823802, -0.4570008, 2.2903876]
model_intercept = 31.367017986676288

def valid_data ():
    data = pd.read_csv(path)

    features = [column for column in data.columns if column not in ["RIAGENDR", 'RIDAGEYR', 'RACETH', 'bmxbmi']]

    X = data.drop(features, axis=1)
    y = data[features[0]]

    # t_model = pickle.load(open('trained_model.sav', 'rb'))
    y_predicted = np.dot(X, model_coef) + model_intercept

    df = pd.DataFrame({'Actual_waist': y, 'Predicted_waist': y_predicted, 'Error': abs(y_predicted - y)
                          , 'Squared_Error': ((y_predicted - y) ** 2)})
    df1 = df.loc[(df['Error'] > (df['Error'].mean()) + 2 * (df['Error'].std())) | (df['Error'] < (df['Error'].mean()) - 2 * (df['Error'].std()))]

    # print(df1[])
    df2 = pd.concat([data.loc[df1.index], df1['Predicted_waist']], axis=1, sort=False)
    # df2 = pd.merge([data.loc[df1.index], df1[y_predicted]])
    if df1.shape[0]/data.shape[0]*100 > 5:
        print('The Input File has more than %5 invalid data')
        print(df2)

valid_data()
