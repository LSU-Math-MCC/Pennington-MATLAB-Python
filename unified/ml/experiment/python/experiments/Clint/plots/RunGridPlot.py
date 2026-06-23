import pandas as pd
from utilities.plotutils import hist_df, param_grid_plot, graph_predicted_vs_actual
import numpy as np
import math

from sklearn.preprocessing import normalize, MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_excel('../reports/TargetRegTrials_StykuDataSet_3.xlsx')

mask = (df['mean_train_r2'] >= 0) & (df['mean_train_r2'] <= 1) &\
       (df['mean_test_r2'] >= 0) & (df['mean_test_r2'] <= 1) &\
       (df['target'] == 'TOTAL_LEAN')
df = df.loc[mask]
param_grid_plot(df,
                [['target','dataset','regressor'],
                 ['SEX','bmi','age', 'volumes', 'measurements', 'a_b']],
                ['mean_train_r2', 'mean_test_r2'],
                ncols=3
                ).show()
