import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso

from data_transformers import column_filter, mean_body_part_transformer, age_transformer, bmi_transformer, average_transformer

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 199
pd.options.display.width = None

essential_transformers = [mean_body_part_transformer(False)] + [average_transformer(cname, cname + "\\d+") for cname in ["ArmR", "ThighR", "Waist", "Hip"]]
# List of functions

loader_params = dict(
    data_transformers=essential_transformers,
    label_cname="TOTAL_FAT",
    scaler_config={"SEX": LabelBinarizer,
                   "age": MinMaxScaler,
                   "default": StandardScaler},
)

loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        "M": essential_transformers + [column_filter("SEX", "M")],
         "F": essential_transformers + [column_filter("SEX", "F")],
         "M/F": essential_transformers,
    },
)

# blood glucose, total bone mineral content (in grams) (bmc), lean body mass
#regressor = MLPRegressor(max_iter=600,
 #                        activation='identity',
  #                       batch_size=8,
   #                      solver='lbfgs', #changed from ADAM to lbfgs
    #                     #learning_rate="adaptive",
     #                    hidden_layer_sizes=(1,))

regressor = Lasso(max_iter=1000, alpha=0.009, tol = .0001)
regressor_parameter_grid = dict(
    #activation=['logistic', 'identity'],
    #solver=['sgd', 'adam']
)
