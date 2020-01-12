import numpy as np, pandas as pd, seaborn as sns
import os
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, SelectKBest
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from DataSets import StykuDataSet, DataSet, to_DataSet
from ml_core import run_batch
from utilities.data_transformers import column_filter


def hold_dataset(dataset, rewrite=False):
    name = type(dataset).__name__
    if not os.path.exists('./extracted datasets'):
        os.makedirs('./extracted datasets')
    if not os.path.isfile(f"./extracted datasets/{name}") or rewrite:
        dataset.load_data().to_csv(f"./extracted datasets/{name}.csv")
# hold_dataset(StykuDataSet(include_classes=True))
df = pd.read_csv('./extracted datasets/StykuDataSet.csv')
# df = df.loc[df['SEX'] == 'M']

'''
Mathematical Functions
'''


def info_entropy(c_binned):
    c_PMF = c_binned / float(np.sum(c_binned)) # normalized
    c_PMF = c_PMF[np.nonzero(c_PMF)]  # for speed, and reduces to 1D case
    return -np.dot(c_PMF, np.log2(c_PMF))

def vect_MI(x, y, bins):
    x_bins = np.histogram(x, bins)[0]
    y_bins = np.histogram(y, bins)[0]
    xy_bins = np.histogram2d(x, y, bins)[0]

    Hx = info_entropy(x_bins)
    Hy = info_entropy(y_bins)
    Hxy = info_entropy(xy_bins)
    return Hx + Hy - Hxy

def fast_vect_MI(x, y, bins):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
    xy_bins = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=xy_bins)

def mtrx_MI(m):
    m = m.T  # loop over columns of  m (features) instead of rows (subjects)
    return np.array([[vect_MI(i_feat, j_feat, 5) for j_feat in m] for i_feat in m]) # best number of bins?

'''
Information-Based Feature Selection
'''


def mRMR(df, feature_cnames, target_cname, metric='covariance', return_best=1,
         best_heatmap=False, best_pairplot=False):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI  # WIP
    }[metric]
    # the first column is the target, all columns are standardized
    df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean())/x.std(), axis=0)
    df_cov = mtrx_method(df.to_numpy())
    n = len(feature_cnames)
    # create list of boolean sequences each starting with 1 (see range)
    bool_list = [np.array([b == '1' for b in list('{:0{}b}'.format(i, n))]) \
                 for i in range(2**n + 1, 2**(n + 1))]

    mRMR = np.array([[-np.inf, [], 0, 0] for n in range(return_best)])
    for mask in bool_list:
        s = np.count_nonzero(mask) - 1 # number of (non-target) features
        covs = df_cov[mask][:, mask]  # apply mask to rows, then columns
        relevance = np.sum(covs[1:, 0]) / s # target_covs = covs[1:, 0]
        redundancy = np.sum(covs[1:, 1:]) / (s ** 2) # feature_covs = covs[1:, 1:]
        mRMR0 = relevance - redundancy
        try:
            # check for existence of the first index where mRMR0 is higher than the listed score
            idx = np.nonzero(mRMR[:, 0] < mRMR0)[0][0]
            # all higher indices must be moved up
            #print(idx, '\n', mRMR, '\n')
            if idx < return_best - 1:
                for i in range(idx + 1, return_best)[::-1]:  # preform this operation from the top down to avoid over-writing
                    mRMR[i] = mRMR[i - 1]
            # place mRMR0 in the first slot of mRMR for which mRMR[:, 0] < mRMR0
            mRMR[idx] = [mRMR0, np.array(feature_cnames)[mask[1:]], relevance, redundancy]
            #print(mRMR, '\n\n')
        except IndexError:
            continue
    if best_heatmap:
        ax = sns.heatmap(df_cov, xticklabels=[target_cname] + feature_cnames, yticklabels=[target_cname] + feature_cnames)
        for labx, laby in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            text = labx.get_text()
            if text in mRMR[0, 1]:  # lets highlight row 2
                labx.set_weight('bold')
                laby.set_weight('bold')
        plt.show()
    if best_pairplot:
        df_pairplot = df[[target_cname] + mRMR[0, 1].tolist()]
        sns.pairplot(df_pairplot, vars=df_pairplot.columns[:-1], markers="+", kind="reg")
        plt.show()
    return np.array(mRMR)

# df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean())/x.std(), axis=0).to_numpy()
# print(mtrx_MI(df),'\n', '\n',
#       np.array([[fast_vect_MI(i_feat, j_feat, 5) for j_feat in df.T] for i_feat in df.T])
#       , '\n', '\n') = 5

# df = df[[target_cname] + feature_cnames].dropna()
# X = df[feature_cnames].to_numpy()
# y = df[target_cname].to_numpy()
# f_dict = {feature_cnames[idx]: np.array2string(X[0:4, idx]) for idx in range(len(feature_cnames))}
# f_dict_inv = dict(map(reversed, f_dict.items()))
# print(f_dict, dict(f_dict))
#
# X_ksel = SelectKBest(f_regression, k=k).fit_transform(X, y) # chi2 -ValueError: Input X must be non-negative.
# X_cnames = [f_dict_inv[np.array2string(X[0:4, idx])] for idx in range(k)]
# print(X_ksel)
# print(X_cnames)


def plot_coefs(df, feature_cnames, target_cname, models = [LinearRegression()], plot = True):
    '''
    :param models: List of estimators with coef_ attribute.
      LinearRegression(), Ridge(), Lasso()
    '''
    df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    X = df[feature_cnames].to_numpy()
    y = df[target_cname].to_numpy()

    results_df = pd.DataFrame()
    results_df['Features'] = feature_cnames
    results_df.set_index('Features', inplace=True)
    models = models if isinstance(models, list) else [models]
    model_cnames = []
    for mdl in models:
        mdl = mdl.fit(X, y)
        results_df[f'{type(mdl).__name__}'] = mdl.coef_
        model_cnames += [f'{type(mdl).__name__}']
    if len(models) > 1:
        results_df['avg'] = results_df[model_cnames].mean(axis=1)
        results_df = results_df.iloc[results_df['avg'].abs().argsort()[::-1]]
        del results_df['avg']
    if plot:
        results_df.plot(kind='bar')
        plt.title(f'Feature Coefficients for {target_cname}')
        plt.xlabel('Features')
        plt.ylabel('Coefficients')
        plt.show()
    print(results_df)

# estimators = all_estimators()
#
# for name, class_ in estimators:
#     if hasattr(class_, 'coef_'):
#         print(name)

# g = sns.pairplot(df[volumes+['LDL_risk']].dropna(), vars=df.columns[:-1], hue='LDL_risk', markers="+", kind="reg")
# plt.show()


def select_n_best_metric(df, feature_cnames, target_cname, n=3, metric='covariance'):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI  # WIP
    }[metric]
    df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean()) / x.std(), axis=0).to_numpy()
    df_cov = mtrx_method(df)
    target_covs = df_cov[0, 1:]
    cov_indices = np.argsort(abs(target_covs))[::-1]  # find indices of target covs, highest to lowest
    return [feature_cnames[i] for i in cov_indices[0:n]]


def remove_n_worst_metric(df, feature_cnames, target_cname, n=3, metric='covariance'):
    return select_n_best_metric(df, feature_cnames, target_cname, len(feature_cnames) - n, metric)


def select_and_run(df, feature_cnames, target_cname, model, upto=10, graph_r2=True):
    dataset = to_DataSet(df)
    selections = [select_n_best_metric(df, feature_cnames, target_cname, n=n) for n in range(1, upto+1)]
    # selections = [arr.tolist() for arr in mRMR(df, feature_cnames, target_cname, return_best=upto)[:, 1]]
    data_config_dict = {
        'target_cnames': [target_cname],
        'feature_options': {
            'Selected': {', '.join(cname_list): cname_list for cname_list in selections}
        },
        'transform_options': {
            'SEX': {
                "M": [column_filter("SEX", "M")],
                "F": [column_filter("SEX", "F")],
                "M/F": []
            }
        },
        'scalar_config': {
            # "SEX": LabelBinarizer,
            "age": MinMaxScaler,
            "default": StandardScaler
        }
    }
    results = run_batch(dataset, data_config_dict, model, n_cores=1)
    if graph_r2:
        collapse = (('test_r2', ['test', 'r2']), ('train_r2', ['train', 'r2']))
        name = ['split', 'score_type']
        dfs = []
        for col in collapse:
            df0 = results.copy()
            df0[col[1][1]] = df0[col[0]]
            df0[name[0]] = col[1][0]
            df0[name[1]] = col[1][1]
            dfs += [df0]
        results_plot = pd.concat(dfs)

        results_plot['Selection Number'] = results_plot.index.map(lambda x: x % upto)
        sns.catplot(x='Selection Number', y="r2", hue="split", col="SEX", data=results_plot, kind="bar", aspect=.7)
        plt.show()
    pd.set_option('display.max_colwidth', -1)
    print(results['Selected'].head(upto))
    return results


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


bmi = ['BMI1']
sex = ['SEX']
volumes = ["TotalVolume",# "headVolume",
           "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
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

all_features = bmi + volumes + m_all + a_b_all# + sex
feature_cnames = m_common + volumes
target_cname = 'TOTAL_FAT'

# print(mRMR(df, feature_cnames, target_cname, return_best=10))
mRMR(df, feature_cnames, target_cname, return_best=10, best_heatmap=True, best_pairplot=True)
# print(mRMR(df, feature_cnames, target_cname, method='mutual information', return_best=6))

plot_coefs(df, feature_cnames, target_cname, model = [Ridge(), Lasso(alpha=.001)])

print(select_n_best_metric(df, feature_cnames, target_cname, n=2))
print(remove_n_worst_metric(df, feature_cnames, target_cname, n=2))

# select_and_run(df, feature_cnames, target_cname, LinearRegression())