import numpy as np
import pandas as pd
import os
from sklearn.utils.testing import all_estimators
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, f_regression, mutual_info_regression, SelectKBest
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
import matplotlib.pyplot as plt
#from datasets import StykuDataSet


def hold_dataset(dataset, rewrite=False):
    name = type(dataset).__name__
    if not os.path.exists('./extracted datasets'):
        os.makedirs('./extracted datasets')
    if not os.path.isfile(f"./extracted datasets/{name}") or rewrite:
        dataset.load_data().to_csv(f"./extracted datasets/{name}.csv")
#hold_dataset(StykuDataSet())
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


def mRMR(df, feature_cnames, target_cname, method='covariance', return_best=1):
    mtrx_method = {
        'covariance': lambda m: np.cov(m, rowvar=False),
        'mutual information': mtrx_MI  # WIP
    }[method]
    # the first column is the target, all columns are standardized
    df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean())/x.std(), axis=0).to_numpy()
    df_cov = mtrx_method(df)
    # print(mtrx_MI(df),'\n', '\n',
    #       np.array([[fast_vect_MI(i_feat, j_feat, 5) for j_feat in df.T] for i_feat in df.T])
    #       , '\n', '\n')
    n = len(feature_cnames)
    # create list of boolean sequences each starting with 1 (see range)
    bool_list = [np.array([b == '1' for b in list('{:0{}b}'.format(i, n))]) \
                 for i in range(2**n + 1, 2**(n + 1))]

    mRMR = np.array([[0, [], 0, 0] for n in range(return_best)])
    for mask in bool_list:
        s = np.count_nonzero(mask) - 1 # number of (non-target) features
        covs = df_cov[mask][:, mask]  # apply mask to rows, then columns
        relevance = np.sum(covs[1:, 0]) / s # target_covs = covs[1:, 0]
        redundancy = np.sum(covs[1:, 1:]) / (s ** 2) # feature_covs = covs[1:, 1:]
        mRMR0 = relevance - redundancy
        try:
            idx = np.nonzero(mRMR[:, 0] < mRMR0)[0][0]
            # place mRMR0 in the first slot of mRMR for which mRMR[:, 0] < mRMR0
            mRMR[idx] = [mRMR0, np.array(feature_cnames)[mask[1:]], relevance, redundancy]
        except IndexError:
            continue
    return np.array(mRMR)








bmi = ['BMI1']
sex = ['SEX']
volumes = ["TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume", "trunkVolume"]
m_common = ["waist circ", #"hip circ",
            "rThighGirth", "rbicepGirth"]
m_all = ["Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                    "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                    "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                    "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength", "crotchHeight"]
a_b_four = ["waist circ A_B", "hip circ A_B", "rThighGirth A_B", "rbicepGirth A_B"]
a_b_all = ["Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
            "rCalfCirc A_B", "lCalfCirc A_B", "lWristGirth A_B", "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B",
            "lBicepGirth A_B", "rAnkle A_B", "rWristGirth A_B", "Lankle A_B"]

feature_cnames = a_b_all
target_cname = 'TOTAL_FAT'

# print(mRMR(df, feature_cnames, target_cname, return_best=6))
# print(mRMR(df, feature_cnames, target_cname, method='mutual information', return_best=6))
#print(mRMR(df, feature_cnames, target_cname, return_best='all'))

# k = 5
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

feature_cnames = bmi + m_all + a_b_all + volumes

def plot_coefs(df, feature_cnames, target_cname, model = [LinearRegression()]):
    '''
    :param model: List of estimators with coef_ attribute.
      LinearRegression(), Ridge(), Lasso()
    '''
    df = df[[target_cname] + feature_cnames].dropna().apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    X = df[feature_cnames].to_numpy()
    y = df[target_cname].to_numpy()

    results_df = pd.DataFrame()
    results_df['Features'] = feature_cnames
    results_df.set_index('Features', inplace=True)
    model = model if isinstance(model, list) else [model]
    for mdl in model:
        mdl = mdl.fit(X, y)
        results_df[f'{type(mdl).__name__}'] = mdl.coef_
    results_df.plot(kind='bar')
    plt.title(f'Feature Coefficients for {target_cname}')
    plt.xlabel('Features')
    plt.ylabel('Coefficients')
    plt.show()


plot_coefs(df, feature_cnames, target_cname, model = [Ridge(), Lasso(alpha=.01)])

# estimators = all_estimators()
#
# for name, class_ in estimators:
#     if hasattr(class_, 'coef_'):
#         print(name)