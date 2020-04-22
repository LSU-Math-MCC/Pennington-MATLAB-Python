import math, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt

def feature_explore_single(df, feature_cname):
    print(df[feature_cname].describe())
    # df.hist(column=feature_cname,
    #         by='DEM_SEX',
    #         # sharex=True,
    #         histtype='barstacked',
    #         label=feature_cname
    #         )
    dfm = df.loc[df['DEM_SEX'] == 'M']
    dff = df.loc[df['DEM_SEX'] == 'F']
    plt.hist([dfm[feature_cname], dff[feature_cname]],
            histtype='barstacked',
             label=feature_cname)
    plt.show()

    print(
        pd.concat([
        dfm[feature_cname].describe(),
        dff[feature_cname].describe()
    ], keys=['M','F'])
    )


def feature_explore(df, feature_cname, age_groups=False) -> pd.DataFrame:
    """
    Print histograms of columns in feature_cname and return dataframe of statistics on each column.

    :param df: pandas dataframe
    :param feature_cname: Column name from df (or list of column names)
    :param age_groups: Bool. Describe dataframe over age groups
    :return: dataframe of statistics on each feature_cname
    """
    if isinstance(feature_cname, str):
        feature_cname = [feature_cname]
    # df.hist(by='DEM_SEX',
    #         # sharex=True,
    #         )
    dfm = df.loc[df['DEM_SEX'] == 'M']
    dfm = dfm[feature_cname + ['DEM_AGE']]
    dff = df.loc[df['DEM_SEX'] == 'F']
    dff = dff[feature_cname + ['DEM_AGE']]

    n_feat = len(feature_cname)
    if n_feat == 1:
        plt.hist([dfm[feature_cname[0]], dff[feature_cname[0]]],
                 histtype='barstacked')
        plt.show()
    else:
        # if n_feat < 6:
        #     f, axes = plt.subplots(2, n_feat, figsize=(3 * 3, 3 * n_feat / 2))
        # else:
        #     # # More rectangular plots for large feature sets
        #     # n_col = math.ceil(n_feat / math.ceil(math.sqrt(len(feature_cname))))
        #     # f, axes = plt.subplots(n_col, 2*n_col, figsize=(6*n_col, 3*n_col))
        n_row = 2*((n_feat // 5) + 1)
        n_col = min(5, n_feat)
        f, axes = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

        for i, feature in enumerate(feature_cname):
            k = 2 * i
            sns.distplot(dfm[feature].dropna(), color="skyblue", ax=axes[k % n_row, k // n_row])
            k = k + 1
            sns.distplot(dff[feature].dropna(), color="pink", ax=axes[k % n_row, k // n_row])
        plt.show()

    if age_groups:
        # describe by age group if option is True
        age_groups = {**{"18 to 29": [18, 29]},
                      **{str(n) + " to " + str(n + 9): [n, n + 9] for n in range(30, 61, 10)},
                      **{"70 to 90": [70, 90]}}
        dfm_list = []
        dff_list = []
        for group_name, group_range in age_groups.items():
            dfm_list += [dfm.loc[(group_range[0] <= dfm['DEM_AGE']) & (dfm['DEM_AGE'] <= group_range[1])].describe()]
            dff_list += [dff.loc[(group_range[0] <= dff['DEM_AGE']) & (dff['DEM_AGE'] <= group_range[1])].describe()]
        dfm = pd.concat(dfm_list, keys=age_groups.keys())
        dff = pd.concat(dff_list, keys=age_groups.keys())
        df_data = pd.concat([dfm, dff], keys=['M', 'F'])
    else:
        df_data = pd.concat([dfm.describe(), dff.describe()], keys=['M','F'])

    del df_data['DEM_AGE']
    return df_data


df = pd.read_excel('../data/ShapeUp/pdDataStorage_v4.xlsx')
m_common = ['DA_3DO3_CIRC_W', 'DA_3DO3_CIRC_H', 'DA_3DO3_CIRC_Th', 'DA_3DO3_CIRC_B']
ER_all = ['DA_3DO3_ER_Ch', 'DA_3DO3_ER_W', 'DA_3DO3_ER_H', 'DA_3DO3_ER_Th', 'DA_3DO3_ER_C', 'DA_3DO3_ER_Wr', 'DA_3DO3_ER_F', 'DA_3DO3_ER_B', 'DA_3DO3_ER_A']

feature_explore(df, 'DA_3DO3_CIRC_Ch')
df = feature_explore(df, m_common+ER_all,age_groups=True)
print(df)