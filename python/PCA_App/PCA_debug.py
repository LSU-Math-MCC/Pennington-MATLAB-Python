import glob, os, pandas as pd, seaborn as sns, numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import SCORERS
from utilities.PathMaker import PathMan
from utilities.data_transformers import standardize_subject_ids, cut_subject_ids
from python.PCA_App.Extract_Ply import rename_subs, sort_by_gender, extract_ply, split_male, split_female
from python.utilities.DirectoryGrab import DirGrab
from python.PCA_App.PCA_rowwise import principal_component_analysis
from python.PCA_App.point_reduction_script import FileConverter
from python.utilities.FolderToFolder import MoverMan
from PCA_App.runner_PCA_auto import PCA_ml
from python.utilities.PathMaker import PathMan
from python.utilities.data_transformers import cut_subject_ids
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoLars, BayesianRidge, SGDRegressor, ElasticNet, RANSACRegressor
from utilities.common_functions import append_dict, partition, df_reorder_columns, timestamp, require_tuple
from PCA_App.runner_PCA_auto import PCAAutoDataSet
from DataSets import to_DataSet
import matplotlib.ticker as ticker
from utilities.feature_selection import select_and_run_targs_MF, select_targs_MF_SIMA
import string


Path = PathMan()
GITPATH = Path.getter()
PCAPATH = GITPATH + 'python/PCA_App/'


def rename_fitted(scan_path):
    pkgs = glob.glob(scan_path + "*.ply")
    pkgs_dict = {pkg: pkg[:-4] + "_fitted.ply" for pkg in pkgs if pkg[-10:-4] != 'fitted'}
    for k, v in pkgs_dict.items():
        try:
            os.rename(k, v)
        except FileExistsError:
            os.remove(k)

def rename_pkgs(fitted_dir=None):
    fitted_dir = GITPATH + "python/PCA_App/process/fitted/" if fitted_dir is None else fitted_dir

    pkg_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_181101.xlsx", sheet_name="Joints")
    pkg_df = pkg_df[['SubjectID', 'Scan_Package_ID']]
    pkg_df.drop_duplicates('Scan_Package_ID', inplace=True)
    pkg_df.set_index('Scan_Package_ID', inplace=True)
    pkgs = glob.glob(fitted_dir + "pkg-*_fitted.ply")
    pkgs = [pkg[len(fitted_dir):-len("_fitted.ply")] for pkg in pkgs]
    pkg_df = pkg_df.loc[pkgs]
    pkg_df['new'] = True

    subj_ids = glob.glob(fitted_dir + "*ADL*_fitted.ply")
    subj_ids = [id[len(fitted_dir):-len("_fitted.ply")] for id in subj_ids]
    subj_df = pd.DataFrame(data=subj_ids, columns=['SubjectID'])

    df = pd.concat([subj_df, pkg_df], sort=True)
    df['SubjectID'] = standardize_subject_ids(df['SubjectID'])  # Use standard df to avoid overwriting filenames
    df = df.loc[(df['new'] == True)]
    for row in df.iterrows():
        pkg = row[0]
        id = row[1]['SubjectID']
        os.rename(
            fitted_dir + f"{pkg}_fitted.ply",
            fitted_dir + f"{id}_fitted.ply"
        )
        print(f'Renamed {pkg}_fitted.ply to {id}_fitted.ply')


def make_dummy_files():
    subj_ids = glob.glob(GITPATH + "python/PCA_App/process/fitted/*ADL*_fitted.ply")
    subj_ids = [id[len(GITPATH + "python/PCA_App/process/fitted/"):-len("_fitted.ply")] for id in subj_ids]
    for id in subj_ids:
        mkr_file = GITPATH + "python/PCA_App/process/" + id + ".mkr"
        ply_file = GITPATH + "python/PCA_App/process/" + id + ".ply"
        try:
            open(mkr_file, 'x')
        except FileExistsError:
            pass
        try:
            open(ply_file, 'x')
        except FileExistsError:
            pass

def list_to_txt(list, fname="file.txt"):
    with open(fname, 'w') as output:
        for value in list:
            output.write(str(value) + '\n')
def compare_subs(path0, path1, remove=True, process_dir=None):
    process_dir = GITPATH + "python/PCA_App/process/" if process_dir is None else process_dir

    subj_paths0 = glob.glob(path0 + "*ADL*_fitted.ply")
    subj_ids0 = {id[len(path0):-len("_fitted.ply")]: id for id in subj_paths0}
    # subj_ids0 = {id[:-2]: id for id in subj_ids0 if id[:-2] == '_2' else id: id}
    folder0 = path0.split('/')[-3]
    subj_paths1 = glob.glob(path1 + "*ADL*_fitted.ply")
    subj_ids1 = {id[len(path1):-len("_fitted.ply")]: id for id in subj_paths1}
    folder1 = path1.split('/')[-3]

    missing0 = list(set(subj_ids1.keys()) - set(subj_ids0.keys()))
    if len(missing0) > 0:
        print(f'{folder0} is missing {len(missing0)}: {missing0}')
        list_to_txt(missing0, process_dir + f'missing_{folder0}.txt')
    if remove:
        for missing in missing0:
            os.remove(subj_ids1[missing])
            print(f'Missing {missing} from {folder0}, removed from {folder1}')

    missing1 = list(set(subj_ids0.keys()) - set(subj_ids1.keys()))
    if len(missing1) > 0:
        print(f'{folder1} is missing {len(missing1)}: {missing1}')
        list_to_txt(missing1, process_dir + f'missing_{folder1}.txt')
    if remove:
        for missing in missing1:
            os.remove(subj_ids0[missing])
            print(f'Missing {missing} from {folder1}, removed from {folder0}')


def remove_duplicates(fitted_dir=None):
    fitted_dir = GITPATH + "python/PCA_App/process/fitted/" if fitted_dir is None else fitted_dir
    subj_ids = glob.glob(fitted_dir + "*ADL*_fitted.ply")
    subj_ids.reverse()  # Keep B scans over A
    seen = []
    for sub in subj_ids:
        id = sub[len(fitted_dir):][:9]
        if id in seen:
            os.remove(sub)
            print(f'Found duplicate {id}, removed ply: {sub}')
        seen += [id]


def plot_PCs(process_dir=None):
    process_dir = GITPATH + 'python/PCA_App/process/' if process_dir is None else process_dir
    males = pd.read_csv(process_dir + "male_pca.csv")
    g = sns.pairplot(males[['PC1','PC2','PC3','PC4','PC5']])
    g.fig.suptitle("Male PCs")
    plt.show()
    males = pd.read_csv(process_dir + "female_pca.csv")
    g = sns.pairplot(males[['PC1','PC2','PC3','PC4','PC5']])
    g.fig.suptitle("Female PCs")
    plt.show()

def list_diff(l1, l2):
    return list(set(l1) - set(l2))
def list_to_mathematica(lst):
    return str(lst).replace('\'','\"').replace('[','{').replace(']','}')

def get_AB_list(file="male_pca.csv", print_lists=True, a_is_lt=0, process_dir=None):
    process_dir = GITPATH + "python/PCA_App/process/" if process_dir is None else process_dir
    fitted_dir = process_dir + 'fitted/'

    pca_df = pd.read_csv(process_dir + file)
    cndA = (pca_df['PC1'] < a_is_lt)

    subj_locs = glob.glob(fitted_dir + "*ADL*_fitted.ply")

    pca_subjects = cut_subject_ids(pca_df['Unnamed: 0']).to_list()
    pca_subjects = [sub for sub in subj_locs \
                if sub[len(fitted_dir):-len("_fitted.ply")][:9] in pca_subjects]

    Ascans = cut_subject_ids(pca_df.loc[cndA]['Unnamed: 0']).to_list()
    Ascans = [sub for sub in subj_locs \
          if sub[len(fitted_dir):-len("_fitted.ply")][:9] in Ascans]
    Bscans = list(set(pca_subjects) - set(Ascans))
    if print_lists:
        print(f'AScans = {list_to_mathematica(Ascans)};')
        print(f'BScans = {list_to_mathematica(Bscans)};')
    return (Ascans, Bscans)
def get_group_lists(file="male_pca.csv", splits=[0, 750], process_dir=None):
    process_dir = GITPATH + "python/PCA_App/process/" if process_dir is None else process_dir

    splits = splits + [np.inf]
    nsplits = len(splits)
    suffixes = list(string.ascii_uppercase)[:nsplits]
    suffix_pairs = partition(suffixes, 2, 1)
    prevs = []
    scans_splits = []
    for split in splits:
        scans0, scans1 = get_AB_list(file, print_lists=False, a_is_lt=split, process_dir=process_dir)
        scans_splits += [list_diff(scans0, prevs)]
        prevs += scans0
    for i in range(nsplits):
        print(f'{suffixes[i]}Scans = {list_to_mathematica(scans_splits[i])};')
    return scans_splits


def tranform_scans(scans, homogeneous_mtrx):
    for sub in scans:
        pcd = o3d.io.read_triangle_mesh(sub)
        # Find transform in Mathematica notebook MeshFit.nb
        pcd.transform(homogeneous_mtrx)
        o3d.io.write_triangle_mesh(sub, pcd)
        print(f'Transformed: {sub}')


def get_ply(process_dir=None):
    process_dir = GITPATH + 'python/PCA_App/process/' if process_dir is None else process_dir
    fitted_dir = process_dir + 'fitted/'
    grabber = DirGrab(fitted_dir)  # CHANGED
    grabber.grabByExtension(".ply")
    ply_array = grabber.getter()
    print(ply_array)
    # array containing list of all subjects by just subject ID(##ADL####) inside input folder
    subjects = rename_subs(ply_array)
    output_df = extract_ply(subjects, ply_array)  # prints wrote subject ...
    output_df.to_csv(process_dir + 'extracted_ply.csv')


def do_pca(process_dir=None):
    process_dir = GITPATH + 'python/PCA_App/process/' if process_dir is None else process_dir

    gender_csv = pd.read_csv(PCAPATH + 'subject_gender.csv')#.dropna(subset=['Gender'])
    male_subjects = gender_csv.loc[gender_csv['Gender'] == 'Male']['SubjectID'].to_list()
    female_subjects = gender_csv.loc[gender_csv['Gender'] == 'Female']['SubjectID'].to_list()
    print(len(gender_csv), len(male_subjects), len(female_subjects))

    output_df = pd.read_csv(process_dir + 'extracted_ply.csv')
    output_df = output_df.T
    output_df['ID'] = output_df.index
    output_df['ID'] = cut_subject_ids(output_df['ID'])
    male_table = output_df.loc[[(id in male_subjects) for id in output_df['ID']]]
    del male_table['ID']
    female_table = output_df.loc[[(id in female_subjects) for id in output_df['ID']]]
    del female_table['ID']
    print(len(output_df), len(male_table), len(female_table))

    # missed = list(set(output_df['ID'].to_list()) - set(male_table.index.to_list() + female_table.index.to_list()))
    # lst = list(set(output_df['ID'].to_list()) - set(gender_csv['SubjectID'].to_list()))
    # # lst = list(gender_csv.loc[(gender_csv['Gender'] != 'Male') & (gender_csv['Gender'] != 'Female')]['SubjectID'])
    # print(missed)
    # print(lst)

    principal_component_analysis(male_table, process_dir + 'male_pca.csv')
    principal_component_analysis(female_table, process_dir + 'female_pca.csv')


def main(process_dir=None):
    process_dir = GITPATH + 'python/PCA_App/process/' if process_dir is None else process_dir
    fitted_dir = process_dir + 'fitted/'

    # STEP 1 - Clean data, get lists and find transform in Mathematica
    # rename_fitted(fitted_dir)
    # rename_pkgs(fitted_dir)
    # remove_duplicates(fitted_dir)
    # # make_dummy_files()  # only needed to trick PCA_rowwise
    #
    # get_ply(process_dir)
    # do_pca(process_dir)
    # plot_PCs(process_dir)
    #
    # for file in ['male_pca.csv', 'female_pca.csv']:
    #     Ascans, Bscans = get_AB_list(file, a_is_lt=500, process_dir=process_dir)
    #     # for scan in Bscans:
    #     #     sub_id = scan[len(process_dir + "fitted/"):-4]
    #     #     print(sub_id, f'({file[0]})')
    #     #     os.remove(scan)
    #     tranform_scans(
    #         Ascans,
    #         [[-0.707107, -0.707107, 0., 0.],
    #          [0.707107, -0.707107, 0., 0.],
    #          [0., 0., 1., 1.],
    #          [0., 0., 0., 1.]]
    #     )
    #
    # STEP 2 - Edit tranform_scans and apply transform
    # for file in ['male_pca.csv', 'female_pca.csv']:
    #     Ascans, Bscans, Cscans = get_group_lists(file, [0, 400], process_dir=process_dir)
    # for file in ['male_pca.csv', 'female_pca.csv']:
    #     Ascans, Bscans, Cscans = get_group_lists(file, [0, 500], process_dir=process_dir)
    #     # for scan in Bscans:
    #     #     sub_id = scan[len(process_dir + "fitted/"):-4]
    #     #     print(sub_id, f'({file[0]})')
    #     #     os.remove(scan)
    #     tranform_scans(
    #         Ascans,
    #         [[-0.707107, -0.707107, 0., 0.],
    #          [0.707107, -0.707107, 0., 0.],
    #          [0., 0., 1., 1.],
    #          [0., 0., 0., 1.]]
    #     )
    #
    get_ply(process_dir)
    do_pca(process_dir)
    plot_PCs(process_dir)

    PCAAutoDataSet(process_dir + 'male_pca.csv', process_dir + 'female_pca.csv').load_data(process_dir)


    # rename_fitted(fitted_dir)
    # rename_pkgs(fitted_dir)
    compare_subs(
        GITPATH + 'python/PCA_App/process/comparison/manual/fitted/',
        GITPATH + 'python/PCA_App/process/comparison/auto/fitted/',
        # remove=False
    )


    # MANUAL CLASSES
    # for file in ['male_pca.csv', 'female_pca.csv']:
    #     Ascans, Bscans, Cscans = get_group_lists(file, [0, 500], process_dir=process_dir)
    #     tranform_scans(
    #         Ascans,
    #         [[0., 0., 1., 0.],
    #          [1., 0., 0., 0.],
    #          [0., 1., 0., 0.],
    #          [0., 0., 0., 1.]]
    #     )
    #     tranform_scans(
    #         Bscans,
    #         [[-0.707107, -0.707107, 0., 0.],
    #          [0.707107, -0.707107, 0., 0.],
    #          [0., 0., 1., 1.],
    #          [0., 0., 0., 1.]]
    #     )

    # AUTOMATIC CLASSES
    # for file in ['male_pca.csv', 'female_pca.csv']:
    #     Ascans, Bscans, Cscans = get_group_lists(file, [0, 400], process_dir=process_dir)
    #     tranform_scans(
    #         Ascans,
    #         [[-0.707107, -0.707107, 0., 0.],
    #          [0.707107, -0.707107, 0., 0.],
    #          [0., 0., 1., 1.],
    #          [0., 0., 0., 1.]]
    #     )
        # remove B

    # mpca = pd.read_csv('male_pca.csv')
    # mcnd = (mpca['PC1'] > 500)
    # print(mpca.loc[mcnd]['Unnamed: 0'].to_list())
    # mpca = mpca.loc[~mcnd]
    # mpca.set_index('Unnamed: 0', inplace=True)
    # mpca.to_csv('male_pca.csv')
    #
    # fpca = pd.read_csv('female_pca.csv')
    # fcnd = (fpca['PC1'] > 500)
    # print(fpca.loc[fcnd]['Unnamed: 0'].to_list())
    # fpca = fpca.loc[~fcnd]
    # fpca.set_index('Unnamed: 0', inplace=True)
    # fpca.to_csv('female_pca.csv')


def figs(process_dir=None):
    process_dir = GITPATH + 'python/PCA_App/process/' if process_dir is None else process_dir
    fitted_dir = process_dir + 'fitted/'
    sns.set_style('whitegrid')

    # rpca = pd.read_csv('reports/RegTrial_PCAAutoDataSet.csv')
    # g = sns.FacetGrid(rpca, col="target", row='SEX')
    # g = g.map(plt.plot, "PCs", "mean_test_r2", marker=".")
    # g = g.map(plt.plot, "PCs", "mean_train_r2", marker="*")
    # plt.show()
    #
    # mpca = pd.read_csv('male_pca.csv')
    #
    # frpca = rpca.loc[rpca['SEX'] == 'F']
    # g = sns.FacetGrid(frpca, col="target", row="PCs")
    # g = g.map(plt.plot, "PCs", "mean_test_r2", marker=".")
    # plt.show()

    df = pd.read_csv(process_dir + 'pca_ds.csv')
    # df_ext = pd.read_csv(process_dir + 'extracted_ply.csv')
    # ml_subs = df['SubjectID'].to_list()
    # ext_subs = df_ext.columns.to_list()
    # # print(ml_subs)
    # # print(ext_subs)
    # print(list_diff(ext_subs, ml_subs))
    # print(list_diff(ml_subs, ext_subs))

    # df['Race'] = df['Race'].map(lambda x: x.split(' ')[0])
    # g = sns.countplot('Race', hue="SEX", data=df)
    # g.yaxis.set_major_locator(ticker.MultipleLocator(2))
    # plt.show()
    # df['AgeAtEnrollment'] = df['AgeAtEnrollment'].map(int)
    # g = sns.countplot('AgeAtEnrollment', hue="SEX", data=df)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # plt.show()
    # df['BMI1'] = df['BMI1'].fillna(0).map(int)
    # g = sns.countplot('BMI1', hue="SEX", data=df)
    # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    # plt.show()

    # df['WEIGHT'] = df['WEIGHT'].map(lambda x: 1000*x)
    # # df = df.loc[df['SEX'] == "M"]
    # sns.pairplot(df[['WBTOT_FAT','WEIGHT']].dropna())
    # plt.show()

    targets = ['WBTOT_FAT', 'WBTOT_LEAN']
    features = ['PC' + str(n) for n in range(1, 30 + 1)]

    # df = to_DataSet(df, combine_common=True).df
    # df1 = df[targets + features + ['SEX'] + ['SubjectID'] + ['BMI1', 'age1']].dropna()
    # print(df1.loc[df['TOTAL_FAT'] > 10000])
    # print(df1.loc[df['BMI1'] > 40])
    # print(df.loc[(df['WEIGHT'] > 110) & (df['WBTOT_FAT'] < 20000)])


    # for sex, color in zip(['M', 'F'], ['b', 'r']):
    #     df0 = df[targets + features + ['SEX'] + ['SubjectID'] + ['BMI1']].dropna()
    #     df0 = df0.loc[df0['SEX'] == sex]
    #
    #     fig, axes = plt.subplots(nrows=len(targets), ncols=len(features), figsize=(4 * len(features), 4 * len(targets)))
    #     for i in range(len(features) * len(targets)):
    #         feat_i = i % len(features)
    #         targ_i = int(np.floor(i / len(features)))
    #
    #         ix = np.unravel_index(i, axes.shape)  # https://stackoverflow.com/questions/21962508/stuffing-a-pandas-dataframe-plot-into-a-matplotlib-subplot
    #
    #         ax = axes[ix]
    #         # df.plot(kind='scatter', x=features[feat_i], y=targets[targ_i], ax=ax, alpha=0.5, color='b')
    #         sns.regplot(x=features[feat_i], y=targets[targ_i], data=df0, ax=ax, color=color)
    #
    #         if feat_i != 0:
    #             ax.get_yaxis().set_visible(False)
    #         if targ_i != len(targets) - 1:
    #             ax.get_xaxis().set_visible(False)
    #         i += 1
    #     plt.show()


    # results = select_targs_MF_SIMA(df, features, targets)
    # results.to_csv(process_dir + 'fig_1.csv')
    # print(results)
    # df0 = pd.read_csv(process_dir + 'fig_1.csv', index_col=0)
    #
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # for c in [c for c in df0.columns if df0[c].dtype in numerics]:
    #     df0[c] = df0[c].abs()
    #
    # df0 = df0.div(df0.sum(axis=1), axis=0).round(decimals=2) # normalize rows
    # g = sns.heatmap(df0, annot=True, annot_kws={"fontsize":5, 'rotation':-45}, cbar_kws={"orientation": "horizontal"})
    #
    # g.set_aspect("equal")
    # bottom, top = g.get_ylim()
    # g.set_ylim(bottom + 0.5, top - 0.5)
    #
    # g.set_yticklabels(df0.index, fontsize=6, rotation=0)
    # g.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # x_ticks = list(df0.columns)
    # g.set_xticks(np.arange(len(x_ticks)))
    # g.set_xticklabels(x_ticks, fontsize=6, horizontalalignment='center')
    # plt.show()


    model = LinearRegression()
    features = ['PC' + str(n) for n in range(1, 80 + 1)]
    results = select_and_run_targs_MF(df, features, targets, model, upto=40, graph_r2=True)
    results['n_PCS'] = results.index.map(lambda x: x + 1)
    results = results.reset_index(drop=True)
    del results['estimator']
    del results['DataSet']
    all_cols = [c for c in results.columns if c.startswith('all_')]
    results2 = pd.DataFrame()
    for n in range(len(results)):
        df1 = results.iloc[[n]]
        alls = df1[all_cols]
        n_splits = 5
        df2 = pd.DataFrame()
        for i in range(n_splits):
            dict = {}
            for col in all_cols:
                dict = {**dict, **{col[4:]: alls[col].iloc[0][i]}}
            df2 = append_dict(df2, dict)
        for df in [df1, df2]:
            df['m'] = 0
        r00 = pd.merge(df1, df2, on=['m'])
        results2 = pd.concat([results2, r00])
    for col in ['m'] + all_cols:
        del results2[col]
    results2 = results2.loc[results2.index == 0]
    results2.to_csv(process_dir + 'fig_2.csv', index=False)
    print(results2)


    # mdl_df = pd.read_excel('C:\\Users\\Clint\\Downloads\\ShapeUp_Q2_Tables_190711.xlsx', sheet_name='PC+DA Female')
    # targets = [['SUBTOT FAT', 'TOTAL_FAT'], ['SUBTOT LEAN', 'TOTAL_LEAN']]
    # df = pd.read_csv('pca_ds.csv')
    # for target in targets:
    #     mdl_df_targ = mdl_df.loc[mdl_df['Outcome'] == target[0]]
    #     mdl_dict = {k: v for k,v in zip(mdl_df_targ['Predictor'], mdl_df_targ['B'])}
    #     mdl_pcs = [pc for pc in mdl_dict.keys() if pc.startswith('PC')]
    #     mdl_pcs_Bs = np.array([v for k,v in mdl_dict.items() if k.startswith('PC')])
    #     # print(mdl_dict)
    #     # print(mdl_pcs_Bs)
    #     mdl = lambda *in_pcs: mdl_dict['Intercept'] + np.dot(mdl_pcs_Bs, in_pcs)
    #
    #     # ex = np.random.random_sample(len(mdl_pcs))
    #     # print(ex)
    #     # print(mdl(*ex))
    #
    #     df0 = df[mdl_pcs + [target[1]] + ['SEX']].dropna()
    #     df0 = df0.loc[df['SEX'] == 'F']
    #     del df0['SEX']
    #     X = df0[mdl_pcs].to_numpy()
    #     y_true = df0[target[1]].to_numpy()
    #     y_pred = np.apply_along_axis(lambda x: mdl(*x), 1, X)
    #     print(r2_score(y_true, y_pred))


if __name__ == "__main__":
    # process_dir = PCAPATH + 'process/'
    # process_dir = PCAPATH + 'process/comparison/manual/'
    # process_dir = PCAPATH + 'process/comparison/auto/'
    # process_dir = PCAPATH + 'process/manual/'
    # main(process_dir)
    # figs(process_dir)

    process_dir = PCAPATH + 'process/automatic/noon run on paper/'
    # do_pca(process_dir)
    # plot_PCs(process_dir)
    ply_file = pd.read_csv(process_dir+'extracted_ply.csv')

    subjs = ply_file.columns
    dxa = pd.read_excel(GITPATH+ 'python/data/ShapeUp/pdDataStorage_v4.xlsx')
    dxa = dxa[['SubjectID', 'BC_DXA_FAT_TOT']]
    print(dxa)
    print(subjs)
    dxa = dxa.loc[[sub in subjs for sub in dxa['SubjectID']]]
    print(dxa)
    dxa.to_csv('body_composition.csv')
    ply_file = ply_file.T
    ply_file = ply_file[[sub in list(dxa['SubjectID'].tolist()) for sub in ply_file.index]]
    print(len(dxa),len(ply_file))
    ply_file.to_csv('body_coordinates.csv')