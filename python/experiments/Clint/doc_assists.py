import glob, os, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import SCORERS
from utilities.PathMaker import PathMan
from utilities.data_transformers import standardize_subject_ids

Path = PathMan()
GITPATH = Path.getter()


def print_files(rel_dir):
    os.chdir(GITPATH + rel_dir)
    print(GITPATH + rel_dir)
    for file in glob.glob("*"):
        print(file)
    print('\n')

# print_files('python/utilities/')

# print(SCORERS['accuracy'])

# pkg_df = pd.read_excel(GITPATH + "python/data/ShapeUp/Shapeup_Adults_Q2_181101.xlsx", sheet_name="Joints")
# pkg_df = pkg_df[['SubjectID', 'Scan_Package_ID']]
# pkg_df.drop_duplicates('Scan_Package_ID', inplace=True)
# pkg_df.set_index('Scan_Package_ID', inplace=True)
# pkgs = glob.glob(GITPATH + "python/PCA_App/process/fitted/pkg-*_fitted.ply")
# pkgs = [pkg[len(GITPATH + "python/PCA_App/process/fitted/"):-len("_fitted.ply")] for pkg in pkgs]
# pkg_df = pkg_df.loc[pkgs]
# pkg_df['new'] = True
# subj_ids = glob.glob(GITPATH + "python/PCA_App/process/fitted/*ADL*_fitted.ply")
# subj_ids = [id[len(GITPATH + "python/PCA_App/process/fitted/"):-len("_fitted.ply")] for id in subj_ids]
# subj_df = pd.DataFrame(data=subj_ids, columns=['SubjectID'])
# df = pd.concat([subj_df, pkg_df], sort=True)
# df['SubjectID'] = standardize_subject_ids(df['SubjectID'])
# df = df.loc[(df['new'] == True)]
# for row in df.iterrows():
#     pkg = row[0]
#     id = row[1]['SubjectID']
#     os.rename(
#         GITPATH + f"python/PCA_App/process/fitted/{pkg}_fitted.ply",
#         GITPATH + f"python/PCA_App/process/fitted/{id}_fitted.ply"
#     )


# subj_ids = glob.glob(GITPATH + "python/PCA_App/process/fitted/*ADL*_fitted.ply")
# subj_ids = [id[len(GITPATH + "python/PCA_App/process/fitted/"):-len("_fitted.ply")] for id in subj_ids]
# for id in subj_ids:
#     mkr_file = GITPATH + "python/PCA_App/process/" + id + ".mkr"
#     ply_file = GITPATH + "python/PCA_App/process/" + id + ".ply"
#     try:
#         open(mkr_file, 'x')
#     except FileExistsError:
#         pass
#     try:
#         open(ply_file, 'x')
#     except FileExistsError:
#         pass

males = pd.read_csv(GITPATH + "python/PCA_App/male_pca.csv")
g = sns.pairplot(males[['PC1','PC2','PC3','PC4','PC5']])
plt.show()