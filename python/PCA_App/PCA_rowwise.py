import csv
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from python.PCA_App.Extract_Ply import rename_subs, sort_by_gender, extract_ply, split_male, split_female
from python.utilities.DirectoryGrab import DirGrab
from python.PCA_App.run_ganger import GangGang
from python.PCA_App.point_reduction_script import FileConverter
from python.utilities.FolderToFolder import MoverMan
from PCA_App.runner_PCA_auto import PCA_ml
from python.utilities.PathMaker import PathMan

####################   Attempts to expand the field limit for 180k points      ##################################
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)
################################################################################################################

ply_ext = ".ply"  # ext for glob


# Performs PCA on a list of subjects(male or female) inside the dataframe(male_table or female_table) from Ply script
# Saves the PCA results as a 'output_file_name.csv'
def principal_component_analysis(coordinate_file, output_file_name):
    # peeps = list(coordinate_file)  # works when subjects are column names
    peeps = coordinate_file.index  # used when listing subjects as row names
    # print(len(coordinate_file))
    # print(peeps)
    # labels = ['PC' + str(x) for x in range(1, len(peeps) + 1)]

    # Scale data before performing PCA
    x = coordinate_file.loc[:, ].values  # grabs column data
    # x = coordinate_file.loc[0:len(peeps)+1, :]  # grabs row data...maybe not
    scaled_coordinate_data = preprocessing.StandardScaler().fit_transform(x)
    # Perform PCA
    pca = PCA()  # not including (n_components) will keep all components
    # pca.fit(scaled_coordinate_data)
    # pca_coordinate_data = pca.transform(scaled_coordinate_data)
    pca_coordinate_data = pca.fit_transform(scaled_coordinate_data)  # does both
    # print(pca_coordinate_data)

    # pcas = []
    # for j in range(len(peeps)):
        # single_df = pd.DataFrame({peeps[j]: pca_coordinate_data[j]})
        # pcas.append(single_df)
    # pca_df = pd.concat(pcas, axis=0)
    # pca_df = pca_df.T

    # yes, i know this is nasty, but it formats the .csv correctly
    # print(len(peeps))

    pca_df = pd.DataFrame(data=pca_coordinate_data, index=peeps, columns=[f'PC{i+1}' for i in
                                                                           range(len(peeps))])
    # pca_df = pd.DataFrame(data=pca_coordinate_data, index=peeps, columns=[f'PC{i+1}' for i in
    #                                                                       range(len(coordinate_file))])
    # pca_df = pca_df.T
    # print(pca_df.head(5))
    # pca_df = pca_df.reindex([f'PC{i+1}' for i in pca_df.index])  # reindex PC#'s, but deletes data

    # Plot the variance ratio for each principal component on a graph
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Screen Plot')
    plt.show()

    # Plot all data points on a PC graph with two components to see first dependencies in data
    # plt.scatter(pca_df.PC1, pca_df.PC2)
    # plt.title('PCA Example Graph')
    # plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    # plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    # plt.plot(pca_df.PC1, pca_df.PC2, '*')
    # plt.show()                                      # << wrong format...PC# not set as features anymore...now indices

    # Saves the PCA DataFrame to a csv file
    pca_df.to_csv(output_file_name)
    # np.savetxt(output_file_name, pca_coordinate_data, delimiter=",")


def main():
    PRS = FileConverter()
    PRS.easyScript()
    Path = PathMan()
    inputPath = Path.getter() + "python/PCA_App/process"
    outputPath = Path.getter() + "python/PCA_App/process/MKR/output"
    MV = MoverMan(inputPath,outputPath)
    MV.MoveByExt(".ply")


    # identify input path, call parseArgs
    # args = Extract_Ply.parseArgs('C:/Users/domii/Desktop/All_Together_Now/process/fitted/')

    gang3x = GangGang() # run Ganger

    # array containing the paths of all of the input .ply's


    # grabber = DirGrab(Path.getter() + 'python/PCA_App/process/MKR/output/fitted')
    grabber = DirGrab(Path.getter() + 'python/PCA_App/process')


    grabber.grabByExtension(".ply")
    ply_array = grabber.getter()
    print(ply_array)
    # ply_array = glob.glob(args.indir + '/*' + ply_ext)

    # array containing list of all subjects by just subject ID(##ADL####) inside input folder
    subjects = rename_subs(ply_array)

    # Code added by Clint -----

    # -------------------------

    # reference gender .csv file and Ply script to get array of **all male and female subjects in the study
    mf_subs = sort_by_gender('subject_gender.csv')
    male_subjects = mf_subs[0]
    female_subjects = mf_subs[1]

    # defines the dataframe creates by Ply script
    output_df = extract_ply(subjects, ply_array) # prints wrote subject ...
    # output_df = output_df.T
    # print(output_df.head(10))

    # uncomment for debugging -----
    # output_df.to_csv('extracted_ply.csv')
    # output_df = pd.read_csv('extracted_ply.csv')
    # -----------------------------

    # uses output_df to create dataframes of males and females from files **in the input directory
    male_table = split_male(output_df, male_subjects)
    male_table = male_table.T  # transposes dataframe
    # print(male_table.head(5))
    female_table = split_female(output_df, female_subjects)
    female_table = female_table.T  # transposes dataframe
    # print(female_table.head(10))

    # run PCA on male and female subjects >> output .csv of PCA(weights?)
    # input either male or female table
    principal_component_analysis(male_table, 'male_pca.csv')
    principal_component_analysis(female_table, 'female_pca.csv')

    ml = PCA_ml('male_pca.csv','female_pca.csv')

    print(ml.results)
    ml.results.to_csv('reports/RegTrial_PCAAutoDataSet_LinearRegression.csv')

if __name__ == "__main__":
    main()
