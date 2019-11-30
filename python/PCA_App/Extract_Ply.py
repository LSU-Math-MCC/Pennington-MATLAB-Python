import os
import glob
import csv
import pandas as pd
import argparse as ap
from plyfile import PlyData
# *** RUNNING THIS SCRIPT THROUGH PCA.py ***
ply_ext = ".ply"  # ext for glob
# Chop "_A", "_fitted" off of ID names to reference in PCA matrix
# ****** We have 2 scans, so we need to decide which one to use ******
def rename_subs(ply_array):
    subjects = []  # initialize "subjects" as an array
    for i in range(len(ply_array)):  # runs for # of .ply files(subjects) in input directory
        if ply_array[i] is not None:
            if ".ply" in ply_array[i]:  # makes sure the files are .ply files
                subj_base = os.path.basename(ply_array[i])  # path of .ply
                subj_id = os.path.splitext(subj_base)[0]  # subject file name
                if "_A" in subj_id:
                    subj_id = subj_id[:-2]  # removes "_A" from end of Subject ID name
                elif "_C" in subj_id:
                    subj_id = subj_id[:-2]  # removes "_C" from end of Subject ID
                elif "fitt" in subj_id:
                    subj_id = subj_id[:-7]  # removes the excess, but after the first -2, it would read "fitt"
                elif "fitted" in subj_id:
                    subj_id = subj_id[:-7]  # removes "_fitted" from end of Subject ID
                subjects.append(subj_id)

    return subjects


# Outputs arrays containing **all male and female subject ID's in the study
def sort_by_gender(gender_file):
    # Opens gender files
    with open(gender_file) as g:
        # creates csv reader object
        gender_reader = csv.reader(g)

        # Skips first line in subject_gender_file >> skips header names
        next(gender_reader)

        # Creates two arrays with all male and female subjects
        male_subjects = []
        female_subjects = []

        count = 0
        missing_subj = []
        # subject = " "
        for line in gender_reader:
            # subject.clear()
            subject = line[0]
            if line[1] == 'Male':
                male_subjects.append(subject)
                # print(subject)
            elif line[1] == 'Female':
                female_subjects.append(subject)
                # print(subject)
            else:
                count += 1
                missing_subj.append(subject)
        print("No gender data for ", count, "subject(s): ", missing_subj)
        # print(subjects)

    return [male_subjects, female_subjects]


# Extracts vertex values from .ply's in input directory and saves them in a dataframe
def extract_ply(subjects, ply_array):
    num_subs = len(subjects)

    xyz = []
    dfs = []

    for i in range(len(ply_array)):
        xyz.clear()
        if ply_array[i] is not None:
            mesh = PlyData.read(ply_array[i])  # << requires the whole path in (), and reads the .ply
            for j in range(0, 60001):  # runs for all 60,000 (x,y,z) points in .ply file
                for index in range(3):  # runs 3 times to receive the first 3 elements, which are x, y, and z
                    out = mesh.elements[0].data[j]  # elem.[0 or 1...use 1], and data[#} reads the rows or xyz pts
                    xyz.append(out[index])
        current_df = pd.DataFrame({subjects[i]: xyz})
        dfs.append(current_df)
        print("wrote subject", i+1)  # makes sure program running >> can DELETE later
    for l in range(num_subs):
        dfs[l].reset_index(drop=True, inplace=True)  # rids columns of Nan values

    output_df = pd.concat(dfs, axis=1)  # concatenate the data frames together vertically(axis=0)
    # print(list(output_df))

    return output_df


# saves dataframe of male subject vertices from input directory
def split_male(output_df, male_subjects):
    males = []  # array to contain list of dataframes to concatenate later
    subs = list(output_df)  # creates list of the column names from df (subject names)
    subs = [s[:9] for s in subs] # CLINT - so formats match male_subjects
    i = 0
    for person in subs:
        if person in male_subjects:  # checks if each subject is in the list of male subjects
            present_df = pd.DataFrame({person: output_df.iloc[:, i]})  # new df with col name and the stored data
            males.append(present_df)  # append current df to final male df
        i += 1  # manual counter b/c for loop is iterating by strings
    male_table = pd.concat(males, axis=1)  # concatenate the male df's into final male df

    return male_table


# saves dataframe of female subject vertices from input directory
def split_female(output_df, female_subjects):
    females = []  # array to contain list of dataframes to concatenate later
    subs = list(output_df)  # creates list of column names from df (subject names)
    subs = [s[:9] for s in subs]  # CLINT - so formats match male_subjects
    i = 0
    for person in subs:
        if person in female_subjects:  # checks if each subject is in the list of female subjects
            present_df = pd.DataFrame({person: output_df.iloc[:, i]})
            females.append(present_df)
        i += 1
    female_table = pd.concat(females, axis=1)

    return female_table


# set input directory for glob
def parseArgs(path):
    parser = ap.ArgumentParser(description='Create CSV Matrix')
    parser.add_argument('-i', '--indir', help='Input directory with target scans')
    parser.add_argument('-o', '--outdir', help='Output directory for matrix appendices')
    args = parser.parse_args()

    args.indir = path

    return args

# Running this script through PCA.py
# def main():

    # args = parseArgs('C:/Users/domii/Desktop/Running_PCA/Ply_Files')  # sets input directory for glob

    # ply_array = glob.glob(args.indir + '/*' + ply_ext)  # array containing the paths of all of the input .ply's

    # subjects = rename_subs(ply_array)  # define list of subjects in input directory

    # mf_subs = sort_by_gender('subject_gender.csv')
    # male_subjects = mf_subs[0]  # define list of male subjects in study
    # female_subjects = mf_subs[1]  # define list of female subjects in study

    # output_df = extract_ply(subjects, ply_array)  # define total df of all files in folder

    # male_table = split_male(output_df, male_subjects)  # define male df with data
    # print(male_table)

    # female_table = split_female(output_df, female_subjects)  # define female df with data
    # print(female_table)

    # print(list(male_table))
    # print(list(female_table))


# if __name__ == '__main__':  # no clue but it should help
    # main()
