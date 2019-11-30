import csv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


def sort_by_gender(coordinate_file, subjects_file, gender_file):
    # Opens gender files
    with open(gender_file) as g:
        # creates csv reader object
        gender_reader = csv.reader(g)

        # Creates two arrays with all male and female subjects
        male_subjects = []
        female_subjects = []

        for line in gender_reader:
            subject = line[0]
            if line[1] == 'Male':
                male_subjects.append(subject)
            elif line[1] == 'Female':
                female_subjects.append(subject)

    # Divide the coordinate_data_file into two seperate csv-files

    # Opens coordinate and subject file
    with open(coordinate_file) as c, open(subjects_file) as s:
        coordinate_reader = csv.reader(c)
        subject_reader = csv.reader(s)

        # Determine number of subjects and reset reader to first line
        s.seek(0)

        # Create two new files with coordinate_data and subjects for both male and female
        m = open('male_coordinate_data.csv', 'w', newline='')
        f = open('female_coordinate_data.csv', 'w', newline='')
        m_s = open('male_subjects.csv', 'w', newline='')
        f_s = open('female_subjects.csv', 'w', newline='')
        male_data_writer = csv.writer(m)
        female_data_writer = csv.writer(f)
        male_subject_writer = csv.writer(m_s)
        female_subject_writer = csv.writer(f_s)

        # Split coordinate_data and subjects into two seperate files for both male and female
        for coordinate_line in coordinate_reader:
            subject = next(subject_reader)[0]
            if subject in male_subjects:
                male_data_writer.writerow(coordinate_line)
                male_subject_writer.writerow([subject])
            elif subject in female_subjects:
                female_data_writer.writerow(coordinate_line)
                female_subject_writer.writerow([subject])
            else:
                print('Error: Cannot find gender information for subject ' + subject + '.')


# Performs PCA on a given subject and coordinate file with the specified format
# Saves the PCA results as a 'output_file_name.csv'
def principal_component_anaylsis(subject_file, coordinate_file, output_file_name):
    # open subject and coordinate file
    with open(subject_file) as s, open(coordinate_file) as c:
        subject_reader = csv.reader(s)
        coordinate_reader = csv.reader(c)

        # Determine given subjects
        subjects = []
        for line in subject_reader:
            subjects.append(line[0])

        coordinates = []
        for line in coordinate_reader:
            coordinates.append(line)

        # create DataFrame to perform pca on
        a = [len(x) for x in coordinates]
        bound = min([len(x) for x in coordinates])
        coordinate_data = pd.DataFrame(columns=["c" + str(i) for i in range(0, bound)],
                                       data=[coordinate[0:bound] for coordinate in coordinates],
                                       index=subjects)

        # # Fill in DataFrame from the coordinate file
        # for subject, coordinate in zip(subjects, coordinates):
        #     coordinate_data.loc[subject] = coordinate

    # Scale data before performing PCA
    # scaled_coordinate_data = preprocessing.scale(coordinate_data)
    scaled_coordinate_data = preprocessing.StandardScaler().fit_transform(coordinate_data)

    # Perform PCA
    pca = PCA()
    pca.fit(scaled_coordinate_data)
    pca_coordinate_data = pca.transform(scaled_coordinate_data)

    # Plot the variance ratio for each principal component on a graph
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    plt.show()

    # Create a DataFrame with the new principal component weights
    pca_df = pd.DataFrame(pca_coordinate_data, index=subjects, columns=labels)

    # Plot all data points on a PC graph with two components to see first dependencies in data
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA Example Graph')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.plot(pca_df.PC1, pca_df.PC2, '*')
    plt.show()

    # Saves the PCA DataFrame to a csv file
    pca_df.index.name = "SubjectID"
    pca_df.to_csv(path_or_buf=output_file_name)


# Seperates the data by sex and performs PCA on each of them
subjects = sort_by_gender('coordinate_data.csv', 'subjects.csv', 'subject_gender.csv')
principal_component_anaylsis('male_subjects.csv', 'male_coordinate_data.csv', 'male_pca.csv')
principal_component_anaylsis('female_subjects.csv', 'female_coordinate_data.csv', 'female_pca.csv')
