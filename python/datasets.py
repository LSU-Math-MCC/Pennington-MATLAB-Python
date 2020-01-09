import time
'''
FUTURE NOTES: need version specific code
'''
import pandas as pd
import warnings
from datetime import date
from abc import abstractmethod

from utilities.data_transformers import discrete_class, standardize_subject_ids, cut_subject_ids
from utilities.folder_searcher import Searcher
from utilities.DirectoryGrab import DirGrab
from utilities.data_merger import ListMaker, ListStandardizer, CBDrowMaker, MergeMan
from utilities.PathMaker import PathMan

Path = PathMan()
GITPATH = Path.getter()


'''
DataSet Creation and Modification Code
'''


class DataSet:
    def __init__(self):
        self.df = self.load_data()

    @abstractmethod
    # Processing Methods
    def load_data(self):
        pass

    def extract_data(self,
                     feature_cnames,
                     target_cname,
                     scaler_config={},
                     data_transformers=[],
                     blacklist_sids=[]):
        df = self.df.copy()
        df = df.set_index('SubjectID', verify_integrity=True)
        df = df[~df.index.isin(blacklist_sids)]

        feature_cnames = feature_cnames.copy()
        df = self.__transform(df, data_transformers, feature_cnames)
        # print(df[feature_cnames])
        df = self.__prune(df, feature_cnames + [target_cname])
        # print(feature_cnames + [label_cname])
        # print(df[label_cname])
        x = df[feature_cnames]
        y = df[[target_cname]]
        return ExtractedData(x, y, DataFrameScaler(scaler_config))

    def __transform(self, df, data_transformers, feature_cnames):
        for transformer in data_transformers:
            ret = transformer(df, feature_cnames)
            if isinstance(ret, pd.DataFrame):
                df = ret
        return df

    def __prune(self, df, cnames_to_keep):
        # abort if columns are missing
        missing_cnames = [cname for cname in cnames_to_keep if cname not in df.columns.values]
        if len(missing_cnames) > 0:
            print(f"Missing columns: {missing_cnames}")
            exit()

        # remove columns not in use
        df = df[cnames_to_keep]
        #df.drop([cname for cname in df.columns if cname not in cnames_to_keep],
        #        axis='columns', inplace=True)

        # remove rows with empty values
        df = df.replace([0, "", "nan"], None).dropna()
        # cnames_to_drop = []
        # for index, row in df.iterrows():
        #   empty = {}
        #   for cname, cvalue in row.iteritems():
        #       cvalue = row[cname]
        #       if cvalue == 0 or cvalue == "" or cvalue == "nan" or pd.isnull(cvalue):
        #           empty[cname] = cvalue
        #   if len(empty) > 0:
        #       #print(f"dropping SubjectID '{index}' due to empty values: {empty}")
        #       cnames_to_drop.append(index)
        # df.drop(cnames_to_drop, inplace=True)
        return df

    # Import Methods
    # Returns a datasheet without duplicates. The duplicates are also aggregated for the different values.
    # This goes into StykyDataset
    def common_dataframes(self, include_classes=False, prune_hist=True, prune_fast=True, n_classes=2):
        questionnaire_df = pd.read_csv(GITPATH + 'python/data/Questionnaire.csv')
        questionnaire_df['SubjectID'] = cut_subject_ids(questionnaire_df['SubjectID'])

        dexa_df = pd.read_excel(GITPATH + "python/data/DXAnooutliers.xlsx", na_values=['#N/A'])
        dexa_df = dexa_df.dropna(axis=0, subset=['TOTAL_FAT', 'TOTAL_LEAN', 'TOTAL_PFAT'])
        dexa_df['TRUNK_BMC'] = dexa_df.apply(
            lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
        dexa_df['TOTAL_PLEAN'] = dexa_df.apply(lambda x: x['TOTAL_LEAN']/x['TOTAL_MASS'], axis=1)
        dexa_df['SubjectID'] = cut_subject_ids(dexa_df['SubjectID'])

        manual_df = pd.read_csv(GITPATH + "python/data/Manual.csv")
        manual_df['SubjectID'] = cut_subject_ids(manual_df['SubjectID'])

        blood_df = pd.read_csv(GITPATH + "python/data/Blood.csv")
        blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])
        if include_classes:
            if prune_hist:
                # Remove people with family histories for HBA1C
                blood_df['SubjectID'] = cut_subject_ids(blood_df['SubjectID'])
                blood_df = blood_df.merge(
                    pd.read_excel('data/Shapeup_Adults_Q2_Fixed_meeting_5-24-19.xlsx', sheet_name='History'),
                    on='SubjectID', how='outer')
                blood_df = blood_df.loc[blood_df['Fam_Diabetes'] != 'Yes']
                blood_df['SubjectID'] = standardize_subject_ids(blood_df['SubjectID'])
            if n_classes == 2:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40])  # 2 (>60) is healthy
            if n_classes == 3:
                # Diabetes risks
                blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100, 125])  # 0 is healthy
                blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6, 6.4])  # 0 is healthy
                # Heart risks
                blood_df['LDL_risk'] = discrete_class(blood_df, 'LDL', [130, 160])  # 0 is healthy
                blood_df['HDL_risk'] = discrete_class(blood_df, 'HDL', [40, 60])  # 2 (>60) is healthy

        bia_df = pd.read_excel(GITPATH + "python/data/Shapeup_Adults_Q2_181101.xlsx", sheet_name="BIA")
        # bia_df = bia_df.rename(columns={"_BFM_Body_Fat_Mass_": "TOTAL_FAT", "_LBM_Lean_Body_Mass_": "TOTAL_LEAN"})
        # bia_df['_AGE'] = bia_df['_AGE'].astype(float)
        bia_df = bia_df.groupby(['SubjectID'], as_index=False).aggregate('mean')

        #a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")
        #a_over_b_df['SubjectID'] = cut_subject_ids(a_over_b_df['SubjectID'])

        dfs = [questionnaire_df, dexa_df, manual_df, blood_df, bia_df]
        combined_df = None
        for df in dfs:
            df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='SubjectID', how='outer')

        combined_df['age'] = combined_df['BIRTHDATE'].astype(str).map(
            lambda row: date.today().year - int(row[2:4]) - 1900 if row != 'NaT' else "")
        combined_df['age'] = combined_df.apply(
            lambda row: date.today().year - row['qff_dob_year'] if row['age'] == "" else row['age'], axis=1)
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')

        if include_classes and prune_fast:
            # Remove subjects that did not fast
            combined_df = combined_df.loc[(combined_df['TRIG'] <= 180) | (combined_df['BMI1'] >= 45)]  # Remove subjects w/ TRIG > 180 & BMI < 45

        return combined_df

    # Returns a merged common datasheet where dexa has duplicate subject ids and the others are duplicated per subject id.
    # This goes into StykyDataset_2
    def common_dataframes_2(self):
        questionnaire_df = pd.read_csv(GITPATH + 'python/data/Questionnaire.csv')
        dexa_df = pd.read_csv(GITPATH + "python/data/DXA.csv", na_values=["#N/A"])
        dexa_df['TRUNK_BMC'] = dexa_df.apply(
            lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
        dexa_df['TOTAL_PLEAN'] = dexa_df.apply(lambda x: x['TOTAL_LEAN']/x['TOTAL_MASS'], axis=1)
        manual_df = pd.read_csv(GITPATH + "python/data/Manual.csv")
        blood_df = pd.read_csv(GITPATH + "python/data/Blood.csv")
        # NEW: Creating numeric health classes
        #blood_df['HBA1C_risk'] = discrete_class(blood_df, '_HBA1C', [5.6, 6.4])
        #blood_df['GLU_risk'] = discrete_class(blood_df, 'GLU', [100, 125])
        #a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")

        combo = blood_df.merge(manual_df, how='outer', on='SubjectID')
        #combo = combo.merge(a_over_b_df, how='outer', on='SubjectID')
        combo = combo.merge(questionnaire_df, how='outer', on='SubjectID')
        combo = dexa_df.merge(combo, how='outer', on='SubjectID', copy=True)
        return combo

    # def search_update(self, prefix):
    #     Path = PathMan()
    #     strange_path = Path.getter() + "python\data"
    #     #print(strange_path)
    #     dir = DirGrab(strange_path)
    #     dir.grabFromPrefix(prefix)
    #     files = dir.getter()
    #     Search = Searcher()
    #     file_list = Search.GreatestValue(files)
    #
    #     #print(file_list[0])
    #     dataset_location = file_list[0]
    #     dataset_df = pd.read_excel(dataset_location, na_values=["[]", 0])
    #     return dataset_df


'''
Usable Datasets
'''


# this class takes the mean of both the duplicate dexa values and styku subject ids .
class StykuDataSet(DataSet):
    def __init__(self, include_BIA=False, include_classes=False):
        self.include_BIA = include_BIA
        self.classes = include_classes
        super().__init__()

    # looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def search_update():
        Path = PathMan()
        strange_path = Path.getter() + "python/data"
        #print(strange_path)
        dir = DirGrab(strange_path)
        dir.grabFromPrefix('ObjOrganizerStyku')
        files = dir.getter()
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        # print(file_list[0])
        styku_location = file_list[0]
        styku = pd.read_excel(styku_location, na_values=["[]", 0])

        #styku = pd.read_excel('data\ObjOrganizerStyku_v12.xlsx', na_values=["[]", 0])
        return styku

    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            elif 'A/B' in col_name:
                df[col_name] = df[col_name]
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if isinstance(x, float) else x)
        return df

    def load_data(self):
        # proposed method
        styku_df = StykuDataSet.search_update()
        # added this to account for the names with "205A.obj" in the styku datasheet. Please remove if that gets fixed or if we get ones with "xxxB.obj"
        styku_df['Name'] = styku_df['Name'].map(
            lambda names: "02ADL0" + names[0:3] + "_A" if names[0] != "0" else names)
        styku_df['SubjectID'] = cut_subject_ids(styku_df['Name'])
        styku_df = styku_df.drop(styku_df.columns[[0, 1, 2, 3, 4]], axis=1)

        styku_df = styku_df.groupby(['SubjectID'], as_index=False).aggregate('mean')

        styku_df = self.standardize_units(styku_df)
        styku_df.rename(columns={col: f"{col.replace('/', '_')}" for col in styku_df.columns},
                        inplace=True)
        combined_df = super().common_dataframes(include_classes=self.classes)
        combined_df = combined_df.merge(styku_df, on='SubjectID', how='outer')

        combined_df['BMI_act'] = combined_df.apply(
            lambda row: row['bmi_calculated'] if row['BMI1'] == 0 or row['BMI1'] == "" or row['BMI1'] == "nan" or pd.isnull(row['BMI1']) else row['BMI1'], axis=1)
        combined_df['BMI_act'] = pd.to_numeric(combined_df['BMI_act'], errors='ignore')

        combined_df['BMI'] = combined_df.apply(
            lambda row: row['BMI_y'] if row['BMI_act'] == 0 or row['BMI_act'] == "" or row[
                'BMI_act'] == "nan" or pd.isnull(row['BMI_act']) else row['BMI_act'], axis=1)
        combined_df['BMI'] = pd.to_numeric(combined_df['BMI'], errors='ignore')

        combined_df = combined_df.drop(columns="BMI_act")
        return combined_df


# this class does a full extention of the dataset by taking both duplicate subject ids on styku and dxa
class StykuDataSet_2(DataSet):
    def __init__(self):
        super().__init__()

    # looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            elif 'A/B' in col_name:
                df[col_name] = df[col_name]
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if isinstance(x, float) else x)
        return df

    def load_data(self):
        Styku_df = StykuDataSet.search_update()  # loads DataFrame

        Styku_df = self.standardize_units(Styku_df)
        Styku_df.rename(columns={col: f"{col.replace('/', '_')}" for col in Styku_df.columns},
                        inplace=True)
        Styku_df['SubjectID'] = ListMaker(Styku_df.Name)  # creates Subject ID column from Name column
        Styku_df = Styku_df.drop(Styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        modData = CBDrowMaker(Styku_df)  # creates modified DataFrame
        modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # standardizes subject ID column

        Combo_df= super().common_dataframes_2()  # loads the Combined DataFrame
        modData_2 = CBDrowMaker(Combo_df)  # creates a modified Dataframe
        modData_2['SubjectID'] = ListStandardizer(modData_2.SubjectID)  #

        combined_df = MergeMan(modData, modData_2)

        #combined_df['BIRTHDATE'] = combined_df['qff_dob_year']


        combined_df['age'] = combined_df['BIRTHDATE'].astype(str).map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if row != 'nan' else "")
        combined_df['age'] = combined_df.apply(
            lambda row: date.today().year - row['qff_dob_year'] if row['age'] == "" else row['age'], axis=1)

        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
        return combined_df


# this class takes the mean of the duplicate dexa values and copies them onto the duplicate subject ids on styku
class StykuDataSet_3(DataSet):
    def __init__(self):
        super().__init__()

    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            elif 'A/B' in col_name:
                df[col_name] = df[col_name]
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if isinstance(x, float) else x)
        return df

    def load_data(self):
        Styku_df = StykuDataSet.search_update()  # loads DataFrame
        Styku_df = self.standardize_units(Styku_df)
        Styku_df.rename(columns={col: f"{col.replace('/', '_')}" for col in Styku_df.columns},
                        inplace=True)

        Styku_df['SubjectID'] = standardize_subject_ids(Styku_df['Name'])
        Styku_df['SubjectID2'] = Styku_df['SubjectID'].apply(lambda x: x[0:len("02ADL0153")])

        #There are issues here that need to fixed. This will NOT assign people without styku measurements with subject ids


        combined_df = super().common_dataframes()
        combined_df = combined_df.rename(columns = {'SubjectID':'SubjectID2'})
        result = Styku_df.merge(combined_df, on="SubjectID2", how="outer", copy=True)
        result = result.drop(columns="SubjectID2")
        result['age'] = result['BIRTHDATE'].astype(str).map(
            lambda row: date.today().year - int(row[2:4]) - 1900 if row != 'NaT' else "")

        result['age'] = result.apply(
            lambda row: date.today().year - row['qff_dob_year'] if row['age'] == "" else row['age'], axis=1)

        result['age'] = pd.to_numeric(result['age'], errors='ignore')
        result = result.drop(result.columns[[0, 1, 2, 3, 4]], axis=1)

        return result


class SS20DataSet(DataSet):
    def __init__(self, include_classes=False):
        self.classes = include_classes
        super().__init__()

    def search_update():
        strange_path = Path.getter() + "python\data"
        dir = DirGrab(strange_path)
        dir.grabFromPrefix('ObjOrganizerSS20')
        files = dir.getter()
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        SS20_location = file_list[0]
        #print(SS20_location)
        SS20 = pd.read_excel(SS20_location, na_values=["[]", "", 0])
        return SS20

    # Volumes in L and Lengths in cm
    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x: x * 10 ** -6)
            elif 'A/B' in col_name:
                df[col_name] = df[col_name]
            else:
                df[col_name] = df[col_name].map(lambda x: x / 10 if type(x) is float else x)
        return df

    def load_data(self):
        SS20_df = SS20DataSet.search_update()
        SS20_df['SubjectID'] = standardize_subject_ids(SS20_df['Name'])
        SS20_df = SS20_df.drop(SS20_df.columns[[0, 1, 2, 3]], axis=1)
        # SS20_df = SS20_df.groupby(['SubjectID'], as_index=False).aggregate('mean')
        SS20_df = self.standardize_units(SS20_df)
        SS20_df.rename(columns={col: f"{col.replace('/', '_')}" for col in SS20_df.columns}, inplace=True)
        combined_df = super().common_dataframes(include_classes=self.classes)
        combined_df = combined_df.merge(SS20_df, on='SubjectID', how='outer')
        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
        combined_df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
        combined_df = combined_df.groupby(['SubjectID'], as_index=False).aggregate('mean')

        return combined_df
        

class CombinedDataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        # Pair subjects with data from common datasets
        # ISSUE: Less than 50 subjects from SS20 are paired with DXA and results are awful
        styku_df, ss20_df = StykuDataSet().load_data(), SS20DataSet().load_data()
        # Distinguish between SS20 and Styku subject IDs (prevents duplicate index error)
        #   SE: Why is map() better than apply() for a single column?
        #   map() is for Series (i.e. single columns) and operates on one cell at a time, while apply() is for DataFrame, and operates on a whole row at a time.
        styku_df['SubjectID'] = styku_df['SubjectID'].map(lambda id: "Styku_" + id)

        ss20_df['SubjectID'] = ss20_df['SubjectID'].map(lambda id: "SS20_" + id)

        #styku_df = styku_df.apply(lambda x: x * 16.3871 / 1000 if x is int else x)  # Convert from in3 to L
        #ss20_df = ss20_df.apply(lambda x: x * 10 ** -6 if x is int else x) # Convert from mm3 to L

        # Return concatenated dataframe with index
        return pd.concat([styku_df, ss20_df], sort=False)


class NhanesDataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
            df = pd.read_excel(GITPATH + "python/data/NHANES/NHANES 07OCT2016active.xls", sheet_name="DXA_Imp_1")
            df['RIDAGEYR'] = df['RIDAGEYR'].astype(float) # fixes DataConversionWarning by MinMaxScaler

            # Standardize units

            #df.reset_index(inplace=True)
            df = df.rename(columns={"RIDAGEYR": "age", "bmxbmi": "BMI1", "RIAGENDR": "SEX",
                               "BMXLEG": "rLegLength", "BMXARML": "RArmLength", "bmxwaist": "waist circ",
                               "BMXARMC": "rbicepGirth", "BMXCALF": "rCalfCirc", "BMXTHICR": "rThighGirth",
                               "DXDTOFAT": "TOTAL_FAT", "DXDTOLE": "TOTAL_LEAN", "DXDTOPF": "TOTAL_PFAT",
                                "lbxglu": "GLU", "lbdldl": "LDL", "lbdhdl": "HDL", "lbxtr": "TRIG",
                                "SEQN": "SubjectID"})
            df = df.loc[df.age >= 18]

            # Standardize units
            df['GLU'] = df['GLU'].map(lambda x: 100*x)
            df['LDL'] = df['LDL'].map(lambda x: 100*x)
            df['HDL'] = df['HDL'].map(lambda x: 100*x)
            df['TRIG'] = df['TRIG'].map(lambda x: 100*x)

            # Remove subjects with family history of diabetes
            # df = df.loc[df['MCQ250A'] != 2]

            # Remove subjects that did not fast
            # df = df.loc[(df['TRIG'] <= 180) | (df['BMI1'] >= 45)]  # Remove subjects w/ TRIG > 180 & BMI < 45
            print(len(df))

            # Diabetes risks
            df['GLU_risk'] = discrete_class(df, 'GLU', [100, 125])  # 0 is healthy
            # df['GLU_risk'] = discrete_class(df, 'GLU', [100])  # 0 is healthy
            # df['HBA1C_risk'] = discrete_class(df, '_HBA1C', [5.6, 6.4])  # 0 is healthy

            # Heart risks
            df['LDL_risk'] = discrete_class(df, 'LDL', [130, 160])  # 0 is healthy
            # df['LDL_risk'] = discrete_class(df, 'LDL', [130])  # 0 is healthy
            df['HDL_risk'] = discrete_class(df, 'HDL', [40, 60])  # 2 (>60) is healthy
            # df['HDL_risk'] = discrete_class(df, 'HDL', [60])  # 2 (>60) is healthy
            df['HDL_risk'] = df['HDL_risk'].map(lambda x: abs(int(x)-2) if x != 'nan' else 'nan')  # Convert (>60) to 0 as healthy
            return df


class PCADataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        pca_male = pd.read_excel('data/Shapeup_Adults_Q2_181101.xlsx', sheet_name='PC_Weights_Male')
        pca_female = pd.read_excel('data/Shapeup_Adults_Q2_181101.xlsx', sheet_name='PC_Weights_Female')

        # SEX is assigned below
        combined_df = super().common_dataframes().merge(pd.concat([pca_male, pca_female], sort=False), on='SubjectID', how='inner')

        combined_df['age'] = combined_df['BIRTHDATE'].map(
            lambda row: date.today().year - int(row[-2:]) - 1900 if isinstance(row, str) else "")
        combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])
        return combined_df


# this class takes the styku dataset and drops all the columns that we dont use for the target (for paper use)
class TrimmedStykuDataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        cnames = ['TOTAL_FAT', 'TOTAL_LEAN', 'TOTAL_PFAT', 'TOTAL_PLEAN',
                  'age', 'BMI',
                  "TotalVolume", "headVolume", "rArmVolume", "lArmVolume", "rLegVolume", "lLegVolume",
                  "trunkVolume",
                  "Chest circ", "waist circ", "hip circ", "rThighGirth", "lThighGirth",
                  "rCalfCirc", "lCalfCirc", "rWristGirth", "lWristGirth", "rForearm",
                  "lForearmGirth", "rbicepGirth", "lBicepGirth", "rAnkle", "Lankle",
                  "LarmLength", "RArmLength", "CollarScalp", "TrunkLength", "lLegLength", "rLegLength",
                  "crotchHeight",
                  "Chest circ A_B", "waist circ A_B", "hip circ A_B", "rThighGirth A_B", "lThighGirth A_B",
                  "rCalfCirc A_B", "lCalfCirc A_B", "rWristGirth A_B", "lWristGirth A_B",
                  "rForearm A_B", "lForearmGirth A_B", "rbicepGirth A_B", "lBicepGirth A_B", "rAnkle A_B",
                  "Lankle A_B"]


        result = StykuDataSet()
        return result.load_data().dropna(axis=0, subset=cnames)


'''
Data Extraction and Column Scaler Code
'''


class ExtractedData:
    def __init__(self, x, y, scaler):
        self.x = x # x column
        self.y = y # y column
        self.scaler = scaler # scalar
        self.x_scaled = scaler.transform(x)
        self.y_scaled = scaler.transform(y)


class DataFrameScaler:
    def __init__(self, scaler_dict):
        '''
        This function is passed scalar_dict of form {cname: scalar function} where cname
        is either the name of a column in some dataframe or 'default'. Then the transform
        function is applied to the dataframe df; each column is scaled appropriately.
        '''
        self.default_scaler = None
        self.column_scalers = {} # dictionary
        if "default" in scaler_dict and scaler_dict["default"] is not None:
            self.default_scaler = scaler_dict["default"]
        for column_tuple, scaler_class in scaler_dict.items():
            if column_tuple == "default":
                continue
            scaler = scaler_class() if scaler_class is not None else None #scalar assingment eithe none or key value

            # verify column_tuple is iterable
            if not isinstance(column_tuple, tuple):
                column_tuple = (column_tuple,)

            for column in column_tuple:
                self.column_scalers[column] = scaler # dictionary assingment

    def __get_column_scaler(self, column):
        # sets self.column_scalers[column] to default or other entry in scaler_dict
        if column not in self.column_scalers and self.default_scaler is not None:
            self.column_scalers[column] = (self.default_scaler)()
        if column in self.column_scalers and self.column_scalers[column] is not None:
            yield self.column_scalers[column]

    def transform(self, df):
        df = df.copy()

        # Setup (fit?) scalars
        for column in df.columns: # enumerate of str cnames
            for scaler in self.__get_column_scaler(column):  # Returns list of scalars to fit to column
                frame = df[column].to_frame() # convert column to df
                if hasattr(scaler, "partial_fit"):
                    scaler.partial_fit(frame)
                else:
                    scaler.fit(frame) # fit scalar to df, e.g. compute the mean and std to be used for later scaling.

        # Apply scalars to columns
        for column in df.columns:
            for scaler in self.__get_column_scaler(column): # Returns list of scalars to transform column
                df[column] = scaler.transform(df[column].to_frame())
        return df


'''
DataSet automatic creation code
'''


def to_Dataset(df, subject_cname='SubjectID'):
    class auto_dataset(DataSet):
        def load_data(self):
            try:
                # find subject IDs and combine with DXA, Blood, and Questionaire data
                df['SubjectID'] = cut_subject_ids(df[subject_cname])  # remove scan suffixes
                if subject_cname != 'SubjectID':
                    del df[subject_cname]  # The next step expects numeric columns

                # TODO: Combine label columns differently than numeric columns
                clean_df = df.replace([0, "[]", "", "nan"], None).dropna()
                clean_df = clean_df.drop_duplicates('SubjectID')
                # clean_df = clean_df.groupby(clean_df['SubjectID'], as_index=False).aggregate('mean')  # take average of duplicates

                combined_df = super().common_dataframes()
                combined_df = combined_df.merge(clean_df, on='SubjectID', how='outer')
                return combined_df
            except KeyError:
                # if there are no subject indentifiers, do not combine with DXA
                warnings.warn(f'Could not find subject identifier column \"{subject_cname}\", loading without DXA.',
                              stacklevel=4)
                return df
    return auto_dataset()