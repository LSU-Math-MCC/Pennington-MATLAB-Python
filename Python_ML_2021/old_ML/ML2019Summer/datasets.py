import pandas as pd
from common import MLPData, DataFrameScaler
from abc import abstractmethod
from Floating_methods_pujan import Searcher, DirGrab, birthday, append_class, standardize_subject_ids , standardize_subject_ids_2,ListMaker,ListStandardizer,CBDrowMaker, MergeMan, animate


class DataSet:
    def __init__(self):
        self.df = self.load_data()

    @abstractmethod
    # Processing Methods
    def load_data(self):
        pass

    def extract_data(self,
                     feature_cnames,
                     label_cname,
                     scaler_config={},
                     data_transformers=[],
                     blacklist_sids=[],
                     eval_type='regressor'):
        df = self.df.copy()

        df = df.set_index('SubjectID', verify_integrity=True)
        df = df[~df.index.isin(blacklist_sids)]

        feature_cnames = feature_cnames.copy()
        df = self.__transform(df, data_transformers, feature_cnames)
        self.__prune(df, feature_cnames + [label_cname])
        x = df[feature_cnames]
        y = df[[label_cname]]
        return MLPData(x, y, DataFrameScaler(scaler_config))

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
        df.drop([cname for cname in df.columns if cname not in cnames_to_keep],
                axis='columns', inplace=True)

        # remove rows with empty values
        cnames_to_drop = []
        for index, row in df.iterrows():
            empty = {}
            for cname, cvalue in row.iteritems():
                cvalue = row[cname]
                if cvalue == 0 or cvalue == "" or pd.isnull(cvalue):
                    empty[cname] = cvalue
            if len(empty) > 0:
                #print(f"dropping SubjectID '{index}' due to empty values: {empty}")
                cnames_to_drop.append(index)
        df.drop(cnames_to_drop, inplace=True)

    # Import Methods
    #the First one returns a datasheet without duplicates. The duplicates are also aggregated for the different values.This goes into StykyDataset
    def common_dataframes(self):
            questionnaire_df = pd.read_csv('data/Questionnaire.csv')
            questionnaire_df['SubjectID'] = standardize_subject_ids_2(questionnaire_df['SubjectID'])

            dexa_df = pd.read_csv("data/DXA.csv", na_values=["#N/A"])
            dexa_df['TRUNK_BMC'] = dexa_df.apply(
                lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
            dexa_df['SubjectID'] = standardize_subject_ids_2(dexa_df['SubjectID'])

            manual_df = pd.read_csv("data/Manual.csv")
            manual_df['SubjectID'] = standardize_subject_ids_2(manual_df['SubjectID'])

            blood_df = pd.read_csv("data/Blood.csv")
            blood_df['SubjectID'] = standardize_subject_ids_2(blood_df['SubjectID'])
            # NEW: Creating numeric health classes
            blood_df = append_class(blood_df, '_HBA1C', [5.6, 6.4])
            blood_df = append_class(blood_df, 'GLU', [100, 125])

            a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")
            a_over_b_df['SubjectID'] = standardize_subject_ids_2(a_over_b_df['SubjectID'])
            dfs = [questionnaire_df, dexa_df, manual_df, blood_df, a_over_b_df]

            combined_df = None
            for df in dfs:
                df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
                if combined_df is None:
                    combined_df = df
                else:
                    combined_df = combined_df.merge(df, on='SubjectID', how='outer')
            return combined_df
    #This returns a merged common datasheet where dexa has duplicate subject ids and the others are duplicated per subject id. This goes into StykyDataset_2
    def common_dataframes_2(self):
            questionnaire_df = pd.read_csv('data/Questionnaire.csv')
            dexa_df = pd.read_csv("data/DXA.csv", na_values=["#N/A"])
            dexa_df['TRUNK_BMC'] = dexa_df.apply(
                lambda x: x['LRIB_BMC'] + x['RRIB_BMC'] + x['T_S_BMC'] + x['L_S_BMC'] + x['PELV_BMC'], axis=1)
            manual_df = pd.read_csv("data/Manual.csv")
            blood_df = pd.read_csv("data/Blood.csv")
            # NEW: Creating numeric health classes
            blood_df = append_class(blood_df, '_HBA1C', [5.6, 6.4])
            blood_df = append_class(blood_df, 'GLU', [100, 125])
            a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")


            combo = blood_df.merge(a_over_b_df, how='outer', on='SubjectID')
            combo = combo.merge(manual_df, how='outer', on='SubjectID')
            combo = combo.merge(questionnaire_df, how='outer', on='SubjectID')
            combo = dexa_df.merge(combo, how='outer', on='SubjectID', copy=True)
            return combo


class StykuDataSet(DataSet):
    def __init__(self):
        super().__init__()

    # looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def search_update():
        path = "data\\"
        strange_path = "data\\ObjOrganizerStyku*.xlsx"
        dir = DirGrab(path)
        files = dir._glob_grabber(strange_path)
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        #print(file_list[1])
        styku_location = file_list[1]
        styku = pd.read_excel(styku_location, na_values=["[]", 0])
        return styku

    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if type(x) is float else x)
        return df

    def load_data(self):
        # proposed method
        styku_df = StykuDataSet.search_update()
        styku_df['SubjectID'] = standardize_subject_ids_2(styku_df['Name'])
        styku_df = styku_df.drop(styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        styku_df = styku_df.groupby(styku_df['SubjectID'], as_index=False).aggregate('mean')
        styku_df = self.standardize_units(styku_df)

        combined_df = super().common_dataframes()

        combined_df = combined_df.merge(styku_df, on='SubjectID', how='outer')
        combined_df['age'] = combined_df.apply(lambda row: birthday(row.BIRTHDATE), axis=1)
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
        return combined_df


class StykuDataSet_2(DataSet):
    def __init__(self):
        super().__init__()

    # looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def search_update():
        path = "data\\"
        strange_path = "data\\ObjOrganizerStyku*.xlsx"
        dir = DirGrab(path)
        files = dir._glob_grabber(strange_path)
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        #print(file_list[1])
        styku_location = file_list[1]
        styku = pd.read_excel(styku_location, na_values=["[]", 0])
        return styku

    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
            else:
                df[col_name] = df[col_name].map(lambda x: x * 2.54 if type(x) is float else x)
        return df

    def load_data(self):
        Styku_df = StykuDataSet.search_update()  # loads DataFrame
        Styku_df = self.standardize_units(Styku_df)
        Styku_df['SubjectID'] = ListMaker(Styku_df.Name)  # creates Subject ID column from Name column
        Styku_df = Styku_df.drop(Styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        modData = CBDrowMaker(Styku_df)  # creates modified DataFrame
        modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # standardizes subject ID column

        Combo_df= super().common_dataframes_2()  # loads the Combined DataFrame
        modData_2 = CBDrowMaker(Combo_df)  # creates a modified Dataframe
        modData_2['SubjectID'] = ListStandardizer(modData_2.SubjectID)  #

        combined_df = MergeMan(modData, modData_2)

        combined_df['age'] = combined_df.apply(lambda row: birthday(row.BIRTHDATE), axis=1)
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
        return combined_df


class SS20DataSet(DataSet):
    def __init__(self):
        super().__init__()

    def search_update():
        path = "data\\"
        strange_path = "data\\ObjOrganizerSS20*.xlsx"
        dir = DirGrab(path)
        files = dir._glob_grabber(strange_path)
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        SS20_location = file_list[0]
        print(SS20_location)
        SS20 = pd.read_excel(SS20_location, na_values=["[]", 0])
        return SS20

    # Volumes in L and Lengths in cm
    def standardize_units(self, df):
        for col_name in df.columns:
            if 'Volume' in col_name:
                df[col_name] = df[col_name].map(lambda x: x * 10 ** -6)
            else:
                df[col_name] = df[col_name].map(lambda x: x / 10 if type(x) is float else x)
        return df

    def load_data(self):
        SS20_df = SS20DataSet.search_update()
        SS20_df['SubjectID'] = standardize_subject_ids(SS20_df['Name'])
        SS20_df = SS20_df.drop(SS20_df.columns[[0, 1, 2, 3]], axis=1)
        SS20_df = SS20_df.groupby(SS20_df['SubjectID'], as_index=False).aggregate('mean')
        SS20_df = self.standardize_units(SS20_df)
        combined_df = super().common_dataframes_2()
        combined_df = combined_df.merge(SS20_df, on='SubjectID', how='outer')
        combined_df['age'] = combined_df.apply(lambda row: birthday(row.BIRTHDATE), axis=1)
        combined_df['age'] = pd.to_numeric(combined_df['age'], errors='ignore')
        return combined_df
        

class NhanesDataSet(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        df = pd.read_csv("data/NHANES/NHANES-adults.csv")
        df.index.name = "SubjectID"
        df.reset_index(inplace=True)
        return df


class MalePcaPoints(DataSet):
    def __init__(self):
        super().__init__()

    def load_data(self):
        pca = pd.read_csv("data/PC_Weights_Male.csv")
        dxa = pd.read_csv("data/DXA.csv")
        return pca.merge(dxa, on="SubjectID", how='inner')


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
        return pd.concat([styku_df, ss20_df])


