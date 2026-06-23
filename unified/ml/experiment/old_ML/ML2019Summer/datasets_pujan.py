import pandas as pd
from common import MLPData, DataFrameScaler
from abc import abstractmethod
from Floating_methods_pujan import Searcher, DirGrab, birthday


def standardize_subject_ids(series):
    seen = []
    id_len = len("02ADL0153")
    def map_name(name):
        name = name.upper()
        id = name[0:id_len]
        return id
    return series.apply(map_name)


class DataSet:
    def __init__(self):
        self.df = self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def extract_data(self,
                     feature_cnames,
                     label_cname,
                     scaler_config={},
                     data_transformers=[],
                     blacklist_sids=[]):
        df = self.df.copy()

        df = df.set_index('SubjectID', verify_integrity=True)
        df = df[~df.index.isin(blacklist_sids)]

        #aggregation_functions = {'price': 'sum', 'amount': 'sum', 'name': 'first'}

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
                print(f"dropping SubjectID '{index}' due to empty values: {empty}")
                cnames_to_drop.append(index)
        df.drop(cnames_to_drop, inplace=True)


    #Will look into the name column and uses it to pull out the


class ShapeUpDataSet(DataSet):
    def __init__(self):
        super().__init__()
    #looks into the data file and provides the lastest dataset from marceline's team. It also standardizes the name and makes a new column called subject_id
    def search_update():

        path = "data\\"
        strange_path = "data\\ObjOrganizer*.xlsx"
        dir = DirGrab(path)
        files = dir._glob_grabber(strange_path)
        Search = Searcher()
        file_list = Search.GreatestValue(files)

        styku_location = file_list[1]
        styku = pd.read_excel(styku_location, na_values=["[]", 0])

        #code for SS20 -------------------------------------------
        # List2 = List[1]
        #ss20 =  pd.read_excel(List2)
        # remove rows with "0" values
        #ss20['SubjectID'] = standardize_subject_ids(ss20['Name'])
        # styku = styku.set_index('SubjectID')
        # styku = styku.drop(styku.columns[[0, 1, 2, 3, 4]],
         #                  axis=1)  # print(styku.columns[[0,1,2,3,4]]) # list of column names given indexes [0,1,2,3,4]
        return styku


    #method to reference all the databases and then yeilds them to form a list
    def spreadsheets(self):
        # ----------------------------------------------------------

        styku_df = ShapeUpDataSet.search_update()
        styku_df['SubjectID'] = standardize_subject_ids(styku_df['Name'])
        styku_df = styku_df.drop(styku_df.columns[[0, 1, 2, 3, 4]], axis=1)
        # styku_df = styku_df[(styku_df != 0).all(axis=1)
        styku_df = styku_df.groupby(styku_df['SubjectID'], as_index=False).aggregate('mean')

        # styku_df = pd.read_csv("data/measurements_for_purposes.csv")
        # styku_df = styku_df.rename(columns={styku_df.columns[0]: "SubjectID"})
        # styku_df = styku_df[(styku_df != 0).all(axis=1)]  # remove rows with "0" values
        # ----------------------------------------------------------
        yield styku_df

        #styku = pd.read_excel('data/Volume_edit.xlsx', na_values=["[]", 0])
        #styku['SubjectID'] = standardize_subject_ids(styku['Name'])
        #styku = styku.drop(styku.columns[[0, 1]], axis=1)
        #yield styku

        questionnaire_df = pd.read_csv('data/Questionnaire.csv')
        questionnaire_df['SubjectID'] = standardize_subject_ids(questionnaire_df['SubjectID'])
        yield questionnaire_df

        dexa_df = pd.read_csv("data/DXA.csv", na_values=["#N/A"])
        dexa_df['SubjectID'] = standardize_subject_ids(dexa_df['SubjectID'])
        yield dexa_df

        manual_df = pd.read_csv("data/Manual.csv")
        manual_df['SubjectID'] = standardize_subject_ids(manual_df['SubjectID'])
        yield manual_df

        blood_df = pd.read_csv("data/Blood.csv")
        blood_df['SubjectID'] = standardize_subject_ids(blood_df['SubjectID'])
        yield blood_df

        a_over_b_df = pd.read_csv("data/Styku_a_over_b.csv")
        a_over_b_df['SubjectID'] = standardize_subject_ids(a_over_b_df['SubjectID'])
        yield a_over_b_df

    def load_data(self):
        dfs = list(self.spreadsheets())
        combined_df = None
        for df in dfs:
            df.drop_duplicates(subset='SubjectID', keep='last', inplace=True)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.merge(df, on='SubjectID', how='outer')
        # print(combined_df['SubjectID'])
        # print("Standard")
        # combined_df['SubjectID'] = standardize_subject_ids(combined_df['SubjectID'])
        # print(combined_df['SubjectID'])
        # exit(1)
        # adds a new column that changes birthdate to year bypassing age transformer function
        combined_df['age'] = combined_df.apply(lambda row: birthday(row.BIRTHDATE), axis=1)
        # print(combined_df.apply(lambda row: complete_bmi(row)), axis=0)
        # combined_df['Derived_BMI'] = combined_df.apply(lambda row: complete_bmi(row))

        #combined_df.to_excel("data/combined_df.xlsx")
        # combined_df['BIRTHDATE'] = combined_df.BIRTHDATE.astype(str)
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

