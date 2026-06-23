import pandas as pd
from datasets import DataSet

class Map():
    """
     compact way to use dot notation
    """
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def collect(self, exclude_keys=[]):
        all_values = []
        Map.__collect(self.__dict__, exclude_keys, all_values)
        return all_values

    @staticmethod
    def __collect(dict, exclude_keys, l):
        for key, val in dict.items():
            if key in exclude_keys:
                continue
            if isinstance(val, str):
                l.append(val)
            else:
                Map.__collect(val.__dict__, exclude_keys, l)


left_arm  = Map(name="Left Arm", short_name="LARM",
                dexa =Map(volume="LARM_volume", fat_mass="LARM_FAT", lean_mass="LARM_LEAN"),
                styku=Map(volume="Styku_lArmVolume"),
                ss20 =Map(volume="SS20_lArmVolume"))
right_arm = Map(name="Right Arm", short_name="RARM",
                dexa =Map(volume="RARM_volume", fat_mass="RARM_FAT", lean_mass="RARM_LEAN"),
                styku=Map(volume="Styku_rArmVolume"),
                ss20 =Map(volume="SS20_rArmVolume"))
left_leg  = Map(name="Left Leg", short_name="LLEG",
                dexa =Map(volume="LLEG_volume", fat_mass="L_LEG_FAT", lean_mass="L_LEG_LEAN"),
                styku=Map(volume="Styku_lLegVolume"),
                ss20 =Map(volume="SS20_lLegVolume"))
right_leg = Map(name="Right Leg", short_name="RLEG",
                dexa =Map(volume="RLEG_volume", fat_mass="R_LEG_FAT", lean_mass="R_LEG_LEAN"),
                styku=Map(volume="Styku_rLegVolume"),
                ss20 =Map(volume="SS20_rLegVolume"))
head      = Map(name="Head", short_name="HEAD",
                dexa =Map(volume="HEAD_volume", fat_mass="HEAD_FAT", lean_mass="HEAD_LEAN"),
                styku=Map(volume="Styku_headVolume"),
                ss20 =Map(volume="SS20_headVolume"))
trunk     = Map(name="Trunk", short_name="TRUNK",
                dexa =Map(volume="TRUNK_volume", fat_mass="TRUNK_FAT", lean_mass="TRUNK_LEAN"),
                styku=Map(volume="Styku_trunkVolume"),
                ss20 =Map(volume="SS20_trunkVolume"))
body_parts = [left_arm, right_arm, left_leg, right_leg, head, trunk]


class Crap:
    def __init__(self):
        self.series = None

    '''Alex's Summary, this code takes a series of strings and cuts all the strings to length 9, the conditional
    if statement is hardcoded for editing subject id's provided by Obj files. if the subject id would end with
    _A the suffix is cut off. if it ends with _B the suffix is cut off and appended with _2'''
def standardize_subject_ids( series):
    seen = []  # list initializer
    id_len = len("02ADL0153")  # len method called on string len(0)

    def map_name(name):  # internal definition map_name method // Params = name : type (string)
        name = name.upper()  # .upper method on name // produces capital notation of string
        id = name[0:id_len]  # id  = string // slices string, starts at 0, ends at the length 9
        seen.append(id)  # appends id to the list
        if len(name) > id_len:  # if the id name is greater than length 9 // redundant name has been sliced to be the exact length
            suffix = name[id_len:id_len + 2]  # slices string from to id length
            if suffix == "_A":  # boolean logic for containing _A
                return id  # return id value
            elif suffix == "_B":  # if it contains B
                return id + "_2"  # returns id string + _2
        return id if seen.count(id) == 1 else f"{id}_{seen.count(id)}"  # returns id if id exists only once. else format

    return series.apply(map_name)  # retruns the appendend map name to the series
    '''
    Alex's Summary this method loads several excel files into dataframes, and then appends the data frames to a list
    and returns them,
    it calls the previous method on the subject id column to edit the id. 
    '''
def sources(scans=True):
    sources = []  # list initialization
    sources.append(pd.read_csv("/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data/DXA.csv"))  # appends csv data to list
    sources.append(pd.read_excel("/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data/Pennington_BodPod.xlsx"))  # see line above
    sources.append(pd.read_excel("/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/python/data/DXA_Q2_plusRegional_brief.xlsx").rename(
        columns={"Subject ID": "SubjectID"}))  # see above

    if scans:  # default value true
        for scan in "SS20", "Styku":  # iterating through the files
            df = pd.read_excel(f"/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data/ObjOrganizer{scan}.xlsx",
                               na_values="[]")  # reads excel file and loads it into
            names = df["Name"]  # list of names derived from data frame names column
            df.rename(columns={col: f"{scan}_{col}" for col in df.columns},
                      inplace=True)  # renames the data frame to the source value
            df["SubjectID"] = standardize_subject_ids(names)  # calls the prior method to name the column
            sources.append(df)  # appends the data frame to the list
    return sources  # returns the appended list


'''
Alex's Summary: A helper method used to convert all volume measurements to Liters
Params: df = Dataframe, body_parts = A defined variable mapping above.
returns: the converted dataframe
'''
def standardize_units(df, body_parts):  #
    for bp in body_parts:
        df[bp.styku.volume] = df[bp.styku.volume] * (2.54 ** 3)  # in3 to cm3
        df[bp.styku.volume] = df[bp.styku.volume] / 1000  # cm3 to L
        df[bp.dexa.volume] = df[bp.dexa.volume] / 1000  # cm3 to L
        df[bp.ss20.volume] = df[bp.ss20.volume] / 1000000  # ??? to L
       # df[dexa_total_volume] = df[dexa_total_volume] / 1000 intentionally commented out by blake
    return df

'''
Alex's Summary: SubClass that inherits The DataSet Class from datasets.py,
Overides the Abstract Method load Data
load method loads a dataframe from combined sources from the sources method
it standardizes the units of meausurements into liters and returns the dataframe
'''
class ModifiedDataSet(DataSet):
    def load_data(self):
        combined_df = None  # sets variable to none
        for df in sources(): # loop through list of dataframes
            df["SubjectID"] = standardize_subject_ids(df["SubjectID"]) # standardized the name of subject id's
            # :: Note from Alex: Subject Id's have been standardized by method call // Redundant
            combined_df = combined_df.merge(df, on="SubjectID", how="outer", suffixes=('_1', '_2')) if combined_df is not None else df  # look up later
        return standardize_units(combined_df, body_parts)  # return returns the combined data frame with the the body parts


def volume_transformer(field):  #
    def transformer(df, feature_columns):  #
        df[field + "_cuberoot"] = df[field] ** (1/3)  # data frame transforme
        df[field + "_cuberoot"] = df[field] ** (1/3)
    return transformer

body_part_features = (
    (left_arm,   ["leftArmLength", "l_forearmgirth", "l_bicepgirth", "l_wristgirth"], ["l_forearm_a_over_b", "l_bicep_a_over_b", "l_wrist_a_over_b"]),
    (right_arm,  ["rightArmLength", "r_forearmgirth", "r_bicepgirth", "r_wristgirth"], ["r_forearm_a_over_b", "r_bicep_a_over_b", "r_wrist_a_over_b"]),
    (left_leg,   ["leftLegLength", "rCalfCircumference", "r_ankle_girth", "lThighGirth"], ["lCalf_a_over_b", "l_ankle_a_over_b", "lThigh_a_over_b"]),
    (right_leg,  ["rightLegLength", "rCalfCircumference", "r_ankle_girth", "rThighGirth"], ["rCalf_a_over_b", "r_ankle_a_over_b", "rThigh_a_over_b"]),
)

essential_transformers = shapeup_common.essential_transformers

loader_params = dict(
    data_transformers=essential_transformers,
    scaler_config={"SEX": LabelEncoder,
                   # "age": MinMaxScaler,
                   "default": StandardScaler},
)

loader_param_grid = dict(
    data_transformers={
        "__cols": "SEX",
        "M": essential_transformers + [column_filter("SEX", "M")],
        "F": essential_transformers + [column_filter("SEX", "F")],
        "M/F": essential_transformers,
    },
)

def __main__():
    # serious = pd.Series(dummylist)
    crap = Crap()
    # serious = crap.standardize_subject_ids(serious)
    # print(serious)
    MDF = ModifiedDataSet()

    MDF.load_data().to_excel("Alex'sExcelSheet.xlsx") # produces a 263 by 963 excel file

if __name__ == "__main__":
    __main__()
