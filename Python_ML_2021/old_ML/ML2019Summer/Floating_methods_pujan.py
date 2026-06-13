# import os # imports pythons os module
import glob
from datetime import date
# Searcher Class
# Searching class for retrieval of top 2 files based on version number
# Params
# input based on standardized input developed by team A of Escowell
# standardized input starts with styku or SS20 and contains v for version and a number representing the current version
# any deviation from standardized input will produce errors
import re # regular expression
# from DirectoryGrab import DirGrab

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

class Searcher:  #  File searcher
    def GreatestValue(self, valuelist : list) -> list:  #  takes in a list of string
        mirrorlist = []  # string initialization to hold values
        for name in valuelist: # searches through value list for version numbers
            mirrorlist.append(self.work(name))  # gets a value based on numeric identities in the file name
        valuetuple = zip(valuelist,mirrorlist)  # zips the values and names into a tuple
        topval = 0  # variable initialization : type int
        topname = ""  # variable initialization : type string
        secondval = 0  # variable initialization : type int
        secondname = ""  # variable initialization : type string
        for name, value in valuetuple:  # parse through tuple for 2 highest values
            if value > topval:
                secondval = topval
                topval = value
                topname = name
            if value > secondval & value != topval:
                secondval = value
                secondname = name

        mirrorlist.clear()  # recylces list
        mirrorlist.append(topname)  # appending the top name
        mirrorlist.append(secondname)  # appending the second highest name
        return mirrorlist  # return list

    def work(self,name : str) -> int:  # method for calculating values from strings, takes a string passes an int
        splitlip = re.split('[a-z.]+', name, flags=re.IGNORECASE)  # splits the string using regular expression.
        #  currently amounts alphabetic characters and periods update as needed
        numstr = ""
        numstr = numstr.join(splitlip)  # variable assingment type str
        splitlip = list(numstr)
        if name.__contains__("SS20"):  # SS20 exception
            splitlip[0] = str(0)  # changes 2 to 0
            splitlip[1] = str(0) # take this out and it breaks, changes 0 to 0
        if splitlip.__contains__("_"):
            while splitlip.__contains__("_"):
                splitlip.pop(splitlip.index("_"))
        if splitlip.__contains__('\\'):
            while splitlip.__contains__("\\"):
                splitlip.pop(splitlip.index("\\"))
        numstr = ""
        numstr = numstr.join(splitlip)
        return int(numstr)  # return integer representation


# IMPORTANT
# pathname passed to be referenced must contatin YOUR pathname to your target directory,
# if the pathname is not passed correctly you will have errors

class DirGrab: # class name
    def __init__(self,path):  # init method, takes a pathname
        self.path = path  # keeps the path in class variables

    #def os_grabber(self) -> list:  # grab method that returns a list of filenames from the given path directory
    #    files = []  # list initialization
    #    # r-root, d-directory, f = files
    #    for r , d, f in os.walk(self.path): # for loop
    #       for file in f:  # nested for loop
    #           # if ()
    #            files.append(file)  # appending filenames to return list

    #   return files  # return

    def glob_grabber(self, strange_path) -> list:
        files = []
        files = glob.glob(strange_path)
        return files

def birthday(birthd):
    today = date.today()
    if type(birthd) == str:
        return today.year - int(birthd[-2:]) - 1900
    else:
        return ""

def perlimbstring(old_list):
    string = 'Styku_'
    my_new_list = [string + x for x in old_list]
    #string2 = 'SS20_'
    #my_new_list2 = [string2 + x for x in old_list]
    return my_new_list
    #return (my_new_list + my_new_list2)

def column_filter(cname, keep):
    def ret(df, feature_columns):
        if cname in feature_columns:
            feature_columns.remove(cname)
        return df[df[cname] == keep]
    return ret

def standardize_units(df, body_parts):
    for bp in body_parts:
        df[bp.styku.volume] = df[bp.styku.volume] * (2.54 ** 3)  # in3 to cm3
        df[bp.styku.volume] = df[bp.styku.volume] / 1000  # cm3 to L
        df[bp.dexa.volume] = df[bp.dexa.volume] / 1000  # cm3 to L
        #df[bp.ss20.volume] = df[bp.ss20.volume] / 1000000  # mm3 to L
        #df[dexa_total_volume] = df[dexa_total_volume] / 1000
    return df

def standardize_subject_ids(series):
    seen = []
    id_len = len("02ADL0153")  # standard length of ids

    def map_name(name):
        if name is not str:
            name = str(name)
        name = name.upper()
        id = name[0:id_len]  # takes name column and extracts ids for beginning of string
        seen.append(id)
        if len(name) > id_len:
            suffix = name[id_len:id_len + 2]  # next 2 characters after id
            if suffix == "_A":
                return id
            elif suffix == "_B":
                return id + "_2"
        return id if seen.count(id) == 1 else f"{id}_{seen.count(id)}"
        # possibilities: id, id_2, id_n

    return series.apply(map_name)

def standardize_units_2(df):
    for col_name in df.columns:
        if 'Volume' in col_name:
            df[col_name] = df[col_name].map(lambda x:  x * 16.3871 / 1000)
        else:
            df[col_name] = df[col_name].map(lambda x: x * 2.54 if type(x) is float else x)
    return df

def HD_classification(rowvalue):
    if rowvalue < 6:
        return 1
    elif (6<= rowvalue <= 7):
        return 2
    elif rowvalue > 7:
        return 3
    else:
        return

