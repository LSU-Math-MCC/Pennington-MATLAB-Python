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
import numpy as np
import math
import pandas as pd
import time
import itertools

import sys

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

def standardize_subject_ids_2(series):
    id_len = len("02ADL0153")
    def map_name(name):
        name = name.upper()
        id = name[0:id_len]
        return id
    return series.apply(map_name)

#Changing the Volume of the dataset to be standardized to be Cm and ML

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

# Append classes with

def append_class(df, cname, segmentation, classnames=None, defaultclass='nan'):
    if classnames is None:
        classnames = list(range(len(segmentation) + 1))
    assert np.all(np.diff(segmentation) > 0), 'Segmentation must be strictly increasing.'
    assert len(classnames) == len(segmentation) + 1, \
        'Number of classes must be one more than the number of floats in segmentation.'

    internal_bounds = partition(segmentation, 2, 1)[:-1]
    conditions = [lambda x: x < segmentation[0]]
    conditions += [lambda x, l = lower, u = upper: (l <= x) & (x < u)
                   for lower, upper in internal_bounds]
    conditions += [lambda x: segmentation[-1] <= x]
    condlist = lambda x: [cnd(x) for cnd in conditions]

    df[f'{cname}_class'] = np.select(condlist(df[cname]), classnames, default=defaultclass)
    return df

# Partition Python list with optional offset.
def partition(list_, n, offset=0):
    return [list_[i:i + n] for i in range(0, len(list_), n - offset)]

def Simpleslicer(pie: str) -> str:
    return pie[0:9]  # super simple list slicer for Styku name column
'''
creates a list from a list with an applied function
'''
def ListMaker(mylist):
    newList = []  # new list to be made
    for stringValue in mylist:  # for loop
        newList.append(Simpleslicer(stringValue)) #function to be applied ot list
    return newList  # return the list
'''
method for standardizing subject ID's with _1 or _2
'''
def ListStandardizer(myList):
    newList = []  # creates a new list
    lastVisited = []  # creates a list for last visited
    for value in myList:  # for loop
        if value not in lastVisited:  # if the value is an original
            lastVisited.append(value)  # look at value
            newList.append(str(value + "_1"))  # append suffix
        else:  # if value is a copy
            newList.append(str(value + "_2"))  # append suffix
    return newList  # return list


'''
method for standardizing a series, similar to ListStandardizer
'''
def Standardizer(serial):
    lastVisited = []  # see above
    newSeries = serial  # copy of series
    for i in range(len(serial)):  # for loop
        if serial.loc[i] in lastVisited:  # if the value is a copy
            serial.loc[i] = (serial.loc[i] + "_2")  # append suffix
        elif serial.loc[i] not in lastVisited:  # if the value is an original
            lastVisited.append(serial.loc[i])  # look at value
            serial.loc[i] = (serial.loc[i] + "_1")  # append suffix
    return serial  # return serial

'''
Method for inserting a row into a dataframe
takes a dataframe, splits the dataframe by the row number
merges the two dataframe with the new row number appended
'''
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df

'''
Method for altering a dataframe of a specific format.
assumes that the dataframe contains a column named SubjectID
assumes that each value in the subject ID is not standardized by a standardization method
returns a datframe that contains 2 rows per one ID value
'''
def CBDrowMaker(product: pd.DataFrame()) -> pd.DataFrame():
    modDf = product.copy()# dataframe copy to be returned
    Seen= [] # List of values already Seen
    flag = False # Boolean flag for Boolean Logic, initial value False
    #print(product)# prints input DataFrame
    iterator = product.iterrows()# creates an iterator for the input Datframe
    changeCounter = 0 # integer marker used to return data about changes
    altIndex = 0# integer incrementer used to represent the size of the modified dataframe index
    for index, series in iterator:# for loop
        changeCounter = 0# reset change Counter
        if index == 0:# if on the first row,
            #print("Zero index")# print indicator of the Zero index
            flag = True # sets the flag to be true
            Seen.append(product.at[index,'SubjectID']) # appends the first Subject ID
            pass# passes through the iteration
        else:# if not the first row
            #print(product.at[index,'SubjectID'])  # print the Subject Id to be changed
            if product.at[index,'SubjectID'] in Seen:  # if the row has been visited already
                #if flag == False: # checks to see that it is not a duplicate of a duplicat
                    #print("Delete Row")# print indication of a row to be deleted
                flag = False  # sets the flag false, indicates that a repeated value was the last enter
                #print("found prev value")  # console indicator of a prev value
                for index2,value in series.items():  # nested for loop for the row
                    tvalue = False  # boolean flag for empty value in current row
                    tvalue2 = False # boolean flag for empty value in previous row
                    if isinstance(series.at[index2],float):  # if current (row,column) value is a number
                        if math.isnan(series.at[index2]):  # if the value is empty
                            tvalue = True  # raise flag
                    elif series.at[index2] is pd.NaT:  # if the value is not a number and empty
                        tvalue = True  # raise flag
                    if isinstance(prevRow.at[index2],float):  # if prev (row,column) value is a number
                        if math.isnan(prevRow.at[index2]):  # if the value is empty
                            tvalue2 = True  # raise flag
                    elif prevRow.at[index2] is pd.NaT:  # if the value is not a number and empty
                        tvalue2 = True  # raise flag
                    if tvalue == True:  # if current row value is empty
                        if tvalue2 == False:  # if prev row value is not
                            #print("series[index2] = prevRow[index2]")  # indication in console for change
                            modDf.at[index + altIndex,index2] = prevRow[index2]  # modifies dataframe
                            changeCounter += 1  # increments change counter
                    else:  # if curren row value has a value
                        if tvalue2 == True:  # and the prev row does not
                            #print("prevRow[index2] = series[index2]")  # console indication for change
                            modDf.at[index - 1 + altIndex,index2] = series[index2]  # modifies dataframe
                            changeCounter += 1  # increments change counter
            else:  # if current subject ID not visited
                if flag == True:  # if the last subject value was an original
                    #print("Copy Previous Row")  # console indication of copy
                    modDf = Insert_row(index + altIndex,modDf,prevRow)  # copy row in new dataframe
                    altIndex += 1  # increment the modified dataframes index
                flag = True  # sets flag to True, indicating an original value was last
                Seen.append(product.at[index,'SubjectID'])  # appends value to subject ID
        prevRow =  series  # prevRow updater
        prevIndex = index  # prev Index updater (unused)
        #print("number of changes :", str(changeCounter))  # console indication of changes
        #print("Index incrementation: ",altIndex)  # console indication of Index Incrementation
    #print(modDf.head(n=40))  # prints modified data head for console verification
    return modDf  # returns the modified dataframe


def MergeMan(Styku:pd.DataFrame(),CBD: pd.DataFrame()) -> pd.DataFrame():  # dataframe Merging
    CBDcopy = CBD.copy()  # load CBD dataframe
    Stykucopy = Styku.copy()  # load styku dataframe
    copy = CBDcopy.merge(Stykucopy,on='SubjectID',how='outer',copy=True)  # merge the two dataframes
    # CBDcopy.merge(CBDcopy,on='SubjectID',how='outer')
    # copy = pd.concat([CBDcopy,Stykucopy],ignore_index=True,)
    copy = copy.drop_duplicates(subset='SubjectID')
    return copy  # return the dataframe



def animate(done):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rloading ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!     ')

'''SYNTAX FOR ANIMATE
import threading, time

d = False
t = threading.Thread(target=animate(d))
t.start()


#your long code goes here

time.sleep(10)
d = True


'''