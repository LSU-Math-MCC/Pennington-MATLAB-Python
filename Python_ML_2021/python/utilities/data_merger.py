import pandas as pd
import math
import re

#global SCopy
# global CCopy

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
                #    print("Delete Row")# print indication of a row to be deleted
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

def loadStyku():
    '''
    SCopy = pd.DataFrame(pd.read_excel("ML-MATLAB-master/data/ObjOrganizerStyku.xlsx"))
    SCopy['SubjectID'] = ListStandardizer(ListMaker(SCopy.Name))
    '''
    data = pd.DataFrame(pd.read_excel("ObjOrganizerStyku_v6.xlsx"))  # loads DataFrame
    # data = pd.read_excel("ML-MATLAB-master/data/ObjOrganizerStyku.xlsx",index_col=False)
    data['SubjectID'] = ListMaker(data.Name)  # creates Subject ID column from Name column
    modData = CBDrowMaker(data)  # creates modified DataFrame
    modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # standardizes subject ID column
    # data.loc['SubjectID'] = Standardizer(data['SubjectID'])
    return modData  # returns the DataFrame

def loadCBD():#Combo, Blood, Dexa
    data = pd.read_excel("Dxa_Blood.xlsx")  # loads the DataFrame
    # data['SubjectID'] = ListStandardizer(data.SubjectID)
    modData = CBDrowMaker(data)  # creates a modified Dataframe
    modData['SubjectID'] = ListStandardizer(modData.SubjectID)  # Standardizes the Subject Id Column
    return modData  # returns the Dataframe
    '''
    CCopy = pd.read_excel("Alex_sExcelSheet.xlsx")
    CCopy['SubjectID'] = ListStandardizer(CCopy.SubjectID)
    '''

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


def workDammit(list):  # proof that the the above algorithms work
    lastVisited = []
    for each in list:
        if each in lastVisited:
            print(True)
        elif not each in lastVisited:
            print(False)
            lastVisited.append(each)


def switch():  # empty method (ignore)
    return None

def MergeMan(Styku:pd.DataFrame(),CBD: pd.DataFrame()) -> pd.DataFrame():  # dataframe Merging
    CBDcopy = CBD.copy()  # load CBD dataframe
    Stykucopy = Styku.copy()  # load styku dataframe
    copy = CBDcopy.merge(Stykucopy,on='SubjectID',how='outer',copy=True)  # merge the two dataframes
    # CBDcopy.merge(CBDcopy,on='SubjectID',how='outer')
    # copy = pd.concat([CBDcopy,Stykucopy],ignore_index=True,)
    copy = copy.drop_duplicates(subset='SubjectID')
    return copy  # return the dataframe

def MrDoitAll():
    error = MergeMan(loadStyku(), loadCBD())  # load method
    error.to_excel("Merged.xlsx")  # copy file to excel

def __main__():
    # error = loadCBD()
    # error = MergeMan()
    #oops = loadStyku()
    #oops.to_excel("ModStyku.xlsx")
    # error.to_excel("Merged.xlsx")
    # error = loadStyku()
    error = MergeMan(loadStyku(),loadCBD())  # load method
    error.to_excel("Merged.xlsx")  # copy file to excel

if __name__ == "__main__":
    __main__()
