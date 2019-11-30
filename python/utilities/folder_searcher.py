# Searcher Class
# Searching class for retrieval of top 2 files based on version number
# Params
# input based on standardized input developed by team A of Escowell
# standardized input starts with styku or SS20 and contains v for version and a number representing the current version
# any deviation from standardized input will produce errors
import re # regular expression
# from DirectoryGrab import DirGrab

'''
Note for programmer, programm needs to be able to remove any excess tokens or charachters from the name grabber
needs revisitation and edits
'''
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
        #print("Method work: variable splitlip = :",splitlip)
        if name.__contains__("SS20"):  # SS20 exception
            splitlip[0] = str(0)  # changes 2 to 0
            splitlip[1] = str(0) # take this out and it breaks, changes 0 to 0
        if splitlip.__contains__("-"):
            while splitlip.__contains__("-"):
                splitlip.pop(splitlip.index("-"))
        if splitlip.__contains__("_"):
            while splitlip.__contains__("_"):
                splitlip.pop(splitlip.index("_"))
        if splitlip.__contains__("\\"):
            while splitlip.__contains__("\\"):
                splitlip.pop(splitlip.index("\\"))
        if splitlip.__contains__("/"):
            while splitlip.__contains__("/"):
                splitlip.pop(splitlip.index("/")) # removes /
        if splitlip.__contains__(":"):
            while splitlip.__contains__(":"):
                splitlip.pop(splitlip.index(":")) # removes :
        #print(splitlip)
        numstr = ""
        numstr = numstr.join(splitlip)
        #print("pulling version :",numstr)
        return int(numstr)  # return integer representation


def __main__(): # testing

    ExList = []
    ExList.append("Stykufilenamev_123.xlx")
    ExList.append("Examplefilenamev124.xlx")
    ExList.append("Examplefilenamev223.xlx")
    ExList.append("SS20filenamev125.xlx")
    ExList.append("\\SS20_437.xlx")
    ExList.append("SS20v111.xss")
    ExList.append("\\Stykuv036.docx")
    ExList.append("stykuv224.docx")

    Search = Searcher()
    lista = Search.GreatestValue(ExList)
    #for each in lista:
    #    print(each)


if __name__ == "__main__":
    __main__()
