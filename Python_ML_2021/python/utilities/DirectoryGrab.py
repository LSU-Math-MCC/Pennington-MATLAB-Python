# import os # imports pythons os module
import glob
'''
DirectoryGrab module
Created by Alex Mensen-Johnson
Last Edited 07/03/2019
Used for Extracting the contents of a Directory.
Dependencies glob2==0.6

'''
# IMPORTANT
# pathname passed to be referenced must contatin YOUR pathname to your target directory,
# if the pathname is not passed correctly you will have errors
'''
File for grabbing ObjOrganizer files from a targeted directory.
Paramater path: a string containing the target directory 
Usage: 
'''
class DirGrab: # class name
    def __init__(self,path):  # init method, takes a pathname
        self.path = path  # keeps the path in class variables
        self.files = None

    '''
    Method grabByExtension
    Parameter ext: a string containing the target extension for the file grab
    Usage: Reads the directory and stores a file list of the input type 
    '''
    def grabByExtension(self,ext = ".txt",bypass = False):
        self._emptyStringException(ext)
        if bypass is False:
            self._safety()
        edit = "/*"
        check = self._glob_grabber(self.path + edit + ext)
        if len(check)  == 0 :
            raise Exception("No Files returned in grab for : ",ext)
        self.files = check
    '''
    Method grabFromPrefix
    Paramater searchphrase: a string containing a prefix to match to file name
    Usage: Reads the directory and stores a file list  of files that begin with the passed prefix
    '''
    def grabFromPrefix(self, searchphrase,bypass = False):
        self._emptyStringException(searchphrase)
        if bypass is False:
            self._safety()
        edit = "/"
        asterik = "*"
        check = self._glob_grabber(self.path + edit + searchphrase + asterik)
        if len(check) == 0:
            raise Exception("No Files returned in grab")
        self.files = check
    '''
    Method grabFromSuffix
    Paramater searchphrase: a string containing a suffix to mathc to a file name
    Usage: Reads the directory and stores a file list of files that contain the passed suffix
    '''
    def grabFromSuffix(self, searchphrase,bypass = False):
        self._emptyStringException(searchphrase)
        if bypass is False:
            self._safety()
        edit = "/*"
        asterik = "*"
        check = self._glob_grabber(self.path + edit + searchphrase + asterik)
        if len(check) == 0:
            raise Exception("No Files returned in grab")
        self.files = check
    '''
    Method _emptyStringException
    Paramater string: the string to be checked
    Usage: Private method for Raising Exceptions
    '''
    def _emptyStringException(self,string):
        if len(string) == 0:
            raise Exception("EmptyStringException")
    '''
    Method _safety
    Usage: A safety check to prevent overwriting previous file lists
    '''
    def _safety(self):
        if self.files is not None:
            safe = str(input("Attempting to overwrite existing class files, Are You sure?\n Enter 1 to proceed, any other key to stop\n"))
            print("Your input: " + safe)
            print("You can bypass this check by passing a True boolean parameter to the Method")
            if not safe == "1":
                raise Exception("run terminated by user")
    '''
    Method _glob_grabber
    Paramater path: a string containing the phrase used by glob to grab files
    Usage: Private Method for grabbing files
    Returns: a list of files that match the specifications of the string
    '''
    def _glob_grabber(self, path) -> list:
        files = []
        files = glob.glob(path)
        return files
    '''
    Method: namecleaner
    Paramater dirtylist: an list passed for file removal
    Paramater keeper: a string containing the phrase for extraction
    Paramater PujCleaner: a bool that denotes a true or false value
    returns a file list without any unwanted files from the list
    '''

    def namecleaner(self, dirtylist, keeper = "ObjOrganizer", PujCleaner=False) -> list:
        cleanlist = []
        for file in dirtylist:
            if keeper in file:
                cleanlist.append(file)
        if len(cleanlist) == 0:
            print("Directory doesnt contain file with {",keeper,"}. List size 0")
            return cleanlist
        count = 0
        if PujCleaner is True:
            for file in cleanlist:
                cleanlist[count] = file[(file.find("Obj")):len(file) ]
                count += 1
        return cleanlist
    '''
    Method getter
    Usage: a method to pass a stored file list to the user
    Returns: a list of files stored with in the class
    '''
    def getter(self):
        return self.files


def __main__():
    path = "/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data"

    strange_path = "D:\ML\python\PCA_App\process"

    dir = DirGrab(strange_path)
    dir.grabByExtension(".mkr")
    files = dir.getter()
    print(dir.files)
    dir.grabByExtension(".ply")
    files = dir.getter()
    print(files)
    dir.grabFromSuffix("00#3")
    print(dir.getter())
    dir.grabFromPrefix("01ADL001")
    print(dir.getter())


if __name__ == "__main__":
    __main__()