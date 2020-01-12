# import os # imports pythons os module
import glob
import sys
sys.path.append('/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/MyPy')
import Searcher
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




def __main__():
    path = "/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data"
    strange_path = "/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/data/ObjOrganizer*.xlsx"
    dir = DirGrab(path)
    files = dir.glob_grabber(strange_path)
    for file in files:
        print(file)


if __name__ == "__main__":
    __main__()