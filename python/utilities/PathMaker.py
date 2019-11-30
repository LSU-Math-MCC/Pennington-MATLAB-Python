import os
import sys

LEVELS = 1
class PathMan():
    def __init__(self,levels = LEVELS):
        os.chdir(os.path.abspath(os.path.split(sys.argv[0])[0]))
        cwd = os.getcwd()
        #print("FOLDERSEARCHER:",cwd)
        self.levels = levels
        self.path = None
        self._pathCutter(cwd)

    def _pathCutter(self,string):
        lister = string.split("\\")

        for i in range(len(lister)):
            if lister[i]=="python":
                j=i
        #print(j)

        #length = len(lister)  # - level
        #print(length - j)

        path = ""
        for i in range(j):
            path = path + lister[i] + "\\"
        self.path = path
        #print(path)

    def getter(self):
        return self.path



if __name__ == '__main__':
    man = PathMan()
    #print(man.getter())
