import os
import sys
from platform import system

LEVELS = 1
class PathMan():
    def __init__(self,levels = LEVELS):
        os.chdir(os.path.abspath(os.path.split(sys.argv[0])[0]))
        cwd = os.getcwd()
        # print("FOLDERSEARCHER:", cwd)
        self.split_str = {
            'Windows': '\\',
            'Linux': '/',
            'Darwin': '/'  # Mac
        }[system()]  # Clint - added this attribute to fix some cross-platforming issues
        self.levels = levels
        self.path = None
        self._pathCutter(cwd)

    def _pathCutter(self,string):
        lister = string.split(self.split_str)

        for i in range(len(lister)):
            if lister[i]=="python":
                j=i
        #print(j)

        #length = len(lister)  # - level
        #print(length - j)

        path = ""
        for i in range(j):
            path = path + lister[i] + self.split_str
        self.path = path
        #print(path)

    def getter(self):
        return self.path



if __name__ == '__main__':
    man = PathMan()
    #print(man.getter())
