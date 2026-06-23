from python.utilities.DirectoryGrab import DirGrab
from shutil import copyfile
import os

'''
class MoverMan
Created By Alex Mensen-Johnson
Paramater inputpath: a string containing the input path for multiple file moves
Paramater outputpath: a string containing the output destination for multiple file move operations
Usage a class used to perform a multifile move operation from one folder to the next
dependencies
python.utilities.DirectoryGrab==1.00
shutil
os

'''
class MoverMan:
    def __init__(self,inputpath = "/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/PCA/PCAsubjects/",outputpath = "/Users/idky/PycharmProjects/EscoWell/ML-MATLAB-master/PCA/PCAsubjects/MKR"):
        self.DG = DirGrab(inputpath)
        self.contents = None
        self.targetFolder = outputpath
        self.inputFolderFiles = None
    '''
    Method Work
    Usage: the function to move all target files to the target folder
    '''
    def MoveByExt(self,ext = None,bypass = False):
        if bypass is False:
            self._TwoTime()
        if ext is None:
            raise Exception("Empty Argument passed")
        self.DG.grabByExtension(ext,bypass)
        self.inputFolderFiles = self.DG.getter()
        self._Work()

    def MoveByPrefix(self,prefix = None,bypass = False):
        if bypass is False:
            self._TwoTime()
        if prefix is None:
            raise Exception("Empty Argument Passed")
        self.DG.grabFromPrefix(prefix)
        self.inputFolderFiles = self.DG.getter()
        self._Work()

    def _TwoTime(self):
        if not self.inputFolderFiles is None:
            raise Exception('Move Operation already performed, run ended for safety')

    def _Work(self):
        if len(self.inputFolderFiles) == 0:
            raise Exception("Empty List input problem bla")
        for file in self.inputFolderFiles:
            name = os.path.basename(file)
            outName = self.targetFolder + "/" + name
            copyfile(file,outName)
        print("Successfull move operation performed")


def __main__():
    Move = MoverMan()
    Move._Work()

if __name__ == "__main__":
    __main__()