from python.utilities.DirectoryGrab import DirGrab
'''
Class DirectoryFunction
Paramater targetDirectory: a string of the target directory specified for operations
Usage: hardcoded class for managing ply and mkr files for the PCA_App
feed your directory into the target directory, do the proper linking and it you should be good, please contact me asap with any error messages
'''
class DirectoryFunction:
    '''
    Method __init__
    Paramater targetDirectory: the directory for which all operations will be executed
    '''
    def __init__(self,targetDirectory = None):
        if targetDirectory is None:
            raise Exception("No Directory Passed")
        self.MKREXTENSION = ".mkr"
        self.PLYEXTENSION = ".ply"
        Director = DirGrab(targetDirectory)
        Director.grabByExtension('.mkr')
        self.mkrfiles = Director.getter()
        Director.grabByExtension(self.PLYEXTENSION,True)
        self.plyfiles = Director.getter()
        return None
    '''
    Method getMkrFiles
    Returns a list of mkr files from the target directory
    '''
    def getMkrFiles(self):
        return self.mkrfiles
    '''
    Method getplyfilse
    Returns a list of ply files fom the target directory
    '''
    def getplyfiles(self):
        return self.plyfiles

def __main__():
    newFunc = DirectoryFunction('D:\ML\python\PCA_App\process')
    print(newFunc.getMkrFiles())

if __name__ == '__main__':
    __main__()