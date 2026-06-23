import os
from python.utilities.DirectoryProcessor import DirectoryFunction
import ntpath
from python.utilities.PathMaker import PathMan
from python.utilities.DirectoryGrab import DirGrab

'''
Module for converting Directories
USER GUIDE:
Created By Alex Mensen-Johnson
Python module for creating edited mkr files
The following variables outside of the class may need to be edited to represent your directory,
    with the evolution of this code expect this to be removed and replaced with class variables
MARK = AN indication of an edited file, if this needs to be removed, create an empty string for the time being
SUBJECTFOLDER = target directory of un edited PCA Subjects, change this to represent your directory
MKREXTENSION = marker file extension, created for ease of file creation
MKRBRANCH = branched marker folder, note that this will be the destination for your files
OUTPUTFOLDER = self explanitory, change only if neccessary. most likely not necessary
SECURITYMEASURES = seecurity protocol for write functions
SECURITYMEASURES2 = part 2 of security protocol
TEMPLATEINPUT = Target of template file, change this to fit your directory
TEMPLATEOUTPUT = Target output of template file, change this to fit your directory and naming conventions
MKRFILES = SEE DIRECTORY FUNCTION, the passed parameters or all the marker files in a directory containing marker
    and ply files. you will have to make change DIRECTORY FUNCTION to represent your structure

'''

MARK = ""
x = PathMan()
print('\n' + x.getter())
SUBJECTFOLDER = x.getter() + "python/PCA_App/process"
MKREXTENSION = ".mkr"
MKRBRANCH = SUBJECTFOLDER + "/MKR"
OUTPUTFOLDER = MKRBRANCH + "/output"
SECURITYMEAUSURES = "Are you sure you want to create the following Directory?"
SECURITYMEAUSURES2 = "Enter 1 for yes and 0 for no, any other entry is an automatic NO: "
TEMPLATEINPUT = x.getter() + "python/PCA_App/afterMarker/template-60k-m-adjust.mkr"
TEMPLATEOUTPUT = x.getter() + "python/PCA_App/afterMarker/templateModified.mkr"
DirGar = DirectoryFunction(x.getter() + "python/PCA_App/process")
MKRFILES = DirGar.getMkrFiles()
HARDCODEPOINTS = [30, 42, 55, 65, 76]
ALLPOINTS = range(1, 75)


class FileConverter:
    def __init__(self, mkrFiles=MKRFILES,
                 Mark=MARK,
                 ext=MKREXTENSION,
                 Folder=MKRBRANCH,
                 outputFolder=OUTPUTFOLDER,
                 templateInput=TEMPLATEINPUT,
                 templateOutput=TEMPLATEOUTPUT,
                 ):

        self.MkrFiles = mkrFiles
        self.MkrFileContent = []
        self.Mark = Mark
        self.ext = ext
        self.Folder = Folder
        self.outputFolder = outputFolder
        self.MkrOutput = []
        self.templateInput = templateInput
        self.templateContents = None
        self.templateOutput = templateOutput
        self.pointers = None
        self.contents = None
        self.output = None

    '''
    load file helper method, loads files and returns the contents by line. DO NOT USE
    Python naming conventions make methods with leading underscores only accessible by the class
    aka this variable is private aka if you use it and it doesnt work it means you didnt read and i hope you 
    feel dumb

    '''

    def _loadFile(self, file):
        subFile = open(file, 'r')
        contents = subFile.readlines()
        subFile.close()
        return contents

    '''
    helper method to copy the contents of mkr files, DNE, contains underscore, see above
    '''

    def _copyMkrContents(self):
        if len(self.MkrFiles) == 0:
            print("MKR Files not Grabbed")
        for content in self.MkrFiles:
            self.MkrFileContent.append(self._loadFile(content))

    '''
    TemplateLoader, loads the template file into the class, DNE, contains underscore, see above
    '''

    def _TemplateLoader(self):
        newFile = open(self.templateInput, "r")
        self.templateContents = newFile.readlines()
        newFile.close()

    '''
    feedPoints feeds a list of points into the class, if anything but integers are passed then it will raise an error
    if any points are above the size of the mkr files then the run will break, so dont do that.
    DNE, contains underscore, see above
    '''

    def _feedPoints(self, pointlist):
        for point in pointlist:
            if isinstance(pointlist, int):
                raise Exception("List should contain only integers: type passed: ", type(point))
        self.pointers = pointlist

    '''
    templateFileMaker is the creator of the target template file, if the file has not been read it will raise an error
    if points have not been fed then it will raise an error,
    success is denoted py template file created
    DNE, contains underscore, see above
    '''

    def _templateFileMaker(self):
        if self.templateContents is None:
            raise Exception("Template File not loaded properly")
        if self.pointers is None:
            raise Exception("Points have not been Selected")
        tempFile = open(self.templateOutput, 'w+')
        tempFile.write("1\n")
        tempFile.write(str(len(self.pointers)) + "\n")
        for line in range(len(self.templateContents)):
            if line in self.pointers:
                tempFile.write(self.templateContents[line])
        print("template File created")
        tempFile.close()

    '''
    mkrFileMaker, creates the mkr file raises exceptions if the file is not loaded or points have not been fed
    DNE,underscore,blah blah blah
    '''

    def _mkrFileMaker(self):
        if self.contents is None:
            raise Exception("content File Not loaded properly")
        if self.pointers is None:
            raise Exception("Points have not been selected")
        subFile = open(self.output, 'w+')
        subFile.write("1\n" + str(len(self.pointers)) + "\n")
        for line in range(len(self.contents)):
            if line in self.pointers:
                subFile.write(self.contents[line])
        subFile.close()

    '''
    creates the MkrFileName
    DNE
    '''

    def MkrFileName(self, fileName):
        filename = ntpath.basename(fileName)
        step1 = os.path.splitext(filename)
        step2 = step1[0]
        final = self.outputFolder + "/" + step2 + self.Mark + self.ext
        return final

    '''
    outputFolderMaker makes the outputFOLDER
    DNE,underscore
    '''

    def _outputFolderMaker(self):
        if not os.path.exists(OUTPUTFOLDER):
            security = int(input(SECURITYMEAUSURES + "\n" + SECURITYMEAUSURES2))
            if not security == 1:
                raise ("RUN TERMINATED BY USER")
            os.mkdir(self.Folder)
            os.mkdir(self.outputFolder)
            print("directory " + self.outputFolder + " created")

        else:
            print("directory " + self.outputFolder + " already exists")

    '''
    Folder Walker, Does the walk across the folder
    DNE,underscore
    '''

    def _MkrWalker(self):
        count = 0
        if len(self.MkrFileContent) == 0:
            print("Empty MkrFileContent")
        for each in self.MkrFileContent:
            self.output = self.MkrFileName(self.MkrFiles[count])
            self.contents = each
            self._mkrFileMaker()
            print(self.output + " created")
            count += 1

    '''
    this script runs the above methods in the correct order
    it takes in a list of points
    this is the only method you need to call.
    '''

    def easyScript(self, points=ALLPOINTS):
        self._copyMkrContents()
        self._TemplateLoader()
        self._feedPoints(points)
        self._templateFileMaker()
        self._outputFolderMaker()
        self._MkrWalker()


'''
    def FileMaker(self):
        if self.contents is None:
            raise Exception("Subject File not loaded properly")
        if self.templateContents is None:
            raise Exception("template File not loaded properly")
        if self.pointers is None:
            raise Exception("Points have not been Selected")
        subFile = open(self.output,'w+')
        tempFile = open(self.templateOutput,'w+')
        subFile.write("1\n")
        tempFile.write("1\n")
        subFile.write(str(len(self.pointers)) + "\n")
        tempFile.write(str(len(self.pointers)) + "\n")
        for line in range(len(self.contents)):
            if line in self.pointers:
                subFile.write(self.contents[line + 1])
        for line in range(len(self.templateContents)):
            if line in self.pointers:
                tempFile.write(self.templateContents[line + 1])
        subFile.close()
        tempFile.close()
'''


def __main__():
    ElDios = FileConverter()
    ElDios.easyScript()
    print("done")


if __name__ == '__main__':
    __main__()