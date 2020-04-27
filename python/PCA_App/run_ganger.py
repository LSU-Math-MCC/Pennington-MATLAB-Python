import argparse as ap
import glob
import os
import time
import psutil
import tkinter as tk
from tkinter import filedialog
from os import system
from utilities.DirectoryGrab import DirGrab
from utilities.PathMaker import PathMan

Path = PathMan()
GANGER_PATH = Path.getter() + 'python/PCA_App/ganger/'
GANGER_BINARY = 'ganger.exe'
STARTUP_FILE = GANGER_PATH + 'startup.txt'

MESH_EXT = '.ply'
MKR_EXT = '.mkr'
DEFAULT_OUTDIR = '/fitted/'
DEFAULT_OUTPUT_SUFFIX = '_fitted'

# TEMPLATETRANS = Path.getter() + 'python/PCA_App/ganger/data/template-60k-m_rot_trans.ply'
# TEMPLATEMODIFIED = Path.getter() + 'python/PCA_App/ganger/data/template-60k-m-adjust.mkr'

# concurring = int(input("Enter number of Simultaneous runs: "))
concurring = 4 # NEED TO CHANGE BACK FOR REAL PIPELINE

class GangGang:
    def __init__(self):
        exarray = self.identify_points()
        self.in_meshes = exarray[0]
        self.in_points = exarray[1]
        self.out_meshes = exarray[2]
        self.run_ganger()


    def run_ganger(self):
        counter = 0
        # the following code writes a start up file for the list of instructions to feed to ganger. StartMatch refers to the settings we use to initailize the fitting. each line refers to a round of fitting.
        for i in range(len(self.in_meshes)):
            with open(STARTUP_FILE, 'w') as f:
                print(self.out_meshes[i])
                if self.in_points[i] is not None:
                    f.write('loadMesh 0 '+ 'data/template-60k-m_rot_trans.ply' + '\n')
                    f.write('loadMarkers 0 '+ 'data/template-60k-m-adjust.mkr' + '\n')
                    f.write('loadMesh 1 ' + self.Bin_meshes[i] + '\n')
                    f.write('loadMarkers 1 ' + self.in_points[i] + '\n')
                    #f.write('show 1'+ '\n')

                    f.write('startMatch 1 1   0   0' + '\n')
                    f.write('startMatch 0 2   0   0.2 800' + '\n')
                    f.write('startMatch 0 10  0   0.2 40' + '\n')
                    f.write('startMatch 0 10  0.2 0.2 100' + '\n')
                    f.write('startMatch 0 5   1   0.3 100' + '\n')
                    f.write('startMatch 0 0.5 10  0.5 100' + '\n')
                    f.write('startMatch 0 0.1 20  0.3 20' + '\n')
                    #f.write('show 1'+ '\n')
                    f.write('saveMesh ' + self.out_meshes[i] + '\n\n')

            self.open_ganger()

            time.sleep(10)

            self.close_display()

            counter = counter + 1

            # counts total number of fittings initialized
            print(counter)

            self.num()

            start = 0

            # after opening each iteration of ganger, check the number running
            if len(self.num()) == concurring:
                start += 1
            # while at 'concurring' value, continually check the # running, and when it drops below 'concurring' value, exit
            while start != 0:
                check = []
                for p in psutil.process_iter(attrs=['name']):
                    if 'ganger' in p.info['name']:
                        check.append(1)
                if len(check) < concurring:
                    break


    # closes the GUI window
    def close_display(self):
        system('taskkill /FI "WINDOWTITLE eq ganger" ')


    # create array containing #-elements of "ganger" processes
    def num(self):
        gang = []
        for p in psutil.process_iter(attrs=['name']):
            if 'ganger' in p.info['name']:
                gang.append(1)
                # print(gang)
        return gang


    # runs ganger
    def open_ganger(self):
        print('Starting Ganger...')

        # change to ganger directory
        dir0 = os.getcwd()
        os.chdir(GANGER_PATH)
        # os.startfile will open a separate batch window instead of running it through the PyCharm 'Run' terminal
        # rather than os.system(GANGER_BINARY). Need this to run multiple at once
        os.startfile(GANGER_BINARY)

        # return to original directory
        os.chdir(dir0)


    def identify_points(self):
        print('No input directory specified - please select one now')
        root = tk.Tk()
        root.withdraw()
        indir = filedialog.askdirectory()
        outdir = indir + DEFAULT_OUTDIR

        # create output(fitted) directory if it doesn't already exist
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # get input meshes and create MKR files
        in_meshes = glob.glob(indir + '/*' + MESH_EXT)  # array containing the paths of all of the input .ply's
        in_points = []  # creates "in_points" array
        out_meshes = []  # creates "out_meshes" array

        for mesh in in_meshes:
            mesh_base = os.path.basename(mesh)  # path of .ply input
            mesh_base_root = os.path.splitext(mesh_base)[0]  # subject ID_A/B
            mkr_file = os.path.splitext(mesh)[0] + MKR_EXT  # path of .mkr input
            in_points.append(mkr_file)  # path of .mkr input = "mkr_file"
            out_mesh = outdir + mesh_base_root + DEFAULT_OUTPUT_SUFFIX + MESH_EXT  # path of fitted .ply in fit folder
            out_meshes.append(out_mesh)  # path of fitted .ply in fitted folder = "out_mesh"

        return [in_meshes, in_points, out_meshes]


def __main__():
    gang3x = GangGang()

# Uncomment for pipeline
if __name__ == "__main__":
   __main__()

