import glob, os
from sklearn.metrics import SCORERS
from utilities.PathMaker import PathMan

Path = PathMan()
GITPATH = Path.getter()


def print_files(rel_dir):
    os.chdir(GITPATH + rel_dir)
    print(GITPATH + rel_dir)
    for file in glob.glob("*"):
        print(file)
    print('\n')

print_files('python/utilities/')

print(SCORERS['accuracy'])