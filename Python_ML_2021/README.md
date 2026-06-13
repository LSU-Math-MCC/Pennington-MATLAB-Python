# ML-MATLAB
# list of Packages to be installed and loaded when installing Octave to run Avatar.m:
# - pkg install -forge geometry
# - pkg intall -forge statistics
# - pkg load geometry
# - pkg load statistics


# How to call the Avatar.m file in octave:
# Avatar Styku_01.obj
# For Octave to run, don't use following MATLAB instructions:
# ------------------------------------------------------------
# - round with precision (round alone is okay)
# - replace "incenter(triangulation([1,2,3],p),1)" with "centroid(delaunay(p),1)"

# leg Volume is commmented out because of the error (need to fix the logic)
# after it is working test in octave.