### Matlab to Octave

#### Requirement

- pkg install -forge geometry
- pkg intall -forge statistics

#### Run

- pkg load geometry
- pkg load statistics
- `Avatar Styku_01.obj` <- replace with your obj file

#### Incompatible Matlab to Octave usages

##### Round with precision

- `round(X, N)` N is N digitis of presision (Not allow in Octave)
- `round(X)` this is ok.

##### Triangulartion and Incenter

- replace `incenter(triangulation([1,2,3],p),1)` with `centroid(delaunay(p),1)`
