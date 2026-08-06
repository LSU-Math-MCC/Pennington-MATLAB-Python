"""MATLAB-faithful Python port of Avatar.m.

Verified bit-identical to the MATLAB/Octave reference across 34 measurements
and 14 landmarks on five test meshes.

Typical use::

    from avatar_conversion import MatlabAvatar, load_obj

    v, f = load_obj("scan.obj")
    avatar = MatlabAvatar(v, f).run()
    print(avatar.measurements["chestGirth"])
    print(avatar.landmarks["crotch"])
"""

from .matlab_avatar import MatlabAvatar
from .mesh_io import load_obj

__all__ = ["MatlabAvatar", "load_obj"]
__version__ = "1.0.0"
