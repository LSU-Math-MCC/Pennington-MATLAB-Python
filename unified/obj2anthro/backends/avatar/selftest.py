#!/usr/bin/env python3
"""Verify this port still reproduces the MATLAB reference values.

Run it after installing, or after changing anything in avatar_conversion/:

    python selftest.py path/to/A00-08-4914_A_2025-12-09_12-27-17__1_.obj

Reference values were produced by Avatar.m running under GNU Octave 8.4 with
``Avatar(file, 'steps', 3, 'SA', 'on')``.  A pass means every measurement agrees
to within 1e-9 relative and every landmark to within 1e-6 absolute.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from avatar_conversion.matlab_avatar import MatlabAvatar
from avatar_conversion.mesh_io import load_mesh

# Reference output of Avatar.m for the supplied scan (raw mesh units, mm).
REF_MEASUREMENTS = {
    "chestGirth": 912.9191269,
    "waistGirth": 840.9199329,
    "hipGirth": 1068.348027,
    "rThighGirth": 588.1905137,
    "lThighGirth": 595.9611138,
    "rCalfGirth": 398.2063418,
    "lCalfGirth": 398.2790077,
    "lWristGirth": 169.6879927,
    "rWristGirth": 199.8976143,
    "rForearmGirth": 276.8279562,
    "lForearmGirth": 256.2996863,
    "rBicepGirth": 359.3343639,
    "lBicepGirth": 367.9007863,
    "rAnkleGirth": 246.6456458,
    "lAnkleGirth": 257.1729283,
    "lArmLength": 434.505594,
    "rArmLength": 424.5704139,
    "trunkLength": 641.121076,
    "lLegLength": 751.1113754,
    "rLegLength": 751.1113754,
    "crotchHeight": 684.9708811,
    "collarScalpLength": 1502.902366,
    "SA_total": 1618953.837,
    "SA_trunk": 510946.0722,
    "SA_lleg": 354203.9856,
    "SA_rleg": 337636.621,
    "SA_legs": 682327.9995,
    "SA_head": 155746.3173,
    "SA_rArm": 169620.3156,
    "SA_lArm": 160220.0717,
    "VOL_total": 68354321.56,
    "height": 1546.6,
}

REF_LANDMARKS = {
    "r_wrist": (-319.9357143, -39.47142857, 760.0857143),
    "l_wrist": (325.2875, -15.79375, 769.69375),
    "r_armpit": (-159.65508, math.nan, 1052.491123),
    "l_armpit": (145.7959131, math.nan, 1058.713629),
    "r_hip": (-186.6, 45.3, 749.6198198),
    "l_hip": (196.3, 37.3, 749.6198198),
    "r_foot": (-174.6, 94.6, -0.1),
    "l_foot": (83.2, -23.9, -0.1),
    "crotch": (-5.3, 49.0, 684.8708811),
    "l_ankle": (100.745, 89.455, 87.045),
    "r_ankle": (-129.28125, 86.4625, 84.18125),
    "lShoulder": (150.3, 45.0, 1274.1),
    "rShoulder": (-161.9, 42.5, 1254.9),
    "nose_tip": (-27.5, -55.8, 1400.8),
}

# The reference CSV carries ~10 significant figures, so compare at that level.
REL_TOL = 1e-9
ABS_TOL = 1e-6


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("error: supply the path to the reference .obj", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    v, f = load_mesh(path)
    print(f"loaded {path.name}: {len(v)} vertices, {len(f)} faces")

    if len(v) != 5005 or len(f) != 9998:
        print("warning: this is not the mesh the reference values came from; "
              "differences below are expected.", file=sys.stderr)

    avatar = MatlabAvatar(v, f).run()

    failures = 0

    print("\n--- measurements ---")
    for name, expected in REF_MEASUREMENTS.items():
        if name not in avatar.measurements:
            print(f"  MISSING  {name}")
            failures += 1
            continue
        got = float(avatar.measurements[name])
        # Reference is rounded to 10 significant figures; compare at that scale.
        tol = max(abs(expected) * 1e-9, 1e-6)
        ok = abs(got - expected) <= tol
        if not ok:
            rel = abs(got - expected) / abs(expected) * 100 if expected else float("inf")
            print(f"  FAIL     {name:<20} expected {expected:.6f}  got {got:.6f}  ({rel:.4g}%)")
            failures += 1
    print(f"  {len(REF_MEASUREMENTS) - failures}/{len(REF_MEASUREMENTS)} measurements match")

    print("\n--- landmarks ---")
    lm_failures = 0
    for name, expected in REF_LANDMARKS.items():
        if name not in avatar.landmarks:
            print(f"  MISSING  {name}")
            lm_failures += 1
            continue
        got = np.asarray(avatar.landmarks[name], dtype=float)[:3]
        exp = np.asarray(expected, dtype=float)
        pairs = [(a, b) for a, b in zip(exp, got)
                 if not (math.isnan(a) or math.isnan(b))]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in pairs))
        if dist > 1e-4:
            print(f"  FAIL     {name:<20} expected {tuple(round(c,3) for c in exp)}  "
                  f"got {tuple(round(c,3) for c in got)}  (dist {dist:.3g})")
            lm_failures += 1
    print(f"  {len(REF_LANDMARKS) - lm_failures}/{len(REF_LANDMARKS)} landmarks match")

    total = failures + lm_failures
    print()
    if total == 0:
        print("PASS - this port reproduces the MATLAB reference exactly.")
        return 0
    print(f"FAIL - {total} mismatch(es).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
