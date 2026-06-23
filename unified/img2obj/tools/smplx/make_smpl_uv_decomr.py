"""Build BLADE's required body_models/smpl/smpl_uv_decomr.npz from a standard SMPL UV .obj.

BLADE's UVRenderer wants an npz with keys {verts_uv (NUM_VT,2), faces_uv (F,3 into verts_uv),
vt2v (NUM_VT,) UV-vertex -> SMPL-vertex}. That's the mmhuman3d smpl_uv.npz layout; a SMPL UV obj
(e.g. smpl_uv_20200910/smpl_uv.obj) carries all of it: `vt u v` lines give verts_uv, and
`f v/vt ...` corners pair each vt index with its v index (-> faces_uv + vt2v).

Usage: python tools/smplx/make_smpl_uv_decomr.py /path/to/smpl_uv.obj /out/smpl_uv_decomr.npz
"""
import sys
import numpy as np


def main(obj_path, out_path):
    vt, faces_uv, faces_v, vt2v = [], [], [], {}
    with open(obj_path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "vt":
                vt.append((float(p[1]), float(p[2])))
            elif p[0] == "f":
                uv_corner, v_corner = [], []
                for c in p[1:4]:
                    bits = c.split("/")
                    v_idx = int(bits[0]) - 1
                    t_idx = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                    uv_corner.append(t_idx)
                    v_corner.append(v_idx)
                    vt2v[t_idx] = v_idx
                faces_uv.append(uv_corner)
                faces_v.append(v_corner)
    verts_uv = np.asarray(vt, np.float32)
    faces_uv = np.asarray(faces_uv, np.int64)
    faces = np.asarray(faces_v, np.int64)                 # SMPL triangle topology (v indices)
    vt2v_arr = np.asarray([vt2v[i] for i in range(len(verts_uv))], np.int64)
    np.savez(out_path, verts_uv=verts_uv, faces_uv=faces_uv, faces=faces, vt2v=vt2v_arr)
    print(f"wrote {out_path}: verts_uv {verts_uv.shape}, faces_uv {faces_uv.shape}, "
          f"faces {faces.shape}, vt2v {vt2v_arr.shape} (max v {vt2v_arr.max()})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
