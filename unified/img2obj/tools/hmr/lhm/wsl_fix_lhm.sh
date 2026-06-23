#!/usr/bin/env bash
# Post-build fixups for the LHM env: install chumpy (whose ancient setup.py breaks
# under build isolation) and re-install the rest of requirements.txt that pip skipped
# after chumpy failed. Also patch chumpy for modern numpy.
set -u
LOG() { echo "===FIXSTAGE=== $* ($(date +%H:%M:%S))"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lhm
cd "$HOME/LHM" || exit 1

LOG "ensure build tooling"
pip install --quiet "setuptools<70" wheel pip Cython numpy

LOG "install chumpy (no build isolation)"
pip install --no-build-isolation --no-deps chumpy || pip install --no-build-isolation chumpy

# patch chumpy for numpy>=1.24 (removed np.bool/int/float/object/unicode/str aliases)
CH=$(python -c "import chumpy, os; print(os.path.dirname(chumpy.__file__))" 2>/dev/null)
if [ -n "$CH" ] && [ -f "$CH/__init__.py" ]; then
  LOG "patching chumpy __init__ for modern numpy ($CH)"
  sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/' "$CH/__init__.py"
fi

LOG "verify chumpy import"
python -c "import chumpy; print('chumpy OK', chumpy.__version__)" || echo "chumpy still broken"

LOG "re-install remaining requirements"
pip install -r requirements.txt
echo "===FIX_REQS_EXIT=== $?"

# CUDA-extension source builds need --no-build-isolation so setup.py sees torch.
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6"          # RTX 3080 Ti (Ampere)
export MAX_JOBS=4

LOG "build pytorch3d (slow)"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
echo "===PYTORCH3D_EXIT=== $?"

LOG "build sam2"
pip install --no-build-isolation "git+https://github.com/hitsz-zuoqi/sam2/"
echo "===SAM2_EXIT=== $?"

LOG "build diff-gaussian-rasterization"
pip install --no-build-isolation "git+https://github.com/ashawkey/diff-gaussian-rasterization/"
echo "===DIFFGAUSS_EXIT=== $?"

LOG "build simple-knn"
pip install --no-build-isolation "git+https://github.com/camenduru/simple-knn/"
echo "===SIMPLEKNN_EXIT=== $?"

LOG "import smoke test"
python - <<'PY'
mods = ["torch","chumpy","smplx","einops","roma","accelerate"]
for m in mods:
    try:
        __import__(m); print("OK", m)
    except Exception as e:
        print("FAIL", m, repr(e)[:120])
for m in ["pytorch3d","diff_gaussian_rasterization","simple_knn"]:
    try:
        __import__(m); print("OK", m)
    except Exception as e:
        print("FAIL", m, repr(e)[:120])
PY
LOG "FIX DONE"
