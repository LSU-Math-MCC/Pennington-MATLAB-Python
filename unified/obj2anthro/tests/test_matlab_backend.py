"""The MATLAB backend must hand Avatar.m the mesh as-is and fail honestly.

The two tests that used to live here covered ``_matlab_input_path``, a helper
that fan-triangulated quads into a temp copy before MATLAB saw the file. It was
removed: Avatar.m has its own OBJ reader, and rewriting the input meant the
reference was measuring a mesh that differed from the one on disk. The tests
below pin the behaviour that replaced it.
"""

from pathlib import Path

import pytest

from unified.obj2anthro.backend_registry import BACKENDS, MatlabBackend
from unified.obj2anthro.pipeline import AUTO_BACKENDS

ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "backends" / "matlab" / "py2mat_avatar_measure.m"
)


def test_registered_under_both_names():
    assert isinstance(BACKENDS["matlab"], MatlabBackend)
    assert isinstance(BACKENDS["matlab-full"], MatlabBackend)
    assert BACKENDS["matlab"].steps == (3,)
    assert BACKENDS["matlab-full"].steps == (1, 2, 3)
    assert "matlab" in AUTO_BACKENDS


def test_backend_does_not_rewrite_the_input_mesh():
    """Avatar.m must read the path it was given, not a normalised copy."""
    assert not hasattr(MatlabBackend, "_matlab_input_path")
    source = Path(BACKENDS["matlab"].run.__code__.co_filename).read_text(encoding="utf-8")
    assert "matlab_input = obj_file" in source


def test_adapter_calls_avatar_once_with_vol_sa_on():
    text = ADAPTER.read_text(encoding="utf-8")
    calls = [line for line in text.splitlines() if "avatar = Avatar(" in line]
    assert len(calls) == 1, f"expected a single Avatar call, found {calls}"
    assert "'Vol_SA', 'on'" in calls[0]


def test_adapter_has_no_reduced_configuration_retry():
    """A mesh Avatar.m cannot process must surface as a failure, not a partial row."""
    text = ADAPTER.read_text(encoding="utf-8")
    assert "'Vol_SA', 0" not in text
    assert "volumeAreaAvailable" not in text
    assert "meshTotals" not in text


def test_missing_engine_reports_how_to_install_it():
    backend = MatlabBackend()
    try:
        backend._import_engine()
    except RuntimeError as exc:
        assert "3.9-3.11" in str(exc)
    except Exception:  # pragma: no cover - engine present in this interpreter
        pytest.skip("MATLAB Engine is importable here")
