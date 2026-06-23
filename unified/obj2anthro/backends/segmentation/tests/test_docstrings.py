import os
import sys
import pkgutil
import importlib
import doctest

# Add tests/ to sys.path so we can import the shared test_utils helper
# tests_dir = os.path.abspath(os.path.dirname(__file__))
# if tests_dir not in sys.path:
#     sys.path.insert(0, tests_dir)
from tests.helpers import add_project_root_to_sys_path

# Ensure project root on sys.path so `src` package imports work
add_project_root_to_sys_path(levels_up=1)


def iter_src_modules(package_name='src'):
    """Yield (module_name, module) for all importable modules under `src` package."""
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        # Nothing to test
        return

    package_path = package.__path__
    for finder, name, ispkg in pkgutil.walk_packages(package_path, package_name + '.'):
        try:
            mod = importlib.import_module(name)
        except Exception:
            # Import errors should fail the test
            raise
        yield name, mod


def test_doctests_for_src_modules():
    """Run doctest.testmod on each module under `src`.

    This test will fail if any example in a docstring does not execute
    or if an import error occurs while loading a module.
    """
    failures = 0
    tests = 0
    for name, mod in iter_src_modules():
        # ignore large or third-party style helper modules if needed later
        r = doctest.testmod(mod, verbose=False)
        tests += r.attempted
        failures += r.failed
    assert failures == 0, f"{failures} doctest examples failed across {tests} checks"
