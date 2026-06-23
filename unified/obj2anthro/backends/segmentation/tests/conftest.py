import os
import sys


def pytest_sessionstart(session):
    """Ensure the tests directory (project/tests) is on sys.path so
    test helpers like `from utils import ...` can be imported during
    collection.

    This mirrors the previous pattern of test files inserting tests/
    into sys.path manually.
    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
