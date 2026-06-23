import os
import sys


def add_project_root_to_sys_path(levels_up=1):
    """Ensure project root is on sys.path for tests.

    levels_up: how many directories to go up from the tests directory to reach
    the project root. Default 1 for tests/; use 2 for tests/mesh_orientation/.
    """
    this_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(this_dir, *(['..'] * levels_up)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
