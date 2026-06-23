CONTRIBUTING
============

This file summarizes how to set up a development environment, run tests, and contribute to the project.

Python version
--------------

This project targets Python 3.12 for maximum compatibility with the MATLAB Engine for Python. You can also run tests on newer Python versions for most pure-Python checks, but MATLAB engine integration is only supported on Python 3.12.

Creating a virtual environment
------------------------------

Create a virtual environment with your system python (use any name, e.g. `.venv`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Note: Do not commit your virtual environment directory. Add common names to `.gitignore` (already added).

Running tests
-------------

After activating your venv and installing requirements, run:

```bash
pytest -q tests
```

To generate a coverage report (requires pytest-cov):

```bash
pytest --cov=src --cov-report=term --cov-report=xml tests
```

MATLAB Engine notes
-------------------

MATLAB Engine for Python supports up to Python 3.12. If you want to run MATLAB comparison tests, use Python 3.12 and ensure MATLAB is installed and `matlab.engine` is available in your environment.

Some tests will be skipped when MATLAB helpers like `plyread` are missing. See `.github/INCOMPLETE_TESTS.md` for known gaps.

Contributions
-------------

Please create feature branches, run tests locally, and open a PR. CI will run tests on push/PR (see `.github/workflows/ci.yml`).

Thank you for contributing!
