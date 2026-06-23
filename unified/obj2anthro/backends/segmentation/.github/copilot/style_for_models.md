# Style guidance for models and contributors

This file documents the conventions we used when porting MATLAB helpers to Python and what downstream models or contributors should expect.

## Project structure

The `src/` directory is organized by domain:

- **`src/body/`** - Core body representation and anatomical region classes
  - `body.py` - Main `Body` class that orchestrates all body parts
  - **`src/body/anatomical_regions/`** - Individual body part implementations
    - `anatomical_region.py` - Base class for all body parts
    - `head/head.py` - Head mesh extraction and measurements
    - `trunk/trunk.py` - Trunk mesh extraction, landmarks (crotch, armpits, hips, collar), and measurements
    - `arms/arm.py` - Arm mesh extraction, landmarks (shoulder, wrist), and measurements (per side: left/right)
    - `legs/leg.py` - Leg mesh extraction, landmarks (foot, ankle), and measurements (per side: left/right)

- **`src/mesh/`** - Mesh processing utilities
  - `mesh.py` - `Mesh` class for cleaning, normalizing, orienting, and aligning meshes
  - `boolean_ops.py` - Boolean operations on meshes (`mesh_difference`, `mesh_intersection`)

- **`src/utils/`** - General utilities
  - `convexity_search.py` - Convexity-based landmark detection

- **`src/main.py`** - Demo/example script showing how to use the library

## Architecture patterns

### Core pattern: Private static methods with caching

Body part calculations follow a consistent pattern:

- **Private static methods** (e.g., `_locate_shoulder`, `_get_submesh`, `_measure_arm_length`) are decorated with `@staticmethod` and `@cache`
- These methods accept a `trimesh.Trimesh` object as the first parameter (and sometimes additional parameters like `side: LEFT_OR_RIGHT`)
- They return the computed result (e.g., a numpy array for landmarks, a float for measurements, a trimesh object for submeshes)
- The `@cache` decorator ensures expensive computations happen only once - results are memoized based on input parameters

**Pattern to follow:**

```python
@staticmethod
@cache
def _locate_landmark(mesh: trimesh.Trimesh, side: LEFT_OR_RIGHT = "left") -> np.ndarray:
    """Compute landmark location."""
    # ... computation logic ...
    return np.array([x, y, z])
```

### Public interface: Instance methods calling static methods

- **Property-based access**: Body parts expose `landmarks`, `measurements`, and `mesh` as `@property` decorated methods
- These properties are normal instance methods (using `self`) that call the private static methods
- Instance methods pass `self.body_mesh` (or `self.mesh`) to the static methods, along with any needed parameters like `self.side`

**Pattern to follow:**

```python
@property
def landmarks(self) -> dict:
    """Public interface for landmarks."""
    return {
        "shoulder": self._locate_shoulder(self.body_mesh, self.side),
        "wrist": self._locate_wrist(self.body_mesh, self.side),
    }
```

### Additional patterns

- **Lazy evaluation**: Landmarks and measurements are computed on-demand when accessed via properties, not at instantiation time.

- **Cross-part dependencies**: Body parts can import and use methods from other parts (e.g., `Arm` uses `Trunk._locate_armpits`, `Leg` uses `Trunk._locate_hips`). The caching system ensures each calculation happens only once regardless of how many times it's referenced.

- **Separation of concerns**: Keep computation logic in private static methods. This makes them testable, cacheable, and reusable across different body part instances.

## Naming conventions

- Use PEP8 / snake_case for function and variable names (e.g. `rotate_person`, `coords_dim1`).

- Avoid camelCase and MATLAB-style names (e.g., `getVOnLine` -> `get_v_on_line`).

- Keep parameter names descriptive but concise. Prefer `x, y, z` when unambiguous, otherwise `coords_dim1`/`coords_dim2` is acceptable in helper utilities.

- Private/internal methods use leading underscore (e.g., `_get_submesh`, `_locate_crotch`).

- Body part sides use string literals: `"left"` or `"right"` (type hinted as `LEFT_OR_RIGHT`).

## Docstrings and examples

- Use NumPy-style or Google-style docstrings with clear `Parameters`, `Returns`, `Raises`, and `Examples` sections.

- Write small, runnable examples in the docstring for each public function. These examples are used in automated doctest checks.

- Keep descriptions compact; the code should be self-explanatory with the docstring acting as a contract.

- Document algorithm steps in multi-step functions (see `Trunk._locate_hips` for an example).

## Testing expectations

- Add unit tests under `tests/` using pytest.

- Include a `tests/test_docstrings.py` that runs `doctest` across public modules to ensure examples in docstrings execute correctly.

- Prefer unit tests that exercise the boundary cases (scalars vs arrays, shape mismatches, empty inputs).

## Implementation notes

- Prefer `numpy` operations for vectorized math.

- Use `trimesh` for mesh operations (slicing, splitting, boolean operations).

- Use `scipy.spatial.cKDTree` for efficient nearest-neighbor queries.

- Keep I/O and heavy visualization separate from computation to make functions easy to unit test.

- Return Python scalars for scalar inputs when the function is expected to return a scalar in normal usage. This matches MATLAB's single-value behavior in the porting context.

- For mesh extraction: use `mesh.slice_plane()` for cutting, `mesh.split()` for separating connected components, and `mesh_difference()` for boolean subtraction.

## Packaging & imports

- For local testing, either insert the project root to `sys.path` in tests, or add a minimal `pyproject.toml` and `pip install -e .` in the venv to make `src` importable.

- To avoid pytest collection collisions, do not keep `test_*.py` files inside `src/`.

- Imports within `src/body/anatomical_regions/` use relative imports (e.g., `from ..trunk import Trunk`).

- Circular import handling: Import classes inside methods when needed to avoid module-level circular dependencies.

If you want a shorter, machine-friendly JSON description of these rules for consumption by a model, ask and I will generate it.
