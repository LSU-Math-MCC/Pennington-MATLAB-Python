# 3D Anthropometric Analysis Project

A Python toolkit for analyzing 3D human body scans and extracting anthropometric measurements. This project processes `.obj` and `.ply` mesh files to compute body measurements for research applications.

---

## Project Overview

### What This Project Is

This is a **student research project** that:
- Processes 3D human body scan meshes (`.obj`, `.ply` files)
- Automatically identifies anatomical landmarks (shoulders, hips, crotch, etc.)
- Extracts body measurements (arm length, leg length, circumferences)
- Provides a Python implementation of anthropometric analysis algorithms

The project was developed as part of **MATH 4020** at Louisiana State University to understand and replicate anthropometric measurement algorithms originally implemented in MATLAB.

### Purpose and Limitations

**Purpose**: To translate the MATLAB codebase used by Pennington Research Institute to Python for improved scalability, maintainability, cost efficiency, and code quality.

**Current Limitations**:
- Unusual body types may lead to incorrect results (the code needs more testing with diverse data)
- No validation against physical measurements (units may be off or inaccurate as a result)

---

## Codebase Layout

```
pennington-math-4020/
├── src/                          # Main Python source code
│   ├── main.py                   # Demo script showing usage
│   ├── body/                     # Body scan processing
│   │   ├── body.py               # Body class - the core class for body analysis
│   │   └── anatomical_regions/   # Body part implementations
│   │       ├── anatomical_region.py  # Abstract base class
│   │       ├── arms/arm.py       # Arm segmentation and measurements
│   │       ├── legs/leg.py       # Leg segmentation and measurements
│   │       ├── trunk/trunk.py    # Trunk segmentation and measurements
│   │       └── head/head.py      # Head segmentation
│   ├── mesh/                     # Mesh utilities
│   │   ├── mesh.py               # Mesh cleaning and orientation
│   │   └── boolean_ops.py        # Mesh boolean operations (difference, intersection)
│   └── utils/                    # Utility functions
│       └── convexity_search.py   # Convexity-based landmark finding
│
├── tests/                        # Test suite
│   ├── test_docstrings.py        # Validates docstring examples
│   ├── mesh_cleaning/            # Tests for mesh operations
│   └── mesh_orientation/         # Tests for mesh orientation
│
├── model_files/                  # Sample 3D meshes
│   ├── cow.ply                   # Test mesh for development
│   └── man.obj                   # Human body scan sample
│
├── matlab/                       # Original MATLAB implementation
│   ├── original_code/            # 7000+ line MATLAB Avatar class
│   └── not_original_code/        # Test scripts for MATLAB code
│
├── dev_only/                     # Development utilities (ignore for now)
│   └── dependency_graph/         # Visualization tools
│
└── requirements.txt              # Python dependencies

```

### Component Responsibilities

**`src/body/body.py`** - The `Body` class is your main entry point. It:
- Loads a mesh file
- Cleans and orients the mesh
- Creates instances of each anatomical region (head, trunk, arms, legs)
- Exposes landmarks and measurements through dictionaries

**`src/body/anatomical_regions/`** - Each body part is a class:
- Segments its portion from the full body mesh
- Locates anatomical landmarks
- Computes measurements (lengths, circumferences)
- Each class caches results to avoid recomputation

**`src/mesh/mesh.py`** - The `Mesh` base class handles:
- Mesh cleaning (removing degenerate faces, filling holes)
- Mesh normalization (scaling for numerical stability)
- Orientation detection (ensuring Z-axis is vertical)

**`src/mesh/boolean_ops.py`** - Boolean operations:
- `mesh_difference(A, B)` - removes vertices of B from A
- `mesh_intersection(A, B)` - keeps only shared vertices
- Used to separate body parts (e.g., remove arms from trunk)

**`src/utils/convexity_search.py`** - Landmark finding:
- Finds anatomical landmarks by searching for convex/concave regions
- Used to locate armpits, crotch, etc.

---

## How to Run the Code

### Prerequisites

- Python 3.12 (recommended for MATLAB compatibility, though 3.10+ works for pure Python)
- Basic understanding of Python virtual environments
- No MATLAB required for Python code

### Setup with Python venv

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BearGotGit/pennington-math-4020.git
   cd pennington-math-4020
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Upgrade pip and install dependencies**:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Demo

Run the `src/main.py` file from the project root to see a demo:

```bash
python -m src.main
```

This will:
- Load `model_files/man.obj`
- Create a `Body` instance
- Extract landmarks and measurements
- Display a 3D visualization with colored body parts

**Expected output**: A 3D viewer window showing the body mesh with landmarks marked as colored spheres.

### Basic Usage Example

```python
from src.body import Body

# Load a body scan
body = Body("model_files/man.obj")

# Access measurements
left_arm_length = body.measurements["left arm"]["left arm length"]
chest_circ = body.measurements["trunk"]["chest circumference"]

print(f"Left arm length: {left_arm_length:.2f} units")
print(f"Chest circumference: {chest_circ:.2f} units")

# Access landmarks
crotch = body.landmarks["trunk"]["crotch"]
left_shoulder = body.landmarks["left arm"]["shoulder"]

print(f"Crotch position: {crotch}")
print(f"Left shoulder position: {left_shoulder}")

# Access body part meshes
trunk_mesh = body.subregion_meshes["trunk"]
print(f"Trunk volume: {trunk_mesh.volume:.2f}")
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run just docstring tests
python -m pytest tests/test_docstrings.py -v

# Run with coverage report
python -m pytest --cov=src --cov-report=term tests/
```

---

## Learning Path

This project is designed to help you learn:

### 1. Python Concepts
- **Object-oriented design**: Abstract base classes, inheritance
- **Properties and caching**: Using `@property` and `@cache` decorators
- **Type hints**: `Literal` types, generics

### 2. 3D Mesh Processing
- **trimesh library**: Core concepts for working with triangle meshes
- **Mesh operations**: Slicing, boolean operations, nearest neighbor queries
- **Spatial data structures**: KD-trees for efficient point queries

### 3. Computational Geometry
- **Landmark detection**: Convexity analysis, ray casting
- **Mesh segmentation**: Plane slicing, connected components
- **Geometric measurements**: Circumferences via cross-sections

### Key External Resources

**Python Built-ins**:
- [`@staticmethod`](https://docs.python.org/3/library/functions.html#staticmethod) - Why some methods don't need `self`
- [`@cache` decorator](https://docs.python.org/3/library/functools.html#functools.cache) - Automatic memoization
- [`@property`](https://docs.python.org/3/library/functions.html#property) - Computed attributes
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html) - Interface enforcement

**trimesh Documentation**:
- [trimesh.Trimesh](https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh) - Core mesh class
- [Mesh properties](https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh.is_watertight) - `is_watertight`, `is_volume`, etc.
- [Slicing operations](https://trimsh.org/trimesh.base.html#trimesh.base.Trimesh.slice_plane) - `slice_plane`, `section`
- [Ray queries](https://trimsh.org/trimesh.ray.html) - Ray casting for intersections
- [Geometric operations](https://trimsh.org/trimesh.path.html) - Path lengths, areas

**Scientific Computing**:
- [NumPy basics](https://numpy.org/doc/stable/user/absolute_beginners.html) - Arrays, vectorization
- [SciPy spatial](https://docs.scipy.org/doc/scipy/reference/spatial.html) - KD-trees, convex hulls

---

## Skeletons & Design Debt

This section documents architectural shortcuts and known issues that future maintainers should understand.

### Why Watertight Meshes Matter

A **watertight mesh** has no holes or gaps - every edge is shared by exactly two faces. This is critical because:
- Volume calculations are only valid for watertight meshes
- Boolean operations assume closed surfaces
- Slicing operations may fail on meshes with holes

**Our approach**: We use `trimesh.fill_holes()` during mesh cleaning, but this is a simple heuristic that may not work for large holes. The original MATLAB code had a sophisticated hole-filling algorithm (`holeFilling_Fit3D`) that we have not yet replicated.

**Why we didn't use trimesh's built-in boolean operations**: 
- `trimesh.boolean.difference()` exists but uses external libraries (manifold3d)
- We implemented simpler vertex-based operations for learning purposes
- Our implementation is less robust but easier to understand and debug
- For production use, consider switching to `trimesh.boolean` methods

### Custom Mesh Utilities in `mesh/`

**`mesh/boolean_ops.py`** - Our boolean operations:
- Work by comparing vertices with KD-tree queries
- Are **not true CSG operations** - they don't compute new geometry
- Assume meshes are already roughly separated (not deeply interpenetrating)
- May fail if meshes share many vertices along boundaries

**When to use them**: For simple subtraction of clearly separated regions (like removing an arm from a body).

**When NOT to use them**: For complex intersecting shapes, use trimesh's built-in boolean operations instead.

**`mesh/mesh.py`** - The `Mesh` class:
- Normalizes vertex coordinates (mean=0, std=1) for numerical stability
- Auto-detects mesh orientation but assumes upright standing pose
- Uses heuristics that may fail on unusual poses (sitting, bent over, etc.)

### Limitations of Landmark/Body-Part Finding Algorithms

**Current approach**:
1. Find major axis → assume this is height (Z-axis)
2. Find convex regions → assume these are armpits, crotch
3. Slice at landmarks → separate body parts

**Known issues**:
- **Assumes standard pose**: Arms down, legs apart, standing upright
- **Heuristic-based**: No machine learning or training data
- **Brittle**: Small mesh issues can cascade to measurement failures
- **No validation**: We don't verify that found landmarks make anatomical sense

**Example failures**:
- Arms raised: armpit detection fails
- Legs together: crotch detection fails  
- Mesh holes near landmarks: convexity search fails

**Future improvements** would require:
- Training data with labeled landmarks
- Machine learning for robust landmark detection
- Better handling of non-standard poses

### Why Reading Pseudocode Is Essential

Many functions include **pseudocode** in their docstrings. This is intentional because:

1. **Algorithms are complex**: Anthropometric measurements involve multi-step geometric operations
2. **Code is dense**: NumPy operations compress many steps into single lines
3. **Debugging is hard**: You need to understand the intended logic to fix bugs

**When debugging**, follow this process:
1. Read the pseudocode to understand the high-level algorithm
2. Add print statements at each pseudocode step
3. Visualize intermediate results (use `trimesh.Scene()` to show meshes)
4. Compare with expected behavior

**Example**: If `_locate_armpits` fails, read its pseudocode, then visualize:
- The horizontal slices being analyzed
- The convexity scores at each height
- The final landmark positions

Without understanding the algorithm, you're just guessing.

### No User Interface for File Submission

**The problem**: Currently, there is no user interface for end users to submit mesh files. Users must interact directly with Python scripts, which is not practical for non-technical users like researchers or clinicians.

**Recommendation**: Future developers should create a **web application** to provide a user-friendly interface for file uploads and measurement results.

**Suggested approach**:
- Use **Flask** or **Flask-RESTful** (FlaskAPI) to create a simple web backend
- Flask is lightweight and integrates easily with Python projects
- Example minimal setup (adjust imports based on your project structure):

```python
from flask import Flask, request, jsonify
from body import Body  # Adjust import path based on Flask app location
import tempfile
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'.obj', '.ply'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@app.route('/upload', methods=['POST'])
def upload_mesh():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Invalid file type. Use .obj or .ply"}), 400
    
    # Save to temporary file and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        try:
            body = Body(tmp.name)
            measurements = body.measurements
            landmarks = body.landmarks
            return jsonify({
                "measurements": measurements,
                "landmarks": landmarks  # Serialize as needed for your data types
            })
        finally:
            os.unlink(tmp.name)

if __name__ == '__main__':
    app.run(debug=True)
```

**Note**: This is a simplified example. For production use, add proper input validation, authentication, error handling, and file size limits. See [Flask Security Considerations](https://flask.palletsprojects.com/en/latest/security/).

**Why Flask**: 
- Simple to learn and implement
- Minimal boilerplate code
- Large ecosystem of extensions (Flask-CORS, Flask-RESTful)
- Well-documented with extensive tutorials

**Frontend options**:
- Simple HTML form for file upload
- React/Vue.js for a more interactive experience
- Streamlit for rapid prototyping (Python-only solution)

**Resources**:
- [Flask Quickstart](https://flask.palletsprojects.com/en/latest/quickstart/)
- [Flask-RESTful](https://flask-restful.readthedocs.io/)
- [Streamlit](https://streamlit.io/) - Alternative for Python-only web apps

---

## Recommended Onboarding Timeline

This is a suggested path for new students joining the project:

### Week 1-2: Environment Setup and Exploration
1. **Set up Python environment** (see "How to Run the Code")
2. **Run the demo** (`python -m src.main`)
3. **Read this README** completely
4. **Read external docs**: Python decorators, trimesh basics

### Week 3-4: Code Reading and Understanding
1. **Read `src/body/body.py`** - Understand the main workflow
2. **Read `src/mesh/mesh.py`** - Understand cleaning and orientation
3. **Pick ONE anatomical region** (suggest starting with `head.py` - it's simplest)
   - Read the class docstring
   - Read function docstrings with pseudocode
   - Trace execution with print statements
4. **Run tests** (`pytest tests/`) and understand what they validate

### Week 5-6: Data Acquisition (CRITICAL)
**This is the highest priority for future work.**

**The problem**: We spent this semester without proper test data, which severely limited progress.

**What you need**:
1. **Real body scan files** from Styku, Fit3D, or similar scanners
2. **Ground truth measurements** - manual measurements of the same subjects
3. **Variety**: Different body types, heights, ages, poses
4. **Quality**: High-quality watertight meshes

**How to get data**:
- Contact LSU Pennington Biomedical Research Center
- Contact scanner manufacturers for sample datasets
- Check public anthropometric databases (CAESAR, etc.)
- Create synthetic data with MakeHuman or similar tools (less ideal)

**Why this matters**: Without real data, you cannot validate measurements or improve algorithms.

### Week 7-8: MATLAB Comparison (Major Challenge)

**The MATLAB problem**:
- Original code is in `matlab/original_code/Avatar.m` (~7000 lines, monolithic)
- We could not get it running this semester
- Original developers gave advice that didn't work
- Email communication was slow

**What to try**:
1. **Brute-force MATLAB version testing**: Try R2016a, R2018b, R2020a, R2023a
   - The code may work in an older version we didn't test
2. **Contact original developers again**: Be persistent, schedule a video call
3. **Simplify the problem**: Extract just one measurement (e.g., arm length) into a standalone script
4. **Use GNU Octave**: Try `matlab/original_code/Avatar_Octave.m` (compatibility version)
5. **Document everything**: When you hit errors, save full error logs and environment info

**Why this matters**: Comparing Python vs MATLAB results is the only way to validate correctness.

**Time estimate**: This could take 2-4 weeks of trial and error.

### Week 9+: Contributing Improvements

Once you understand the codebase:
1. **Fix a known bug** (see "Known Bugs/Issues" section)
2. **Add a new measurement** (e.g., neck circumference)
3. **Improve a landmark algorithm** (e.g., make armpit detection more robust)
4. **Add validation** (compare results with ground truth if you have data)

---

## Known Bugs / Issues

### Data Issues

**No real body scan data**:
- Only have `model_files/man.obj` which has quality issues
- Cannot validate measurements against ground truth
- Cannot test algorithm robustness across body types

**`man.obj` mesh quality**:
- Has repeated faces and degenerate triangles
- Mesh cleaning removes >10% of faces
- Too few vertices remain after cleaning for some operations
- See `matlab/not_original_code/README.md` for detailed error logs

**Action needed**: Acquire high-quality body scan dataset (Week 5-6 priority).

### Body Indices Issues

**Armpit detection**:
- Uses `convexity_search()` which is sensitive to mesh quality
- May fail if arms are raised or mesh has holes near armpits
- File: `src/body/anatomical_regions/trunk/trunk.py:_locate_armpits()`

**Crotch detection**:
- Assumes legs are apart (standard pose)
- Fails on `man.obj` when mesh repair is skipped (too few vertices in region)
- File: `src/body/anatomical_regions/trunk/trunk.py:_locate_crotch()`

**Shoulder landmarks**:
- Found by taking highest point of arm mesh (simple heuristic)
- Not anatomically precise - should be at glenohumeral joint
- File: `src/body/anatomical_regions/arms/arm.py:_locate_shoulder()`

**General issue**: All landmark detection is heuristic-based, not learned from data.

### Mesh Issues

**Boolean operations are not true CSG**:
- `mesh_difference()` just removes shared vertices
- Doesn't compute new intersection edges
- May leave gaps or create non-watertight meshes
- File: `src/mesh/boolean_ops.py`

**Mesh orientation detection assumptions**:
- Assumes major axis is vertical (fails for lying-down poses)
- Assumes left-right symmetry (fails for asymmetric meshes)
- File: `src/mesh/mesh.py:orient_mesh()`

**Normalization side effects**:
- Vertices are scaled to mean=0, std=1
- Must denormalize before interpreting measurements
- Easy to forget and get incorrect values
- File: `src/mesh/mesh.py:normalize()`, `denormalize()`

### Development Life / Refactoring Opportunities

**Circular dependencies**:
- `Arm` needs `Trunk` (for armpits), `Trunk` needs `Arm` (for shoulders)
- Currently handled with runtime imports inside methods
- This pattern works well and doesn't cause issues if the algorithms are logically sound

**Caching**:
- `@cache` decorator used extensively to avoid recomputation
- Debugging is aided by visualization tools (e.g., `trimesh.Scene()`)
- Visualization is the greatest debugging aid for 3D mesh processing

**Inconsistent error handling**:
- Some functions print warnings, others raise exceptions, others fail silently
- No consistent logging strategy
- Hard to diagnose failures in production
- Consider: Add proper logging framework (e.g., Python `logging` module)

**No configuration system**:
- Magic numbers scattered throughout (tolerances, thresholds, slice widths)
- Hard to tune algorithms without editing code
- Consider: Add `config.py` or YAML configuration file

**Incomplete test coverage**:
- Tests exist for mesh orientation but not for landmark detection
- Docstring examples fail (see `pytest tests/test_docstrings.py`)
- Need integration tests with real body scan data
- Consider: Add pytest fixtures for common test meshes

**Documentation gaps**:
- Function docstrings exist but some lack examples
- No end-to-end workflow documentation
- No troubleshooting guide
- This README addresses some gaps but more detailed guides needed

**MATLAB code not integrated**:
- Python and MATLAB implementations are completely separate
- No automated comparison tests
- Unknown if results match
- Highest priority for validation

---

## Testing

See `CONTRIBUTING.md` for detailed testing instructions.

**Quick start**:
```bash
python -m pytest tests/                     # Run all tests
python -m pytest tests/test_docstrings.py  # Run docstring tests
python -m pytest --cov=src tests/          # Run with coverage
```

**Note**: Docstring examples are validated by `tests/test_docstrings.py` using Python's `doctest` module. All examples in function docstrings should be runnable.

---

## Contributing

See `CONTRIBUTING.md` for:
- Virtual environment setup
- Testing procedures  
- MATLAB Engine integration notes
- Contribution workflow

**In short**:
1. Create a feature branch
2. Make changes with tests
3. Run `pytest tests/`
4. Open a pull request

---

## License

[Add appropriate license]

---

## Contact

**Louisiana State University - MATH 4020**  
*3D Anthropometric Analysis Research Project*

For questions about this code, contact the course instructor or the development team:

**Development Team (Fall 2025)**:
- Berend Grandt (beraulndt+pennington@gmail.com)
- Kim Nguyen
- Grishma Srestha
- Shelby Primeaux
- Matthew Lemoine

---

## Acknowledgments

- **Original MATLAB Implementation**: Previous LSU research team
- **Current Development**: MATH 4020 Project Team (Fall 2025)
- **Advisors**: Dr. Nadejda Drenska and Dr. Peter Wolenski

This project is a learning exercise in anthropometric analysis and 3D mesh processing.
