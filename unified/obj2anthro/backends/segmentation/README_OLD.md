# MATH4997 - 3D Anthropometric Analysis Project

A comprehensive toolkit for analyzing 3D human body scans and extracting anthropometric measurements. This project processes `.obj` and `.ply` mesh files from various 3D body scanners to compute standardized body measurements for research applications.

## 🎯 Project Overview

This codebase implements automated anthropometric analysis of 3D human body scans. The core `Avatar` class processes raw 3D meshes and extracts over 50+ body measurements including circumferences, volumes, surface areas, and anatomical landmarks.

### Key Features

- **Multi-Scanner Support**: Works with Styku, Fit3D, SS20, and other 3D body scanners
- **Automated Orientation**: Intelligently orients meshes to standard position regardless of scan orientation  
- **Comprehensive Measurements**: Extracts chest, waist, hip circumferences, total volume, surface area, and more
- **Batch Processing**: Process multiple subjects with standardized data export
- **Cross-Platform**: MATLAB implementation with GNU Octave compatibility

## 📁 Project Structure

```text
├── matlab/                     # Core MATLAB implementation
│   ├── original_code/         # Original Avatar implementation
│   │   ├── Avatar.m           # Main anthropometric analysis class
│   │   ├── Avatar_Octave.m    # GNU Octave compatible version
│   │   ├── build_matrice_UNiversalPaper.m  # Batch processing pipeline
│   │   ├── BuildExcelFromStruct.m # Data export to Excel
│   │   └── buildObjfrom_mat.m # Mesh export utilities
│   └── not_original_code/     # Testing and demo scripts
│       ├── run_avatar_demo.m  # Full pipeline test
│       ├── run_avatar_demo_skip_repair.m  # Skip mesh repair test
│       ├── inspect_avatar_step1.m  # Cleaning only test
│       └── run_avatar_demo_raw.m   # Raw mesh inspection
├── src/                        # Core Python implementations
│   ├── Avatar.py              # Python port of Avatar class
│   └── mesh_orientation/      # Mesh processing utilities
├── tests/                      # Python test suite
│   └── mesh_orientation/      # Unit tests for mesh utilities
├── model_files/                # Sample 3D mesh files
│   ├── cow.ply                # Test meshes for development
│   ├── man.obj                # Human body scan samples
│   └── *.ply, *.obj           # Various test models
├── dev_only/                   # Development utilities (not for production)
│   ├── dependency_graph/      # Dependency visualization tool
│   │   ├── topo_sort.py       # Excel → interactive graph visualizer
│   │   ├── dependencies.xlsx  # Excel template for dependencies
│   │   └── generated/         # Generated outputs (HTML, PNG, lib/)
│   ├── trimesh_demo/          # 3D mesh visualization examples
│   ├── tinkering/             # Experimental code
│   ├── misc/                  # Analysis scripts and notes
│   └── research/              # Academic references and papers
└── src_old/                    # Deprecated/archived code
```

## 🚀 Quick Start

### MATLAB Usage

```matlab
% First, add the original_code directory to your MATLAB path
cd matlab/
addpath('original_code')

% Load and analyze a 3D body scan
avatar = Avatar('../model_files/man.obj');

% Access measurements
total_volume = avatar.volume.total;
chest_circ = avatar.chestCircumference.value;
waist_circ = avatar.waistCircumference.value;

% View results
avatar.show();  % Display 3D mesh with measurements
```

### Python Usage

```python
# Set up environment
source .venv/bin/activate

# Run visualization examples (in dev_only/)
cd dev_only/trimesh_demo/
python cow_point_show.py      # Basic mesh + point visualization
python cow_upright.py         # Orientation correction demo
python cow_show.py           # Simple mesh viewer
```

### Batch Processing

```matlab
% Process multiple subjects (requires pre-loaded data structure)
run('matlab/build_matrice_UNiversalPaper.m');

% Export results to Excel
run('matlab/BuildExcelFromStruct.m');
```

## 🔧 Installation

### MATLAB Requirements

- MATLAB R2016a or later
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox (recommended)

### Python Requirements

```bash
pip install -r requirements.txt
# Key packages: trimesh, numpy, matplotlib
```

## 📊 Measurements Extracted

The Avatar class computes:

**Circumferences**: Chest, waist, hip, neck, bicep, forearm, thigh, calf, ankle
**Volumes**: Total body volume, segmental volumes
**Surface Areas**: Total and regional surface areas  
**Anatomical Landmarks**: Key body points for measurement standardization
**Proportions**: Body ratios and anthropometric indices

## 🔬 Research Applications

This toolkit has been used for:

- Anthropometric population studies
- Clothing size standardization research  
- Medical anthropometry analysis
- Sports science body composition studies

## 📈 Scanner Compatibility

Tested with output from:

- **Styku** body scanners
- **Fit3D** ProScanner  
- **SS20** systems
- **Human Solutions** medical scanners
- Generic `.obj` and `.ply` mesh files

## 🛠️ Development

### Core Algorithm (`Avatar.m`)

1. **Mesh Loading**: Supports `.obj` and `.ply` formats
2. **Orientation Correction**: `fixOrientation()` standardizes pose
3. **Measurement Pipeline**: Automated extraction of 50+ measurements
4. **Quality Control**: Error handling and measurement validation

### Key Functions

- `fixOrientation()`: Intelligent mesh orientation using dimensional analysis
- `meshPoly()`: Polygon mesh processing utilities
- `ellipse_fit()`: Anatomical landmark fitting
- `read_ply()`/`readObj()`: Mesh file I/O

## 📝 Citation

If you use this code in research, please cite:

```text
[Add appropriate academic citation when published]
```

## 👥 Contributors

- **Original MATLAB Implementation**: Previous LSU team
- **Current Development**: MATH4997 Project Team
- **Python Port**: In development

## 📄 License

[Add appropriate license]

---

**Louisiana State University - MATH 4020/4997**  
*3D Anthropometric Analysis Research Project*

## 📊 Dependency Graph Visualization

The `dev_only/dependency_graph/topo_sort.py` tool visualizes function dependencies from an Excel sheet as an interactive hierarchical tree.

### Quick Usage

1. **Prepare Excel file** with sheet named `TopoSort`:
   - Column `Function`: Function names (comma-separated if multiple)
   - Column `Dependencies`: Direct dependencies only (comma-separated)

2. **Generate interactive HTML** (default):
   ```bash
   cd dev_only/dependency_graph
   python3 topo_sort.py dependencies.xlsx
   ```
   Creates `generated/dependency_graph.html` with:
   - Hierarchical tree layout in topological order
   - Zoom/pan controls with mouse or navigation buttons
   - Automatic transitive reduction (removes redundant edges)
   - All assets (HTML + lib/ folder) in `generated/` directory

3. **Generate static PNG**:
   ```bash
   cd dev_only/dependency_graph
   python3 topo_sort.py dependencies.xlsx --format png
   ```
   Creates `generated/dependency_graph.png`

4. **Show topological order in labels**:
   ```bash
   cd dev_only/dependency_graph
   python3 topo_sort.py dependencies.xlsx --show-order
   ```

### Installation

```bash
pip install pandas openpyxl networkx matplotlib pyvis
```

### Features

- **Transitive Reduction**: Automatically removes redundant edges (e.g., if A→B→C and A→C exist, removes A→C)
- **Hierarchical Layout**: Nodes arranged by topological level for clear dependency flow
- **Interactive HTML**: Scrollable, zoomable graph with navigation controls
- **Static PNG**: High-resolution export for documentation
- **Clean Output**: All generated files (HTML, PNG, lib/ assets) go to `generated/` subdirectory

### Example Files

- Excel template: `dev_only/dependency_graph/dependencies.xlsx`
- Generated outputs: `dev_only/dependency_graph/generated/`
- Detailed guide: `dev_only/dependency_graph/README.md`

## 🧪 Testing & Debugging

### Python Tests

Quick steps to run the Python test-suite locally:

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run the test-suite with pytest from the repository root:

```bash
python -m pytest tests # or maybe python3 
```

3. (Optional) Generate coverage report (requires pytest-cov):

```bash
python -m pytest --cov=src --cov-report=term --cov-report=xml tests
```

### MATLAB Avatar Construction Tests

📖 **See [`matlab/not_original_code/README.md`](matlab/not_original_code/README.md) for detailed testing instructions and available demo scripts.**

The `matlab/not_original_code/` directory contains several test scripts for Avatar construction at different processing stages.

### Known Issues with `model_files/man.obj`

⚠️ The sample mesh has several issues that cause Avatar construction to fail:

**Issue #1: Mesh Repair Failure (Line 4159 in Avatar.m)**
- **Location:** `holeFilling_Fit3D` function
- **Error:** `"Colon operands must be real scalars"` at `for lll=1:size(XX)`
- **Cause:** `size(XX)` returns `[2 1]` but loop expects scalar. Should be `size(XX,1)` 
- **Impact:** Prevents full pipeline execution (steps [1,2,3])

**Issue #2: Landmark Detection Failure (Line 2478 in Avatar.m)**
- **Location:** `findMaxMin` function called by `getCrotch`
- **Error:** `"Unable to perform assignment because the size of the left side is 1-by-1 and the size of the right side is 0-by-1"`
- **Cause:** `vIdxOnLine{i}` is empty for some partitions after mesh cleaning removes too many vertices
- **Impact:** Fails when skipping mesh repair (`steps=[1,3]`)

**Issue #3: Mesh Quality Problems**
- Repeated faces and bad-shaped faces detected during cleaning
- `removeBoundaryProblems` removes >10% of faces (warning at line 3145)
- Reduced from original mesh to only 268 vertices and 95 faces after cleaning

**Recommendations:**
1. Test with higher-quality mesh files (from Styku/Fit3D scanners)
2. Use external mesh repair tools (MeshLab, trimesh) before Avatar construction
3. For development/testing, use `steps=[1]` to inspect mesh quality first
4. Ask original developers about expected mesh quality standards and repair parameters
