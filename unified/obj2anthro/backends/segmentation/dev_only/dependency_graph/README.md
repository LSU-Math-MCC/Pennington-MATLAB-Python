# Dependency Graph Visualization

This directory contains tools and data for visualizing function dependencies as an interactive hierarchical tree.

## Directory Structure

```
dev_only/dependency_graph/
├── topo_sort.py          # Visualization tool
├── dependencies.xlsx     # Excel template with function dependencies
├── generated/            # All generated outputs
│   ├── dependency_graph.html  # Interactive visualization
│   ├── dependency_graph.png   # Static PNG (optional)
│   └── lib/              # JavaScript dependencies (pyvis assets)
└── README.md            # This file
```

## Excel Format

The `TopoSort` sheet should have two columns:

| Function | Dependencies |
|----------|--------------|
| functionA, functionB | dependency1, dependency2 |
| functionC | functionA |

- **Function**: One or more function names (comma-separated)
- **Dependencies**: Direct dependencies only (not transitive)

### Important: List Only Direct Dependencies

❌ **Wrong** - listing all ancestors:
```
Function: myFunction
Dependencies: grandparent, parent, sibling  # Too many!
```

✅ **Correct** - only immediate parent(s):
```
Function: myFunction
Dependencies: parent  # Only direct dependency
```

The tool automatically performs **transitive reduction** to remove redundant edges, but starting with direct dependencies gives the cleanest tree structure.

## Usage

### Generate Interactive HTML (default)

```bash
# From this directory
python3 topo_sort.py dependencies.xlsx
```

This creates `generated/dependency_graph.html` with:
- Hierarchical tree layout flowing top-to-bottom
- Topological ordering (dependencies before dependents)
- Navigation buttons for zoom/pan
- Mouse wheel zoom and click-drag panning
- Mouse wheel zoom and click-drag panning
- Transitive reduction (removes redundant edges automatically)
- All assets in `generated/` subdirectory (keeps workspace clean)

### Generate Static PNG

```bash
python3 topo_sort.py dependencies.xlsx --format png
```

Creates `generated/dependency_graph.png`

### Show Topological Order Numbers

```bash
python3 topo_sort.py dependencies.xlsx --show-order
```

Adds `(N)` after each node label showing its position in topological order.

### Custom Output Path

```bash
# HTML output to custom location
python3 topo_sort.py dependencies.xlsx --out custom_name.html

# PNG output to custom location
python3 topo_sort.py dependencies.xlsx --out ../other_dir/graph.png
```

**Note**: When using custom paths, the `lib/` folder will be created relative to the output HTML file.

## Features

### Transitive Reduction

Automatically removes redundant edges. For example:

```
Before: A→B, B→C, A→C (3 edges)
After:  A→B, B→C      (2 edges, A→C removed as redundant)
```

This creates a cleaner tree visualization while preserving all dependency information.

### Hierarchical Layout

Nodes are positioned by **topological level**:
- Level 0: Root nodes (no dependencies)
- Level 1: Nodes depending only on Level 0
- Level 2: Nodes depending on Level 0 or 1
- etc.

Nodes at the same level are arranged horizontally with consistent spacing.

### Interactive Controls (HTML output)

- **Zoom**: Mouse wheel or navigation buttons
- **Pan**: Click and drag, or use navigation buttons
- **Reset**: Double-click background to reset view
- **Keyboard**: Arrow keys for navigation

## Installation

Required Python packages:

```bash
pip install pandas openpyxl networkx matplotlib pyvis
```

## Troubleshooting

### Graph has too many edges / looks messy

**Solution**: Edit your Excel file to include only **direct dependencies**, not all ancestors. The tool will compute transitive relationships automatically.

### Nodes overlap

**Solution**: The tool uses hierarchical layout with spacing designed for ~50 nodes. For very large graphs (100+ nodes), consider:
1. Breaking the graph into sub-graphs by functional area
2. Increasing `nodeSpacing` and `levelSeparation` in the code
3. Using the interactive HTML output and zooming out

### Graph has cycles / topological sort fails

**Error message**: `"Graph contains cycles, example cycle: [A, B, C, A]"`

**Solution**: Your dependencies have a circular reference. Check the Excel file and remove the cycle (e.g., if A depends on B and B depends on A, one of those must be wrong).

## Advanced Usage

### Run from Python script

```python
import sys
sys.path.insert(0, 'dev_only/dependency_graph')
from topo_sort import read_excel_to_graph, topo_sort, draw_graph

# Load graph
G = read_excel_to_graph('dev_only/dependency_graph/dependencies.xlsx')

# Get topological order
order = topo_sort(G)
print(order)

# Generate visualization (output to generated/)
draw_graph(G, path='generated/output.html', interactive=True, show_order=True)
```

### Disable transitive reduction

```python
G = read_excel_to_graph('file.xlsx', reduce_transitive=False)
```

## Example

Given this Excel data:

| Function | Dependencies |
|----------|--------------|
| crotch | r_foot, l_foot |
| r_armpit | crotch |
| hipCircumference | r_armpit, crotch |

The tool will:
1. Create nodes: `r_foot`, `l_foot`, `crotch`, `r_armpit`, `hipCircumference`
2. Create edges: `r_foot→crotch`, `l_foot→crotch`, `crotch→r_armpit`, `r_armpit→hipCircumference`, `crotch→hipCircumference`
3. Apply transitive reduction: remove `crotch→hipCircumference` (redundant via `r_armpit`)
4. Arrange hierarchically:
   - Level 0: `r_foot`, `l_foot`
   - Level 1: `crotch`
   - Level 2: `r_armpit`
   - Level 3: `hipCircumference`

---

*Tool maintained as part of MATH 4020/4997 - 3D Anthropometric Analysis Project*
