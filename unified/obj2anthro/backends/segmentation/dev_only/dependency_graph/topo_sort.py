"""Tools to build and visualize a dependency graph from an Excel sheet.

The expected sheet is named "TopoSort" with two columns:
- "Function": one or more function names separated by commas (each a node)
- "Dependencies": zero or more function names separated by commas (each an inbound edge)

Each dependency creates a directed edge from dependency -> function.

This module exposes:
- read_excel_to_graph(path, sheet_name='TopoSort') -> networkx.DiGraph
- topo_sort(graph) -> list of nodes in topological order (raises on cycle)
- draw_graph(graph, path=None, figsize=(12,8)) -> matplotlib figure (optionally saves to file)

Requires: pandas, networkx, matplotlib. Graphviz layout used if available.
"""

from __future__ import annotations

from typing import Iterable, List, Optional
from pathlib import Path

import logging
import os

import networkx as nx
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception as e:  # pragma: no cover - environment dependent
    raise ImportError("pandas is required to read Excel files: pip install pandas openpyxl") from e

logger = logging.getLogger(__name__)


def _split_and_clean(cell: Optional[str]) -> List[str]:
    """Split a comma-separated cell into tokens, trimming whitespace.

    Returns an empty list for empty/NaN inputs.
    """
    if cell is None:
        return []
    # pandas may deliver a float('nan') for empty cells
    try:
        if pd.isna(cell):
            return []
    except Exception:
        pass
    if not isinstance(cell, str):
        cell = str(cell)
    parts = [p.strip() for p in cell.split(",")]
    return [p for p in parts if p]


def read_excel_to_graph(path: str, sheet_name: str = "TopoSort", reduce_transitive: bool = True) -> nx.DiGraph:
    """Read the Excel sheet and return a directed graph.

    Each function in the "Function" column becomes a node. For each dependency
    listed in the corresponding "Dependencies" cell, an edge dependency -> function
    is added.

    Examples
    --------
    >>> # given a sheet where one row has Function="A,B" and Dependencies="C"
    >>> # the graph will contain nodes A,B,C and edges C->A and C->B

    Parameters
    ----------
    path: str
        Path to Excel file (xls/xlsx)
    sheet_name: str
        Sheet name (default "TopoSort")
    reduce_transitive: bool
        If True, remove redundant transitive edges to create a cleaner tree-like graph.
        For example, if A->B, B->C, and A->C all exist, the A->C edge is removed.

    Returns
    -------
    nx.DiGraph
        Directed graph including isolated nodes
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Expect columns named 'Function' and 'Dependencies' (case-sensitive)
    if "Function" not in df.columns or "Dependencies" not in df.columns:
        raise ValueError("Sheet must contain 'Function' and 'Dependencies' columns")

    G = nx.DiGraph()

    # First pass: ensure all functions are added as nodes (including multiple per cell)
    for idx, row in df.iterrows():
        funcs = _split_and_clean(row.get("Function"))
        for f in funcs:
            G.add_node(f)

    # Second pass: add dependencies edges
    for idx, row in df.iterrows():
        funcs = _split_and_clean(row.get("Function"))
        deps = _split_and_clean(row.get("Dependencies"))
        # Add dependency nodes as well (in case they never appear in Function column)
        for d in deps:
            if d not in G:
                G.add_node(d)
        # For each pair dependency -> function
        for f in funcs:
            for d in deps:
                G.add_edge(d, f)

    # Remove transitive edges to create a cleaner tree-like visualization
    if reduce_transitive:
        original_edges = len(G.edges())
        G = nx.transitive_reduction(G)
        # transitive_reduction returns a new graph without node attributes, so we need to copy them
        reduced = nx.DiGraph()
        for node in G.nodes():
            reduced.add_node(node)
        for u, v in G.edges():
            reduced.add_edge(u, v)
        logger.info(f"Transitive reduction: {original_edges} edges -> {len(reduced.edges())} edges")
        return reduced

    return G


def topo_sort(G: nx.DiGraph) -> List[str]:
    """Return a topological ordering of the graph nodes.

    Raises a ValueError if the graph contains a cycle. The returned list
    preserves node identity (strings) as provided in the Excel sheet.
    """
    try:
        order = list(nx.topological_sort(G))
        return order
    except nx.NetworkXUnfeasible as exc:
        # graph has at least one directed cycle
        cycles = list(nx.simple_cycles(G))
        logger.error("Graph contains cycles: %s", cycles)
        raise ValueError(f"Graph contains cycles, example cycle: {cycles[0]}") from exc


def draw_graph(
    G: nx.DiGraph,
    path: Optional[str] = None,
    figsize=(12, 8),
    orientation: str = "TB",
    show_order: bool = False,
    interactive: bool = False,
) -> plt.Figure:
    """Draw the dependency graph and optionally save to `path`.

    Parameters
    ----------
    G:
        Directed graph
    path:
        Optional path to save output (PNG for static, HTML for interactive)
    figsize:
        Matplotlib figure size (only for static PNG)
    orientation:
        'TB' (top->bottom) or 'LR' (left->right)
    show_order:
        If True, append the topological index to node labels
    interactive:
        If True, create an interactive HTML visualization with pyvis

    Returns
    -------
    matplotlib.figure.Figure (for static) or None (for interactive)
    """
    # If interactive requested, try to use pyvis (better interactivity than mpl)
    if interactive:
        try:
            from pyvis.network import Network

            net = Network(height="900px", width="100%", directed=True, notebook=False)
            
            # Compute topo order for positioning
            try:
                topo_order = list(nx.topological_sort(G))
            except Exception:
                topo_order = list(G.nodes())
            
            # Compute levels
            levels = {}
            for n in topo_order:
                preds = list(G.predecessors(n))
                if not preds:
                    levels[n] = 0
                else:
                    levels[n] = max(levels[p] for p in preds) + 1
            
            # Add nodes with hierarchical positioning
            for node in G.nodes():
                level = levels.get(node, 0)
                label = f"{node} ({topo_order.index(node) + 1})" if show_order else str(node)
                net.add_node(
                    node,
                    label=label,
                    level=level,
                    color="#78C0E0",
                    font={"size": 20},
                    shape="dot",
                    size=30,
                )
            
            # Add edges
            for u, v in G.edges():
                net.add_edge(u, v, arrows="to", color="#333333", width=2)
            
            # Use hierarchical layout
            net.set_options("""
            {
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "UD",
                  "sortMethod": "directed",
                  "nodeSpacing": 200,
                  "levelSeparation": 200,
                  "treeSpacing": 150
                }
              },
              "physics": {
                "enabled": false
              },
              "interaction": {
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true
              }
            }
            """)
            
            # Get the script's directory to ensure output always goes to the right place
            script_dir = Path(__file__).parent
            
            if path and path.endswith(".html"):
                out_html = path
            else:
                out_html = str(script_dir / "generated" / "dependency_graph.html")
            
            # Ensure output directory exists
            out_path = Path(out_html)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Change to output directory so pyvis puts lib/ folder there
            original_cwd = os.getcwd()
            try:
                os.chdir(out_path.parent)
                net.write_html(out_path.name)
                logger.info("Wrote interactive graph to %s", out_html)
            finally:
                os.chdir(original_cwd)
            
            return None
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("pyvis interactive output failed, falling back to static figure: %s", exc)

    # Compute topo ordering/levels - always compute for hierarchical layout
    try:
        topo_order = list(nx.topological_sort(G))
    except Exception:
        topo_order = list(G.nodes())  # fallback if graph has cycles

    # Compute level (distance from sources) for each node
    levels = {}
    for n in topo_order:
        preds = list(G.predecessors(n))
        if not preds:
            levels[n] = 0
        else:
            levels[n] = max(levels[p] for p in preds) + 1

    # Create hierarchical tree layout based on topological levels
    # Group nodes by level
    level_groups = {}
    for node, level in levels.items():
        if level not in level_groups:
            level_groups[level] = []
        level_groups[level].append(node)

    # Position nodes: Y coordinate = level, X coordinate = position within level
    pos = {}
    y_spacing = 4.5  # vertical spacing between levels (increased)
    x_spacing = 4.0  # horizontal spacing between nodes at same level (increased)

    for level, nodes in level_groups.items():
        # Sort nodes within each level by their position in topo order for consistency
        nodes_sorted = sorted(nodes, key=lambda n: topo_order.index(n))
        
        # Center the nodes horizontally
        width = (len(nodes_sorted) - 1) * x_spacing
        start_x = -width / 2
        
        for i, node in enumerate(nodes_sorted):
            x = start_x + i * x_spacing
            y = -level * y_spacing  # negative so tree flows top to bottom
            
            if orientation == "LR":
                pos[node] = (y, x)  # swap for left-right
            else:
                pos[node] = (x, y)  # top-bottom (default)

    # Create a clean matplotlib figure with larger size for better spacing
    fig, ax = plt.subplots(figsize=(16, 12))

    # Node styling: flat color, thin black border (no bold outlines)
    node_color = "#78C0E0"
    node_size = 1200
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    # Draw edges with visible arrows that STOP before entering the nodes
    # Use min_source_margin and min_target_margin to make arrows visible
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=2.0,
        edge_color="#333333",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=25,
        min_target_margin=25,
        ax=ax,
    )

    # Labels: optionally append topo index to make order visible
    labels = {}
    if show_order:
        idx_map = {n: i + 1 for i, n in enumerate(topo_order)}
        for n in G.nodes():
            labels[n] = f"{n} ({idx_map.get(n, '?')})"
    else:
        for n in G.nodes():
            labels[n] = str(n)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=30, font_family="sans-serif", ax=ax)

    ax.set_axis_off()
    # Add padding around the graph to prevent cropping
    ax.margins(0.15)

    if path:
        # Only save PNG if we have a valid path and it's not an HTML request
        if path.endswith(".html"):
            # Can't save as HTML without pyvis, skip file output
            logger.warning("HTML output requested but pyvis not available. No file saved.")
        else:
            # Ensure output directory exists
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight", dpi=200)
            logger.info("Saved graph visualization to %s", path)
    
    return fig


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build and visualize dependency graph from Excel")
    p.add_argument("excel", help="Path to Excel file")
    p.add_argument("--sheet", default="TopoSort", help="Sheet name (default TopoSort)")
    p.add_argument("--out", help="Output path (.html for interactive, .png for static). Default: dependency_graph.html")
    p.add_argument("--format", choices=["html", "png"], default="html", help="Output format (default: html)")
    p.add_argument("--show-order", action="store_true", help="Show topological order numbers in labels")
    args = p.parse_args()

    G = read_excel_to_graph(args.excel, sheet_name=args.sheet)
    try:
        order = topo_sort(G)
        print("Topological order:")
        for n in order:
            print(n)
    except ValueError as e:
        print("Topological sort failed:", e)

    # Determine output path and format
    # Get the script's directory to ensure output always goes to the right place
    script_dir = Path(__file__).parent
    
    if args.out:
        output_path = args.out
        is_interactive = output_path.endswith(".html") or args.format == "html"
    else:
        # Default to generated/ subdirectory relative to script location
        if args.format == "html":
            output_path = str(script_dir / "generated" / "dependency_graph.html")
        else:
            output_path = str(script_dir / "generated" / "dependency_graph.png")
        is_interactive = args.format == "html"

    draw_graph(G, path=output_path, show_order=args.show_order, interactive=is_interactive)
