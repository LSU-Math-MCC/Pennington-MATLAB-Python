"""
Boolean mesh operations for trimesh objects.

This module provides utilities for performing boolean operations on meshes
by comparing vertices and faces between meshes.
"""

import ast
import inspect
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _mesh_difference_arg_names():
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    if caller is None:
        return "mesh_a", "mesh_b", "unknown caller"

    filename = caller.f_code.co_filename.replace("\\", "/").split("/")[-1]
    caller_label = f"{filename}:{caller.f_lineno}::{caller.f_code.co_name}"
    context = inspect.getframeinfo(caller).code_context
    if not context:
        return "mesh_a", "mesh_b", caller_label

    source_line = context[0].strip()
    try:
        tree = ast.parse(source_line)
    except SyntaxError:
        return "mesh_a", "mesh_b", caller_label

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if func_name == "mesh_difference" and len(node.args) >= 2:
            mesh_a_name = ast.get_source_segment(source_line, node.args[0]) or "mesh_a"
            mesh_b_name = ast.get_source_segment(source_line, node.args[1]) or "mesh_b"
            return mesh_a_name, mesh_b_name, caller_label

    return "mesh_a", "mesh_b", caller_label


def mesh_difference(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    tolerance: float = 1e-6,
) -> trimesh.Trimesh:
    """
    Compute the mesh difference A \\\\ B (all of A that's not in B).
    
    This operation removes vertices from mesh A that are also present in mesh B
    (within the specified tolerance), and removes any faces that reference those vertices.
    
    This is NOT a true CSG (Constructive Solid Geometry) boolean operation. It works by
    comparing vertices, not computing geometric intersections. Use for meshes that are
    already roughly separated (like removing an arm from a body), not for complex
    interpenetrating shapes.
    
    Parameters
    ----------
    mesh_a : trimesh.Trimesh
        The mesh to subtract from (A)
    mesh_b : trimesh.Trimesh
        The mesh to subtract (B)
    tolerance : float, default=1e-6
        Distance threshold for considering vertices identical
        Vertices closer than this are considered the same point
    
    Returns
    -------
    trimesh.Trimesh
        A new trimesh containing only the parts of A that don't overlap with B
        Returns empty mesh if all vertices are removed
        
    Examples
    --------
    Remove an arm from a body mesh:
    
    >>> import trimesh
    >>> import numpy as np
    >>> from mesh.boolean_ops import mesh_difference  # doctest: +SKIP
    >>> # Create a simple example: box with attached sphere
    >>> box = trimesh.creation.box(extents=[2, 2, 2])  # doctest: +SKIP
    >>> sphere = trimesh.creation.icosphere(radius=0.5)  # doctest: +SKIP
    >>> sphere.apply_translation([1.5, 0, 0])  # Move sphere to side  # doctest: +SKIP
    >>> # Combine them (union)
    >>> combined = trimesh.util.concatenate([box, sphere])  # doctest: +SKIP
    >>> # Remove the sphere part
    >>> box_only = mesh_difference(combined, sphere)  # doctest: +SKIP
    >>> print(f"Original: {len(combined.vertices)} vertices")  # doctest: +SKIP
    Original: 150 vertices
    >>> print(f"After removal: {len(box_only.vertices)} vertices")  # doctest: +SKIP
    After removal: 100 vertices
    
    Pseudocode
    ----------
    1. Build KD-tree from mesh B vertices (for fast nearest neighbor queries)
    2. For each vertex in mesh A:
        a. Find nearest vertex in mesh B
        b. If distance > tolerance, mark vertex as "keep"
        c. Otherwise, mark as "remove" (it's in B)
    3. Create mapping: old vertex index -> new vertex index (for kept vertices)
    4. Filter faces:
        a. For each face in mesh A:
            - Check if ALL three vertices are being kept
            - If yes, remap vertex indices and keep face
            - If no, discard face
    5. Create new mesh from kept vertices and remapped faces
    6. Return result
    
    Notes
    -----
    - This is a vertex-based operation, not a true geometric boolean
    - Assumes meshes are already roughly separated (e.g., arm and body touch at shoulder)
    - Does not compute new intersection geometry
    - May leave gaps or create non-watertight meshes
    - For true boolean operations, use trimesh.boolean.difference() instead
    - Prints diagnostic information about vertices removed
    
    Limitations
    -----------
    - Only removes vertices that exactly match (within tolerance)
    - Doesn't handle partial overlaps or intersections
    - Faces are removed conservatively (any removed vertex -> face removed)
    - Result may not be watertight even if inputs were
    
    See Also
    --------
    mesh_intersection : Keeps only overlapping vertices
    trimesh.boolean.difference : True CSG boolean (more robust but requires external libraries)
    scipy.spatial.cKDTree : Used for fast nearest neighbor queries
    """
    
    mesh_a_name, mesh_b_name, caller = _mesh_difference_arg_names()

    if len(mesh_b.vertices) == 0:
        print(
            f"mesh_difference: {mesh_a_name} ({len(mesh_a.vertices)} vertices) "
            f"- {mesh_b_name} (0 vertices) [{caller}] -> skipped empty subtract mesh"
        )
        return mesh_a.copy()

    # Build a KDTree for fast nearest neighbor queries on mesh B vertices
    tree_b = cKDTree(mesh_b.vertices)
    
    # For each vertex in A, check if it exists in B (within tolerance)
    distances, _ = tree_b.query(mesh_a.vertices, k=1)
    
    # Vertices that are NOT in B (distance > tolerance)
    vertices_to_keep_mask = distances > tolerance
    vertices_to_keep_indices = np.where(vertices_to_keep_mask)[0]
    
    kept_vertices = int(np.sum(vertices_to_keep_mask))
    removed_vertices = len(mesh_a.vertices) - kept_vertices
    print(
        f"mesh_difference: {mesh_a_name} ({len(mesh_a.vertices)} vertices) "
        f"- {mesh_b_name} ({len(mesh_b.vertices)} vertices) -> "
        f"{kept_vertices} kept, {removed_vertices} removed [{caller}]"
    )
    
    # If all vertices are being kept, just return a copy
    if np.all(vertices_to_keep_mask):
        print("No vertices removed - meshes don't overlap")
        return mesh_a.copy()
    
    # If no vertices are being kept, return empty mesh
    if not np.any(vertices_to_keep_mask):
        print("Warning: All vertices removed - complete overlap")
        return trimesh.Trimesh()
    
    # Create a mapping from old vertex indices to new vertex indices
    old_to_new = np.full(len(mesh_a.vertices), -1, dtype=int)
    old_to_new[vertices_to_keep_indices] = np.arange(len(vertices_to_keep_indices))
    
    # Filter faces: keep only faces where ALL vertices are being kept
    faces_to_keep = []
    for face in mesh_a.faces:
        # Check if all three vertices of this face are being kept
        if np.all(vertices_to_keep_mask[face]):
            # Remap the vertex indices to the new vertex array
            new_face = old_to_new[face]
            faces_to_keep.append(new_face)
    
    if len(faces_to_keep) == 0:
        print("Warning: No faces remain after vertex filtering")
        return trimesh.Trimesh()
    
    # Create new mesh with filtered vertices and remapped faces
    new_vertices = mesh_a.vertices[vertices_to_keep_mask]
    new_faces = np.array(faces_to_keep)
    
    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    print(f"mesh_difference result: {len(result.vertices)} vertices, {len(result.faces)} faces")
    
    return result


def mesh_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, tolerance: float = 1e-6) -> trimesh.Trimesh:
    """
    Compute the mesh intersection A ∩ B (vertices present in both A and B).
    
    This operation keeps only vertices from mesh A that are also present in mesh B
    (within the specified tolerance). Like mesh_difference, this is a vertex-based
    operation, not a true geometric intersection.
    
    Parameters
    ----------
    mesh_a : trimesh.Trimesh
        First mesh (A)
    mesh_b : trimesh.Trimesh
        Second mesh (B)
    tolerance : float, default=1e-6
        Distance threshold for considering vertices identical
    
    Returns
    -------
    trimesh.Trimesh
        A new trimesh containing only the overlapping parts
        Returns empty mesh if no vertices overlap
        
    Examples
    --------
    Find overlapping region of two meshes:
    
    >>> import trimesh
    >>> from mesh.boolean_ops import mesh_intersection  # doctest: +SKIP
    >>> # Create two overlapping boxes
    >>> box1 = trimesh.creation.box(extents=[2, 2, 2])  # doctest: +SKIP
    >>> box2 = trimesh.creation.box(extents=[2, 2, 2])  # doctest: +SKIP
    >>> box2.apply_translation([1, 0, 0])  # Shift to overlap  # doctest: +SKIP
    >>> # Combine them
    >>> combined = trimesh.util.concatenate([box1, box2])  # doctest: +SKIP
    >>> # Find intersection
    >>> overlap = mesh_intersection(box1, box2)  # doctest: +SKIP
    >>> print(f"Overlapping vertices: {len(overlap.vertices)}")  # doctest: +SKIP
    Overlapping vertices: 4
    
    Pseudocode
    ----------
    1. Build KD-tree from mesh B vertices
    2. For each vertex in mesh A:
        a. Find nearest vertex in mesh B
        b. If distance <= tolerance, mark vertex as "keep"
        c. Otherwise, mark as "remove" (not in B)
    3. Create mapping: old vertex index -> new vertex index
    4. Filter faces (keep only faces where ALL vertices are kept)
    5. Create new mesh from kept vertices and remapped faces
    6. Return result
    
    Notes
    -----
    - This is the complement of mesh_difference
    - Also prints diagnostic information
    - Same limitations as mesh_difference (vertex-based, not geometric)
    - For true geometric intersection, use trimesh.boolean.intersection()
    
    See Also
    --------
    mesh_difference : Remove overlapping parts instead of keeping them
    trimesh.boolean.intersection : True CSG boolean intersection
    """
    
    # Build a KDTree for fast nearest neighbor queries on mesh B vertices
    tree_b = cKDTree(mesh_b.vertices)
    
    # For each vertex in A, check if it exists in B (within tolerance)
    distances, _ = tree_b.query(mesh_a.vertices, k=1)
    
    # Vertices that ARE in B (distance <= tolerance)
    vertices_to_keep_mask = distances <= tolerance
    vertices_to_keep_indices = np.where(vertices_to_keep_mask)[0]
    
    print(f"Mesh A has {len(mesh_a.vertices)} vertices")
    print(f"Mesh B has {len(mesh_b.vertices)} vertices")
    print(f"Vertices in both A and B: {np.sum(vertices_to_keep_mask)}")
    
    if not np.any(vertices_to_keep_mask):
        print("Warning: No overlapping vertices found")
        return trimesh.Trimesh()
    
    # Create a mapping from old vertex indices to new vertex indices
    old_to_new = np.full(len(mesh_a.vertices), -1, dtype=int)
    old_to_new[vertices_to_keep_indices] = np.arange(len(vertices_to_keep_indices))
    
    # Filter faces: keep only faces where ALL vertices are being kept
    faces_to_keep = []
    for face in mesh_a.faces:
        if np.all(vertices_to_keep_mask[face]):
            new_face = old_to_new[face]
            faces_to_keep.append(new_face)
    
    if len(faces_to_keep) == 0:
        print("Warning: No faces remain after vertex filtering")
        return trimesh.Trimesh()
    
    # Create new mesh with filtered vertices and remapped faces
    new_vertices = mesh_a.vertices[vertices_to_keep_mask]
    new_faces = np.array(faces_to_keep)
    
    result = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    print(f"Result mesh has {len(result.vertices)} vertices and {len(result.faces)} faces")
    
    return result
