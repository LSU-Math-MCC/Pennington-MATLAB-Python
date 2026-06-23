import trimesh
import numpy as np
from sklearn.cluster import DBSCAN



def get_significant_min_values_along_axis(
        vertices, 
        scan_axis=0,        # 0 for x, 1 for y, 2 for z
        height_axis=2,      # 2 for z, etc.
        resolution=200, 
        threshold_ratio=0.25, 
        slice_width=0.005
    ):
        scan_coords = vertices[:, scan_axis]
        height_coords = vertices[:, height_axis]

        height_range = height_coords.max() - height_coords.min()
        threshold = height_range * threshold_ratio

        steps = np.linspace(scan_coords.min(), scan_coords.max(), resolution)
        min_values = []

        for val in steps:
            mask = np.abs(scan_coords - val) < slice_width
            slice_verts = vertices[mask]

            if len(slice_verts) == 0:
                continue

            slice_heights = slice_verts[:, height_axis]
            height_span = slice_heights.max() - slice_heights.min()

            if height_span < threshold:
                continue

            min_val = slice_heights.min()
            min_values.append((val, min_val))

        return min_values

class Avatar:
    def __init__(self, mesh_path):
        self.mesh = trimesh.load(mesh_path)
        self.v = self.mesh.vertices
        self.scene = trimesh.Scene(self.mesh)
        self.clean_mesh()
        self.align_model()
        self.mark_extrema()
        self.mark_feet()
        self.mark_crotch()
        self.mark_armpits()
        #self.draw_bounding_box()
        #self.draw_axes()
        self.scene.show()

    def clean_mesh(self):
        self.mesh.remove_duplicate_faces()
        self.mesh.remove_unreferenced_vertices()
        self.mesh.fill_holes()
        self.mesh.update_faces(self.mesh.nondegenerate_faces(height=1e-8))

    def align_model(self):
        bounds = self.mesh.bounds
        extents = bounds[1] - bounds[0]
        major_axis = np.argmax(extents)

        rotation_to_z = self._rotation_matrix_to_axis(major_axis, target_axis=2)
        self.mesh.apply_transform(rotation_to_z)

        center_z = np.mean(self.mesh.bounds[:, 2])
        z_dist = np.abs(self.mesh.vertices[:, 2] - center_z)
        closest_mask = z_dist < 0.01
        cross_section_vertices = self.mesh.vertices[closest_mask]

        lines = []
        errors = []
        for axis in [0, 1]:
            side_center = np.mean(self.mesh.bounds[:, axis])
            point_on_line = np.zeros(3)
            point_on_line[axis] = side_center
            point_on_line[2] = center_z

            direction = np.zeros(3)
            direction[2] = 1

            diffs = cross_section_vertices - point_on_line
            projections = np.outer(diffs @ direction, direction)
            reconstructed = point_on_line + projections
            mse = np.mean(np.linalg.norm(cross_section_vertices - reconstructed, axis=1) ** 2)
            lines.append((point_on_line, direction))
            errors.append(mse)

        x_axis = np.argmin(errors)
        rotation_to_x = self._rotation_matrix_to_axis(x_axis, target_axis=0)
        self.mesh.apply_transform(rotation_to_x)

        center = np.mean(self.mesh.vertices, axis=0)
        self.mesh.apply_translation(-center)

    def _rotation_matrix_to_axis(self, source_axis, target_axis):
        if source_axis == target_axis:
            return np.eye(4)
        axes = np.eye(3)
        v1 = axes[source_axis]
        v2 = axes[target_axis]
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
        R = np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))
        M = np.eye(4)
        M[:3, :3] = R
        return M

    def mark_extrema(self):
        bounds = self.mesh.bounds
        colors = [
            [255, 0, 0, 255],
            [0, 0, 255, 255],
            [0, 255, 0, 255]
        ]
        for i in range(3):
            for j, val in enumerate(bounds[:, i]):
                marker = trimesh.creation.icosphere(radius=0.01)
                pos = np.zeros(3)
                pos[i] = val
                marker.apply_translation(pos)
                marker.visual.face_colors = colors[i]
                self.scene.add_geometry(marker)



    def mark_feet(self):
        v = np.asarray(self.mesh.vertices)
        z = v[:, 2]
        lower_mask = z < np.percentile(z, 10)
        lower = v[lower_mask]
        left = lower[lower[:, 0] < 0]
        right = lower[lower[:, 0] > 0]
        left_foot = left[np.argmin(left[:, 2])] if len(left) > 0 else None
        right_foot = right[np.argmin(right[:, 2])] if len(right) > 0 else None
        for foot in [left_foot, right_foot]:
            if foot is not None:
                marker = trimesh.creation.icosphere(radius=0.01)
                marker.apply_translation(foot)
                marker.visual.face_colors = [128, 0, 128, 255]
                self.scene.add_geometry(marker)

    def get_crotch(self):
        verts = self.mesh.vertices
        min_zs = get_significant_min_values_along_axis(verts, 0, 2)

        if not min_zs:
            print("No valid slices with sufficient height spread.")
            return None

        crotch_x, crotch_z = max(min_zs, key=lambda item: item[1])

        candidate_verts = verts[np.abs(verts[:, 0] - crotch_x) < 0.01]
        candidate_verts = candidate_verts[np.abs(candidate_verts[:, 2] - crotch_z) < 0.01]
        crotch_y = candidate_verts[:, 1].mean() if len(candidate_verts) > 0 else 0.0

        return np.array([crotch_x, crotch_y, crotch_z])
    
    def mark_crotch(self):
        crotch_point = self.get_crotch()
        if crotch_point is None:
            return

        print(f"Crotch found at: {crotch_point}")

        marker = trimesh.creation.icosphere(radius=0.2)
        marker.apply_translation(crotch_point)
        marker.visual.face_colors = [255, 165, 0, 255]
        self.scene.add_geometry(marker)
    
    def get_armpits(self):
        verts = self.mesh.vertices
        crotch = self.get_crotch()
        if crotch is None:
            return None, None

        crotch_x = crotch[0]
        left_verts = verts[verts[:, 0] < crotch_x]
        right_verts = verts[verts[:, 0] > crotch_x]

        def find_armpit(region_verts):
            min_zs = get_significant_min_values_along_axis(region_verts, 0, 2,200,.01)
            if not min_zs:
                return None
            armpit_x, armpit_z = max(min_zs, key=lambda item: item[1])
            candidate_verts = region_verts[np.abs(region_verts[:, 0] - armpit_x) < 0.01]
            candidate_verts = candidate_verts[np.abs(candidate_verts[:, 2] - armpit_z) < 0.01]
            armpit_y = candidate_verts[:, 1].mean() if len(candidate_verts) > 0 else 0.0
            return np.array([armpit_x, armpit_y, armpit_z])

        return find_armpit(left_verts), find_armpit(right_verts)

    def mark_armpits(self):
        left, right = self.get_armpits()

        if left is not None:
            print(f"Left armpit: {left}")
            marker = trimesh.creation.icosphere(radius=0.2)
            marker.apply_translation(left)
            marker.visual.face_colors = [0, 255, 255, 255]  # cyan
            self.scene.add_geometry(marker)

        if right is not None:
            print(f"Right armpit: {right}")
            marker = trimesh.creation.icosphere(radius=0.2)
            marker.apply_translation(right)
            marker.visual.face_colors = [255, 0, 255, 255]  # magenta
            self.scene.add_geometry(marker)




    def draw_axes(self):
        origin = np.array([0.0, 0.0, 0.0])
        length = 50
        axes = [
            ([length, 0, 0], [255, 0, 0, 255]),
            ([0, length, 0], [0, 0, 255, 255]),
            ([0, 0, length], [0, 255, 0, 255])
        ]
        for vec, color in axes:
            line = trimesh.load_path(np.array([[origin, origin + vec]]))
            line.colors = np.tile(color, (len(line.entities), 1))
            self.scene.add_geometry(line)
    

mesh_path = "model_files/man.obj"
avatar = Avatar(mesh_path)
