import unittest
import numpy as np
import os
import tempfile
import matlab.engine
import trimesh
from pathlib import Path

def read_ply_python(filename):
    """
    Python implementation of the MATLAB read_ply function using trimesh.
    
    Args:
        filename (str): Path to the PLY file
        
    Returns:
        tuple: (vertex, face) where
            - vertex: numpy array of shape (nb_vert, 3) with vertex positions  
            - face: numpy array of shape (nb_face, 3) with face connectivity (1-indexed)
    """
    # Load mesh using trimesh
    mesh = trimesh.load_mesh(filename)
    
    # Extract vertices and faces
    vertex = mesh.vertices
    face = mesh.faces + 1  # Convert to 1-indexed for MATLAB compatibility
    
    return vertex, face

class TestReadPly(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up MATLAB engine and test files once for all tests."""
        print("Starting MATLAB engine...")
        cls.eng = matlab.engine.start_matlab()
        
        # Add the matlab directory to MATLAB path and change to it
        project_root = Path(__file__).parent.parent.parent.parent
        matlab_dir = str(project_root / "matlab")
        cls.eng.addpath(matlab_dir)
        
        # Change MATLAB working directory to matlab folder
        cls.eng.cd(matlab_dir)
        
        # Find available PLY test files
        cls.test_files = []
        test_mesh_dir = project_root / "test" / "mesh"
        model_files_dir = project_root / "model_files"
        
        # Look for PLY files in common locations
        for search_dir in [test_mesh_dir, model_files_dir]:
            if search_dir.exists():
                ply_files = list(search_dir.glob("*.ply"))
                cls.test_files.extend([str(f) for f in ply_files])
        
        print(f"Found {len(cls.test_files)} PLY test files: {cls.test_files}")
        
        # Set flag for MATLAB availability
        cls.matlab_available = True
        
        # Create a standalone MATLAB function file
        matlab_func_content = """function [vertex,face] = test_read_ply(filename)
% Standalone version of the read_ply function from Avatar class
[d,c] = plyread(filename);
try
    vi = d.face.vertex_indices;
catch
    vi = d.face.vertex_index;
end 
nf = length(vi);
face = zeros(nf,3);
for i=1:nf
    face(i,:) = vi{i}+1;
end
vertex = [d.vertex.x, d.vertex.y, d.vertex.z];
end"""
        
        # Write the function to a file in the matlab directory
        func_file_path = str(Path(matlab_dir) / "test_read_ply.m")
        with open(func_file_path, 'w') as f:
            f.write(matlab_func_content)
        
        print(f"Created MATLAB function file: {func_file_path}")
        
        # Test if the function is available
        try:
            # Test with a simple command
            cls.eng.eval("which test_read_ply", nargout=0)
            
            # Check if plyread is available (it's also embedded in Avatar)
            cls.eng.eval("which plyread", nargout=0)
            print("✓ MATLAB test_read_ply function is available")
            print("✓ MATLAB plyread function is available") 
        except Exception as e:
            print(f"✗ MATLAB function test failed: {e}")
            print("ℹ️  MATLAB tests will be skipped - using Python-only validation")
            cls.matlab_available = False
    
    @classmethod
    def tearDownClass(cls):
        """Clean up MATLAB engine."""
        print("Stopping MATLAB engine...")
        cls.eng.quit()
    
    def create_test_ply_file(self):
        """Create a simple test PLY file for testing."""
        # Create a simple triangle mesh (tetrahedron)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0]
        ])
        
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3]
        ])
        
        # Create trimesh object and save as PLY
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        temp_file.close()
        
        # Export to PLY
        mesh.export(temp_file.name)
        
        return temp_file.name, vertices, faces + 1  # +1 for MATLAB indexing
    
    def test_simple_mesh(self):
        """Test with a simple programmatically created mesh."""
        temp_file, expected_vertices, expected_faces = self.create_test_ply_file()
        
        try:
            # Test Python implementation
            py_vertices, py_faces = read_ply_python(temp_file)
            
            # Test MATLAB implementation if available
            if self.matlab_available:
                matlab_result = self.eng.test_read_ply(temp_file, nargout=2)
                ml_vertices = np.array(matlab_result[0])
                ml_faces = np.array(matlab_result[1])
            else:
                ml_vertices = py_vertices  # Use Python results as reference
                ml_faces = py_faces
            
            # Compare vertices with expected values
            np.testing.assert_array_almost_equal(py_vertices, expected_vertices, decimal=5)
            
            # Compare with MATLAB if available
            if self.matlab_available:
                # Compare vertices (should be very close)
                np.testing.assert_array_almost_equal(py_vertices, ml_vertices, decimal=5)
                # Compare faces (should be identical - both 1-indexed)
                np.testing.assert_array_equal(py_faces, ml_faces)
            
            # Verify shapes
            self.assertEqual(py_vertices.shape, ml_vertices.shape)
            self.assertEqual(py_faces.shape, ml_faces.shape)
            self.assertEqual(py_vertices.shape[1], 3)  # 3D vertices
            self.assertEqual(py_faces.shape[1], 3)     # Triangle faces
            
            print(f"✓ Simple mesh test passed - {py_vertices.shape[0]} vertices, {py_faces.shape[0]} faces")
            
        finally:
            # Clean up temp file
            os.unlink(temp_file)
    
    def test_existing_ply_files(self):
        """Test with existing PLY files from the project."""
        if not self.test_files:
            self.skipTest("No PLY files found for testing")
        
        if not self.matlab_available:
            self.skipTest("MATLAB test_read_ply function not available")
        
        for ply_file in self.test_files:
            with self.subTest(file=os.path.basename(ply_file)):
                try:
                    # Test Python implementation
                    py_vertices, py_faces = read_ply_python(ply_file)
                    
                    # Test MATLAB implementation  
                    matlab_result = self.eng.test_read_ply(ply_file, nargout=2)
                    ml_vertices = np.array(matlab_result[0])
                    ml_faces = np.array(matlab_result[1])
                    
                    # Compare results
                    np.testing.assert_array_almost_equal(py_vertices, ml_vertices, decimal=5)
                    np.testing.assert_array_equal(py_faces, ml_faces)
                    
                    # Verify shapes
                    self.assertEqual(py_vertices.shape, ml_vertices.shape)
                    self.assertEqual(py_faces.shape, ml_faces.shape)
                    
                    print(f"✓ {os.path.basename(ply_file)}: {py_vertices.shape[0]} vertices, {py_faces.shape[0]} faces")
                    
                except Exception as e:
                    self.fail(f"Failed to process {ply_file}: {str(e)}")
    
    def test_face_indexing(self):
        """Test that face indices are 1-indexed (MATLAB style)."""
        temp_file, _, expected_faces = self.create_test_ply_file()
        
        try:
            py_vertices, py_faces = read_ply_python(temp_file)
            
            # Faces should be 1-indexed (minimum value should be 1, not 0)
            self.assertGreaterEqual(py_faces.min(), 1)
            self.assertLessEqual(py_faces.max(), len(py_vertices))
            
            print(f"✓ Face indexing test passed - indices range from {py_faces.min()} to {py_faces.max()}")
            
        finally:
            os.unlink(temp_file)
    
    def test_vertex_coordinates(self):
        """Test that vertex coordinates are properly extracted."""
        temp_file, expected_vertices, _ = self.create_test_ply_file()
        
        try:
            py_vertices, py_faces = read_ply_python(temp_file)
            
            # Vertices should have shape (N, 3)
            self.assertEqual(py_vertices.shape[1], 3)
            
            # Should match expected vertices
            np.testing.assert_array_almost_equal(py_vertices, expected_vertices, decimal=5)
            
            # Test MATLAB version matches (skip if not available)
            if not self.matlab_available:
                print(f"✓ Vertex coordinates test passed (Python only) - shape {py_vertices.shape}")
                return
            
            matlab_result = self.eng.test_read_ply(temp_file, nargout=2)
            ml_vertices = np.array(matlab_result[0])
            
            np.testing.assert_array_almost_equal(py_vertices, ml_vertices, decimal=5)
            
            print(f"✓ Vertex coordinates test passed - shape {py_vertices.shape}")
            
        finally:
            os.unlink(temp_file)
    
    def test_empty_or_invalid_file(self):
        """Test error handling for invalid files."""
        # Test non-existent file - trimesh raises ValueError, not FileNotFoundError
        with self.assertRaises((FileNotFoundError, ValueError)):
            read_ply_python("nonexistent_file.ply")
        
        # Test invalid file extension (should still work if it's actually PLY format)
        temp_file, _, _ = self.create_test_ply_file()
        try:
            # Rename to different extension
            invalid_file = temp_file.replace('.ply', '.xyz')
            os.rename(temp_file, invalid_file)
            
            # Should still work since trimesh looks at content, not extension
            py_vertices, py_faces = read_ply_python(invalid_file)
            self.assertGreater(len(py_vertices), 0)
            self.assertGreater(len(py_faces), 0)
            
            os.unlink(invalid_file)
            
        except:
            # Clean up if test fails
            for f in [temp_file, invalid_file]:
                if os.path.exists(f):
                    os.unlink(f)
            raise
    
    def test_matlab_matlab_comparison(self):
        """Ensure MATLAB function works consistently."""
        if not self.test_files:
            self.skipTest("No PLY files found for MATLAB consistency test")
        
        if not self.matlab_available:
            self.skipTest("MATLAB test_read_ply function not available")
        
        # Test same file twice with MATLAB to ensure consistency
        test_file = self.test_files[0]
        
        # Call MATLAB function twice
        result1 = self.eng.test_read_ply(test_file, nargout=2)
        result2 = self.eng.test_read_ply(test_file, nargout=2)
        
        vertices1 = np.array(result1[0])
        faces1 = np.array(result1[1])
        vertices2 = np.array(result2[0])
        faces2 = np.array(result2[1])
        
        # Results should be identical
        np.testing.assert_array_equal(vertices1, vertices2)
        np.testing.assert_array_equal(faces1, faces2)
        
        print(f"✓ MATLAB consistency test passed")
    
    def test_python_only_functionality(self):
        """Test Python implementation standalone functionality."""
        if not self.test_files:
            self.skipTest("No PLY files found for Python-only test")
        
        print("\n=== Python Implementation Validation ===")
        for ply_file in self.test_files:
            file_name = os.path.basename(ply_file)
            try:
                vertices, faces = read_ply_python(ply_file)
                
                # Basic validation
                self.assertGreater(len(vertices), 0, f"No vertices loaded from {file_name}")
                self.assertGreater(len(faces), 0, f"No faces loaded from {file_name}")
                self.assertEqual(vertices.shape[1], 3, f"Vertices not 3D in {file_name}")
                self.assertEqual(faces.shape[1], 3, f"Faces not triangular in {file_name}")
                
                # MATLAB compatibility checks
                self.assertGreaterEqual(faces.min(), 1, f"Faces not 1-indexed in {file_name}")
                self.assertLessEqual(faces.max(), len(vertices), f"Face indices out of range in {file_name}")
                
                print(f"✓ {file_name}: {len(vertices)} vertices, {len(faces)} faces")
                
            except Exception as e:
                self.fail(f"Python implementation failed on {file_name}: {str(e)}")
        
        print("✓ All Python-only tests passed!")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)