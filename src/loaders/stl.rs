use std::path::Path;

use crate::error::IoError;
use crate::types::SurfaceMesh;

/// Decode an STL file into neutral surface mesh data.
pub fn mesh_from_path(path: &Path) -> Result<SurfaceMesh, IoError> {
    #[cfg(feature = "stl")]
    {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let indexed = stl_io::read_stl(&mut reader)
            .map_err(|error| IoError::Parse(format!("stl: {error}")))?;

        let triangle_count = indexed.faces.len();
        let mut mesh_data = SurfaceMesh::default();
        mesh_data.positions = Vec::with_capacity(triangle_count * 3);
        mesh_data.normals = Vec::with_capacity(triangle_count * 3);
        mesh_data.indices = Vec::with_capacity(triangle_count * 3);

        for (face_index, face) in indexed.faces.iter().enumerate() {
            for &vertex_index in &face.vertices {
                let vertex = indexed.vertices[vertex_index];
                let normal = face.normal;
                mesh_data.positions.push([vertex[0], vertex[1], vertex[2]]);
                mesh_data.normals.push([normal[0], normal[1], normal[2]]);
            }

            let base = (face_index * 3) as u32;
            mesh_data.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }

        if mesh_data.positions.is_empty() {
            return Err(IoError::Parse("stl contains no triangles".into()));
        }

        Ok(mesh_data)
    }

    #[cfg(not(feature = "stl"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "stl",
            context: "STL mesh decoding",
        })
    }
}
