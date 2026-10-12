use std::path::Path;

use viewport_lib_types::data::mesh::MeshData;

use crate::error::IoError;

/// Decode an STL file into neutral surface mesh data.
pub fn mesh_from_path(path: &Path) -> Result<MeshData, IoError> {
    #[cfg(feature = "stl")]
    {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let indexed = stl_io::read_stl(&mut reader)
            .map_err(|error| IoError::Parse(format!("stl: {error}")))?;

        let triangle_count = indexed.faces.len();
        let mut mesh_data = MeshData::default();
        mesh_data.positions = Vec::with_capacity(triangle_count * 3);
        mesh_data.normals = Vec::with_capacity(triangle_count * 3);
        mesh_data.indices = Vec::with_capacity(triangle_count * 3);

        for (face_index, face) in indexed.faces.iter().enumerate() {
            let corners = face.vertices.map(|index| {
                let vertex = indexed.vertices[index];
                [vertex[0], vertex[1], vertex[2]]
            });
            let stored = [face.normal[0], face.normal[1], face.normal[2]];
            let normal = facet_normal(stored, corners);
            for corner in corners {
                mesh_data.positions.push(corner);
                mesh_data.normals.push(normal);
            }

            let base = (face_index * 3) as u32;
            mesh_data
                .indices
                .extend_from_slice(&[base, base + 1, base + 2]);
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

/// The unit normal for one facet.
///
/// Many exporters write `facet normal 0 0 0` and leave the normal to the
/// reader, and some write one that is not unit length. A stored normal with a
/// usable direction is kept and normalised, since it is what the file says;
/// one with no direction is computed from the triangle's winding. A degenerate
/// triangle with no stored normal gets +Z, so every normal is unit length.
fn facet_normal(stored: [f32; 3], corners: [[f32; 3]; 3]) -> [f32; 3] {
    if let Some(normal) = glam::Vec3::from(stored).try_normalize() {
        return normal.to_array();
    }
    let [a, b, c] = corners.map(glam::Vec3::from);
    (b - a)
        .cross(c - a)
        .try_normalize()
        .unwrap_or(glam::Vec3::Z)
        .to_array()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    #[test]
    fn a_zero_normal_is_computed_from_the_winding() {
        assert_eq!(facet_normal([0.0; 3], TRIANGLE), [0.0, 0.0, 1.0]);
        let reversed = [TRIANGLE[0], TRIANGLE[2], TRIANGLE[1]];
        assert_eq!(facet_normal([0.0; 3], reversed), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn a_stored_normal_is_kept_and_normalised() {
        assert_eq!(facet_normal([0.0, 0.0, 3.0], TRIANGLE), [0.0, 0.0, 1.0]);
        // Kept even against the winding: the file's normal is the file's call.
        assert_eq!(facet_normal([0.0, 0.0, -1.0], TRIANGLE), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn a_degenerate_facet_with_no_normal_gets_up() {
        let line = [[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_eq!(facet_normal([0.0; 3], line), [0.0, 0.0, 1.0]);
    }
}
