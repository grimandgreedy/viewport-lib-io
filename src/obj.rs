use std::path::Path;

use crate::error::IoError;
use crate::scene::{IoMaterial, IoMesh, IoScene, TextureSource};

/// Decode an OBJ file into a CPU-side scene.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    #[cfg(feature = "obj")]
    {
        let options = tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        };

        let (models, materials) = tobj::load_obj(path, &options)
            .map_err(|error| IoError::Parse(format!("obj: {error}")))?;
        let materials = materials.map_err(|error| IoError::Parse(format!("mtl: {error}")))?;

        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));

        let materials = materials
            .into_iter()
            .map(|material| IoMaterial {
                name: material.name,
                base_color: material.diffuse.unwrap_or([0.7, 0.7, 0.7]),
                metallic: material.shininess.unwrap_or(0.0).clamp(0.0, 1.0),
                roughness: 0.5,
                opacity: material.dissolve.unwrap_or(1.0),
                base_color_texture: material.diffuse_texture.and_then(|texture| {
                    if texture.is_empty() {
                        None
                    } else {
                        Some(TextureSource::File(base_dir.join(texture)))
                    }
                }),
                normal_map_texture: material.normal_texture.and_then(|texture| {
                    if texture.is_empty() {
                        None
                    } else {
                        Some(TextureSource::File(base_dir.join(texture)))
                    }
                }),
                ao_texture: None,
            })
            .collect();

        let meshes = models
            .into_iter()
            .map(|model| {
                let mesh = model.mesh;
                let mut mesh_data = viewport_lib::MeshData::default();

                mesh_data.positions = mesh
                    .positions
                    .chunks_exact(3)
                    .map(|values| [values[0], values[1], values[2]])
                    .collect();
                mesh_data.indices = mesh.indices;

                if mesh.normals.is_empty() {
                    mesh_data.normals = vec![[0.0, 1.0, 0.0]; mesh_data.positions.len()];
                } else {
                    mesh_data.normals = mesh
                        .normals
                        .chunks_exact(3)
                        .map(|values| [values[0], values[1], values[2]])
                        .collect();
                }

                if !mesh.texcoords.is_empty() {
                    mesh_data.uvs = Some(
                        mesh.texcoords
                            .chunks_exact(2)
                            .map(|values| [values[0], values[1]])
                            .collect(),
                    );
                }

                IoMesh {
                    name: model.name,
                    mesh_data,
                    material_index: mesh.material_id,
                    ..IoMesh::default()
                }
            })
            .collect();

        Ok(IoScene {
            meshes,
            materials,
            ..IoScene::default()
        })
    }

    #[cfg(not(feature = "obj"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "obj",
            context: "OBJ scene decoding",
        })
    }
}
