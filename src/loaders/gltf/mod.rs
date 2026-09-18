//! glTF and GLB decoding.
//!
//! `scene_from_path` and `scene_from_slice` are the entrypoints; both walk the
//! document once and hand each piece to the module that owns it: `node` for the
//! scene graph, `primitive` for geometry, `material` for materials and
//! textures, `skin` for skeletons and weights, `animation` for clips, and
//! `axis` for the Y-up to Z-up conversion every one of them feeds into.

use std::path::Path;

use crate::error::IoError;
use crate::types::{IoScene, MAX_JOINTS};

mod animation;
mod axis;
mod material;
mod node;
mod primitive;
mod skin;

use animation::{convert_animations, convert_morph_animations};
use axis::reorient_mesh_z_up;
use material::convert_material;
use node::collect_node;
use skin::convert_skeletons;

/// Decode a glTF or GLB file into a CPU-side scene.
///
/// Reads the file at `path` and delegates to [`scene_from_slice`]. The
/// parent directory is used as the base for resolving external buffers and
/// image URIs.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    #[cfg(feature = "gltf")]
    {
        if !path.exists() {
            return Err(IoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }
        let bytes = std::fs::read(path)?;
        let parent_dir = path.parent().unwrap_or(Path::new("."));
        scene_from_slice(&bytes, Some(parent_dir))
    }

    #[cfg(not(feature = "gltf"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "gltf",
            context: "glTF scene decoding",
        })
    }
}

/// Decode a glTF or GLB blob already held in memory.
///
/// `base` is the directory used to resolve external buffer / image URIs.
/// Pass `None` when the caller has no filesystem context (in-memory tests,
/// packed bundles, network resolvers); external references will then fail
/// to resolve with [`IoError::Parse`]. Self-contained GLBs and embedded
/// data URIs work regardless of `base`.
pub fn scene_from_slice(data: &[u8], base: Option<&Path>) -> Result<IoScene, IoError> {
    #[cfg(feature = "gltf")]
    {
        let gltf = gltf::Gltf::from_slice_without_validation(data)
            .map_err(|error| IoError::Parse(format!("glTF load failed: {error:?}")))?;
        let blob = gltf.blob.clone();
        let buffers = gltf::import_buffers(&gltf, base, blob)
            .map_err(|error| IoError::Parse(format!("glTF buffers failed: {error:?}")))?;
        let document = gltf.document;
        let images = gltf::import_images(&document, base, &buffers).unwrap_or_default();

        // External texture URIs are resolved relative to `base`. When `base`
        // is `None` we fall back to the current working directory only as a
        // last resort; data URIs and embedded textures still work either way.
        let texture_base = base.unwrap_or(Path::new("."));

        let materials = document
            .materials()
            .map(|material| convert_material(&material, &images, texture_base))
            .collect();

        // Reject skeletons that overflow the fixed-size skinning palette
        // before any decoding work commits. The per-vertex joint index type
        // is `[u8; 4]`, so anything past MAX_JOINTS cannot be referenced
        // anyway.
        for skin in document.skins() {
            let count = skin.joints().count();
            if count > MAX_JOINTS {
                return Err(IoError::Parse(format!(
                    "skin '{}' has {count} joints, exceeds MAX_JOINTS = {MAX_JOINTS}",
                    skin.name().unwrap_or("<unnamed>"),
                )));
            }
        }

        // Skeletons must be built before meshes/animations so the index map
        // (glTF node index -> joint index within a skeleton) is available.
        let (skeletons, joint_lookup) = convert_skeletons(&document, &buffers);

        let mut meshes = Vec::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    &buffers,
                    glam::Mat4::IDENTITY,
                    None,
                    &joint_lookup,
                    None,
                    &mut meshes,
                );
            }
        }

        // glTF uses right-handed Y-up; viewport-lib-io emits right-handed
        // Z-up. The conversion happens once, here, on every piece of data
        // that carries an orientation: vertex positions, normals, tangents,
        // per-mesh transforms, joint inverse-bind matrices, and animation
        // samples (see convert_skeletons / convert_animations). After this
        // point the entire IoScene is in Z-up; downstream consumers do not
        // re-rotate.
        for mesh in &mut meshes {
            reorient_mesh_z_up(mesh);
        }

        let animations = convert_animations(&document, &buffers, &joint_lookup);
        let morph_animations = convert_morph_animations(&document, &buffers);

        Ok(IoScene {
            meshes,
            materials,
            skeletons,
            animations,
            morph_animations,
            ..IoScene::default()
        })
    }

    #[cfg(not(feature = "gltf"))]
    {
        let _ = (data, base);
        Err(IoError::MissingFeature {
            feature: "gltf",
            context: "glTF scene decoding",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_io_testkit::synth::temp_dir;

    #[cfg(feature = "gltf")]
    #[test]
    fn accepts_material_only_required_extensions() {
        let dir = temp_dir("gltf_required_extensions");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "extensionsRequired": ["KHR_materials_unlit"],
  "extensionsUsed": ["KHR_materials_unlit"],
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [{
    "extensions": {
      "KHR_materials_unlit": {}
    }
  }],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.meshes.len(), 1);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn scene_from_slice_matches_scene_from_path() {
        // Reuse the same fixture-building pattern as the skin test but at a
        // smaller scale: one triangle, no skin, no materials.
        let dir = temp_dir("gltf_slice_vs_path");
        let gltf_path = dir.join("triangle.gltf");
        let bin_path = dir.join("triangle.bin");

        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "triangle.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let from_path = scene_from_path(&gltf_path).unwrap();
        let bytes = std::fs::read(&gltf_path).unwrap();
        let from_slice = scene_from_slice(&bytes, Some(dir.as_path())).unwrap();

        assert_eq!(from_path.meshes.len(), from_slice.meshes.len());
        let a = &from_path.meshes[0].mesh.positions;
        let b = &from_slice.meshes[0].mesh.positions;
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            for k in 0..3 {
                assert!((pa[k] - pb[k]).abs() < 1e-6);
            }
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn scene_from_slice_without_base_rejects_external_buffer() {
        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "external.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" }
  ]
}"#;
        let err = scene_from_slice(json.as_bytes(), None)
            .expect_err("external buffer should fail to resolve without base");
        match err {
            IoError::Parse(msg) => assert!(
                msg.contains("buffers") || msg.contains("external") || msg.contains("Uri"),
                "expected a buffer-resolution error, got: {msg}",
            ),
            other => panic!("expected IoError::Parse, got {other:?}"),
        }
    }

    // --- Weight normalisation ---
}
