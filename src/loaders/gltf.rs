use std::path::Path;

use crate::error::IoError;
use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AlphaMode, AnimationTrackValues, IoMaterial, IoMesh, IoScene, Joint, Skeleton, SkinWeights,
    SurfaceMesh, TextureData, TextureSource, MAX_JOINTS,
};

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

        Ok(IoScene {
            meshes,
            materials,
            skeletons,
            animations,
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

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "viewport_lib_io_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn loads_external_base_color_texture() {
        let dir = temp_dir("gltf_external_texture");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");
        let png_path = dir.join("albedo.png");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&png_path, png_bytes).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
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
    "pbrMetallicRoughness": {
      "baseColorTexture": { "index": 0 }
    }
  }],
  "textures": [{ "sampler": 0, "source": 0 }],
  "samplers": [{}],
  "images": [{ "uri": "albedo.png" }],
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
        let material = scene.materials.first().unwrap();
        match material.base_color_texture.as_ref() {
            Some(TextureSource::Decoded(image)) => {
                assert_eq!(image.width, 1);
                assert_eq!(image.height, 1);
                assert_eq!(image.rgba, vec![255, 0, 0, 255]);
            }
            other => panic!("expected decoded external texture, got {other:?}"),
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_file(png_path);
        let _ = std::fs::remove_dir(dir);
    }

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
    fn reads_emissive_alpha_mode_and_double_sided() {
        let dir = temp_dir("gltf_material_render_state");
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
  "materials": [
    {
      "name": "emissive_cutout",
      "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
      "emissiveFactor": [0.1, 0.2, 0.3],
      "alphaMode": "MASK",
      "alphaCutoff": 0.7,
      "doubleSided": true
    },
    {
      "name": "plain_opaque",
      "pbrMetallicRoughness": { "baseColorFactor": [0.5, 0.5, 0.5, 1.0] }
    }
  ],
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
        assert_eq!(scene.materials.len(), 2);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;

        let m = &scene.materials[0];
        assert!(close(m.emissive[0], 0.1) && close(m.emissive[1], 0.2) && close(m.emissive[2], 0.3));
        assert_eq!(m.alpha_mode, crate::types::AlphaMode::Mask(0.7));
        assert!(m.double_sided);

        // A plain material keeps the neutral defaults.
        let p = &scene.materials[1];
        assert_eq!(p.emissive, [0.0, 0.0, 0.0]);
        assert_eq!(p.alpha_mode, crate::types::AlphaMode::Opaque);
        assert!(!p.double_sided);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn converts_spec_gloss_materials_to_metal_rough() {
        let dir = temp_dir("gltf_spec_gloss");
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
  "extensionsUsed": ["KHR_materials_pbrSpecularGlossiness"],
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
  "materials": [
    {
      "name": "dielectric_sg",
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {
          "diffuseFactor": [0.8, 0.2, 0.2, 0.75],
          "specularFactor": [0.0, 0.0, 0.0],
          "glossinessFactor": 0.3
        }
      }
    },
    {
      "name": "metal_sg",
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {
          "diffuseFactor": [0.0, 0.0, 0.0, 1.0],
          "specularFactor": [1.0, 1.0, 1.0],
          "glossinessFactor": 0.9
        }
      }
    },
    {
      "name": "plain_mr",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.1, 0.2, 0.3, 1.0],
        "metallicFactor": 0.5,
        "roughnessFactor": 0.25
      }
    }
  ],
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
        assert_eq!(scene.materials.len(), 3);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;

        // Zero specular: metallic 0, roughness = 1 - glossiness, base colour
        // is diffuse / (1 - 0.04), opacity from the diffuse alpha.
        let d = &scene.materials[0];
        assert!(close(d.metallic, 0.0), "dielectric metallic {}", d.metallic);
        assert!(
            close(d.roughness, 0.7),
            "dielectric roughness {}",
            d.roughness
        );
        assert!(close(d.base_color[0], 0.8 / 0.96));
        assert!(close(d.base_color[1], 0.2 / 0.96));
        assert!(close(d.opacity, 0.75));

        // Full white specular, black diffuse: solves to metallic 1 with the
        // specular colour as base.
        let m = &scene.materials[1];
        assert!(close(m.metallic, 1.0), "metal metallic {}", m.metallic);
        assert!(close(m.roughness, 0.1), "metal roughness {}", m.roughness);
        assert!(close(m.base_color[0], 1.0));

        // The plain metallic-roughness material is untouched by the branch.
        let p = &scene.materials[2];
        assert!(close(p.metallic, 0.5));
        assert!(close(p.roughness, 0.25));
        assert!(close(p.base_color[2], 0.3));

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn loads_skin_weights_skeleton_and_animation() {
        // Build a minimal skinned glTF: one triangle, two joints (root and
        // child), one rotation track on the child.
        let dir = temp_dir("gltf_skin");
        let gltf_path = dir.join("rig.gltf");
        let bin_path = dir.join("rig.bin");

        // Binary layout (little-endian):
        //   positions: 3 vec3 (36 bytes)
        //   indices:   3 u32  (12 bytes)
        //   joints:    3 [u8;4] (12 bytes)
        //   weights:   3 vec4  (48 bytes)
        //   inv_bind:  2 mat4  (128 bytes) — identity, identity
        //   anim_in:   2 f32   (8 bytes)   — [0.0, 1.0]
        //   anim_out:  2 quat  (32 bytes)  — [identity, +90deg around X]
        // Total: 276 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        // Joints: all weighted to joint 1 (the child).
        for _ in 0..3 {
            bin.extend_from_slice(&[1u8, 0, 0, 0]);
        }
        // Weights: all 1.0 on slot 0.
        for _ in 0..3 {
            for v in [1.0f32, 0.0, 0.0, 0.0] {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Two inverse-bind matrices: identity, identity.
        for _ in 0..2 {
            let m = glam::Mat4::IDENTITY.to_cols_array();
            for v in m {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Animation input times.
        for v in [0.0f32, 1.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        // Animation output: two quaternions.
        let q0 = glam::Quat::IDENTITY.to_array();
        let q1 = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2).to_array();
        for q in [q0, q1] {
            for v in q {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        assert_eq!(bin.len(), 276);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0, 1] }],
  "nodes": [
    { "mesh": 0, "skin": 0, "name": "skinned_mesh" },
    { "name": "root", "children": [2] },
    { "name": "child" }
  ],
  "meshes": [{
    "primitives": [{
      "attributes": {
        "POSITION": 0,
        "JOINTS_0": 2,
        "WEIGHTS_0": 3
      },
      "indices": 1
    }]
  }],
  "skins": [{
    "name": "rig",
    "joints": [1, 2],
    "inverseBindMatrices": 4
  }],
  "animations": [{
    "name": "bend",
    "channels": [{
      "sampler": 0,
      "target": { "node": 2, "path": "rotation" }
    }],
    "samplers": [{
      "input": 5,
      "output": 6,
      "interpolation": "LINEAR"
    }]
  }],
  "buffers": [{ "uri": "rig.bin", "byteLength": 276 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,   "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36,  "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48,  "byteLength": 12 },
    { "buffer": 0, "byteOffset": 60,  "byteLength": 48 },
    { "buffer": 0, "byteOffset": 108, "byteLength": 128 },
    { "buffer": 0, "byteOffset": 236, "byteLength": 8 },
    { "buffer": 0, "byteOffset": 244, "byteLength": 32 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4" },
    { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4" },
    { "bufferView": 4, "componentType": 5126, "count": 2, "type": "MAT4" },
    { "bufferView": 5, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0], "max": [1] },
    { "bufferView": 6, "componentType": 5126, "count": 2, "type": "VEC4" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();

        // Skin weights round-tripped onto the mesh.
        let mesh = scene.meshes.first().expect("mesh missing");
        assert_eq!(mesh.skeleton_index, Some(0));
        let sw = mesh
            .mesh
            .skin_weights
            .as_ref()
            .expect("skin weights missing");
        assert_eq!(sw.joint_indices.len(), 3);
        assert_eq!(sw.joint_weights.len(), 3);
        for ji in &sw.joint_indices {
            assert_eq!(ji[0], 1, "expected all vertices weighted to child joint");
        }

        // One skeleton with two joints in topological order.
        assert_eq!(scene.skeletons.len(), 1);
        let sk = &scene.skeletons[0];
        assert_eq!(sk.joints.len(), 2);
        assert!(sk.joints[0].parent.is_none(), "root should have no parent");
        assert_eq!(sk.joints[1].parent, Some(0), "child should point at root");

        // One animation clip with one rotation track on the child joint.
        assert_eq!(scene.animations.len(), 1);
        let clip = &scene.animations[0];
        assert_eq!(clip.skeleton_index, 0);
        assert!((clip.duration - 1.0).abs() < 1e-5);
        assert_eq!(clip.tracks.len(), 1);
        let track = &clip.tracks[0];
        assert_eq!(track.joint, 1);
        assert_eq!(track.channel, AnimationChannel::Rotation);
        assert_eq!(track.sampler.interpolation, AnimationInterpolation::Linear);
        assert_eq!(track.sampler.times, vec![0.0, 1.0]);
        match &track.sampler.values {
            AnimationTrackValues::Quat(q) => {
                assert_eq!(q.len(), 2);
                assert!(q[0].dot(glam::Quat::IDENTITY).abs() > 0.9999);
            }
            _ => panic!("expected Quat values"),
        }

        // Z-up reorientation: glTF positions (0, 1, 0) and (1, 0, 0) become
        // (0, 0, 1) and (1, 0, 0) once the loader emits Z-up data.
        let positions = &mesh.mesh.positions;
        assert_eq!(positions.len(), 3);
        assert!((positions[0][0] - 0.0).abs() < 1e-5);
        assert!((positions[0][1] - 0.0).abs() < 1e-5);
        assert!((positions[0][2] - 0.0).abs() < 1e-5);
        assert!((positions[1][0] - 1.0).abs() < 1e-5);
        assert!((positions[1][1] - 0.0).abs() < 1e-5);
        assert!((positions[1][2] - 0.0).abs() < 1e-5);
        assert!((positions[2][0] - 0.0).abs() < 1e-5);
        assert!((positions[2][1] - 0.0).abs() < 1e-5);
        assert!((positions[2][2] - 1.0).abs() < 1e-5);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn loads_vertex_colours_from_color_0() {
        // Minimal triangle carrying a COLOR_0 (VEC4 f32) attribute.
        let dir = temp_dir("gltf_colour");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        // Binary layout (little-endian):
        //   positions: 3 vec3  (36 bytes) offset 0
        //   indices:   3 u32   (12 bytes) offset 36
        //   colours:   3 vec4  (48 bytes) offset 48
        // Total: 96 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let colours = [
            [1.0f32, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.5],
        ];
        for c in colours {
            for v in c {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        assert_eq!(bin.len(), 96);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "coloured_tri" }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0, "COLOR_0": 2 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 96 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 48 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let mesh = scene.meshes.first().expect("mesh missing");
        let got = mesh.mesh.colours.as_ref().expect("colours missing");
        assert_eq!(got.len(), 3);
        // Colours are direction-independent, so the Z-up reorientation leaves
        // them exactly as authored.
        for (g, e) in got.iter().zip(colours.iter()) {
            for k in 0..4 {
                assert!(
                    (g[k] - e[k]).abs() < 1e-5,
                    "colour mismatch: {g:?} vs {e:?}"
                );
            }
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    // --- Z-up reorientation helpers ---

    #[test]
    fn vec3_y_axis_maps_to_z_axis() {
        let v = reorient_vec3([0.0, 1.0, 0.0]);
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_z_axis_maps_to_negative_y_axis() {
        let v = reorient_vec3([0.0, 0.0, 1.0]);
        assert!((v[0]).abs() < 1e-6);
        assert!((v[1] + 1.0).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn vec3_x_axis_unchanged() {
        let v = reorient_vec3([1.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn tangent_xyz_rotates_but_w_preserved() {
        let t = reorient_tangent([0.0, 1.0, 0.0, -1.0]);
        assert!((t[2] - 1.0).abs() < 1e-6);
        assert!((t[3] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn quat_y_axis_rotation_becomes_z_axis_rotation() {
        let q_y_up = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let q_z_up = reorient_quat(q_y_up);
        let expected = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!(q_z_up.abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn quat_x_axis_rotation_unchanged() {
        let q = glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let r = reorient_quat(q);
        assert!(r.abs_diff_eq(q, 1e-5));
    }

    #[test]
    fn scale_swaps_y_and_z_components() {
        let s = reorient_scale(glam::Vec3::new(2.0, 3.0, 5.0));
        assert_eq!(s, glam::Vec3::new(2.0, 5.0, 3.0));
    }

    #[test]
    fn affine_translation_along_y_lands_along_z() {
        let m = glam::Mat4::from_translation(glam::Vec3::Y * 2.0);
        let m_z = reorient_affine_mat4(m);
        let p = m_z.transform_point3(glam::Vec3::ZERO);
        assert!((p - glam::Vec3::Z * 2.0).length() < 1e-5);
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

    #[test]
    fn weights_summing_above_one_renormalise() {
        let normalised = normalise_skin_weights([0.5, 0.5, 0.5, 0.5]);
        let sum: f32 = normalised.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for v in normalised {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn weights_summing_to_two_renormalise_to_one() {
        let normalised = normalise_skin_weights([1.0, 1.0, 0.0, 0.0]);
        let sum: f32 = normalised.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((normalised[0] - 0.5).abs() < 1e-6);
        assert!((normalised[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn zero_weight_vertex_falls_back_to_full_bind_on_joint_zero() {
        let normalised = normalise_skin_weights([0.0, 0.0, 0.0, 0.0]);
        assert_eq!(normalised, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn near_zero_weights_below_threshold_fall_back() {
        let normalised = normalise_skin_weights([1e-9, 1e-9, 0.0, 0.0]);
        assert_eq!(normalised, [1.0, 0.0, 0.0, 0.0]);
    }

    // --- MAX_JOINTS enforcement ---

    #[cfg(feature = "gltf")]
    #[test]
    fn skin_exceeding_max_joints_is_rejected() {
        // Build a glTF with one mesh and a skin referencing MAX_JOINTS + 1
        // joint nodes. We only need the structure to parse; the buffer can
        // be a stub since we error out before reading skinning data.
        let n_joints = MAX_JOINTS + 1;

        // Buffer: 3 positions (36 bytes), 3 indices (12 bytes), 3 joint
        // tuples (12 bytes), 3 weights (48 bytes), n_joints identity
        // inverse-bind matrices (n_joints * 64 bytes).
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..3 {
            bin.extend_from_slice(&[0u8, 0, 0, 0]);
        }
        for _ in 0..3 {
            for v in [1.0f32, 0.0, 0.0, 0.0] {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        for _ in 0..n_joints {
            for v in identity {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }

        let positions_len = 36;
        let indices_len = 12;
        let joints_len = 12;
        let weights_len = 48;
        let ib_len = n_joints * 64;
        let buffer_len = positions_len + indices_len + joints_len + weights_len + ib_len;
        assert_eq!(bin.len(), buffer_len);

        let positions_offset = 0;
        let indices_offset = positions_offset + positions_len;
        let joints_offset = indices_offset + indices_len;
        let weights_offset = joints_offset + joints_len;
        let ib_offset = weights_offset + weights_len;

        // Joint node indices: 1..=n_joints. Node 0 is the mesh node.
        let mut joint_indices: Vec<String> = Vec::with_capacity(n_joints);
        for j in 0..n_joints {
            joint_indices.push((j + 1).to_string());
        }
        let joint_list = joint_indices.join(",");

        // Each joint is its own glTF node. Keep them flat (no parents) so
        // the test focuses on the count check.
        let mut nodes_json = String::from(r#"{ "mesh": 0, "skin": 0 }"#);
        for _ in 0..n_joints {
            nodes_json.push_str(",{}");
        }

        let json = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{nodes_json}],
  "meshes": [{{
    "primitives": [{{
      "attributes": {{ "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3 }},
      "indices": 1
    }}]
  }}],
  "skins": [{{
    "joints": [{joint_list}],
    "inverseBindMatrices": 4
  }}],
  "buffers": [{{ "byteLength": {buffer_len} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {positions_offset}, "byteLength": {positions_len} }},
    {{ "buffer": 0, "byteOffset": {indices_offset},   "byteLength": {indices_len}, "target": 34963 }},
    {{ "buffer": 0, "byteOffset": {joints_offset},    "byteLength": {joints_len} }},
    {{ "buffer": 0, "byteOffset": {weights_offset},   "byteLength": {weights_len} }},
    {{ "buffer": 0, "byteOffset": {ib_offset},        "byteLength": {ib_len} }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 4, "componentType": 5126, "count": {n_joints}, "type": "MAT4" }}
  ]
}}"#
        );

        // Pack as GLB so the binary buffer travels with the JSON and the
        // loader does not need a base path for an external `.bin`.
        let glb = make_glb(json.as_bytes(), &bin);
        let err = scene_from_slice(&glb, None).expect_err("should reject oversize skin");
        match err {
            IoError::Parse(msg) => assert!(
                msg.contains("MAX_JOINTS") && msg.contains(&format!("{n_joints}")),
                "expected MAX_JOINTS error citing the joint count, got: {msg}",
            ),
            other => panic!("expected IoError::Parse, got {other:?}"),
        }
    }

    /// Pack a glTF JSON + binary buffer into a self-contained GLB blob.
    /// Used by tests that want a `scene_from_slice`-friendly fixture with
    /// no external `.bin` file on disk.
    #[cfg(feature = "gltf")]
    fn make_glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
        // glTF 2.0 GLB layout: 12-byte header, 8-byte JSON chunk header,
        // padded JSON body, 8-byte BIN chunk header, padded BIN body.
        fn pad4(len: usize) -> usize {
            (4 - (len & 3)) & 3
        }
        let json_pad = pad4(json.len());
        let bin_pad = pad4(bin.len());
        let json_len = json.len() + json_pad;
        let bin_len = bin.len() + bin_pad;
        let total = 12 + 8 + json_len + 8 + bin_len;

        let mut out = Vec::with_capacity(total);
        // Header.
        out.extend_from_slice(&0x46546C67u32.to_le_bytes()); // "glTF"
        out.extend_from_slice(&2u32.to_le_bytes()); // version
        out.extend_from_slice(&(total as u32).to_le_bytes());
        // JSON chunk.
        out.extend_from_slice(&(json_len as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        out.extend_from_slice(json);
        for _ in 0..json_pad {
            out.push(b' ');
        }
        // BIN chunk.
        out.extend_from_slice(&(bin_len as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
        out.extend_from_slice(bin);
        for _ in 0..bin_pad {
            out.push(0);
        }
        out
    }

    #[test]
    fn affine_inverse_pair_is_identity() {
        let product = Y_UP_TO_Z_UP * Y_UP_TO_Z_UP_INV;
        let i = glam::Mat4::IDENTITY;
        for c in 0..4 {
            for r in 0..4 {
                assert!((product.col(c)[r] - i.col(c)[r]).abs() < 1e-6);
            }
        }
    }
}

fn collect_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_world: glam::Mat4,
    parent_mesh_index: Option<usize>,
    joint_lookup: &JointLookup,
    parent_joint: Option<(usize, usize)>,
    out: &mut Vec<IoMesh>,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;

    // If this node is itself a joint, it overrides the inherited joint for
    // any descendant meshes. Otherwise we keep walking down with the parent's
    // joint context (so a mesh four nodes below a bone still gets attached
    // to that bone for rigid bone-parented rigs).
    let my_joint = joint_lookup.get(&node.index()).copied().or(parent_joint);

    let my_first_index = out.len();
    let mut this_node_has_mesh = false;

    if let Some(mesh) = node.mesh() {
        // glTF binds a skin to the node that references a mesh, so a real
        // skinned mesh has the skin reference on its own node. `skin indices
        // are global` and `convert_skeletons` builds skeletons in glTF skin
        // order, so we can use the glTF index directly.
        let explicit_skeleton = node.skin().map(|s| s.index());

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if let Some(mut imported) =
                convert_primitive(&primitive, buffers, &mesh, primitive_index)
            {
                if explicit_skeleton.is_some() {
                    // Real glTF skinning: JOINTS_0 / WEIGHTS_0 already
                    // populated on the primitive. Keep the mesh transform as
                    // its scene-graph world matrix.
                    imported.skeleton_index = explicit_skeleton;
                    imported.transform = world;
                } else if let Some((skeleton_idx, joint_idx)) = my_joint {
                    // Rigid bone-parented mesh: synthesize 100%-weight
                    // skinning to the nearest joint ancestor. Bake the mesh's
                    // current world transform into the vertex data so the
                    // skinned output at bind pose equals the original world
                    // position.
                    rigidly_attach_to_joint(&mut imported, world, skeleton_idx, joint_idx);
                } else {
                    imported.transform = world;
                }

                if !this_node_has_mesh {
                    imported.parent_index = parent_mesh_index;
                    this_node_has_mesh = true;
                } else {
                    imported.parent_index = Some(my_first_index);
                }
                out.push(imported);
            }
        }
    }

    let child_parent = if this_node_has_mesh {
        Some(my_first_index)
    } else {
        parent_mesh_index
    };

    for child in node.children() {
        collect_node(
            &child,
            buffers,
            world,
            child_parent,
            joint_lookup,
            my_joint,
            out,
        );
    }
}

/// Attach every vertex of `imported` rigidly to a single joint. Used for
/// glTF rigs that animate by bone-parenting meshes rather than per-vertex
/// skinning (no JOINTS_0/WEIGHTS_0, the mesh node is a descendant of a joint
/// in the scene graph).
///
/// The mesh's bind-world transform is baked into positions and normals, the
/// scene transform is set to identity, and synthesized skin weights put
/// 100% influence on `joint_index`. At bind pose the skinning matrix is
/// identity so the rendered position equals the original world position; as
/// the joint animates, the mesh follows it.
fn rigidly_attach_to_joint(
    imported: &mut IoMesh,
    world: glam::Mat4,
    skeleton_index: usize,
    joint_index: usize,
) {
    let world3 = glam::Mat3::from_mat4(world);
    // Normal matrix: transpose of inverse of the upper-left 3x3. For uniform
    // scale this is rotation only; for non-uniform scale it accounts for the
    // skew. If the matrix is singular (degenerate scale), fall back to the
    // raw upper-left and let `normalize_or_zero` clean up.
    let normal_mat = if world3.determinant().abs() > 1e-8 {
        world3.inverse().transpose()
    } else {
        world3
    };

    for p in &mut imported.mesh.positions {
        let v = world.transform_point3(glam::Vec3::from(*p));
        *p = v.to_array();
    }
    for n in &mut imported.mesh.normals {
        let v = normal_mat * glam::Vec3::from(*n);
        *n = v.normalize_or_zero().to_array();
    }

    let count = imported.mesh.positions.len();
    imported.mesh.skin_weights = Some(SkinWeights {
        joint_indices: vec![[joint_index as u8, 0, 0, 0]; count],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; count],
    });
    imported.skeleton_index = Some(skeleton_index);
    imported.transform = glam::Mat4::IDENTITY;
}

fn convert_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mesh: &gltf::Mesh,
    primitive_index: usize,
) -> Option<IoMesh> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    if positions.is_empty() {
        return None;
    }

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let normals = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| compute_vertex_normals(&positions, &indices));

    let uvs = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect());

    let tangents = reader.read_tangents().map(|iter| iter.collect());

    // COLOR_0: per-vertex colour. glTF stores it as vec3 or vec4 in u8, u16, or
    // f32; `into_rgba_f32` normalises all of them to `[f32; 4]` (expanding vec3
    // with alpha 1.0).
    let colours: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect());

    // Skin attributes. glTF stores joint indices as either u8 or u16; we
    // normalise to u8 because the runtime substrate uses [u8; 4] today. Joint
    // indices above 255 are clamped with a warning.
    let joint_indices_u16: Option<Vec<[u16; 4]>> =
        reader.read_joints(0).map(|iter| iter.into_u16().collect());
    let joint_weights: Option<Vec<[f32; 4]>> =
        reader.read_weights(0).map(|iter| iter.into_f32().collect());
    let skin_weights = match (joint_indices_u16, joint_weights) {
        (Some(ji), Some(jw)) => Some(SkinWeights {
            joint_indices: ji
                .into_iter()
                .map(|q| [q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8])
                .collect(),
            joint_weights: jw.into_iter().map(normalise_skin_weights).collect(),
        }),
        _ => None,
    };

    let material_index = primitive.material().index();

    let base_name = mesh
        .name()
        .map(std::borrow::ToOwned::to_owned)
        .unwrap_or_else(|| format!("mesh_{}", mesh.index()));
    let name = if mesh.primitives().len() > 1 {
        format!("{base_name}.{primitive_index}")
    } else {
        base_name
    };

    let mut mesh_data = SurfaceMesh::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    mesh_data.uvs = uvs;
    mesh_data.tangents = tangents;
    mesh_data.colours = colours;
    mesh_data.skin_weights = skin_weights;

    Some(IoMesh {
        name,
        mesh: mesh_data,
        material_index,
        transform: glam::Mat4::IDENTITY,
        two_sided: primitive.material().double_sided(),
        parent_index: None,
        ..IoMesh::default()
    })
}

fn convert_material(
    material: &gltf::Material,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> IoMaterial {
    // KHR_materials_pbrSpecularGlossiness assets get the reference lossy
    // conversion to metallic-roughness; everything else reads the standard
    // metallic-roughness properties.
    let (base_color, metallic, roughness, opacity, base_color_texture, metallic_roughness_texture) =
        if let Some(sg) = material.pbr_specular_glossiness() {
            let diffuse = sg.diffuse_factor();
            let (rgb, metallic) = spec_gloss_to_metal_rough(
                [diffuse[0], diffuse[1], diffuse[2]],
                sg.specular_factor(),
            );
            // The diffuse texture stands in for the base colour texture. This
            // is part of the lossy factor-level conversion: the specular-
            // glossiness texture is not converted, so per-texel specular
            // variation is dropped and only the factors steer metallic and
            // roughness. There is no metallic-roughness texture on this path.
            let texture = sg
                .diffuse_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
            (
                rgb,
                metallic,
                1.0 - sg.glossiness_factor(),
                diffuse[3],
                texture,
                None,
            )
        } else {
            let pbr = material.pbr_metallic_roughness();
            let base_color_factor = pbr.base_color_factor();
            let texture = pbr
                .base_color_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
            let mr_texture = pbr
                .metallic_roughness_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
            (
                [
                    base_color_factor[0],
                    base_color_factor[1],
                    base_color_factor[2],
                ],
                pbr.metallic_factor(),
                pbr.roughness_factor(),
                base_color_factor[3],
                texture,
                mr_texture,
            )
        };

    let emissive = material.emissive_factor();
    let emissive_texture = material
        .emissive_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(material.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };
    let double_sided = material.double_sided();

    // Read the strength scalars before consuming the texture references. Absent
    // textures leave the scalars at the glTF defaults of 1.0.
    let normal_texture = material.normal_texture();
    let normal_scale = normal_texture.as_ref().map_or(1.0, |t| t.scale());
    let normal_map_texture = normal_texture
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    let occlusion_texture = material.occlusion_texture();
    let occlusion_strength = occlusion_texture.as_ref().map_or(1.0, |t| t.strength());
    let ao_texture = occlusion_texture
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("material_{}", material.index().unwrap_or(0))),
        base_color,
        metallic,
        roughness,
        emissive,
        opacity,
        alpha_mode,
        double_sided,
        base_color_texture,
        metallic_roughness_texture,
        normal_map_texture,
        normal_scale,
        ao_texture,
        occlusion_strength,
        emissive_texture,
    }
}

/// Reference lossy conversion of specular-glossiness factors to
/// metallic-roughness, following the KHR_materials_pbrSpecularGlossiness
/// conversion published with the glTF spec: solve metallic from the specular
/// brightness (a dielectric F0 of 0.04), then reconstruct the base colour by
/// blending the diffuse- and specular-derived candidates by metallic^2.
fn spec_gloss_to_metal_rough(diffuse: [f32; 3], specular: [f32; 3]) -> ([f32; 3], f32) {
    const DIELECTRIC_SPECULAR: f32 = 0.04;
    const EPS: f32 = 1e-4;

    // Perceived brightness per the reference implementation.
    let brightness = |c: [f32; 3]| -> f32 {
        (0.299 * c[0] * c[0] + 0.587 * c[1] * c[1] + 0.114 * c[2] * c[2]).sqrt()
    };
    let max_component = |c: [f32; 3]| -> f32 { c[0].max(c[1]).max(c[2]) };

    let one_minus_specular_strength = 1.0 - max_component(specular);
    let diffuse_brightness = brightness(diffuse);
    let specular_brightness = brightness(specular);

    let metallic = if specular_brightness < DIELECTRIC_SPECULAR {
        0.0
    } else {
        let a = DIELECTRIC_SPECULAR;
        let b = diffuse_brightness * one_minus_specular_strength / (1.0 - DIELECTRIC_SPECULAR)
            + specular_brightness
            - 2.0 * DIELECTRIC_SPECULAR;
        let c = DIELECTRIC_SPECULAR - specular_brightness;
        let d = (b * b - 4.0 * a * c).max(0.0);
        ((-b + d.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
    };

    let mut base = [0.0f32; 3];
    for i in 0..3 {
        let from_diffuse = diffuse[i] * one_minus_specular_strength
            / (1.0 - DIELECTRIC_SPECULAR)
            / (1.0 - metallic).max(EPS);
        let from_specular =
            (specular[i] - DIELECTRIC_SPECULAR * (1.0 - metallic)) / metallic.max(EPS);
        base[i] =
            (from_diffuse + (from_specular - from_diffuse) * metallic * metallic).clamp(0.0, 1.0);
    }
    (base, metallic)
}

fn image_to_texture_source(
    texture: &gltf::Texture,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> Option<TextureSource> {
    let source = texture.source();
    let index = source.index();

    if let Some(data) = images.get(index) {
        return Some(TextureSource::Decoded(TextureData {
            width: data.width,
            height: data.height,
            rgba: to_rgba8(data),
        }));
    }

    match source.source() {
        gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => {
            Some(TextureSource::File(parent_dir.join(uri)))
        }
        _ => None,
    }
}

fn to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;

    match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 3 * 4);
            for rgb in data.pixels.chunks_exact(3) {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            rgba
        }
        Format::R8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
            for &r in &data.pixels {
                rgba.extend_from_slice(&[r, r, r, 255]);
            }
            rgba
        }
        Format::R8G8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 2 * 4);
            for rg in data.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            rgba
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let bytes_per_pixel = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * bytes_per_pixel * 2;
                let mut channels = [0u8; 4];
                for c in 0..bytes_per_pixel {
                    let lo = data.pixels.get(base + c * 2).copied().unwrap_or(0);
                    let hi = data.pixels.get(base + c * 2 + 1).copied().unwrap_or(0);
                    channels[c] = (u16::from_le_bytes([lo, hi]) >> 8) as u8;
                }
                match bytes_per_pixel {
                    1 => {
                        channels[1] = channels[0];
                        channels[2] = channels[0];
                        channels[3] = 255;
                    }
                    2 => {
                        channels[2] = 0;
                        channels[3] = 255;
                    }
                    3 => channels[3] = 255,
                    _ => {}
                }
                rgba.extend_from_slice(&channels);
            }
            rgba
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * channels * 4;
                let mut out = [0u8; 4];
                for c in 0..channels.min(4) {
                    let bytes = [
                        data.pixels.get(base + c * 4).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 1).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 2).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 3).copied().unwrap_or(0),
                    ];
                    let value = f32::from_le_bytes(bytes);
                    out[c] = (value.clamp(0.0, 1.0) * 255.0) as u8;
                }
                if channels < 4 {
                    out[3] = 255;
                }
                rgba.extend_from_slice(&out);
            }
            rgba
        }
    }
}

// ---------------------------------------------------------------------------
// Y-up to Z-up reorientation
//
// glTF stores everything in right-handed Y-up. viewport-lib-io exposes a
// right-handed Z-up scene. The conversion is a +90 degree rotation about the
// X-axis: Y -> Z, Z -> -Y, X unchanged. Every orientation-bearing piece of
// data (positions, normals, tangents, mesh transforms, inverse-bind matrices,
// animation samples) is rotated once at load.
// ---------------------------------------------------------------------------

/// +90 degree rotation about X as a Mat4.
const Y_UP_TO_Z_UP: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
    glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Inverse of [`Y_UP_TO_Z_UP`]: -90 degrees about X, i.e. its transpose.
const Y_UP_TO_Z_UP_INV: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, -1.0, 0.0),
    glam::Vec4::new(0.0, 1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

/// Rotate a position or direction vector from Y-up into Z-up:
/// `(x, y, z) -> (x, -z, y)`.
fn reorient_vec3(v: [f32; 3]) -> [f32; 3] {
    [v[0], -v[2], v[1]]
}

/// Rotate a tangent vec4: xyz is direction, w is bitangent sign and stays.
fn reorient_tangent(t: [f32; 4]) -> [f32; 4] {
    [t[0], -t[2], t[1], t[3]]
}

/// Conjugate an affine transform by the Y-up to Z-up rotation: `R * M * R^-1`.
fn reorient_affine_mat4(m: glam::Mat4) -> glam::Mat4 {
    Y_UP_TO_Z_UP * m * Y_UP_TO_Z_UP_INV
}

/// Y-up to Z-up rotation as a quaternion (used for animation rotation tracks).
fn y_up_to_z_up_quat() -> glam::Quat {
    glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
}

/// Conjugate a unit quaternion by the Y-up to Z-up rotation: `R * q * R^-1`.
fn reorient_quat(q: glam::Quat) -> glam::Quat {
    let r = y_up_to_z_up_quat();
    r * q * r.conjugate()
}

/// Permute the components of a per-axis scale vector under the X-axis +90
/// rotation. Anisotropic scales remain correct because the rotation is
/// axis-aligned; arbitrary-axis non-uniform scale is not representable as a
/// scale vector in any frame.
fn reorient_scale(s: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::new(s.x, s.z, s.y)
}

/// Normalise a vertex's four blend weights so they sum to 1. A vertex with
/// vanishingly small total influence (degenerate authoring, or weights that
/// got rounded to zero on quantisation) falls back to a full-weight bind to
/// joint 0 so the runtime never has to divide by zero or render a missing
/// vertex.
fn normalise_skin_weights(w: [f32; 4]) -> [f32; 4] {
    let sum = w[0] + w[1] + w[2] + w[3];
    if sum > 1e-6 {
        let inv = 1.0 / sum;
        [w[0] * inv, w[1] * inv, w[2] * inv, w[3] * inv]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Rotate a single [`IoMesh`]'s vertex attributes and world transform into
/// Z-up. Skin weights are indices and per-vertex scalars, so they need no
/// reorientation.
fn reorient_mesh_z_up(mesh: &mut IoMesh) {
    for p in &mut mesh.mesh.positions {
        *p = reorient_vec3(*p);
    }
    for n in &mut mesh.mesh.normals {
        *n = reorient_vec3(*n);
    }
    if let Some(tangents) = mesh.mesh.tangents.as_mut() {
        for t in tangents.iter_mut() {
            *t = reorient_tangent(*t);
        }
    }
    mesh.transform = reorient_affine_mat4(mesh.transform);
}

/// (skeleton_index, joint_index_within_skeleton) for each glTF node that is a
/// joint of any skin. Animation channels use this to look up which joint they
/// target.
type JointLookup = std::collections::HashMap<usize, (usize, usize)>;

/// Build one [`Skeleton`] per glTF skin. Joints are emitted in topological
/// order so each parent index is less than its own. Returns the skeletons and
/// a lookup from glTF node index to (skeleton_index, joint_index).
fn convert_skeletons(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> (Vec<Skeleton>, JointLookup) {
    let mut skeletons = Vec::new();
    let mut lookup: JointLookup = std::collections::HashMap::new();

    // Build a glTF node -> parent map. glTF only encodes children, so we walk
    // every node's children list to invert it.
    let mut node_parent: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for node in document.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }

    for skin in document.skins() {
        let skin_idx = skin.index();
        let joints_in_skin: Vec<gltf::Node> = skin.joints().collect();
        if joints_in_skin.is_empty() {
            // glTF allows empty skins in principle but the runtime cannot use
            // them. Emit an empty skeleton placeholder so skin indices line up.
            skeletons.push(Skeleton::default());
            continue;
        }

        // Set of glTF node indices that belong to this skin.
        let skin_member: std::collections::HashSet<usize> =
            joints_in_skin.iter().map(|n| n.index()).collect();

        // Read the inverse bind matrices, indexed in glTF skin-joint order.
        let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
        let inverse_binds: Vec<glam::Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(|m| glam::Mat4::from_cols_array_2d(&m)).collect())
            .unwrap_or_else(|| vec![glam::Mat4::IDENTITY; joints_in_skin.len()]);

        // Map glTF skin-joint index -> parent's glTF skin-joint index (or
        // None if the parent is not a member of this skin).
        let gltf_idx_to_skin_pos: std::collections::HashMap<usize, usize> = joints_in_skin
            .iter()
            .enumerate()
            .map(|(i, n)| (n.index(), i))
            .collect();
        let parent_in_skin: Vec<Option<usize>> = joints_in_skin
            .iter()
            .map(|node| {
                node_parent.get(&node.index()).and_then(|p| {
                    if skin_member.contains(p) {
                        gltf_idx_to_skin_pos.get(p).copied()
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Topo-order joints so parents precede children. Use DFS from each
        // root (joint with no in-skin parent), pre-order.
        let n = joints_in_skin.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, p) in parent_in_skin.iter().enumerate() {
            if let Some(pi) = p {
                children[*pi].push(i);
            }
        }
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        for root in 0..n {
            if parent_in_skin[root].is_none() {
                dfs_preorder(root, &children, &mut order, &mut visited);
            }
        }
        // Any joints not reachable from a root (e.g. cycles, which are
        // malformed) get appended at the end so we never lose data.
        for i in 0..n {
            if !visited[i] {
                order.push(i);
                visited[i] = true;
            }
        }

        // Build the final Skeleton in topo order.
        let mut skin_pos_to_joint: Vec<usize> = vec![0; n];
        for (joint_idx, &skin_pos) in order.iter().enumerate() {
            skin_pos_to_joint[skin_pos] = joint_idx;
        }

        let mut joints = Vec::with_capacity(n);
        for &skin_pos in &order {
            let node = &joints_in_skin[skin_pos];
            let parent = parent_in_skin[skin_pos].map(|p| skin_pos_to_joint[p] as u8);
            let inverse_bind_y_up = inverse_binds
                .get(skin_pos)
                .copied()
                .unwrap_or(glam::Mat4::IDENTITY);
            joints.push(Joint {
                name: node.name().unwrap_or_default().to_string(),
                parent,
                inverse_bind: reorient_affine_mat4(inverse_bind_y_up),
            });
        }

        // Populate the global lookup.
        for (skin_pos, &joint_idx) in skin_pos_to_joint.iter().enumerate() {
            let gltf_node_idx = joints_in_skin[skin_pos].index();
            lookup.insert(gltf_node_idx, (skin_idx, joint_idx));
        }

        skeletons.push(Skeleton {
            name: skin.name().unwrap_or_default().to_string(),
            joints,
        });
    }

    (skeletons, lookup)
}

fn dfs_preorder(
    node: usize,
    children: &[Vec<usize>],
    order: &mut Vec<usize>,
    visited: &mut [bool],
) {
    if visited[node] {
        return;
    }
    visited[node] = true;
    order.push(node);
    for &c in &children[node] {
        dfs_preorder(c, children, order, visited);
    }
}

/// Convert glTF animations into [`AnimationClip`]s. Each animation is split
/// per target skeleton (channels naming nodes not in any skin are skipped).
fn convert_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    joint_lookup: &JointLookup,
) -> Vec<AnimationClip> {
    let mut out: Vec<AnimationClip> = Vec::new();

    for animation in document.animations() {
        // Group tracks by skeleton index.
        let mut per_skeleton: std::collections::HashMap<usize, (Vec<AnimationTrack>, f32)> =
            std::collections::HashMap::new();

        for channel in animation.channels() {
            let target = channel.target();
            let node_idx = target.node().index();
            let &(skeleton_idx, joint_idx) = match joint_lookup.get(&node_idx) {
                Some(v) => v,
                None => continue, // channel targets a non-joint; skip
            };

            let gltf_channel = match target.property() {
                gltf::animation::Property::Translation => AnimationChannel::Translation,
                gltf::animation::Property::Rotation => AnimationChannel::Rotation,
                gltf::animation::Property::Scale => AnimationChannel::Scale,
                gltf::animation::Property::MorphTargetWeights => continue, // not supported yet
            };

            let sampler = channel.sampler();
            let interp = match sampler.interpolation() {
                gltf::animation::Interpolation::Step => AnimationInterpolation::Step,
                gltf::animation::Interpolation::Linear => AnimationInterpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => AnimationInterpolation::CubicSpline,
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let times: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            if times.is_empty() {
                continue;
            }
            let clip_end = times.last().copied().unwrap_or(0.0);

            // Animation samples describe a joint's local transform; they
            // are reoriented into Z-up here so the player can consume them
            // without any per-frame conversion. Translations rotate,
            // rotations conjugate, scales permute their Y and Z components
            // (axis-aligned permutation under the X-axis 90 rotation).
            let values = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    AnimationTrackValues::Vec3(
                        iter.map(|v| {
                            let r = reorient_vec3(v);
                            glam::Vec3::from_array(r)
                        })
                        .collect(),
                    )
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    AnimationTrackValues::Vec3(
                        iter.map(|v| reorient_scale(glam::Vec3::from_array(v)))
                            .collect(),
                    )
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    AnimationTrackValues::Quat(
                        iter.into_f32()
                            .map(|q| reorient_quat(glam::Quat::from_array(q)))
                            .collect(),
                    )
                }
                _ => continue,
            };

            let entry = per_skeleton
                .entry(skeleton_idx)
                .or_insert_with(|| (Vec::new(), 0.0));
            entry.0.push(AnimationTrack {
                joint: joint_idx,
                channel: gltf_channel,
                sampler: AnimationSampler {
                    interpolation: interp,
                    times,
                    values,
                },
            });
            entry.1 = entry.1.max(clip_end);
        }

        let base_name = animation
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("animation_{}", animation.index()));

        for (skeleton_idx, (tracks, duration)) in per_skeleton {
            if tracks.is_empty() {
                continue;
            }
            out.push(AnimationClip {
                name: base_name.clone(),
                duration,
                skeleton_index: skeleton_idx,
                tracks,
            });
        }
    }

    out
}

fn compute_vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let (i0, i1, i2) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let v0 = glam::Vec3::from_array(positions[i0]);
        let v1 = glam::Vec3::from_array(positions[i1]);
        let v2 = glam::Vec3::from_array(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0);
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }

    normals
        .into_iter()
        .map(|n| {
            let n = n.normalize_or_zero();
            [n.x, n.y, n.z]
        })
        .collect()
}
