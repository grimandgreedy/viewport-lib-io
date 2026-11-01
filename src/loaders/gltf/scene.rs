//! The document walk: buffers and images first, then materials, skeletons,
//! meshes, animations and lights, each handed to the module that owns it.

use std::path::Path;

use crate::error::IoError;
use crate::types::{IoScene, MAX_JOINTS};

use super::animation::{convert_animations, convert_morph_animations, convert_node_animations};
use super::axis::reorient_mesh_z_up;
use super::material::convert_material;
use super::node::{collect_node, convert_lights, convert_nodes};
use super::skin::convert_skeletons;

/// Decode a glTF or GLB blob into a scene. `base` resolves external buffer and
/// image URIs.
pub(super) fn build_scene(data: &[u8], base: Option<&Path>) -> Result<IoScene, IoError> {
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

    let mut nodes = convert_nodes(&document);
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
                &mut nodes,
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
    let node_animations = convert_node_animations(&document, &buffers, &joint_lookup);
    let morph_animations = convert_morph_animations(&document, &buffers);
    // Node transforms are reoriented as they are read, so the lights'
    // world transforms come out Z-up.
    let lights = convert_lights(&document, &nodes);

    Ok(IoScene {
        meshes,
        materials,
        skeletons,
        animations,
        morph_animations,
        nodes,
        lights,
        node_animations,
        ..IoScene::default()
    })
}
