//! The document walk: buffers and images first, then materials, skeletons,
//! meshes, animations and lights, each handed to the module that owns it.

use std::path::Path;

use crate::error::IoError;
use crate::types::{IoScene, MAX_JOINTS};

use super::animation::{convert_animations, convert_morph_animations, convert_node_animations};
use super::axis::reorient_mesh_z_up;
use super::buffers::load_buffers;
use super::material::convert_material;
use super::node::{collect_node, convert_lights, convert_nodes};
use super::options::GltfLoadOptions;
use super::skin::convert_skeletons;
use super::texture::ImageTable;

/// Decode a glTF or GLB blob into a scene. `base` resolves external buffer and
/// image URIs.
pub(super) fn build_scene(
    data: &[u8],
    base: Option<&Path>,
    options: &GltfLoadOptions,
) -> Result<IoScene, IoError> {
    let gltf = gltf::Gltf::from_slice_without_validation(data)
        .map_err(|error| IoError::Parse(format!("glTF load failed: {error:?}")))?;
    check_required_extensions(&gltf)?;
    let blob = gltf.blob.clone();
    let buffers = load_buffers(&gltf, base, blob)?;
    let document = gltf.document;
    let images = ImageTable::load(&document, base, &buffers, options);

    let materials = document
        .materials()
        .map(|material| convert_material(&material, &images))
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
                &document,
                &node,
                &buffers,
                glam::Mat4::IDENTITY,
                None,
                &joint_lookup,
                None,
                &mut meshes,
                &mut nodes,
            )?;
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

/// Refuse a file that requires a compression extension this build does not
/// decode, naming the feature that does. Other unknown required extensions are
/// let through, since most only add data the loader ignores.
fn check_required_extensions(document: &gltf::Document) -> Result<(), IoError> {
    for name in document.extensions_required() {
        let missing = match name {
            "KHR_draco_mesh_compression" if !cfg!(feature = "gltf-draco") => Some((
                "gltf-draco",
                "glTF meshes compressed with KHR_draco_mesh_compression",
            )),
            "EXT_meshopt_compression" if !cfg!(feature = "gltf-meshopt") => Some((
                "gltf-meshopt",
                "glTF buffers compressed with EXT_meshopt_compression",
            )),
            _ => None,
        };
        if let Some((feature, context)) = missing {
            return Err(IoError::MissingFeature { feature, context });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::scene_from_slice;
    use crate::error::IoError;

    /// Each compression extension, the feature that decodes it, and whether
    /// this build has that feature.
    const COMPRESSION: &[(&str, &str, bool)] = &[
        (
            "KHR_draco_mesh_compression",
            "gltf-draco",
            cfg!(feature = "gltf-draco"),
        ),
        (
            "EXT_meshopt_compression",
            "gltf-meshopt",
            cfg!(feature = "gltf-meshopt"),
        ),
    ];

    #[test]
    fn a_required_compression_extension_names_the_feature_that_decodes_it() {
        for &(extension, feature, enabled) in COMPRESSION {
            let json = format!(
                r#"{{"asset":{{"version":"2.0"}},"extensionsUsed":["{extension}"],
                "extensionsRequired":["{extension}"],"scenes":[{{"nodes":[]}}]}}"#
            );
            match scene_from_slice(json.as_bytes(), None) {
                Err(IoError::MissingFeature { feature: named, .. }) => {
                    assert!(!enabled, "{extension} refused although {feature} is on");
                    assert_eq!(named, feature);
                }
                Ok(_) => assert!(enabled, "{extension} accepted without {feature}"),
                Err(other) => panic!("{extension}: unexpected error {other:?}"),
            }
        }
    }
}
