use std::path::Path;

use crate::error::IoError;
use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, IoMaterial, IoMesh, IoScene, Joint, Skeleton, SkinWeights, SurfaceMesh,
    TextureData, TextureSource,
};

/// Decode a glTF or GLB file into a CPU-side scene.
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
        let gltf = gltf::Gltf::from_slice_without_validation(&bytes)
            .map_err(|error| IoError::Parse(format!("glTF load failed ({}): {error:?}", path.display())))?;
        let blob = gltf.blob.clone();
        let buffers = gltf::import_buffers(&gltf, Some(parent_dir), blob)
            .map_err(|error| IoError::Parse(format!("glTF buffers failed ({}): {error:?}", path.display())))?;
        let document = gltf.document;
        let images = gltf::import_images(&document, Some(parent_dir), &buffers).unwrap_or_default();

        let materials = document
            .materials()
            .map(|material| convert_material(&material, &images, parent_dir))
            .collect();

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

        // glTF uses a right-handed Y-up coordinate system; viewport-lib uses
        // right-handed Z-up. Apply the standard rotation as a left-multiply
        // on every mesh's scene transform. The skinning math, vertex bakes,
        // and joint inverse-bind matrices stay in glTF coordinates -- only
        // the final render transform rotates -- so the conversion is
        // self-contained and reversible.
        for mesh in &mut meshes {
            mesh.transform = Y_UP_TO_Z_UP * mesh.transform;
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
        let _ = path;
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
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49,
            0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06,
            0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44,
            0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00,
            0x01, 0xFF, 0x89, 0x99, 0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
            0x44, 0xAE, 0x42, 0x60, 0x82,
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
        let sw = mesh.mesh.skin_weights.as_ref().expect("skin weights missing");
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

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
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
            if let Some(mut imported) = convert_primitive(&primitive, buffers, &mesh, primitive_index) {
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
        collect_node(&child, buffers, world, child_parent, joint_lookup, my_joint, out);
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

    // Skin attributes. glTF stores joint indices as either u8 or u16; we
    // normalise to u8 because the runtime substrate uses [u8; 4] today. Joint
    // indices above 255 are clamped with a warning. Phase 5 of the skeletal
    // plan will widen this to u16.
    let joint_indices_u16: Option<Vec<[u16; 4]>> = reader
        .read_joints(0)
        .map(|iter| iter.into_u16().collect());
    let joint_weights: Option<Vec<[f32; 4]>> = reader
        .read_weights(0)
        .map(|iter| iter.into_f32().collect());
    let skin_weights = match (joint_indices_u16, joint_weights) {
        (Some(ji), Some(jw)) => Some(SkinWeights {
            joint_indices: ji
                .into_iter()
                .map(|q| [q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8])
                .collect(),
            joint_weights: jw,
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
    let pbr = material.pbr_metallic_roughness();
    let base_color_factor = pbr.base_color_factor();
    let base_color = [base_color_factor[0], base_color_factor[1], base_color_factor[2]];

    let base_color_texture = pbr
        .base_color_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
    let normal_map_texture = material
        .normal_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
    let ao_texture = material
        .occlusion_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("material_{}", material.index().unwrap_or(0))),
        base_color,
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        opacity: base_color_factor[3],
        base_color_texture,
        normal_map_texture,
        ao_texture,
    }
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

/// Rotation that maps right-handed Y-up (glTF) into right-handed Z-up
/// (viewport-lib). +90 degrees around X: Y -> Z, Z -> -Y, X unchanged.
const Y_UP_TO_Z_UP: glam::Mat4 = glam::Mat4::from_cols(
    glam::Vec4::new(1.0, 0.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
    glam::Vec4::new(0.0, -1.0, 0.0, 0.0),
    glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
);

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
            .map(|iter| {
                iter.map(|m| glam::Mat4::from_cols_array_2d(&m)).collect()
            })
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
                node_parent
                    .get(&node.index())
                    .and_then(|p| {
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
            let parent = parent_in_skin[skin_pos]
                .map(|p| skin_pos_to_joint[p] as u8);
            let inverse_bind = inverse_binds
                .get(skin_pos)
                .copied()
                .unwrap_or(glam::Mat4::IDENTITY);
            joints.push(Joint {
                name: node.name().unwrap_or_default().to_string(),
                parent,
                inverse_bind,
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

            let values = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    AnimationTrackValues::Vec3(iter.map(glam::Vec3::from).collect())
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    AnimationTrackValues::Vec3(iter.map(glam::Vec3::from).collect())
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    AnimationTrackValues::Quat(
                        iter.into_f32().map(glam::Quat::from_array).collect(),
                    )
                }
                _ => continue,
            };

            let entry = per_skeleton.entry(skeleton_idx).or_insert_with(|| (Vec::new(), 0.0));
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
