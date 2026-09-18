//! Skeletons, joints, and per-vertex skin weights.

use crate::types::{Joint, Skeleton};

use super::axis::reorient_affine_mat4;
use super::node::dfs_preorder;

/// Normalise a vertex's four blend weights so they sum to 1. A vertex with
/// vanishingly small total influence (degenerate authoring, or weights that
/// got rounded to zero on quantisation) falls back to a full-weight bind to
/// joint 0 so the runtime never has to divide by zero or render a missing
/// vertex.
pub(super) fn normalise_skin_weights(w: [f32; 4]) -> [f32; 4] {
    let sum = w[0] + w[1] + w[2] + w[3];
    if sum > 1e-6 {
        let inv = 1.0 / sum;
        [w[0] * inv, w[1] * inv, w[2] * inv, w[3] * inv]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// (skeleton_index, joint_index_within_skeleton) for each glTF node that is a
/// joint of any skin. Animation channels use this to look up which joint they
/// target.
pub(super) type JointLookup = std::collections::HashMap<usize, (usize, usize)>;

/// Build one [`Skeleton`] per glTF skin. Joints are emitted in topological
/// order so each parent index is less than its own. Returns the skeletons and
/// a lookup from glTF node index to (skeleton_index, joint_index).
pub(super) fn convert_skeletons(
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

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::types::{AnimationChannel, AnimationInterpolation, AnimationTrackValues};
    use viewport_lib_io_testkit::synth::{glb as make_glb, temp_dir};

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
        //   inv_bind:  2 mat4  (128 bytes) - identity, identity
        //   anim_in:   2 f32   (8 bytes)   - [0.0, 1.0]
        //   anim_out:  2 quat  (32 bytes)  - [identity, +90deg around X]
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
}
