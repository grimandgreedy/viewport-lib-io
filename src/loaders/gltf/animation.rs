//! Animation clips: skeletal, node and morph-weight.

use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, MorphWeightClip, NodeAnimationClip, NodeAnimationTrack,
};

use super::axis::{reorient_quat, reorient_scale, reorient_vec3};
use super::skin::JointLookup;

/// Convert glTF animations into [`AnimationClip`]s. Each animation is split
/// per target skeleton; channels naming nodes not in any skin are read by
/// [`convert_node_animations`] instead.
pub(super) fn convert_animations(
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

            let Some((track_channel, sampler)) = read_transform_channel(&channel, buffers) else {
                continue;
            };
            let clip_end = sampler.times.last().copied().unwrap_or(0.0);

            let entry = per_skeleton
                .entry(skeleton_idx)
                .or_insert_with(|| (Vec::new(), 0.0));
            entry.0.push(AnimationTrack {
                joint: joint_idx,
                channel: track_channel,
                sampler,
            });
            entry.1 = entry.1.max(clip_end);
        }

        let base_name = animation
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("animation_{}", animation.index()));

        // Emit in skeleton order. `per_skeleton` is a `HashMap`, so pushing
        // straight out of it put the clips of an animation that drives more
        // than one skeleton in a per-process random order, and a consumer
        // indexing `scene.animations` addressed a different clip on each run.
        let mut per_skeleton: Vec<(usize, (Vec<AnimationTrack>, f32))> =
            per_skeleton.into_iter().collect();
        per_skeleton.sort_by_key(|(skeleton_idx, _)| *skeleton_idx);

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

/// Convert the translation, rotation and scale channels of glTF animations
/// that target nodes outside every skin into [`NodeAnimationClip`]s, one per
/// animation that has any. The node index is the glTF node index, which is
/// the index into `SceneData::nodes`.
pub(super) fn convert_node_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    joint_lookup: &JointLookup,
) -> Vec<NodeAnimationClip> {
    let mut out = Vec::new();
    for animation in document.animations() {
        let mut tracks = Vec::new();
        let mut duration = 0.0f32;
        for channel in animation.channels() {
            let node = channel.target().node().index();
            if joint_lookup.contains_key(&node) {
                continue; // driven through the skeleton's clip
            }
            let Some((track_channel, sampler)) = read_transform_channel(&channel, buffers) else {
                continue;
            };
            duration = duration.max(sampler.times.last().copied().unwrap_or(0.0));
            tracks.push(NodeAnimationTrack {
                node,
                channel: track_channel,
                sampler,
            });
        }
        if tracks.is_empty() {
            continue;
        }
        out.push(NodeAnimationClip {
            name: animation
                .name()
                .map(std::borrow::ToOwned::to_owned)
                .unwrap_or_else(|| format!("animation_{}", animation.index())),
            duration,
            tracks,
        });
    }
    out
}

/// Read one translation, rotation or scale channel into a sampler, reoriented
/// into Z-up. `None` for a morph-weight channel or one without readable
/// keyframes.
///
/// Samples describe a node's local transform; they are reoriented here so a
/// player consumes them without any per-frame conversion. Translations rotate,
/// rotations conjugate, scales permute their Y and Z components (axis-aligned
/// permutation under the X-axis 90 rotation). Each conversion is linear, so
/// cubic-spline in- and out-tangents convert exactly like the values.
fn read_transform_channel(
    channel: &gltf::animation::Channel,
    buffers: &[gltf::buffer::Data],
) -> Option<(AnimationChannel, AnimationSampler)> {
    let track_channel = match channel.target().property() {
        gltf::animation::Property::Translation => AnimationChannel::Translation,
        gltf::animation::Property::Rotation => AnimationChannel::Rotation,
        gltf::animation::Property::Scale => AnimationChannel::Scale,
        gltf::animation::Property::MorphTargetWeights => return None,
    };

    let interpolation = match channel.sampler().interpolation() {
        gltf::animation::Interpolation::Step => AnimationInterpolation::Step,
        gltf::animation::Interpolation::Linear => AnimationInterpolation::Linear,
        gltf::animation::Interpolation::CubicSpline => AnimationInterpolation::CubicSpline,
    };

    let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
    let times: Vec<f32> = reader.read_inputs()?.collect();
    if times.is_empty() {
        return None;
    }

    let values = match reader.read_outputs()? {
        gltf::animation::util::ReadOutputs::Translations(iter) => AnimationTrackValues::Vec3(
            iter.map(|v| glam::Vec3::from_array(reorient_vec3(v)))
                .collect(),
        ),
        gltf::animation::util::ReadOutputs::Scales(iter) => AnimationTrackValues::Vec3(
            iter.map(|v| reorient_scale(glam::Vec3::from_array(v)))
                .collect(),
        ),
        gltf::animation::util::ReadOutputs::Rotations(iter) => AnimationTrackValues::Quat(
            iter.into_f32()
                .map(|q| reorient_quat(glam::Quat::from_array(q)))
                .collect(),
        ),
        gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => return None,
    };

    Some((
        track_channel,
        AnimationSampler {
            interpolation,
            times,
            values,
        },
    ))
}

/// Convert glTF morph-target-weight channels into [`MorphWeightClip`]s: one clip
/// per morph channel, named after its animation. Weights are scalar (no
/// orientation), so nothing is reoriented. `CubicSpline` output keeps its
/// in-tangents, values and out-tangents per keyframe, as glTF stores them.
pub(super) fn convert_morph_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Vec<MorphWeightClip> {
    let mut out: Vec<MorphWeightClip> = Vec::new();

    for animation in document.animations() {
        for (chan_idx, channel) in animation.channels().enumerate() {
            let target = channel.target();
            if !matches!(
                target.property(),
                gltf::animation::Property::MorphTargetWeights
            ) {
                continue;
            }

            let Some(mesh) = target.node().mesh() else {
                continue;
            };
            // All primitives of a mesh share the same morph target count.
            let target_count = mesh
                .primitives()
                .next()
                .map(|p| p.morph_targets().count())
                .unwrap_or(0);
            if target_count == 0 {
                continue;
            }

            let interpolation = match channel.sampler().interpolation() {
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
            let weights: Vec<f32> = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(w)) => {
                    w.into_f32().collect()
                }
                _ => continue,
            };
            if weights.len() != times.len() * target_count * interpolation.values_per_key() {
                // Malformed sampler: skip rather than mis-slice the palette.
                continue;
            }

            let duration = times.last().copied().unwrap_or(0.0);
            let name = animation
                .name()
                .map(std::borrow::ToOwned::to_owned)
                .unwrap_or_else(|| format!("morph_{chan_idx}"));

            out.push(MorphWeightClip {
                name,
                duration,
                mesh_index: mesh.index(),
                target_count,
                interpolation,
                times,
                weights,
            });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::types::{AnimationChannel, AnimationInterpolation, AnimationTrackValues};
    use viewport_lib_io_testkit::synth::temp_dir;

    #[test]
    fn loads_morph_weight_animation() {
        // Triangle with two morph targets and a weights animation over them.
        let dir = temp_dir("gltf_morph_anim");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        // Binary layout (little-endian):
        //   positions:  3 vec3 (36) @0
        //   indices:    3 u32  (12) @36
        //   target0:    3 vec3 (36) @48
        //   target1:    3 vec3 (36) @84
        //   anim times: 2 f32  (8)  @120
        //   anim wts:   4 f32  (16) @128   rows [kf0(t0,t1), kf1(t0,t1)]
        // Total: 144 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..2 {
            for v in [0.1f32, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0] {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [0.0f32, 1.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f32, 0.0, 1.0, 0.5] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bin.len(), 144);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "morph_tri" }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "targets": [{ "POSITION": 2 }, { "POSITION": 3 }]
    }]
  }],
  "animations": [{
    "name": "Smile",
    "channels": [{ "sampler": 0, "target": { "node": 0, "path": "weights" } }],
    "samplers": [{ "input": 4, "output": 5, "interpolation": "LINEAR" }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 144 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,   "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36,  "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 84,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 120, "byteLength": 8 },
    { "buffer": 0, "byteOffset": 128, "byteLength": 16 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,1] },
    { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,1] },
    { "bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0] },
    { "bufferView": 5, "componentType": 5126, "count": 4, "type": "SCALAR" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.morph_animations.len(), 1, "one morph clip expected");
        let clip = &scene.morph_animations[0];
        assert_eq!(clip.name, "Smile");
        assert_eq!(clip.mesh_index, 0);
        assert_eq!(clip.target_count, 2);
        assert!((clip.duration - 1.0).abs() < 1e-5);
        assert_eq!(clip.times, vec![0.0, 1.0]);
        // Row-major [keyframe][target]: kf0 = (0, 0), kf1 = (1.0, 0.5).
        assert_eq!(clip.weights, vec![0.0, 0.0, 1.0, 0.5]);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn cubic_morph_weights_keep_their_tangents() {
        // Triangle with two morph targets and a CUBICSPLINE weights sampler.
        //   positions:  3 vec3 (36) @0
        //   indices:    3 u32  (12) @36
        //   target0:    3 vec3 (36) @48
        //   target1:    3 vec3 (36) @84
        //   anim times: 2 f32  (8)  @120
        //   anim wts:   12 f32 (48) @128  [kf][in | value | out][target]
        // Total: 176 bytes.
        let dir = temp_dir("gltf_morph_cubic");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..2 {
            for v in [0.1f32, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0] {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        for v in [0.0f32, 1.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let weights = [
            // kf0: in-tangents, values, out-tangents
            -0.1f32, -0.2, 0.0, 0.25, 0.1, 0.2, //
            // kf1
            -0.3, -0.4, 1.0, 0.5, 0.3, 0.4,
        ];
        for v in weights {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bin.len(), 176);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "targets": [{ "POSITION": 2 }, { "POSITION": 3 }]
    }]
  }],
  "animations": [{
    "name": "Blink",
    "channels": [{ "sampler": 0, "target": { "node": 0, "path": "weights" } }],
    "samplers": [{ "input": 4, "output": 5, "interpolation": "CUBICSPLINE" }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 176 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,   "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36,  "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 84,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 120, "byteLength": 8 },
    { "buffer": 0, "byteOffset": 128, "byteLength": 48 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,1] },
    { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,1] },
    { "bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0] },
    { "bufferView": 5, "componentType": 5126, "count": 12, "type": "SCALAR" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.morph_animations.len(), 1);
        let clip = &scene.morph_animations[0];
        assert_eq!(clip.interpolation, AnimationInterpolation::CubicSpline);
        assert_eq!(clip.target_count, 2);
        assert_eq!(clip.times, vec![0.0, 1.0]);
        // Every tangent survives, in the order glTF stores them.
        assert_eq!(clip.weights, weights.to_vec());
        assert_eq!(clip.key_weights(0), Some(&[0.0, 0.25][..]));
        assert_eq!(clip.key_weights(1), Some(&[1.0, 0.5][..]));
        assert_eq!(clip.key_weights(2), None);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn reads_node_translation_channel_with_node_target() {
        // A triangle on node 1, a child of node 0, with a translation channel on
        // node 1. Neither node is a joint.
        //   positions:  3 vec3 (36) @0
        //   indices:    3 u32  (12) @36
        //   anim times: 2 f32  (8)  @48
        //   anim xyz:   2 vec3 (24) @56
        // Total: 80 bytes.
        let dir = temp_dir("gltf_node_anim");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f32, 2.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(bin.len(), 80);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [
    { "name": "rig", "children": [1] },
    { "name": "lift", "mesh": 0 }
  ],
  "meshes": [{
    "primitives": [{ "attributes": { "POSITION": 0 }, "indices": 1 }]
  }],
  "animations": [{
    "name": "Rise",
    "channels": [{ "sampler": 0, "target": { "node": 1, "path": "translation" } }],
    "samplers": [{ "input": 2, "output": 3, "interpolation": "LINEAR" }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 80 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 8 },
    { "buffer": 0, "byteOffset": 56, "byteLength": 24 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [2.0] },
    { "bufferView": 3, "componentType": 5126, "count": 2, "type": "VEC3" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert!(scene.animations.is_empty(), "no skeleton, no skeletal clip");
        assert_eq!(scene.node_animations.len(), 1);
        let clip = &scene.node_animations[0];
        assert_eq!(clip.name, "Rise");
        assert!((clip.duration - 2.0).abs() < 1e-6);
        assert_eq!(clip.tracks.len(), 1);

        let track = &clip.tracks[0];
        assert_eq!(track.node, 1);
        assert_eq!(scene.nodes[track.node].name.as_deref(), Some("lift"));
        assert_eq!(scene.nodes[track.node].parent, Some(0));
        assert_eq!(scene.nodes[track.node].meshes, vec![0]);
        assert_eq!(track.channel, AnimationChannel::Translation);
        assert_eq!(track.sampler.interpolation, AnimationInterpolation::Linear);
        assert_eq!(track.sampler.times, vec![0.0, 2.0]);
        // glTF +Y (up) becomes Z-up +Z.
        match &track.sampler.values {
            AnimationTrackValues::Vec3(values) => {
                assert_eq!(values.len(), 2);
                assert!(values[1].abs_diff_eq(glam::Vec3::Z, 1e-6));
            }
            other => panic!("expected Vec3 values, got {other:?}"),
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    // --- Z-up reorientation helpers ---
}
