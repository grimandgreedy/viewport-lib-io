//! Animation clips and morph-weight clips.

use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, MorphWeightClip,
};

use super::axis::{reorient_quat, reorient_scale, reorient_vec3};
use super::skin::JointLookup;

/// Convert glTF animations into [`AnimationClip`]s. Each animation is split
/// per target skeleton (channels naming nodes not in any skin are skipped).
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

            let gltf_channel = match target.property() {
                gltf::animation::Property::Translation => AnimationChannel::Translation,
                gltf::animation::Property::Rotation => AnimationChannel::Rotation,
                gltf::animation::Property::Scale => AnimationChannel::Scale,
                // Morph-weight channels drive a mesh, not a joint: handled
                // separately by `convert_morph_animations`.
                gltf::animation::Property::MorphTargetWeights => continue,
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

/// Convert glTF morph-target-weight channels into [`MorphWeightClip`]s: one clip
/// per morph channel, named after its animation. Weights are scalar (no
/// orientation), so nothing is reoriented. `CubicSpline` output stores
/// (in-tangent, value, out-tangent) per weight; the value component is kept so
/// the clip is a plain keyframe track.
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

            let sampler = channel.sampler();
            let cubic = matches!(
                sampler.interpolation(),
                gltf::animation::Interpolation::CubicSpline
            );
            let interpolation = match sampler.interpolation() {
                gltf::animation::Interpolation::Step => AnimationInterpolation::Step,
                // CubicSpline is collapsed to its value component, so the track
                // reads as Linear downstream.
                _ => AnimationInterpolation::Linear,
            };

            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let times: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };
            if times.is_empty() {
                continue;
            }
            let raw: Vec<f32> = match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(w)) => {
                    w.into_f32().collect()
                }
                _ => continue,
            };
            let weights: Vec<f32> = if cubic {
                raw.chunks_exact(3).map(|c| c[1]).collect()
            } else {
                raw
            };
            if weights.len() != times.len() * target_count {
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

    // --- Z-up reorientation helpers ---
}
