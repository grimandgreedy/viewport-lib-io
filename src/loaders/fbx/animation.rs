//! Animation curves: joint transform tracks and blend-shape weight clips, sampled
//! out of the FBX anim graph.

use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, MorphWeightClip, Skeleton,
};
use fbxcel_dom::v7400::Document;
use std::collections::HashMap;

use super::raw::{
    objects_in_stable_order, read_f32_array, read_i64_array, read_int_property, read_vec3_property,
};
use super::skin::{build_rig_from_limbs, find_or_insert_rig};
use super::transform::euler_to_mat4;

/// One ktime tick is 1 / 46_186_158_000 second. Conventional FBX constant.
#[cfg(feature = "fbx")]
pub(super) const FBX_KTIME_PER_SECOND: f64 = 46_186_158_000.0;

#[cfg(feature = "fbx")]
pub(super) fn extract_animations(
    document: &Document,
    skeletons: &mut Vec<Skeleton>,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
) -> Vec<AnimationClip> {
    let mut anim_stacks: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = Vec::new();
    for obj in objects_in_stable_order(document) {
        let c = obj.class();
        if c == "AnimStack" || c == "AnimationStack" {
            anim_stacks.push(obj);
        }
    }
    if anim_stacks.is_empty() {
        return Vec::new();
    }

    // File-level conversion `C = axis_transform * unit_scale`: the same
    // matrix the mesh leaves and the skin inverse-binds compose with. Joint
    // WORLD transforms must become `W' = C * W`; premultiplying every ROOT
    // joint's local by C achieves that for the whole chain, so only
    // root-joint tracks are converted below. When the file-level transform
    // is identity and the unit scale is 1, all of this is a no-op.
    //
    // NOTE: skeletons and animations are file-scoped, so this is keyed to
    // the FILE-level transform from `get_axis_transform`. Under
    // `AxisPolicy::HeuristicVeto` a mesh leaf may individually veto the
    // axis transform (see `extract_node_transform`); a vetoed skinned leaf
    // paired with a converted skeleton is a pre-existing inconsistency that
    // is not addressed here.
    let conv = *axis_transform * glam::Mat4::from_scale(glam::Vec3::splat(unit_scale));
    // `axis_transform` is a pure rotation (unit scale is carried separately),
    // so its quaternion is exactly the rotation part of C.
    let conv_rot = glam::Quat::from_mat4(axis_transform).normalize();

    // Build a rig skeleton from every LimbNode/Null model in the document so
    // animation tracks have a stable joint indexing. If a skin already produced
    // a skeleton whose bone set matches, reuse it; otherwise append a new one.
    let (rig_skeleton, model_id_to_joint) = build_rig_from_limbs(document, &conv);
    if model_id_to_joint.is_empty() {
        return Vec::new();
    }

    // Find a matching existing skeleton by bone name set, otherwise append.
    let rig_skel_idx = find_or_insert_rig(skeletons, rig_skeleton);
    // Name -> idx in the final rig skeleton (post-insert), for tracks remapping.
    let rig_name_to_idx: HashMap<String, usize> = skeletons[rig_skel_idx]
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.clone(), i))
        .collect();
    // Roots of the final rig skeleton: only their local transforms need the
    // file-level conversion applied (every descendant world inherits it).
    let joint_is_root: Vec<bool> = skeletons[rig_skel_idx]
        .joints
        .iter()
        .map(|j| j.parent.is_none())
        .collect();

    // Cache per-bone PreRotation / PostRotation / RotationOrder so each
    // rotation curve sample can compose them as the FBX bone matrix does:
    //   R_local = PreR * R_anim * inverse(PostR)
    // This matches the bind-pose decomposition (which already bakes in PreR
    // via the bone's TransformLink), so at t=0 the curve agrees with bind.
    let mut bone_rot_ctx: HashMap<i64, (glam::Mat4, glam::Mat4, i32)> = HashMap::new();
    for obj in document.objects() {
        if !model_id_to_joint.contains_key(&obj.object_id().raw()) {
            continue;
        }
        let Some(props) = obj.direct_properties() else {
            continue;
        };
        let pre = read_vec3_property(&props, "PreRotation").unwrap_or(glam::Vec3::ZERO);
        let post = read_vec3_property(&props, "PostRotation").unwrap_or(glam::Vec3::ZERO);
        let order = read_int_property(&props, "RotationOrder").unwrap_or(0);
        let pre_m = euler_to_mat4(pre, 0);
        let post_inv_m = euler_to_mat4(post, 0).inverse();
        bone_rot_ctx.insert(obj.object_id().raw(), (pre_m, post_inv_m, order));
    }

    let mut clips = Vec::new();
    for stack in anim_stacks {
        let name = stack.name().unwrap_or("AnimStack").to_string();
        let mut tracks: Vec<AnimationTrack> = Vec::new();
        let mut max_t: f32 = 0.0;

        // Stack -> Layers (layers connect TO stack: layer is source, stack is dest)
        let layers: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = stack
            .source_objects()
            .filter_map(|c| c.object_handle())
            .filter(|o| {
                let c = o.class();
                c == "AnimLayer" || c == "AnimationLayer"
            })
            .collect();

        for layer in layers {
            // Layer -> CurveNodes (curvenodes connect TO layer)
            let curve_nodes: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = layer
                .source_objects()
                .filter_map(|c| c.object_handle())
                .filter(|o| {
                    let c = o.class();
                    c == "AnimCurveNode" || c == "AnimationCurveNode"
                })
                .collect();

            for cn in curve_nodes {
                // CurveNode -> Model with property label
                let target = cn.destination_objects().find_map(|c| {
                    let label = c.label()?;
                    let obj = c.object_handle()?;
                    let model_id = obj.object_id().raw();
                    let joint_idx = model_id_to_joint.get(&model_id)?;
                    let bone_name = obj.name().unwrap_or("").to_string();
                    Some((*joint_idx, bone_name, label.to_string(), model_id))
                });
                let Some((local_joint_idx, bone_name, property, model_id)) = target else {
                    continue;
                };
                let channel = match property.as_str() {
                    "Lcl Translation" => AnimationChannel::Translation,
                    "Lcl Rotation" => AnimationChannel::Rotation,
                    "Lcl Scaling" => AnimationChannel::Scale,
                    _ => continue,
                };

                // Resolve joint index in the final rig skeleton via name.
                let _ = local_joint_idx;
                let Some(&rig_joint) = rig_name_to_idx.get(&bone_name) else {
                    continue;
                };

                // CurveNode <- Curves labelled "d|X" "d|Y" "d|Z"
                let mut axis_curves: [Option<(Vec<f32>, Vec<f32>)>; 3] = [None, None, None];
                for c in cn.source_objects() {
                    let Some(label) = c.label() else { continue };
                    let axis = match label {
                        "d|X" | "d|X|X" => 0,
                        "d|Y" | "d|Y|Y" => 1,
                        "d|Z" | "d|Z|Z" => 2,
                        _ => continue,
                    };
                    let Some(obj) = c.object_handle() else {
                        continue;
                    };
                    let cc = obj.class();
                    if cc != "AnimCurve" && cc != "AnimationCurve" {
                        continue;
                    }
                    let node = obj.node();
                    let Some(times) = read_i64_array(&node, "KeyTime") else {
                        continue;
                    };
                    let Some(values) = read_f32_array(&node, "KeyValueFloat") else {
                        continue;
                    };
                    if times.is_empty() || times.len() != values.len() {
                        continue;
                    }
                    let times_s: Vec<f32> = times
                        .iter()
                        .map(|t| (*t as f64 / FBX_KTIME_PER_SECOND) as f32)
                        .collect();
                    axis_curves[axis] = Some((times_s, values));
                }
                if axis_curves.iter().all(|c| c.is_none()) {
                    continue;
                }

                // Union times across the three axes.
                let mut union_times: Vec<f32> = Vec::new();
                for c in axis_curves.iter().flatten() {
                    union_times.extend_from_slice(&c.0);
                }
                if union_times.is_empty() {
                    continue;
                }
                union_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                union_times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

                // For each axis, default-value lookup. If curve is missing,
                // hold a constant value (0 for translation/rotation, 1 for scale).
                let default = if matches!(channel, AnimationChannel::Scale) {
                    1.0
                } else {
                    0.0
                };
                let sample_axis = |a: usize, t: f32| -> f32 {
                    match &axis_curves[a] {
                        None => default,
                        Some((ts, vs)) => sample_linear(ts, vs, t),
                    }
                };

                if let Some(last) = union_times.last() {
                    if *last > max_t {
                        max_t = *last;
                    }
                }

                // Root joints carry the file-level conversion: with
                // `C = s * A` (uniform scale s, rotation A), converting the
                // root local `L0' = C * L0 = T'R'S'` decomposes per channel
                // as `t' = s * (A * t)`, `q' = q_A * q`, `s' = s * scale`
                // (scale axes stay in the joint's local frame). Non-root
                // tracks are unchanged: their worlds inherit C via the root.
                let is_root = joint_is_root.get(rig_joint).copied().unwrap_or(false);
                let values = match channel {
                    AnimationChannel::Translation => {
                        let v: Vec<glam::Vec3> = union_times
                            .iter()
                            .map(|t| {
                                let raw = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                if is_root {
                                    // C has no translation part, so this is
                                    // exactly `s * (A * raw)`.
                                    conv.transform_point3(raw)
                                } else {
                                    raw
                                }
                            })
                            .collect();
                        AnimationTrackValues::Vec3(v)
                    }
                    AnimationChannel::Scale => {
                        let v: Vec<glam::Vec3> = union_times
                            .iter()
                            .map(|t| {
                                let raw = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                if is_root { unit_scale * raw } else { raw }
                            })
                            .collect();
                        AnimationTrackValues::Vec3(v)
                    }
                    AnimationChannel::Rotation => {
                        // Compose with PreRotation / PostRotation as FBX does:
                        //   R_local = PreR * R_anim * inverse(PostR)
                        // The bind pose (derived from TransformLink) already
                        // bakes in PreR, so this composition matches it at t=0.
                        let (pre_m, post_inv_m, order) = bone_rot_ctx
                            .get(&model_id)
                            .copied()
                            .unwrap_or((glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, 0));
                        let v: Vec<glam::Quat> = union_times
                            .iter()
                            .map(|t| {
                                let deg = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                let r_anim = euler_to_mat4(deg, order);
                                // Compose PreR * R_anim * PostR^-1 FIRST:
                                // the composed value is the joint's full
                                // local rotation, which is what the root
                                // conversion `L0' = C * L0` transforms.
                                let m = pre_m * r_anim * post_inv_m;
                                let q = glam::Quat::from_mat4(&m).normalize();
                                if is_root {
                                    (conv_rot * q).normalize()
                                } else {
                                    q
                                }
                            })
                            .collect();
                        AnimationTrackValues::Quat(v)
                    }
                };

                tracks.push(AnimationTrack {
                    joint: rig_joint,
                    channel,
                    sampler: AnimationSampler {
                        interpolation: AnimationInterpolation::Linear,
                        times: union_times,
                        values,
                    },
                });
            }
        }

        if tracks.is_empty() {
            continue;
        }
        clips.push(AnimationClip {
            name,
            duration: max_t,
            skeleton_index: rig_skel_idx,
            tracks,
        });
    }
    clips
}

/// Import blend-shape **weight** animation from the FBX anim graph, the morph
/// counterpart of [`extract_animations`]. Each `BlendShapeChannel` carries a
/// `DeformPercent` property (0..100) driven by one `AnimCurve`; this walks
/// `AnimStack -> AnimLayer -> AnimCurveNode` (the same connection traversal the
/// bone tracks use), keeps the curve nodes whose destination property is
/// `DeformPercent`, and keys the curve by its channel's name, which is the same
/// name [`extract_blend_shapes`] gives the morph target.
///
/// `mesh_targets` is `(scene-mesh index, target names in emitted order)` for each
/// mesh that carries blend shapes. Per animation stack, one [`MorphWeightClip`]
/// is emitted for each such mesh with at least one animated target: FBX keys each
/// channel on its own times, so the mesh's animated curves are merged onto a union
/// timeline and every target sampled at each key (an unanimated target holds `0`),
/// yielding the dense row-major `[keyframe][target]` matrix the consumers expect.
#[cfg(feature = "fbx")]
pub(super) fn extract_morph_weight_clips(
    document: &Document,
    mesh_targets: &[(usize, Vec<String>)],
) -> Vec<MorphWeightClip> {
    if mesh_targets.is_empty() {
        return Vec::new();
    }
    let anim_stacks: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> =
        objects_in_stable_order(document)
            .into_iter()
            .filter(|o| matches!(o.class(), "AnimStack" | "AnimationStack"))
            .collect();

    let mut out: Vec<MorphWeightClip> = Vec::new();
    for stack in anim_stacks {
        let stack_name = stack.name().unwrap_or("AnimStack").to_string();

        // Channel name -> its weight curve (times in seconds, values in 0..1).
        let mut curves: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
        for layer in stack
            .source_objects()
            .filter_map(|c| c.object_handle())
            .filter(|o| matches!(o.class(), "AnimLayer" | "AnimationLayer"))
        {
            for cn in layer
                .source_objects()
                .filter_map(|c| c.object_handle())
                .filter(|o| matches!(o.class(), "AnimCurveNode" | "AnimationCurveNode"))
            {
                // Keep only curve nodes driving a channel's DeformPercent; the
                // destination object is the BlendShapeChannel, named like the target.
                let channel_name = cn.destination_objects().find_map(|c| {
                    if c.label() != Some("DeformPercent") {
                        return None;
                    }
                    c.object_handle()?.name().map(str::to_string)
                });
                let Some(channel_name) = channel_name else {
                    continue;
                };

                for c in cn.source_objects() {
                    let Some(label) = c.label() else { continue };
                    if !label.starts_with("d|DeformPercent") {
                        continue;
                    }
                    let Some(obj) = c.object_handle() else {
                        continue;
                    };
                    if !matches!(obj.class(), "AnimCurve" | "AnimationCurve") {
                        continue;
                    }
                    let node = obj.node();
                    let Some(times) = read_i64_array(&node, "KeyTime") else {
                        continue;
                    };
                    let Some(values) = read_f32_array(&node, "KeyValueFloat") else {
                        continue;
                    };
                    if times.is_empty() || times.len() != values.len() {
                        continue;
                    }
                    let times_s: Vec<f32> = times
                        .iter()
                        .map(|t| (*t as f64 / FBX_KTIME_PER_SECOND) as f32)
                        .collect();
                    // FBX stores the weight as a percentage; the runtime wants 0..1.
                    let vals01: Vec<f32> = values.iter().map(|v| v / 100.0).collect();
                    curves.insert(channel_name.clone(), (times_s, vals01));
                    break;
                }
            }
        }
        if curves.is_empty() {
            continue;
        }

        for (mesh_index, names) in mesh_targets {
            if !names.iter().any(|n| curves.contains_key(n)) {
                continue;
            }
            // Union of every animated target's key times across this mesh.
            let mut union_times: Vec<f32> = Vec::new();
            for n in names {
                if let Some((ts, _)) = curves.get(n) {
                    union_times.extend_from_slice(ts);
                }
            }
            union_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            union_times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            if union_times.is_empty() {
                continue;
            }

            let target_count = names.len();
            let mut weights = Vec::with_capacity(union_times.len() * target_count);
            for &t in &union_times {
                for n in names {
                    let w = match curves.get(n) {
                        Some((ts, vs)) => sample_linear(ts, vs, t),
                        None => 0.0,
                    };
                    weights.push(w);
                }
            }
            let duration = union_times.last().copied().unwrap_or(0.0);
            out.push(MorphWeightClip {
                name: stack_name.clone(),
                duration,
                mesh_index: *mesh_index,
                target_count,
                interpolation: AnimationInterpolation::Linear,
                times: union_times,
                weights,
            });
        }
    }
    out
}

#[cfg(feature = "fbx")]
pub(super) fn sample_linear(times: &[f32], values: &[f32], t: f32) -> f32 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    if t >= *times.last().unwrap() {
        return *values.last().unwrap();
    }
    for i in 1..times.len() {
        if t <= times[i] {
            let span = times[i] - times[i - 1];
            if span <= 0.0 {
                return values[i];
            }
            let a = (t - times[i - 1]) / span;
            return values[i - 1] * (1.0 - a) + values[i] * a;
        }
    }
    *values.last().unwrap()
}
