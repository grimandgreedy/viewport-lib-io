//! Properties decoded data has to satisfy, whatever format it came from.
//!
//! A check collects every violation rather than stopping at the first, so one
//! run of a loader tells you everything wrong with its output instead of one
//! thing at a time.

use crate::types::{
    AnimationTrackValues, AttributeDomain, AttributeValues, PointSet, SceneData, Skeleton,
    SurfaceMesh,
};

/// One broken invariant: what was being checked, which rule it broke, and the
/// numbers that show it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// What was checked: a file name, a mesh name, or a path like
    /// `hero.glb / mesh[2] / morph_target[0]`.
    pub subject: String,
    /// Short name of the rule, stable enough to grep for.
    pub rule: &'static str,
    /// The specifics: counts, indices, whatever makes the failure actionable.
    pub detail: String,
}

/// Everything wrong with one decode. Empty means the data is consistent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// Violations found, in the order they were checked.
    pub violations: Vec<Violation>,
}

impl Report {
    /// Nothing wrong.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    /// Panic with every violation listed, or return quietly. The usual way to
    /// use a report from a test.
    #[track_caller]
    pub fn expect_clean(&self) {
        if self.is_clean() {
            return;
        }
        let mut message = format!("{} invariant violations:\n", self.violations.len());
        for v in &self.violations {
            message.push_str(&format!("  [{}] {}: {}\n", v.rule, v.subject, v.detail));
        }
        panic!("{message}");
    }

    /// Fold another report into this one.
    pub fn merge(&mut self, other: Report) {
        self.violations.extend(other.violations);
    }

    fn push(&mut self, subject: &str, rule: &'static str, detail: impl Into<String>) {
        self.violations.push(Violation {
            subject: subject.to_string(),
            rule,
            detail: detail.into(),
        });
    }

    fn check(&mut self, ok: bool, subject: &str, rule: &'static str, detail: impl Into<String>) {
        if !ok {
            self.push(subject, rule, detail);
        }
    }
}

/// Check one mesh in isolation: index bounds, per-vertex array lengths, unit
/// normals, and the skin and morph data hanging off it.
///
/// `subject` names the mesh in any violation, so pass something that locates it
/// in the file (`"hero.glb / mesh[2]"` rather than `"mesh"`).
pub fn check_surface_mesh(subject: &str, mesh: &SurfaceMesh) -> Report {
    let mut report = Report::default();
    let vertices = mesh.positions.len();

    report.check(
        mesh.indices.len() % 3 == 0,
        subject,
        "indices_are_triangles",
        format!("{} indices is not a multiple of 3", mesh.indices.len()),
    );

    if let Some(&max) = mesh.indices.iter().max() {
        report.check(
            (max as usize) < vertices,
            subject,
            "index_in_bounds",
            format!("index {max} addresses {vertices} vertices"),
        );
    }

    // Every per-vertex array is parallel to positions. A mismatch here is the
    // shape of bug that renders as scrambled shading rather than as a crash.
    let mut parallel = vec![("normals", mesh.normals.len())];
    if let Some(uvs) = &mesh.uvs {
        parallel.push(("uvs", uvs.len()));
    }
    if let Some(tangents) = &mesh.tangents {
        parallel.push(("tangents", tangents.len()));
    }
    if let Some(colours) = &mesh.colours {
        parallel.push(("colours", colours.len()));
    }
    for (name, len) in parallel {
        // Normals are allowed to be absent entirely; a partial set is not.
        if name == "normals" && len == 0 {
            continue;
        }
        report.check(
            len == vertices,
            subject,
            "per_vertex_length",
            format!("{name} has {len} entries for {vertices} vertices"),
        );
    }

    for (i, n) in mesh.normals.iter().enumerate() {
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if !length.is_finite() || (length - 1.0).abs() > 1e-3 {
            report.push(
                subject,
                "normal_is_unit",
                format!("normal[{i}] has length {length}"),
            );
            break; // one example is enough to act on
        }
    }

    if let Some(skin) = &mesh.skin_weights {
        report.check(
            skin.joint_indices.len() == vertices,
            subject,
            "per_vertex_length",
            format!(
                "joint_indices has {} entries for {vertices} vertices",
                skin.joint_indices.len()
            ),
        );
        report.check(
            skin.joint_weights.len() == skin.joint_indices.len(),
            subject,
            "skin_arrays_parallel",
            format!(
                "{} weight sets against {} index sets",
                skin.joint_weights.len(),
                skin.joint_indices.len()
            ),
        );
        for (i, w) in skin.joint_weights.iter().enumerate() {
            let sum: f32 = w.iter().sum();
            if (sum - 1.0).abs() > 1e-3 {
                report.push(
                    subject,
                    "skin_weights_sum_to_one",
                    format!("vertex {i} weights sum to {sum}"),
                );
                break;
            }
        }
    }

    for (t, target) in mesh.morph_targets.iter().enumerate() {
        let target_subject = format!("{subject} / morph_target[{t}]");
        report.check(
            target.position_deltas.len() == vertices,
            &target_subject,
            "per_vertex_length",
            format!(
                "{} position deltas for {vertices} vertices",
                target.position_deltas.len()
            ),
        );
        for (name, len) in [
            ("normal_deltas", target.normal_deltas.as_ref().map(Vec::len)),
            (
                "tangent_deltas",
                target.tangent_deltas.as_ref().map(Vec::len),
            ),
        ] {
            if let Some(len) = len {
                report.check(
                    len == target.position_deltas.len(),
                    &target_subject,
                    "morph_deltas_parallel",
                    format!(
                        "{name} has {len} entries against {} position deltas",
                        target.position_deltas.len()
                    ),
                );
            }
        }
    }

    for (name, attribute) in &mesh.attributes {
        // Only point-domain attributes are parallel to vertices; cell, face,
        // and corner attributes are counted against topology this check does
        // not reconstruct.
        if attribute.domain != AttributeDomain::Point {
            continue;
        }
        let len = match &attribute.values {
            AttributeValues::Scalars(v) => v.len(),
            AttributeValues::Colors(v) => v.len(),
            AttributeValues::Vectors(v) => v.len(),
        };
        report.check(
            len == vertices,
            subject,
            "attribute_length",
            format!("point attribute '{name}' has {len} values for {vertices} vertices"),
        );
    }

    report
}

/// Check one skeleton: joints in topological order, parents in range.
pub fn check_skeleton(subject: &str, skeleton: &Skeleton) -> Report {
    let mut report = Report::default();
    for (i, joint) in skeleton.joints.iter().enumerate() {
        let Some(parent) = joint.parent else {
            continue;
        };
        report.check(
            (parent as usize) < skeleton.joints.len(),
            subject,
            "joint_parent_in_bounds",
            format!(
                "joint[{i}] '{}' names parent {parent} of {} joints",
                joint.name,
                skeleton.joints.len()
            ),
        );
        // Downstream code builds world transforms in one pass, which only works
        // if a parent is already resolved by the time its child is reached.
        report.check(
            (parent as usize) < i,
            subject,
            "joints_topologically_ordered",
            format!("joint[{i}] '{}' names parent {parent}", joint.name),
        );
    }
    report
}

/// Check a whole decoded scene: every mesh, every skeleton, and the indices
/// that tie them together.
pub fn check_scene(subject: &str, scene: &SceneData) -> Report {
    let mut report = Report::default();

    for (i, mesh) in scene.meshes.iter().enumerate() {
        let mesh_subject = format!("{subject} / mesh[{i}] '{}'", mesh.name);
        report.merge(check_surface_mesh(&mesh_subject, &mesh.mesh));

        if let Some(material) = mesh.material_index {
            report.check(
                material < scene.materials.len(),
                &mesh_subject,
                "material_index_resolves",
                format!("material {material} of {}", scene.materials.len()),
            );
        }
        if let Some(parent) = mesh.parent_index {
            report.check(
                parent < scene.meshes.len(),
                &mesh_subject,
                "parent_index_resolves",
                format!("parent {parent} of {} meshes", scene.meshes.len()),
            );
            report.check(
                parent != i,
                &mesh_subject,
                "parent_is_not_self",
                format!("mesh {i} is its own parent"),
            );
        }
        if let Some(skeleton_index) = mesh.skeleton_index {
            report.check(
                skeleton_index < scene.skeletons.len(),
                &mesh_subject,
                "skeleton_index_resolves",
                format!("skeleton {skeleton_index} of {}", scene.skeletons.len()),
            );
            // A skinned mesh's joint indices have to address that skeleton.
            if let (Some(skin), Some(skeleton)) =
                (&mesh.mesh.skin_weights, scene.skeletons.get(skeleton_index))
            {
                let joints = skeleton.joints.len();
                if let Some(max) = skin.joint_indices.iter().flatten().max() {
                    report.check(
                        (*max as usize) < joints,
                        &mesh_subject,
                        "joint_index_in_bounds",
                        format!("joint index {max} against {joints} joints"),
                    );
                }
            }
        }
    }

    for (i, skeleton) in scene.skeletons.iter().enumerate() {
        report.merge(check_skeleton(
            &format!("{subject} / skeleton[{i}] '{}'", skeleton.name),
            skeleton,
        ));
    }

    for (i, clip) in scene.animations.iter().enumerate() {
        let clip_subject = format!("{subject} / animation[{i}] '{}'", clip.name);
        report.check(
            clip.skeleton_index < scene.skeletons.len(),
            &clip_subject,
            "skeleton_index_resolves",
            format!(
                "skeleton {} of {}",
                clip.skeleton_index,
                scene.skeletons.len()
            ),
        );
        let joints = scene
            .skeletons
            .get(clip.skeleton_index)
            .map_or(0, |s| s.joints.len());

        for (t, track) in clip.tracks.iter().enumerate() {
            let track_subject = format!("{clip_subject} / track[{t}]");
            report.check(
                track.joint < joints,
                &track_subject,
                "track_joint_in_bounds",
                format!("joint {} of {joints}", track.joint),
            );

            let times = &track.sampler.times;
            report.check(
                !times.is_empty(),
                &track_subject,
                "sampler_has_keyframes",
                "no keyframe times",
            );
            if times.windows(2).any(|w| w[1] < w[0]) {
                report.push(
                    &track_subject,
                    "sampler_times_increase",
                    "keyframe times go backwards",
                );
            }
            if let Some(&last) = times.last() {
                report.check(
                    last <= clip.duration + 1e-3,
                    &track_subject,
                    "duration_covers_keyframes",
                    format!("last key at {last} past duration {}", clip.duration),
                );
            }

            // One value per keyframe, or three under cubic-spline (in-tangent,
            // value, out-tangent).
            let per_key = match track.sampler.interpolation {
                crate::types::AnimationInterpolation::CubicSpline => 3,
                _ => 1,
            };
            let values = match &track.sampler.values {
                AnimationTrackValues::Vec3(v) => v.len(),
                AnimationTrackValues::Quat(v) => v.len(),
            };
            report.check(
                values == times.len() * per_key,
                &track_subject,
                "sampler_values_match_times",
                format!("{values} values for {} times", times.len()),
            );
        }
    }

    for (i, clip) in scene.morph_animations.iter().enumerate() {
        let clip_subject = format!("{subject} / morph_animation[{i}] '{}'", clip.name);
        // `mesh_index` is the source format's mesh index, not a position in
        // `scene.meshes`, so it is not checked here: the two only coincide for
        // formats that emit one mesh per source mesh.
        report.check(
            clip.target_count > 0,
            &clip_subject,
            "morph_clip_has_targets",
            "target_count is zero",
        );
        report.check(
            clip.weights.len() == clip.times.len() * clip.target_count,
            &clip_subject,
            "weights_match_times_by_targets",
            format!(
                "{} weights for {} times of {} targets",
                clip.weights.len(),
                clip.times.len(),
                clip.target_count
            ),
        );
        if clip.times.windows(2).any(|w| w[1] < w[0]) {
            report.push(
                &clip_subject,
                "sampler_times_increase",
                "keyframe times go backwards",
            );
        }
        if let Some(&last) = clip.times.last() {
            report.check(
                last <= clip.duration + 1e-3,
                &clip_subject,
                "duration_covers_keyframes",
                format!("last key at {last} past duration {}", clip.duration),
            );
        }
    }

    report
}

/// Check a point set: per-point arrays parallel to the positions.
pub fn check_point_set(subject: &str, points: &PointSet) -> Report {
    let mut report = Report::default();
    let count = points.positions.len();

    for (name, len) in [
        ("colors", points.colors.len()),
        ("scalars", points.scalars.len()),
    ] {
        // Both are optional in practice: empty means the source carried none.
        if len == 0 {
            continue;
        }
        report.check(
            len == count,
            subject,
            "per_point_length",
            format!("{name} has {len} entries for {count} points"),
        );
    }

    for (name, values) in &points.scalar_attributes {
        report.check(
            values.len() == count,
            subject,
            "per_point_length",
            format!(
                "attribute '{name}' has {} values for {count} points",
                values.len()
            ),
        );
    }

    report
}
