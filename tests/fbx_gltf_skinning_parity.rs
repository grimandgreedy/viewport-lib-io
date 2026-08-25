//! Cross-format skinning parity: the same rigged Fox asset loaded from FBX
//! and from glTF must produce matching geometry both in bind pose and when
//! posed by sampling the "Walk" clip.
//!
//! The FBX loader converts mesh transforms, skeleton inverse-binds and
//! animation curves by the file-level axis transform + unit scale; the glTF
//! loader reorients everything into Z-up at load. If any one of those three
//! (mesh / binds / curves) misses the conversion, the posed geometry
//! diverges (historically the sampled pose came out ~90 degrees rotated
//! against the bind pose because curves stayed in raw file space).
//!
//! The Fox FBX fixture is a Blender export with `UpAxis = Z` and
//! `UnitScaleFactor = 100`, i.e. its raw data is already Z-up at scale 1:
//!
//! - Under [`AxisPolicy::HonourHeader`] the file-level conversion is
//!   identity, so FBX output must equal the glTF output directly.
//! - Under [`AxisPolicy::ForceYUpRaw`] the file-level conversion is a +90
//!   degree X rotation, so FBX output (mesh, binds AND curves all converted
//!   by the same file-level transform) must equal the glTF output rotated
//!   by that same +90 degrees. This is the regression test for the
//!   conversion math: before curves were converted, the sampled pose failed
//!   this by a ~90 degree bind/pose mismatch.
#![cfg(all(feature = "fbx", feature = "gltf"))]

use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec3};
use viewport_lib_io::loaders::fbx::{AxisPolicy, FbxLoadOptions};
use viewport_lib_io::types::{
    AnimationChannel, AnimationClip, AnimationSampler, AnimationTrackValues, SceneData, SceneMesh,
    Skeleton,
};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_gltf_fox() -> SceneData {
    viewport_lib_io::loaders::gltf::scene_from_path(&fixture("fox.glb")).expect("load fox.glb")
}

fn load_fbx_fox(policy: AxisPolicy) -> SceneData {
    viewport_lib_io::loaders::fbx::scene_from_path_with_options(
        &fixture("fox_rigged.fbx"),
        FbxLoadOptions {
            axis_policy: policy,
            ..FbxLoadOptions::default()
        },
    )
    .expect("load fox_rigged.fbx")
}

/// Locate the (first) skinned mesh in a scene.
fn skinned_mesh(scene: &SceneData) -> &SceneMesh {
    scene
        .meshes
        .iter()
        .find(|m| m.skeleton_index.is_some() && m.mesh.skin_weights.is_some())
        .expect("scene has no skinned mesh")
}

/// Locate a clip whose name is `name` or ends with `|name` (FBX stack names
/// are prefixed with the armature object, e.g. `root|Walk`).
fn find_clip<'a>(scene: &'a SceneData, name: &str) -> &'a AnimationClip {
    scene
        .animations
        .iter()
        .find(|c| c.name == name || c.name.ends_with(&format!("|{name}")))
        .unwrap_or_else(|| {
            let names: Vec<&str> = scene.animations.iter().map(|c| c.name.as_str()).collect();
            panic!("clip '{name}' not found; available: {names:?}")
        })
}

/// Bind-pose local transforms derived from the inverse-bind hierarchy:
/// `L_j = W_parent^-1 * W_j = inverse_bind_parent * inverse_bind_j^-1`.
fn bind_locals(skel: &Skeleton) -> Vec<Mat4> {
    skel.joints
        .iter()
        .map(|j| {
            let world = j.inverse_bind.inverse();
            match j.parent {
                Some(p) => skel.joints[p as usize].inverse_bind * world,
                None => world,
            }
        })
        .collect()
}

fn sample_vec3(sampler: &AnimationSampler, t: f32) -> Vec3 {
    let AnimationTrackValues::Vec3(values) = &sampler.values else {
        panic!("expected Vec3 sampler values");
    };
    let (i0, i1, a) = segment(&sampler.times, t);
    values[i0].lerp(values[i1], a)
}

fn sample_quat(sampler: &AnimationSampler, t: f32) -> Quat {
    let AnimationTrackValues::Quat(values) = &sampler.values else {
        panic!("expected Quat sampler values");
    };
    let (i0, i1, a) = segment(&sampler.times, t);
    let q0 = values[i0];
    let mut q1 = values[i1];
    if q0.dot(q1) < 0.0 {
        q1 = -q1;
    }
    q0.slerp(q1, a).normalize()
}

/// Keyframe segment lookup: indices of the bracketing keys and the blend
/// factor between them.
fn segment(times: &[f32], t: f32) -> (usize, usize, f32) {
    assert!(!times.is_empty());
    if t <= times[0] {
        return (0, 0, 0.0);
    }
    if t >= *times.last().unwrap() {
        let last = times.len() - 1;
        return (last, last, 0.0);
    }
    for i in 1..times.len() {
        if t <= times[i] {
            let span = times[i] - times[i - 1];
            let a = if span > 0.0 {
                (t - times[i - 1]) / span
            } else {
                1.0
            };
            return (i - 1, i, a);
        }
    }
    let last = times.len() - 1;
    (last, last, 0.0)
}

/// Compute world transforms for every joint of `skel`, posed by `clip` at
/// time `t`. Channels without a track hold their bind-pose value. The clip's
/// tracks index `clip_skel` (the skeleton the clip targets), which may be a
/// different skeleton object than `skel`; tracks are matched onto `skel` by
/// joint name.
fn posed_worlds(skel: &Skeleton, clip_skel: &Skeleton, clip: &AnimationClip, t: f32) -> Vec<Mat4> {
    let locals = bind_locals(skel);
    let mut trs: Vec<(Vec3, Quat, Vec3)> = locals
        .iter()
        .map(|l| l.to_scale_rotation_translation())
        .collect();

    let name_to_idx: std::collections::HashMap<&str, usize> = skel
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();

    for track in &clip.tracks {
        let name = clip_skel.joints[track.joint].name.as_str();
        let Some(&j) = name_to_idx.get(name) else {
            continue;
        };
        match track.channel {
            AnimationChannel::Translation => trs[j].2 = sample_vec3(&track.sampler, t),
            AnimationChannel::Rotation => trs[j].1 = sample_quat(&track.sampler, t),
            AnimationChannel::Scale => trs[j].0 = sample_vec3(&track.sampler, t),
        }
    }

    let mut worlds: Vec<Mat4> = Vec::with_capacity(skel.joints.len());
    for (i, joint) in skel.joints.iter().enumerate() {
        let (s, r, tr) = trs[i];
        let local = Mat4::from_scale_rotation_translation(s, r, tr);
        let world = match joint.parent {
            Some(p) => worlds[p as usize] * local,
            None => local,
        };
        worlds.push(world);
    }
    worlds
}

/// Linear-blend skin every vertex of `mesh`. `pre` is applied to each vertex
/// before skinning (the FBX loader stores raw file-space vertices plus a
/// node transform that carries the axis/unit conversion; glTF vertices are
/// already in scene space, so it passes identity).
fn skin_positions(mesh: &SceneMesh, skel: &Skeleton, worlds: &[Mat4], pre: Mat4) -> Vec<Vec3> {
    let sw = mesh.mesh.skin_weights.as_ref().expect("skin weights");
    let skin_mats: Vec<Mat4> = worlds
        .iter()
        .zip(&skel.joints)
        .map(|(w, j)| *w * j.inverse_bind)
        .collect();
    mesh.mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let v = pre.transform_point3(Vec3::from(*p));
            let mut acc = Vec3::ZERO;
            let mut wsum = 0.0;
            for k in 0..4 {
                let w = sw.joint_weights[i][k];
                if w > 0.0 {
                    let j = sw.joint_indices[i][k] as usize;
                    acc += w * skin_mats[j].transform_point3(v);
                    wsum += w;
                }
            }
            if wsum > 1e-6 { acc / wsum } else { v }
        })
        .collect()
}

/// Skin the scene's skinned mesh posed by its "Walk" clip at 0.3 x duration.
/// `pre_from_mesh` selects whether the mesh node transform is applied to the
/// vertices before skinning (FBX convention) or not (glTF convention).
fn walk_pose_points(scene: &SceneData, pre_from_mesh: bool) -> Vec<Vec3> {
    let mesh = skinned_mesh(scene);
    let skel = &scene.skeletons[mesh.skeleton_index.unwrap()];
    let clip = find_clip(scene, "Walk");
    let clip_skel = &scene.skeletons[clip.skeleton_index];
    let t = 0.3 * clip.duration;
    let worlds = posed_worlds(skel, clip_skel, clip, t);
    let pre = if pre_from_mesh {
        mesh.transform
    } else {
        Mat4::IDENTITY
    };
    skin_positions(mesh, skel, &worlds, pre)
}

fn bbox(points: impl IntoIterator<Item = Vec3>) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in points {
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

fn gltf_bind_bbox_diag(glb: &SceneData) -> f32 {
    let mesh = skinned_mesh(glb);
    let (min, max) = bbox(mesh.mesh.positions.iter().map(|p| Vec3::from(*p)));
    (max - min).length()
}

fn assert_bbox_close(label: &str, a: (Vec3, Vec3), b: (Vec3, Vec3), tol: f32) {
    for axis in 0..3 {
        let dmin = (a.0[axis] - b.0[axis]).abs();
        let dmax = (a.1[axis] - b.1[axis]).abs();
        assert!(
            dmin <= tol && dmax <= tol,
            "{label}: bbox mismatch on axis {axis} (min diff {dmin:.3}, max diff {dmax:.3}, \
             tol {tol:.3})\n  a: {:?}..{:?}\n  b: {:?}..{:?}",
            a.0,
            a.1,
            b.0,
            b.1
        );
    }
}

/// Assertion 4 of the regression spec: the FBX bind-pose mesh (raw vertices
/// through the mesh node transform; skin matrices are identity at bind by
/// construction) must occupy the same bbox as the glTF bind-pose mesh.
#[test]
fn fbx_bind_pose_matches_gltf() {
    let glb = load_gltf_fox();
    let fbx = load_fbx_fox(AxisPolicy::HonourHeader);

    let glb_mesh = skinned_mesh(&glb);
    let fbx_mesh = skinned_mesh(&fbx);

    let glb_bbox = bbox(glb_mesh.mesh.positions.iter().map(|p| Vec3::from(*p)));
    let fbx_bbox = bbox(
        fbx_mesh
            .mesh
            .positions
            .iter()
            .map(|p| fbx_mesh.transform.transform_point3(Vec3::from(*p))),
    );

    let diag = gltf_bind_bbox_diag(&glb);
    eprintln!("glb bind bbox: {glb_bbox:?}\nfbx bind bbox: {fbx_bbox:?}\ndiag: {diag}");
    assert_bbox_close("bind pose", fbx_bbox, glb_bbox, 0.05 * diag);
}

/// Assertion 3 of the regression spec: sampling the Walk clip at 0.3 x
/// duration and CPU-skinning must land FBX and glTF vertices in the same
/// bbox (same asset, but vertex order differs between exports).
#[test]
fn fbx_walk_pose_matches_gltf() {
    let glb = load_gltf_fox();
    let fbx = load_fbx_fox(AxisPolicy::HonourHeader);

    let glb_bbox = bbox(walk_pose_points(&glb, false));
    let fbx_bbox = bbox(walk_pose_points(&fbx, true));

    let diag = gltf_bind_bbox_diag(&glb);
    eprintln!("glb posed bbox: {glb_bbox:?}\nfbx posed bbox: {fbx_bbox:?}\ndiag: {diag}");
    assert_bbox_close("walk pose t=0.3*duration", fbx_bbox, glb_bbox, 0.05 * diag);
}

/// Regression test for the file-level conversion of skeleton binds and
/// animation curves. `ForceYUpRaw` makes the file-level transform a +90
/// degree X rotation (C = R_x(90)) even though this fixture is Z-up, so
/// every stage — mesh transform, inverse binds (`B' = B * C^-1`), and
/// root-joint animation curves (`L0' = C * L0`) — must carry the SAME
/// conversion. The result must equal the glTF pose rotated by that same
/// C. Before animation curves were converted, the sampled pose stayed in
/// raw file space and this failed with a ~90 degree bind/pose mismatch.
#[test]
fn fbx_forced_axis_walk_pose_is_rotated_gltf() {
    let glb = load_gltf_fox();
    let fbx = load_fbx_fox(AxisPolicy::ForceYUpRaw);

    let c = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);

    // Bind pose under forced conversion: C * (z-up bind mesh).
    let glb_mesh = skinned_mesh(&glb);
    let glb_bind_rot = bbox(
        glb_mesh
            .mesh
            .positions
            .iter()
            .map(|p| c.transform_point3(Vec3::from(*p))),
    );
    let fbx_mesh = skinned_mesh(&fbx);
    let fbx_bind = bbox(
        fbx_mesh
            .mesh
            .positions
            .iter()
            .map(|p| fbx_mesh.transform.transform_point3(Vec3::from(*p))),
    );

    let diag = gltf_bind_bbox_diag(&glb);
    assert_bbox_close("forced-axis bind pose", fbx_bind, glb_bind_rot, 0.05 * diag);

    // Walk pose under forced conversion: C * (z-up posed mesh).
    let glb_posed_rot = bbox(
        walk_pose_points(&glb, false)
            .iter()
            .map(|p| c.transform_point3(*p)),
    );
    let fbx_posed = bbox(walk_pose_points(&fbx, true));

    eprintln!(
        "rotated glb posed bbox: {glb_posed_rot:?}\nfbx posed bbox: {fbx_posed:?}\ndiag: {diag}"
    );
    assert_bbox_close(
        "forced-axis walk pose t=0.3*duration",
        fbx_posed,
        glb_posed_rot,
        0.05 * diag,
    );
}
