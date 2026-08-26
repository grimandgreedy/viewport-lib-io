//! The Khronos AnimatedMorphCube (CC0) exercises glTF morph-target geometry and
//! weight-animation import end to end: two targets on the cube mesh, plus a
//! looping weight clip that blends between them.

#![cfg(feature = "gltf")]

use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn morph_cube_carries_two_targets_and_a_weight_clip() {
    let scene = viewport_lib_io::loaders::gltf::scene_from_path(&fixture("AnimatedMorphCube.glb"))
        .expect("decode AnimatedMorphCube.glb");

    // The cube mesh carries two morph targets, each a full per-vertex delta set.
    let mesh = scene
        .meshes
        .iter()
        .find(|m| !m.mesh.morph_targets.is_empty())
        .expect("a mesh with morph targets");
    assert_eq!(
        mesh.mesh.morph_targets.len(),
        2,
        "cube has two blend shapes"
    );
    for t in &mesh.mesh.morph_targets {
        assert_eq!(
            t.position_deltas.len(),
            mesh.mesh.positions.len(),
            "each target covers every base vertex",
        );
        assert!(
            t.position_deltas.iter().any(|d| d != &[0.0, 0.0, 0.0]),
            "target '{}' should move some vertices",
            t.name,
        );
    }

    // And a weight animation over those two targets.
    let clip = scene
        .morph_animations
        .first()
        .expect("a morph weight animation");
    assert_eq!(clip.target_count, 2);
    assert!(clip.duration > 0.0, "clip has a positive duration");
    assert_eq!(
        clip.weights.len(),
        clip.times.len() * clip.target_count,
        "row-major [keyframe][target] weight table",
    );
    assert!(
        clip.weights.iter().any(|&w| w > 0.5),
        "the animation drives the targets toward full weight",
    );
}
