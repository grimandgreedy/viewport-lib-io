//! End-to-end check that the FBX loader reads a **legacy** (pre-7.5) ARKit facial
//! rig, where the blend shapes are nested `Shape` sub-nodes inside the geometry
//! rather than modern `BlendShape` deformer objects.
//!
//! No such rig is committed: they are large and typically licensed. The test is
//! gated on one being present locally and **skips rather than fails** when it is
//! absent. Point it at a copy with the `VPIO_LEGACY_ARKIT_FBX` environment
//! variable. The format-independent slicing logic is covered by the committed
//! unit tests in `loaders::fbx`.

#![cfg(feature = "fbx")]

use std::path::PathBuf;

/// The 52 canonical ARKit / Live-Link face blend-shape names. A legacy FBX facial
/// rig authored for ARKit names its shapes from this set.
const ARKIT_BLEND_SHAPES: [&str; 52] = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
];

/// Resolve the rig from `VPIO_LEGACY_ARKIT_FBX`. `None` when it is unset or does
/// not name a file (the CI / no-asset case).
fn legacy_arkit_fbx() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("VPIO_LEGACY_ARKIT_FBX")?);
    p.is_file().then_some(p)
}

#[test]
fn legacy_arkit_rig_imports_its_blend_shapes() {
    let Some(path) = legacy_arkit_fbx() else {
        eprintln!("skipping: set VPIO_LEGACY_ARKIT_FBX to a legacy ARKit FBX rig to run this test");
        return;
    };

    let scene = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("decode the FBX rig");

    // Some mesh in the rig carries nested legacy shapes at all: this is the
    // reader under test, and a rig that decodes with none of them is the
    // failure this file exists to catch.
    let with_targets: Vec<_> = scene
        .meshes
        .iter()
        .filter(|m| !m.mesh.morph_targets.is_empty())
        .collect();
    assert!(
        !with_targets.is_empty(),
        "no mesh carried morph targets; the legacy nested Shape nodes were not read",
    );

    // Every target on every such mesh is a dense per-vertex delta set that
    // actually moves its mesh. Nothing here depends on how the rig splits its
    // face across meshes, or on what those meshes are called.
    for m in &with_targets {
        for t in &m.mesh.morph_targets {
            assert_eq!(
                t.position_deltas.len(),
                m.mesh.positions.len(),
                "target '{}' on mesh '{}' covers every base vertex",
                t.name,
                m.name,
            );
            assert!(
                t.position_deltas.iter().any(|d| d != &[0.0, 0.0, 0.0]),
                "target '{}' on mesh '{}' should move some vertices",
                t.name,
                m.name,
            );
        }
    }

    // The ARKit vocabulary survives decoding, counted across the whole rig
    // rather than per mesh: a rig may hold the set on one mesh or spread it
    // over head, teeth, and eyes, and both are correct.
    let all_names: std::collections::HashSet<&str> = with_targets
        .iter()
        .flat_map(|m| m.mesh.morph_targets.iter())
        .map(|t| t.name.as_str())
        .collect();
    let present = ARKIT_BLEND_SHAPES
        .iter()
        .filter(|n| all_names.contains(**n))
        .count();
    assert!(
        present >= 50,
        "expected the full ARKit set across the rig, matched {present}/52",
    );
}
