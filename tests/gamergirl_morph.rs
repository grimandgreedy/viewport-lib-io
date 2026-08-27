//! End-to-end check that the FBX loader reads a **legacy** (pre-7.5) ARKit facial
//! rig, where the blend shapes are nested `Shape` sub-nodes inside the geometry
//! rather than modern `BlendShape` deformer objects. The GamerGirl character
//! (FBX 7.1) stores its 52 ARKit shapes this way.
//!
//! The asset is a large licensed file that is not committed, so this test is
//! **gated on the asset being present locally** — it skips (rather than fails)
//! when the FBX is absent, mirroring the local-only `morph_face` witness. Point
//! it at a copy with the `GAMERGIRL_FBX` environment variable, or drop the file
//! at the Portingale sibling path below. The format-independent slicing logic is
//! covered by the committed unit tests in `loaders::fbx`.

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

/// Resolve the GamerGirl FBX: the `GAMERGIRL_FBX` override, else the Portingale
/// sibling checkout. `None` when neither is present (the CI / no-asset case).
fn gamergirl_fbx() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("GAMERGIRL_FBX") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.join("../Portingale/assets/GamerGirl/Base_mesh/SK_GamerGirl_Nude.fbx");
    candidate.is_file().then_some(candidate)
}

#[test]
fn gamergirl_legacy_arkit_rig_imports_its_blend_shapes() {
    let Some(path) = gamergirl_fbx() else {
        eprintln!(
            "skipping: GamerGirl FBX not found (set GAMERGIRL_FBX or check out Portingale as a sibling)"
        );
        return;
    };

    let scene =
        viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("decode the GamerGirl FBX");

    // The head mesh carries the bulk of the ARKit set as nested legacy shapes.
    let head = scene
        .meshes
        .iter()
        .find(|m| m.name.to_uppercase().contains("HEAD"))
        .expect("a head mesh");
    assert!(
        head.mesh.morph_targets.len() >= 50,
        "head mesh should carry ~51 ARKit shapes, found {}",
        head.mesh.morph_targets.len(),
    );

    // Every target is a dense per-vertex delta set that actually moves the mesh.
    for t in &head.mesh.morph_targets {
        assert_eq!(
            t.position_deltas.len(),
            head.mesh.positions.len(),
            "target '{}' covers every base vertex",
            t.name,
        );
        assert!(
            t.position_deltas.iter().any(|d| d != &[0.0, 0.0, 0.0]),
            "target '{}' should move some vertices",
            t.name,
        );
    }

    // Across the whole character the full ARKit vocabulary is present (the head
    // holds all but tongueOut, which sits on the teeth mesh).
    let all_names: std::collections::HashSet<&str> = scene
        .meshes
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
        "expected the full ARKit set across the character, matched {present}/52",
    );

    // The scene as a whole reads far more than one mesh's worth of shapes.
    let total: usize = scene
        .meshes
        .iter()
        .map(|m| m.mesh.morph_targets.len())
        .sum();
    assert!(
        total >= 80,
        "expected ~88 morph targets across the character, found {total}",
    );
}
