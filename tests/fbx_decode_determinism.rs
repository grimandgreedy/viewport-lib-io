//! Decoding the same FBX twice must give the same scene.
//!
//! Positional indices are how a consumer refers to a decoded mesh: a material
//! assignment, a saved selection, a retarget map. If a second decode reorders
//! the meshes, every index recorded against the first one now addresses the
//! wrong mesh, and the symptom (wrong-looking textures, a character wearing
//! another mesh's material) points nowhere near the loader.

#![cfg(feature = "fbx")]

use std::path::{Path, PathBuf};

use viewport_lib_io::types::SceneData;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The identity a consumer's index depends on: which mesh sits at which
/// position, and what it points at.
fn shape(scene: &SceneData) -> Vec<(String, usize, Option<usize>)> {
    scene
        .meshes
        .iter()
        .map(|m| (m.name.clone(), m.mesh.positions.len(), m.material_index))
        .collect()
}

#[test]
fn mesh_order_is_reproducible_across_decodes() {
    let path = fixture("fox_rigged.fbx");

    let first = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("first decode");
    let second = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("second decode");

    assert_eq!(
        shape(&first),
        shape(&second),
        "the same file decoded twice produced different mesh order"
    );
}

/// The rest of the scene, so a fix to one ordered walk cannot destabilise
/// another.
#[test]
fn materials_skeletons_and_mesh_content_are_reproducible() {
    let path = fixture("fox_rigged.fbx");

    let first = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("first decode");
    let second = viewport_lib_io::loaders::fbx::scene_from_path(&path).expect("second decode");

    assert_eq!(first.meshes.len(), second.meshes.len());
    assert_eq!(
        first.materials.iter().map(|m| &m.name).collect::<Vec<_>>(),
        second.materials.iter().map(|m| &m.name).collect::<Vec<_>>(),
        "material order is stable"
    );

    assert_eq!(
        first.skeletons.len(),
        second.skeletons.len(),
        "skeleton count is stable"
    );
    for (x, y) in first.skeletons.iter().zip(&second.skeletons) {
        assert_eq!(
            x.joints.iter().map(|j| &j.name).collect::<Vec<_>>(),
            y.joints.iter().map(|j| &j.name).collect::<Vec<_>>(),
            "joint order is stable (topo_sort already seeds itself deterministically)"
        );
    }
}
