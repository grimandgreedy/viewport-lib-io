//! FBX conformance over the committed Fox asset.
//!
//! FBX has no writer in the Rust ecosystem, so unlike every other loader here
//! its fixtures cannot be built in memory: this file needs a committed asset,
//! and everything else FBX-specific is covered by pure-function unit tests
//! inside the loader.

#![cfg(feature = "fbx")]

use std::path::{Path, PathBuf};

use viewport_lib_io::loaders::fbx;
use viewport_lib_io::testkit::conformance;
use viewport_lib_io::types::SceneData;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn the_fox_asset_conforms() {
    let path = fixture("fox_rigged.fbx");
    let scene = conformance::scene("fox_rigged.fbx", || fbx::scene_from_path(&path));

    assert!(!scene.meshes.is_empty());
    assert!(!scene.skeletons.is_empty(), "the fox is rigged");
}

/// The ordering guard, in the terms a consumer depends on: which mesh sits at
/// which position and what it points at.
///
/// `fbxcel-dom` yields a document's objects from a hash map, so this is the
/// check that catches a walk that has stopped sorting them.
#[test]
fn mesh_order_is_reproducible_across_decodes() {
    let path = fixture("fox_rigged.fbx");

    let shape = |scene: &SceneData| {
        scene
            .meshes
            .iter()
            .map(|m| (m.name.clone(), m.mesh.positions.len(), m.material_index))
            .collect::<Vec<_>>()
    };

    let first = fbx::scene_from_path(&path).expect("first decode");
    let second = fbx::scene_from_path(&path).expect("second decode");

    assert_eq!(
        shape(&first),
        shape(&second),
        "the same file decoded twice produced different mesh order"
    );
}

#[test]
fn materials_skeletons_and_joints_hold_their_order() {
    let path = fixture("fox_rigged.fbx");
    let first = fbx::scene_from_path(&path).expect("first decode");
    let second = fbx::scene_from_path(&path).expect("second decode");

    assert_eq!(
        first.materials.iter().map(|m| &m.name).collect::<Vec<_>>(),
        second.materials.iter().map(|m| &m.name).collect::<Vec<_>>(),
    );
    assert_eq!(first.skeletons.len(), second.skeletons.len());
    for (a, b) in first.skeletons.iter().zip(&second.skeletons) {
        assert_eq!(
            a.joints.iter().map(|j| &j.name).collect::<Vec<_>>(),
            b.joints.iter().map(|j| &j.name).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn the_axis_policy_reaches_the_output() {
    let path = fixture("fox_rigged.fbx");

    let load = |policy| {
        fbx::scene_from_path_with_options(
            &path,
            fbx::FbxLoadOptions {
                axis_policy: policy,
                ..fbx::FbxLoadOptions::default()
            },
        )
        .expect("decode fbx")
    };
    let honour = load(fbx::AxisPolicy::HonourHeader);
    let forced = load(fbx::AxisPolicy::ForceYUpRaw);

    conformance::scene("fox_rigged.fbx (honour)", || Ok(honour.clone()));
    conformance::scene("fox_rigged.fbx (forced)", || Ok(forced.clone()));

    // The conversion rides the mesh transform, not the vertex positions, which
    // stay in local space. This fixture is authored Z-up, so honouring the
    // header converts by identity while forcing raw Y-up composes a +90 degree
    // X rotation on top of exactly the same chain.
    assert_eq!(
        honour.meshes[0].mesh.positions, forced.meshes[0].mesh.positions,
        "local geometry is the same either way"
    );

    let c = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let expected = c * honour.meshes[0].transform;
    let actual = forced.meshes[0].transform;
    let difference = (expected - actual)
        .to_cols_array()
        .iter()
        .fold(0.0f32, |worst, d| worst.max(d.abs()));
    assert!(
        difference < 1e-4,
        "forcing raw Y-up should compose R_x(90) onto the honoured transform, off by {difference}"
    );
}
