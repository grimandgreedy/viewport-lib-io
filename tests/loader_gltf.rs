//! glTF conformance, over a synthetic GLB and the committed Fox asset. The
//! module's own unit tests cover the conversion details; this is the floor plus
//! the two entrypoints agreeing.

#![cfg(feature = "gltf")]

use std::path::{Path, PathBuf};

use viewport_lib_io::loaders::gltf;
use viewport_lib_io_testkit::{conformance, synth};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A single triangle as a self-contained GLB: three positions, no indices.
fn triangle_glb() -> Vec<u8> {
    let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut bin = Vec::new();
    for p in positions {
        for v in p {
            bin.extend_from_slice(&v.to_le_bytes());
        }
    }
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,
        "scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
        "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}}}}]}}],
        "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
        "min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}}],
        "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{}}}],
        "buffers":[{{"byteLength":{}}}]}}"#,
        bin.len(),
        bin.len()
    );
    synth::glb(json.as_bytes(), &bin)
}

#[test]
fn a_synthetic_glb_conforms() {
    let blob = triangle_glb();
    let scene = conformance::scene("triangle.glb", || gltf::scene_from_slice(&blob, None));

    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.meshes[0].mesh.positions.len(), 3);
}

#[test]
fn the_fox_asset_conforms() {
    let path = fixture("fox.glb");
    let scene = conformance::scene("fox.glb", || gltf::scene_from_path(&path));

    assert!(!scene.meshes.is_empty(), "the fox has geometry");
    assert!(!scene.skeletons.is_empty(), "and a rig");
    assert!(!scene.animations.is_empty(), "and clips");
}

#[test]
fn the_two_entrypoints_agree() {
    let path = fixture("fox.glb");
    let bytes = std::fs::read(&path).expect("read fixture");

    let from_path = gltf::scene_from_path(&path).expect("from path");
    let from_slice = gltf::scene_from_slice(&bytes, path.parent()).expect("from slice");

    let names = |s: &viewport_lib_io::types::SceneData| {
        s.meshes.iter().map(|m| m.name.clone()).collect::<Vec<_>>()
    };
    assert_eq!(names(&from_path), names(&from_slice));
}
