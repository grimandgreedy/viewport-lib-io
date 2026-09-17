//! PLY conformance across both entrypoints: the scene loader for a PLY with
//! faces, and the point-set loader for one without.

#![cfg(feature = "ply")]

use viewport_lib_io::loaders::{ply_points, ply_scene};
use viewport_lib_io::testkit::{conformance, synth};

#[test]
fn ascii_mesh_conforms() {
    let path = synth::temp_path("ply_ascii", "quad.ply");
    synth::write(&path, synth::ply_ascii_quad(false));

    let scene = conformance::scene("quad_ascii.ply", || ply_scene::scene_from_path(&path));

    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(
        scene.meshes[0].mesh.positions.len(),
        4,
        "indexed, not fanned"
    );
    assert_eq!(scene.meshes[0].mesh.indices.len(), 6);
}

#[test]
fn binary_mesh_conforms_and_matches_ascii() {
    let dir = synth::temp_dir("ply_binary");
    let ascii = dir.join("quad_ascii.ply");
    let binary = dir.join("quad_binary.ply");
    synth::write(&ascii, synth::ply_ascii_quad(false));
    synth::write(&binary, synth::ply_binary_quad());

    let from_binary = conformance::scene("quad_binary.ply", || ply_scene::scene_from_path(&binary));
    let from_ascii = ply_scene::scene_from_path(&ascii).expect("ascii ply");

    assert_eq!(
        from_binary.meshes[0].mesh.positions, from_ascii.meshes[0].mesh.positions,
        "the same quad either encoding"
    );
}

#[test]
fn a_face_less_ply_conforms_as_points() {
    let path = synth::temp_path("ply_points", "cloud.ply");
    synth::write(&path, synth::ply_ascii_points(true));

    let points = conformance::points("cloud.ply", || ply_points::point_cloud_from_path(&path));

    assert_eq!(points.positions.len(), 4);
    assert_eq!(points.colors.len(), 4, "per-vertex colours are carried");
}

#[test]
fn the_point_loader_refuses_a_mesh() {
    // Faces mean the caller wants the other entrypoint, and saying so beats
    // returning a point cloud that silently drops the topology.
    let path = synth::temp_path("ply_refuse", "quad.ply");
    synth::write(&path, synth::ply_ascii_quad(false));

    assert!(ply_points::point_cloud_from_path(&path).is_err());
}
