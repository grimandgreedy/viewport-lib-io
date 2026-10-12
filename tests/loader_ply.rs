//! PLY conformance across both entrypoints: the scene loader for a PLY with
//! faces, and the point-set loader for one without.

#![cfg(feature = "ply")]

use viewport_lib_io::loaders::{ply_points, ply_scene};
use viewport_lib_io_testkit::{conformance, synth};

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

/// A mesh PLY's vertex colours land on the mesh, decoded from the file's sRGB
/// bytes to the linear channel `MeshData` holds.
#[test]
fn a_mesh_keeps_its_vertex_colours_in_linear() {
    let path = synth::temp_path("ply_colours", "tri.ply");
    synth::write(
        &path,
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n\
         property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n\
         element face 1\nproperty list uchar int vertex_indices\nend_header\n\
         0 0 0 255 0 0\n1 0 0 128 128 128\n0 1 0 0 0 255\n3 0 1 2\n",
    );

    let scene = conformance::scene("tri.ply", || ply_scene::scene_from_path(&path));
    let colours = scene.meshes[0]
        .mesh
        .vertex_colours
        .as_ref()
        .expect("the file declares colour");
    assert_eq!(colours.len(), 3);
    assert_eq!(colours[0], [1.0, 0.0, 0.0, 1.0]);
    // sRGB 128 is about 0.216 linear; a raw copy would leave it at 0.502.
    assert!((colours[1][0] - 0.2158).abs() < 1e-3, "{:?}", colours[1]);
    assert_eq!(colours[2], [0.0, 0.0, 1.0, 1.0]);
}
