//! STL conformance. STL stores unindexed triangles with a per-face normal, so
//! the two encodings have to agree and the vertex count is three per triangle.

#![cfg(feature = "stl")]

use viewport_lib_io::loaders::stl;
use viewport_lib_io_testkit::{conformance, synth};

#[test]
fn ascii_quad_conforms() {
    let path = synth::temp_path("stl_ascii", "quad.stl");
    synth::write(&path, synth::stl_ascii_quad());

    let mesh = conformance::mesh("quad_ascii.stl", || stl::mesh_from_path(&path));

    assert_eq!(mesh.positions.len(), 6, "two triangles, unindexed");
    assert_eq!(mesh.normals.len(), 6);
}

#[test]
fn binary_quad_conforms() {
    let path = synth::temp_path("stl_binary", "quad.stl");
    synth::write(&path, synth::stl_binary_quad());

    let mesh = conformance::mesh("quad_binary.stl", || stl::mesh_from_path(&path));

    assert_eq!(mesh.positions.len(), 6);
}

#[test]
fn the_two_encodings_decode_alike() {
    let dir = synth::temp_dir("stl_parity");
    let ascii = dir.join("quad_ascii.stl");
    let binary = dir.join("quad_binary.stl");
    synth::write(&ascii, synth::stl_ascii_quad());
    synth::write(&binary, synth::stl_binary_quad());

    let from_ascii = stl::mesh_from_path(&ascii).expect("ascii");
    let from_binary = stl::mesh_from_path(&binary).expect("binary");

    assert_eq!(from_ascii.positions, from_binary.positions);
    assert_eq!(from_ascii.indices, from_binary.indices);
}
