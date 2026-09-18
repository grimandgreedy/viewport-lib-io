//! Gmsh conformance. The two format versions describe the same mesh through
//! entirely different section layouts, so both are run through the floor and
//! compared against each other.

#![cfg(feature = "msh")]

use viewport_lib_io::loaders::msh;
use viewport_lib_io_testkit::{conformance, synth};

#[test]
fn a_v2_quad_conforms() {
    let path = synth::temp_path("msh_v2", "quad.msh");
    synth::write(&path, synth::msh_v2_quad());

    let mesh = conformance::mesh("quad_v2.msh", || msh::mesh_from_path(&path));

    assert_eq!(mesh.positions.len(), 4, "the shared corner is emitted once");
    assert_eq!(mesh.indices.len(), 6, "two triangles");
}

#[test]
fn a_v4_quad_conforms() {
    let path = synth::temp_path("msh_v4", "quad.msh");
    synth::write(&path, synth::msh_v4_quad());

    let mesh = conformance::mesh("quad_v4.msh", || msh::mesh_from_path(&path));

    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(mesh.indices.len(), 6);
}

#[test]
fn the_two_format_versions_decode_alike() {
    let dir = synth::temp_dir("msh_parity");
    let v2 = dir.join("quad_v2.msh");
    let v4 = dir.join("quad_v4.msh");
    synth::write(&v2, synth::msh_v2_quad());
    synth::write(&v4, synth::msh_v4_quad());

    let from_v2 = msh::mesh_from_path(&v2).expect("2.2");
    let from_v4 = msh::mesh_from_path(&v4).expect("4.1");

    assert_eq!(from_v2.positions, from_v4.positions);
    assert_eq!(from_v2.indices, from_v4.indices);
    assert_eq!(from_v2.normals, from_v4.normals);
}

/// The quad the other loaders emit, decoded through Gmsh: same corners, same
/// two triangles. Cross-format agreement is what makes a fixture trustworthy.
#[test]
fn it_decodes_the_same_quad_as_every_other_format() {
    let path = synth::temp_path("msh_quad_parity", "quad.msh");
    synth::write(&path, synth::msh_v2_quad());

    let mesh = msh::mesh_from_path(&path).expect("decode msh");
    assert_eq!(mesh.positions, synth::QUAD_POSITIONS.to_vec());
    assert_eq!(mesh.indices, synth::QUAD_INDICES.to_vec());
}
