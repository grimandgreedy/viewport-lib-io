//! Cross-format parity: one mesh, authored in every format that can express it,
//! must decode to the same thing.
//!
//! This is the strongest shape of test available to a decoding crate, because
//! it checks each loader against an independent description rather than against
//! a transcript nobody re-derives. `testkit::synth` emits the same
//! two-triangle quad for every format on purpose, so the comparison is free.
//!
//! What the formats legitimately differ on is stated here rather than worked
//! around: STL stores unindexed triangles and so has six vertices where the
//! indexed formats have four, and normals are per-format (authored, per-face,
//! or computed) so only their direction is compared.

#![cfg(all(
    feature = "obj",
    feature = "ply",
    feature = "stl",
    feature = "msh",
    feature = "vtk"
))]

use std::path::Path;

use viewport_lib_io::loaders::{msh, obj, ply_scene, stl, vtk};
use viewport_lib_io::types::SurfaceMesh;
use viewport_lib_io_testkit::synth;

/// A mesh flattened to its triangle soup: three positions per triangle, in
/// emitted order. This is the form every format can be compared in, indexed or
/// not.
fn triangles(mesh: &SurfaceMesh) -> Vec<[[f32; 3]; 3]> {
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            [
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            ]
        })
        .collect()
}

/// Write the quad in every format and decode each one back.
fn decode_every_format() -> Vec<(&'static str, SurfaceMesh)> {
    let dir = synth::temp_dir("cross_format_parity");
    let write = |name: &str, bytes: &[u8]| {
        let path = dir.join(name);
        synth::write(&path, bytes);
        path
    };

    let obj_path = write("quad.obj", synth::obj_quad(None).as_bytes());
    let ply_ascii = write("quad_ascii.ply", synth::ply_ascii_quad(false).as_bytes());
    let ply_binary = write("quad_binary.ply", &synth::ply_binary_quad());
    let stl_ascii = write("quad_ascii.stl", synth::stl_ascii_quad().as_bytes());
    let stl_binary = write("quad_binary.stl", &synth::stl_binary_quad());
    let msh_v2 = write("quad_v2.msh", synth::msh_v2_quad().as_bytes());
    let msh_v4 = write("quad_v4.msh", synth::msh_v4_quad().as_bytes());
    let vtk_path = write("quad.vtk", synth::vtk_legacy_quad("density").as_bytes());

    let scene_mesh = |label: &'static str, path: &Path, decode: fn(&Path) -> _| {
        let scene: viewport_lib_io::types::SceneData = decode(path);
        (label, scene.meshes.into_iter().next().expect(label).mesh)
    };

    let vtk_mesh = vtk::datasets_from_path(&vtk_path)
        .expect("vtk")
        .into_iter()
        .find_map(|d| d.surface_mesh)
        .expect("the vtk piece is a surface");

    vec![
        scene_mesh("obj", &obj_path, |p| obj::scene_from_path(p).expect("obj")),
        scene_mesh("ply_ascii", &ply_ascii, |p| {
            ply_scene::scene_from_path(p).expect("ply ascii")
        }),
        scene_mesh("ply_binary", &ply_binary, |p| {
            ply_scene::scene_from_path(p).expect("ply binary")
        }),
        (
            "stl_ascii",
            stl::mesh_from_path(&stl_ascii).expect("stl ascii"),
        ),
        (
            "stl_binary",
            stl::mesh_from_path(&stl_binary).expect("stl binary"),
        ),
        ("msh_v2", msh::mesh_from_path(&msh_v2).expect("msh 2.2")),
        ("msh_v4", msh::mesh_from_path(&msh_v4).expect("msh 4.1")),
        ("vtk", vtk_mesh),
    ]
}

/// Eight decodes of the same quad through five loaders and seven encodings, all
/// producing the same two triangles in the same order.
#[test]
fn every_format_decodes_the_same_quad() {
    let decoded = decode_every_format();
    let expected = triangles(&decoded[0].1);
    assert_eq!(expected.len(), 2, "the fixture is two triangles");

    for (label, mesh) in &decoded {
        assert_eq!(
            triangles(mesh),
            expected,
            "{label} disagrees with {} on the geometry",
            decoded[0].0
        );
    }
}

/// The indexed formats share the quad's corner between its two triangles; STL
/// cannot, so it repeats it. That is the only vertex-count difference between
/// them, and it is the format's, not the loader's.
#[test]
fn only_stl_is_unindexed() {
    for (label, mesh) in decode_every_format() {
        let expected = if label.starts_with("stl") { 6 } else { 4 };
        assert_eq!(
            mesh.positions.len(),
            expected,
            "{label} should emit {expected} vertices"
        );
        assert_eq!(
            mesh.indices.len(),
            6,
            "{label} emits two triangles either way"
        );
    }
}

/// However a format supplies its normals, the quad faces +Z. A loader that
/// flipped a winding or dropped the conversion shows up here.
#[test]
fn every_format_agrees_on_which_way_the_quad_faces() {
    for (label, mesh) in decode_every_format() {
        assert_eq!(
            mesh.normals.len(),
            mesh.positions.len(),
            "{label} emits one normal per vertex"
        );
        for normal in &mesh.normals {
            assert!(normal[2] > 0.99, "{label} should face +Z, got {normal:?}");
        }
    }
}
