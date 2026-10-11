//! OBJ conformance, plus what is peculiar to OBJ: an accompanying `.mtl`, and
//! the 1-based `position/uv/normal` face indices.

#![cfg(feature = "obj")]

use viewport_lib_io::loaders::obj;
use viewport_lib_io_testkit::{conformance, synth};

#[test]
fn quad_conforms() {
    let dir = synth::temp_dir("obj_conformance");
    let path = dir.join("quad.obj");
    synth::write(&path, synth::obj_quad(None));

    let scene = conformance::scene("quad.obj", || obj::scene_from_path(&path));

    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.meshes[0].mesh.indices.len(), 6, "two triangles");
}

#[test]
fn material_library_is_read() {
    let dir = synth::temp_dir("obj_mtl");
    let path = dir.join("quad.obj");
    synth::write(&path, synth::obj_quad(Some("quad.mtl")));
    synth::write(&dir.join("quad.mtl"), synth::mtl_quad());

    let scene = conformance::scene("quad.obj + mtl", || obj::scene_from_path(&path));

    assert_eq!(scene.materials.len(), 1, "the mtl contributes one material");
    assert_eq!(scene.materials[0].name, "quad");
    assert_eq!(scene.meshes[0].material_index, Some(0));
}

#[test]
fn a_file_without_normals_gets_ones_that_follow_the_winding() {
    let dir = synth::temp_dir("obj_no_normals");
    let path = dir.join("quad.obj");
    let mut text = String::new();
    for p in synth::QUAD_POSITIONS {
        text.push_str(&format!("v {} {} {}\n", p[0], p[1], p[2]));
    }
    text.push_str("f 1 2 3\nf 1 3 4\n");
    synth::write(&path, text);

    let scene = conformance::scene("quad.obj without vn", || obj::scene_from_path(&path));

    // The quad winds anticlockwise seen from +Z, so that is where it faces.
    let mesh = &scene.meshes[0].mesh;
    assert_eq!(mesh.normals.len(), mesh.positions.len());
    for n in &mesh.normals {
        assert!((n[2] - 1.0).abs() < 1e-5, "expected +Z, got {n:?}");
    }
}

#[test]
fn a_missing_file_is_an_error_not_an_empty_scene() {
    let path = synth::temp_dir("obj_missing").join("absent.obj");
    assert!(obj::scene_from_path(&path).is_err());
}
