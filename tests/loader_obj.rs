//! OBJ conformance, plus what is peculiar to OBJ: an accompanying `.mtl`, and
//! the 1-based `position/uv/normal` face indices.

#![cfg(feature = "obj")]

use viewport_lib_io::loaders::obj;
use viewport_lib_io::testkit::{conformance, synth};

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
fn a_missing_file_is_an_error_not_an_empty_scene() {
    let path = synth::temp_dir("obj_missing").join("absent.obj");
    assert!(obj::scene_from_path(&path).is_err());
}
