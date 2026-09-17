//! The testkit has to be worth trusting before anything leans on it: every
//! builder's output decodes through its loader and passes the invariants, and
//! every invariant actually fires on data that breaks it.

#![cfg(feature = "testkit")]

use viewport_lib_io::testkit::{check_point_set, check_scene, check_surface_mesh, synth};
use viewport_lib_io::types::{SkinWeights, SurfaceMesh};

fn quad_mesh() -> SurfaceMesh {
    SurfaceMesh {
        positions: synth::QUAD_POSITIONS.to_vec(),
        normals: vec![[0.0, 0.0, 1.0]; 4],
        indices: synth::QUAD_INDICES.to_vec(),
        ..SurfaceMesh::default()
    }
}

#[test]
fn a_consistent_mesh_is_clean() {
    check_surface_mesh("quad", &quad_mesh()).expect_clean();
}

#[test]
fn out_of_bounds_index_is_caught() {
    let mut mesh = quad_mesh();
    mesh.indices[0] = 4; // one past the last vertex
    let report = check_surface_mesh("quad", &mesh);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.rule == "index_in_bounds")
    );
}

#[test]
fn index_count_not_a_multiple_of_three_is_caught() {
    let mut mesh = quad_mesh();
    mesh.indices.pop();
    let report = check_surface_mesh("quad", &mesh);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.rule == "indices_are_triangles")
    );
}

#[test]
fn a_short_per_vertex_array_is_caught() {
    let mut mesh = quad_mesh();
    mesh.uvs = Some(vec![[0.0, 0.0]; 3]); // three uvs for four vertices
    let report = check_surface_mesh("quad", &mesh);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.rule == "per_vertex_length")
    );
}

#[test]
fn an_unnormalised_normal_is_caught() {
    let mut mesh = quad_mesh();
    mesh.normals[2] = [0.0, 0.0, 4.0];
    let report = check_surface_mesh("quad", &mesh);
    assert!(report.violations.iter().any(|v| v.rule == "normal_is_unit"));
}

#[test]
fn skin_weights_that_do_not_sum_to_one_are_caught() {
    let mut mesh = quad_mesh();
    mesh.skin_weights = Some(SkinWeights {
        joint_indices: vec![[0, 0, 0, 0]; 4],
        joint_weights: vec![[0.5, 0.25, 0.0, 0.0]; 4],
    });
    let report = check_surface_mesh("quad", &mesh);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.rule == "skin_weights_sum_to_one")
    );
}

#[test]
fn a_clean_report_lists_nothing() {
    let report = check_surface_mesh("quad", &quad_mesh());
    assert!(report.is_clean());
    assert!(report.violations.is_empty());
}

#[cfg(feature = "obj")]
#[test]
fn obj_builder_decodes_and_is_consistent() {
    let dir = synth::temp_dir("synth_obj");
    let obj = dir.join("quad.obj");
    synth::write(&obj, synth::obj_quad(Some("quad.mtl")));
    synth::write(&dir.join("quad.mtl"), synth::mtl_quad());

    let scene = viewport_lib_io::loaders::obj::scene_from_path(&obj).expect("decode obj");
    check_scene("quad.obj", &scene).expect_clean();
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.meshes[0].mesh.indices.len(), 6, "two triangles");
}

#[cfg(feature = "stl")]
#[test]
fn stl_builders_decode_and_agree_with_each_other() {
    let dir = synth::temp_dir("synth_stl");
    let ascii = dir.join("quad_ascii.stl");
    let binary = dir.join("quad_binary.stl");
    synth::write(&ascii, synth::stl_ascii_quad());
    synth::write(&binary, synth::stl_binary_quad());

    let from_ascii = viewport_lib_io::loaders::stl::mesh_from_path(&ascii).expect("ascii stl");
    let from_binary = viewport_lib_io::loaders::stl::mesh_from_path(&binary).expect("binary stl");
    check_surface_mesh("quad_ascii.stl", &from_ascii).expect_clean();
    check_surface_mesh("quad_binary.stl", &from_binary).expect_clean();

    // The same quad either way: STL stores triangles unindexed, so six vertices.
    assert_eq!(from_ascii.positions.len(), 6);
    assert_eq!(from_ascii.positions, from_binary.positions);
}

#[cfg(feature = "csv")]
#[test]
fn csv_builder_decodes_and_is_consistent() {
    let path = synth::temp_path("synth_csv", "points.csv");
    synth::write(&path, synth::csv_points(8, "density"));

    let points = viewport_lib_io::loaders::csv::point_cloud_from_path(&path).expect("decode csv");
    check_point_set("points.csv", &points).expect_clean();
    assert_eq!(points.positions.len(), 8);
}

#[cfg(feature = "vtk")]
#[test]
fn vtk_builder_decodes_and_is_consistent() {
    let path = synth::temp_path("synth_vtk", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let datasets = viewport_lib_io::loaders::vtk::datasets_from_path(&path).expect("decode vtk");
    assert!(!datasets.is_empty(), "one dataset per piece");
    for (i, dataset) in datasets.iter().enumerate() {
        if let Some(mesh) = dataset.as_surface_mesh() {
            check_surface_mesh(&format!("quad.vtk / dataset[{i}]"), mesh).expect_clean();
        }
        if let Some(points) = dataset.as_point_set() {
            check_point_set(&format!("quad.vtk / dataset[{i}]"), points).expect_clean();
        }
    }
}

#[cfg(feature = "svg")]
#[test]
fn svg_builder_decodes() {
    let art = viewport_lib_io::loaders::svg::vector_from_bytes(
        synth::svg_filled_path("M0 0 H10 V10 H0 Z", "#ff0000").as_bytes(),
    )
    .expect("decode svg");
    assert_eq!(art.shapes.len(), 1);
}

#[cfg(feature = "ply")]
#[test]
fn ply_builders_decode_and_agree_with_each_other() {
    let dir = synth::temp_dir("synth_ply");
    let ascii = dir.join("quad_ascii.ply");
    let binary = dir.join("quad_binary.ply");
    synth::write(&ascii, synth::ply_ascii_quad(false));
    synth::write(&binary, synth::ply_binary_quad());

    let from_ascii =
        viewport_lib_io::loaders::ply_scene::scene_from_path(&ascii).expect("ascii ply");
    let from_binary =
        viewport_lib_io::loaders::ply_scene::scene_from_path(&binary).expect("binary ply");
    check_scene("quad_ascii.ply", &from_ascii).expect_clean();
    check_scene("quad_binary.ply", &from_binary).expect_clean();

    let a = &from_ascii.meshes[0].mesh;
    let b = &from_binary.meshes[0].mesh;
    assert_eq!(a.positions, b.positions, "same quad either encoding");
    assert_eq!(a.indices, b.indices);
}

#[cfg(feature = "ply")]
#[test]
fn ply_vertex_colours_survive_as_a_point_set() {
    let path = synth::temp_path("synth_ply_points", "coloured.ply");
    synth::write(&path, synth::ply_ascii_points(true));

    let points =
        viewport_lib_io::loaders::ply_points::point_cloud_from_path(&path).expect("ply points");
    check_point_set("coloured.ply", &points).expect_clean();
    assert_eq!(points.positions.len(), 4);
    assert_eq!(points.colors.len(), 4, "per-vertex colours are carried");
}

#[cfg(feature = "numpy")]
#[test]
fn npy_builder_decodes_as_a_volume() {
    let path = synth::temp_path("synth_npy", "field.npy");
    let values: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    synth::write(&path, synth::npy_f32(&[2, 3, 4], &values));

    let volume = viewport_lib_io::loaders::numpy::volume_from_path(&path).expect("decode npy");
    assert_eq!(volume.dims, [4, 3, 2], "npy is C-ordered, so dims reverse");
    let field = volume
        .point_fields
        .values()
        .next()
        .expect("the array lands as a point field");
    assert_eq!(field.len(), values.len());
}

#[cfg(feature = "gltf")]
#[test]
fn glb_builder_decodes() {
    // A single triangle: three positions in the buffer, no indices.
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
    let blob = synth::glb(json.as_bytes(), &bin);

    let scene = viewport_lib_io::loaders::gltf::scene_from_slice(&blob, None).expect("decode glb");
    check_scene("synth.glb", &scene).expect_clean();
    assert_eq!(scene.meshes.len(), 1);
    assert_eq!(scene.meshes[0].mesh.positions.len(), 3);
}
