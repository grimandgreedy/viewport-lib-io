//! The checks have to be worth trusting before anything leans on them: every
//! invariant fires on data that breaks it, and none fires on data that does
//! not.
//!
//! That the fixture builders produce decodable files is covered where it
//! belongs, in the per-loader conformance tests.

use viewport_lib_io::types::{SkinWeights, SurfaceMesh};
use viewport_lib_io_testkit::{check_surface_mesh, synth};

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
