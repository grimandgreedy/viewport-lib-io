//! VTK conformance. A VTK file can be several things at once, so the loader
//! returns one dataset per piece and each piece is checked for whichever
//! representations it carries.

#![cfg(feature = "vtk")]

use viewport_lib_io::loaders::vtk;
use viewport_lib_io::testkit::{invariants, synth};

#[test]
fn a_legacy_unstructured_grid_conforms() {
    let path = synth::temp_path("vtk_legacy", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let datasets = vtk::datasets_from_path(&path).expect("decode vtk");
    assert!(!datasets.is_empty(), "one dataset per piece");

    for (i, dataset) in datasets.iter().enumerate() {
        let label = format!("quad.vtk / dataset[{i}]");
        if let Some(mesh) = dataset.as_surface_mesh() {
            invariants::check_surface_mesh(&label, mesh).expect_clean();
        }
        if let Some(points) = dataset.as_point_set() {
            invariants::check_point_set(&label, points).expect_clean();
        }
    }
}

#[test]
fn decoding_is_reproducible() {
    let path = synth::temp_path("vtk_determinism", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let identity = |sets: &[viewport_lib_io::types::DecodedDataSet]| {
        sets.iter()
            .map(|d| {
                (
                    d.name.clone(),
                    d.as_surface_mesh().map(|m| m.positions.len()),
                    d.as_point_set().map(|p| p.positions.len()),
                )
            })
            .collect::<Vec<_>>()
    };

    let first = vtk::datasets_from_path(&path).expect("first decode");
    let second = vtk::datasets_from_path(&path).expect("second decode");
    assert_eq!(identity(&first), identity(&second));
}

#[test]
fn the_point_field_survives() {
    let path = synth::temp_path("vtk_field", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let datasets = vtk::datasets_from_path(&path).expect("decode vtk");
    let found = datasets.iter().any(|d| {
        d.as_structured_volume()
            .and_then(|v| v.scalar_values("density"))
            .is_some()
            || d.as_point_set()
                .is_some_and(|p| p.scalar_attributes.contains_key("density"))
            || d.as_surface_mesh()
                .is_some_and(|m| m.attributes.contains_key("density"))
    });
    assert!(found, "the named point scalar is reachable after decoding");
}
