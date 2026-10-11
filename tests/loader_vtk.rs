//! VTK conformance. A VTK file can be several things at once, so the loader
//! returns one dataset per piece and each piece is checked for whichever
//! representations it carries.

#![cfg(feature = "vtk")]

use viewport_lib_io::loaders::vtk;
use viewport_lib_io_testkit::{invariants, synth};

#[test]
fn a_legacy_unstructured_grid_conforms() {
    let path = synth::temp_path("vtk_legacy", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let datasets = vtk::datasets_from_path(&path).expect("decode vtk");
    assert!(!datasets.is_empty(), "one dataset per piece");

    for (i, dataset) in datasets.iter().enumerate() {
        let label = format!("quad.vtk / dataset[{i}]");
        if let Some(mesh) = dataset.as_surface_mesh() {
            invariants::check_mesh(&label, mesh).expect_clean();
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

/// `vtu` and `vtp` are forwards to the same decode, reachable at their own
/// top-level paths. A consumer naming `loaders::vtu` must keep working whatever
/// the VTK family looks like internally.
#[test]
fn the_family_forwards_reach_the_same_decode() {
    let path = synth::temp_path("vtk_forwards", "quad.vtk");
    synth::write(&path, synth::vtk_legacy_quad("density"));

    let count = |sets: Vec<viewport_lib_io::types::DecodedDataSet>| {
        sets.iter()
            .map(|d| {
                (
                    d.name.clone(),
                    d.as_surface_mesh().map(|m| m.positions.len()),
                )
            })
            .collect::<Vec<_>>()
    };

    let direct = count(vtk::datasets_from_path(&path).expect("decode through vtk"));
    let through_vtu = count(
        viewport_lib_io::loaders::vtu::datasets_from_path(&path).expect("decode through vtu"),
    );
    let through_vtp = count(
        viewport_lib_io::loaders::vtp::datasets_from_path(&path).expect("decode through vtp"),
    );

    assert_eq!(direct, through_vtu);
    assert_eq!(direct, through_vtp);
}

/// An unstructured grid of volume cells becomes a volume mesh. Point data is
/// per node and arrives as the file stores it, not averaged onto the cells;
/// cell data stays per cell.
#[test]
fn a_tetrahedral_grid_keeps_node_and_cell_fields_apart() {
    let path = synth::temp_path("vtk_tet", "tet.vtk");
    synth::write(
        &path,
        "# vtk DataFile Version 3.0\ntet\nASCII\nDATASET UNSTRUCTURED_GRID\n\
         POINTS 4 float\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n\
         CELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n10\n\
         POINT_DATA 4\nSCALARS temperature float 1\nLOOKUP_TABLE default\n1\n2\n3\n4\n\
         CELL_DATA 1\nSCALARS pressure float 1\nLOOKUP_TABLE default\n7\n",
    );

    let datasets = vtk::datasets_from_path(&path).expect("decode vtk");
    let volume = datasets
        .iter()
        .find_map(|d| d.as_volume_mesh())
        .expect("a tet grid decodes to a volume mesh");

    assert_eq!(volume.positions.len(), 4);
    assert_eq!(volume.cells.len(), 1);
    assert_eq!(volume.node_scalars["temperature"], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(volume.cell_scalars["pressure"], vec![7.0]);
    assert!(!volume.cell_scalars.contains_key("temperature"));
}
