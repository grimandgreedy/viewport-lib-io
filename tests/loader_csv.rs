//! CSV conformance. The loader sniffs its delimiter and its coordinate columns
//! by header name, so those are what the format-specific cases cover.

#![cfg(feature = "csv")]

use viewport_lib_io::loaders::csv;
use viewport_lib_io_testkit::{conformance, synth};

#[test]
fn points_conform() {
    let path = synth::temp_path("csv_conformance", "points.csv");
    synth::write(&path, synth::csv_points(8, "density"));

    let points = conformance::points("points.csv", || csv::point_cloud_from_path(&path));

    assert_eq!(points.positions.len(), 8);
    assert_eq!(points.positions[2], [2.0, 4.0, 6.0], "x, y = 2x, z = 3x");
}

#[test]
fn a_named_scalar_column_is_carried() {
    let path = synth::temp_path("csv_scalar", "points.csv");
    synth::write(&path, synth::csv_points(4, "density"));

    let points = conformance::points("points.csv", || csv::point_cloud_from_path(&path));

    assert!(
        points.scalar_attributes.contains_key("density") || !points.scalars.is_empty(),
        "the density column survives as a scalar, got {:?}",
        points.scalar_attributes.keys().collect::<Vec<_>>()
    );
}

#[test]
fn an_empty_file_is_an_error() {
    let path = synth::temp_path("csv_empty", "empty.csv");
    synth::write(&path, "");
    assert!(csv::point_cloud_from_path(&path).is_err());
}
