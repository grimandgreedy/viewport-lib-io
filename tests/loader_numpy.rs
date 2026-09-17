//! NumPy conformance. The format stores shape and dtype in a text header and is
//! C-ordered, so the axis order on the way out is the thing worth pinning.

#![cfg(feature = "numpy")]

use viewport_lib_io::loaders::numpy;
use viewport_lib_io::testkit::synth;

#[test]
fn an_f32_array_decodes_as_a_volume() {
    let path = synth::temp_path("npy_volume", "field.npy");
    let values: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    synth::write(&path, synth::npy_f32(&[2, 3, 4], &values));

    let volume = numpy::volume_from_path(&path).expect("decode npy");

    assert_eq!(volume.dims, [4, 3, 2], "C-ordered, so the axes reverse");
    let field = volume
        .point_fields
        .values()
        .next()
        .expect("the array lands as a point field");
    assert_eq!(field.len(), values.len());
}

#[test]
fn decoding_is_reproducible() {
    let path = synth::temp_path("npy_determinism", "field.npy");
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    synth::write(&path, synth::npy_f32(&[2, 2, 2], &values));

    let first = numpy::volume_from_path(&path).expect("first decode");
    let second = numpy::volume_from_path(&path).expect("second decode");

    assert_eq!(first.dims, second.dims);
    assert_eq!(
        first.point_fields.keys().collect::<Vec<_>>(),
        second.point_fields.keys().collect::<Vec<_>>()
    );
}

#[test]
fn a_truncated_header_is_an_error() {
    let path = synth::temp_path("npy_truncated", "bad.npy");
    synth::write(&path, b"\x93NUMPY\x01\x00");
    assert!(numpy::volume_from_path(&path).is_err());
}
