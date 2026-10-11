//! SVG conformance for the vector entrypoint. There is no mesh or point set to
//! check invariants over, so the floor here is the decode being reproducible
//! and the two entrypoints agreeing.

#![cfg(feature = "svg")]

use viewport_lib_io::loaders::svg;
use viewport_lib_io_testkit::{determinism, synth};

#[test]
fn a_filled_path_decodes_reproducibly() {
    let source = synth::svg_filled_path("M0 0 H10 V10 H0 Z", "#ff0000");
    determinism::decode_twice_identical("filled path", || {
        svg::vector_from_bytes(source.as_bytes()).expect("decode svg")
    });
}

#[test]
fn the_two_entrypoints_agree() {
    let source = synth::svg_filled_path("M0 0 C 0 5, 5 10, 10 10 Z", "#00ff00");
    let path = synth::temp_path("svg_entrypoints", "art.svg");
    synth::write(&path, &source);

    let from_path = svg::vector_from_path(&path).expect("from path");
    let from_bytes = svg::vector_from_bytes(source.as_bytes()).expect("from bytes");

    assert_eq!(from_path, from_bytes);
}

#[test]
fn the_raster_entrypoint_still_works_on_the_same_file() {
    let path = synth::temp_path("svg_raster", "art.svg");
    synth::write(
        &path,
        synth::svg_filled_path("M0 0 H10 V10 H0 Z", "#ff0000"),
    );

    let texture = svg::texture_from_path(&path).expect("rasterize");
    let viewport_lib_io::viewport_lib_types::data::texture::TexturePayload::Rgba8(bytes) =
        texture.payload()
    else {
        panic!("expected an 8-bit payload");
    };
    assert_eq!(
        bytes.len(),
        (texture.width() * texture.height()) as usize * 4,
        "four channels per pixel"
    );
}
