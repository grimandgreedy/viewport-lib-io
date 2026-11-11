//! Aseprite sprite sheet conformance: Aseprite and TexturePacker exports in both JSON
//! layouts clear the floor, each pair decodes the same, and the fields an
//! exporter writes (trim, rotation, pivots, durations, tags, slices) come
//! through.

#![cfg(feature = "aseprite")]

use std::path::PathBuf;

use viewport_lib_io::loaders::aseprite;
use viewport_lib_io::types::{
    AnimationDirection, PixelRect, SpriteSheetData, SpriteSliceKey, TextureSource,
};
use viewport_lib_io::viewport_lib_types::colour::Colour;
use viewport_lib_io_testkit::{conformance, synth};

fn decode(label: &str, contents: &str) -> (PathBuf, SpriteSheetData) {
    let path = synth::temp_path("sprite_sheet", "sheet.json");
    synth::write(&path, contents);
    let sheet = conformance::sprite_sheet(label, || aseprite::sprite_sheet_from_path(&path));
    (path, sheet)
}

#[test]
fn hash_and_array_layouts_decode_the_same() {
    let without_image = |mut s: SpriteSheetData| {
        s.image = None;
        s
    };
    let (_, hash) = decode("aseprite hash", &synth::aseprite_sheet(true));
    let (_, array) = decode("aseprite array", &synth::aseprite_sheet(false));
    assert_eq!(without_image(hash), without_image(array));

    let (_, hash) = decode("texturepacker hash", &synth::texturepacker_sheet(true));
    let (_, array) = decode("texturepacker array", &synth::texturepacker_sheet(false));
    assert_eq!(without_image(hash), without_image(array));
}

#[test]
fn aseprite_frames_keep_their_order_trim_and_durations() {
    let (path, sheet) = decode("aseprite hash", &synth::aseprite_sheet(true));
    assert_eq!(
        sheet.image,
        Some(TextureSource::File(path.parent().unwrap().join("hero.png")))
    );
    assert_eq!((sheet.size, sheet.scale), ([64, 16], 1.0));
    assert!(sheet.app.contains("aseprite"));

    let names: Vec<_> = sheet.frames.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "hero 0.aseprite",
            "hero 1.aseprite",
            "hero 2.aseprite",
            "hero 10.aseprite"
        ],
        "file order, not sorted"
    );
    let trimmed = &sheet.frames[1];
    assert!(trimmed.trimmed && !trimmed.rotated);
    assert_eq!(trimmed.rect, PixelRect::new(16, 0, 12, 14));
    assert_eq!(
        (trimmed.source_size, trimmed.source_offset),
        ([16, 16], [2, 1])
    );
    assert_eq!(trimmed.uv, [0.25, 0.0, 28.0 / 64.0, 14.0 / 16.0]);
    assert_eq!(trimmed.duration_ms, Some(120));
    assert_eq!(trimmed.pivot, None);
}

#[test]
fn aseprite_tags_and_slices_come_through() {
    let (_, sheet) = decode("aseprite array", &synth::aseprite_sheet(false));

    let walk = &sheet.animations[0];
    assert_eq!((walk.name.as_str(), walk.first, walk.last), ("walk", 0, 2));
    assert_eq!(
        (walk.direction, walk.repeat),
        (AnimationDirection::Forward, None)
    );
    let blink = &sheet.animations[1];
    assert_eq!(
        (blink.direction, blink.repeat),
        (AnimationDirection::PingPong, Some(2))
    );
    assert_eq!(blink.colour, Some(Colour::srgb_u8(255, 0, 0, 255)));
    assert_eq!(blink.data, "eyes");

    let hitbox = &sheet.slices[0];
    assert_eq!(
        (hitbox.name.as_str(), hitbox.data.as_str()),
        ("hitbox", "solid")
    );
    assert_eq!(hitbox.colour, Some(Colour::srgb_u8(0, 0, 255, 255)));
    assert_eq!(
        hitbox.keys,
        [
            SpriteSliceKey {
                frame: 0,
                bounds: PixelRect::new(2, 2, 12, 12),
                centre: Some(PixelRect::new(3, 3, 6, 6)),
                pivot: Some([6, 12]),
            },
            SpriteSliceKey {
                frame: 2,
                bounds: PixelRect::new(4, 2, 8, 12),
                centre: None,
                pivot: None,
            },
        ]
    );
}

#[test]
fn texturepacker_rotation_and_pivots_come_through() {
    let (_, sheet) = decode("texturepacker hash", &synth::texturepacker_sheet(true));
    assert_eq!((sheet.size, sheet.scale), ([64, 32], 0.5));
    assert!(sheet.animations.is_empty() && sheet.slices.is_empty());

    let coin = &sheet.frames[0];
    assert_eq!(coin.pivot, Some([0.5, 0.5]));
    assert_eq!(coin.duration_ms, None);

    let sword = &sheet.frames[1];
    assert!(sword.rotated && sword.trimmed);
    assert_eq!(
        sword.rect,
        PixelRect::new(32, 0, 14, 30),
        "turned in the atlas"
    );
    assert_eq!((sword.source_size, sword.source_offset), ([32, 32], [1, 9]));
    assert_eq!(sword.pivot, Some([0.1, 0.9]));
}

#[test]
fn bytes_decode_like_the_path() {
    let contents = synth::aseprite_sheet(true);
    let (path, from_path) = decode("aseprite hash", &contents);
    let from_bytes = aseprite::sprite_sheet_from_bytes(contents.as_bytes(), path.parent()).unwrap();
    assert_eq!(from_bytes, from_path);
}
