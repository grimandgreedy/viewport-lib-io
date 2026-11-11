//! Aseprite sprite sheet conformance: Aseprite and TexturePacker exports in both JSON
//! layouts clear the floor, each pair decodes the same, and the fields an
//! exporter writes (trim, rotation, pivots, durations, tags, slices) come
//! through. Aseprite documents in every colour mode clear the floor through
//! both entrypoints, and a document decodes to what its default export says.

#![cfg(feature = "aseprite")]

use std::path::PathBuf;

use viewport_lib_io::loaders::aseprite;
use viewport_lib_io::loaders::aseprite::{AsepriteLoadOptions, layered_sprite_from_path};
use viewport_lib_io::types::{
    AnimationDirection, PixelRect, SpriteSheetData, SpriteSliceKey, TextureSource,
};
use viewport_lib_io::viewport_lib_types::colour::Colour;
use viewport_lib_io_testkit::synth::{
    AseCel, AseDocument, AseFrame, AseImage, AseLayer, AseSlice, AseSliceKey, AseTag,
};
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

/// A 4x4 sprite of three frames in `depth`, using raw, compressed and linked
/// cels, a multiply layer, a group, a hidden and a reference layer, a tag in
/// each direction, a slice with a centre and a pivot, and pixels twice as wide
/// as tall.
fn hero(depth: u16) -> AseDocument {
    let px = |rgba: [u8; 4], index: u8, n: usize| -> Vec<u8> {
        match depth {
            32 => rgba.repeat(n),
            16 => [rgba[0], rgba[3]].repeat(n),
            _ => vec![index; n],
        }
    };
    let mut doc = AseDocument::new([4, 4], depth);
    doc.pixel_ratio = [2, 1];
    doc.palette = vec![
        [0, 0, 0, 0],
        [220, 60, 60, 255],
        [60, 60, 220, 255],
        [60, 200, 60, 255],
    ];
    doc.layers = vec![
        AseLayer::image("body"),
        AseLayer {
            blend: 1,
            opacity: 200,
            ..AseLayer::image("shade")
        },
        AseLayer {
            kind: 1,
            ..AseLayer::image("arm")
        },
        AseLayer {
            child_level: 1,
            ..AseLayer::image("hand")
        },
        AseLayer {
            flags: 0,
            ..AseLayer::image("old")
        },
        AseLayer {
            flags: 1 | 64,
            ..AseLayer::image("sketch")
        },
    ];
    let body = AseImage::Compressed {
        size: [2, 3],
        pixels: px([220, 60, 60, 255], 1, 6),
    };
    let shade = AseImage::Raw {
        size: [4, 1],
        pixels: px([60, 60, 220, 255], 2, 4),
    };
    let hand = |x| {
        AseCel::new(
            3,
            [x, 0],
            AseImage::Raw {
                size: [1, 1],
                pixels: px([60, 200, 60, 255], 3, 1),
            },
        )
    };
    doc.frames = vec![
        AseFrame {
            duration_ms: 100,
            cels: vec![
                AseCel::new(0, [1, 1], body),
                AseCel::new(1, [0, 2], shade),
                hand(2),
                AseCel::new(
                    4,
                    [0, 0],
                    AseImage::Raw {
                        size: [1, 1],
                        pixels: px([1, 1, 1, 255], 1, 1),
                    },
                ),
                AseCel::new(
                    5,
                    [3, 3],
                    AseImage::Raw {
                        size: [1, 1],
                        pixels: px([1, 1, 1, 255], 1, 1),
                    },
                ),
            ],
        },
        AseFrame {
            duration_ms: 120,
            cels: vec![
                AseCel::new(0, [0, 0], AseImage::Linked(0)),
                AseCel::new(1, [0, 0], AseImage::Linked(0)),
                hand(3),
            ],
        },
        AseFrame {
            duration_ms: 80,
            cels: vec![AseCel::new(0, [0, 0], AseImage::Linked(0))],
        },
    ];
    let tag = |name: &str, from, to, direction| AseTag {
        from,
        to,
        direction,
        name: name.into(),
        ..AseTag::default()
    };
    doc.tags = vec![
        AseTag {
            repeat: 2,
            user: Some(("loop twice".into(), [255, 0, 0, 255])),
            ..tag("walk", 0, 1, 0)
        },
        tag("back", 1, 2, 1),
        tag("bounce", 0, 2, 2),
        tag("bounce back", 0, 2, 3),
    ];
    doc.slices = vec![AseSlice {
        name: "hitbox".into(),
        keys: vec![AseSliceKey {
            frame: 0,
            bounds: [1, 1, 2, 3],
            centre: Some([0, 1, 2, 1]),
            pivot: Some([1, 3]),
        }],
        user: Some(("solid".into(), [0, 0, 255, 255])),
    }];
    doc
}

fn write_document(name: &str, doc: &AseDocument) -> PathBuf {
    let path = synth::temp_path("aseprite_document", name);
    synth::write(&path, doc.to_bytes());
    path
}

#[test]
fn documents_in_every_colour_mode_clear_the_floor() {
    for depth in [32, 16, 8] {
        let path = write_document("hero.aseprite", &hero(depth));
        let label = format!("hero depth {depth}");
        let sheet = conformance::sprite_sheet(&label, || aseprite::sprite_sheet_from_path(&path));
        assert_eq!(sheet.frames.len(), 3, "{label}");
        assert_eq!(sheet.pixel_ratio, [2, 1], "{label}");
        assert!(
            sheet.unsupported.is_empty(),
            "{label}: {:?}",
            sheet.unsupported
        );

        let layered = conformance::layered_sprite(&label, || layered_sprite_from_path(&path));
        assert_eq!(layered.layers.len(), 6, "{label}");
        assert_eq!(layered.frames[1].cels.len(), 3, "{label}");
    }
}

#[test]
fn a_document_decodes_to_what_its_default_export_says() {
    let doc = hero(32);
    let document = write_document("hero.aseprite", &doc);
    let export = synth::temp_path("aseprite_export", "hero.json");
    synth::write(&export, doc.export_json("hero.aseprite"));

    // Set aside what differs by construction: where the atlas is and how it
    // is laid out, which exporter wrote it, and what JSON cannot record.
    let comparable = |mut s: SpriteSheetData| {
        s.image = None;
        s.size = [0, 0];
        s.app.clear();
        s.pixel_ratio = [1, 1];
        s.unsupported.clear();
        for f in &mut s.frames {
            f.rect = PixelRect::default();
            f.uv = [0.0; 4];
        }
        s
    };
    let from_document = comparable(aseprite::sprite_sheet_from_path(&document).unwrap());
    let from_export = comparable(aseprite::sprite_sheet_from_path(&export).unwrap());
    assert_eq!(from_document, from_export);

    let names: Vec<_> = from_document
        .frames
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(
        names,
        ["hero 0.aseprite", "hero 1.aseprite", "hero 2.aseprite"]
    );
    let directions: Vec<_> = from_document
        .animations
        .iter()
        .map(|a| a.direction)
        .collect();
    assert_eq!(
        directions,
        [
            AnimationDirection::Forward,
            AnimationDirection::Reverse,
            AnimationDirection::PingPong,
            AnimationDirection::PingPongReverse
        ]
    );
    assert_eq!(from_document.animations[0].repeat, Some(2));
    assert_eq!(
        from_document.slices[0].keys[0],
        SpriteSliceKey {
            frame: 0,
            bounds: PixelRect::new(1, 1, 2, 3),
            centre: Some(PixelRect::new(0, 1, 2, 1)),
            pivot: Some([1, 3]),
        }
    );
}

#[test]
fn options_reach_a_document_through_the_path_entrypoint() {
    let path = write_document("hero.aseprite", &hero(32));
    let options = AsepriteLoadOptions::default()
        .with_trim(true)
        .with_merge_duplicates(true)
        .with_gutter(1);
    let sheet = conformance::sprite_sheet("hero trimmed", || {
        aseprite::sprite_sheet_from_path_with_options(&path, options)
    });
    // Frame 2 holds only the body, 2x3 at (1, 1).
    let body = &sheet.frames[2];
    assert!(body.trimmed);
    assert_eq!((body.source_size, body.source_offset), ([4, 4], [1, 1]));
    assert_eq!((body.rect.width, body.rect.height), (2, 3));
}

#[test]
fn hidden_and_reference_layers_stay_out_of_the_frames() {
    // The hidden and reference layers each put a pixel in a corner of frame 0
    // that nothing else covers.
    let doc = raw_frame0(&hero(32));
    assert_eq!(doc[..4], [0, 0, 0, 0], "hidden layer's corner is clear");
    let last = doc.len() - 4;
    assert_eq!(
        doc[last..],
        [0, 0, 0, 0],
        "reference layer's corner is clear"
    );
}

/// Frame 0 of a document, flattened, as the sheet's atlas holds it.
fn raw_frame0(doc: &AseDocument) -> Vec<u8> {
    use viewport_lib_io::viewport_lib_types::data::texture::TexturePayload;
    let sheet = aseprite::sprite_sheet_from_bytes(&doc.to_bytes(), None).unwrap();
    let Some(TextureSource::Decoded(atlas)) = &sheet.image else {
        panic!("expected a decoded atlas");
    };
    let TexturePayload::Rgba8(pixels) = atlas.payload() else {
        panic!("expected RGBA8");
    };
    let r = sheet.frames[0].rect;
    let mut out = Vec::new();
    for y in r.y..r.y + r.height {
        let start = ((y * sheet.size[0] + r.x) * 4) as usize;
        out.extend_from_slice(&pixels[start..start + r.width as usize * 4]);
    }
    out
}
