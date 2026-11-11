//! The sheet types, built from the JSON an export writes or from a decoded
//! document.

use std::fmt;
use std::path::Path;

use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use viewport_lib_types::colour::Colour;

use viewport_lib_types::data::texture::TextureData;

use super::{AsepriteLoadOptions, composite, raw};
use crate::error::IoError;
use crate::types::{
    AnimationDirection, PixelRect, SpriteAnimation, SpriteFrame, SpriteSheetData, SpriteSlice,
    SpriteSliceKey, TextureSource,
};

/// The two members a sheet is read from. `frames` keeps the file's order in
/// the hash layout, since animations index frames by it and a sorted map
/// would put `walk 10` before `walk 2`.
struct Document {
    frames: Option<Vec<(Option<String>, Value)>>,
    meta: Value,
}

impl<'de> Deserialize<'de> for Document {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct DocumentVisitor;
        impl<'de> Visitor<'de> for DocumentVisitor {
            type Value = Document;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a sprite sheet object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Document, A::Error> {
                let mut document = Document {
                    frames: None,
                    meta: Value::Null,
                };
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "frames" => document.frames = Some(map.next_value::<Frames>()?.0),
                        "meta" => document.meta = map.next_value()?,
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                Ok(document)
            }
        }
        d.deserialize_map(DocumentVisitor)
    }
}

/// Frames as a list, named by their key in the hash layout.
struct Frames(Vec<(Option<String>, Value)>);

impl<'de> Deserialize<'de> for Frames {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct FramesVisitor;
        impl<'de> Visitor<'de> for FramesVisitor {
            type Value = Frames;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an array or object of frames")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Frames, A::Error> {
                let mut out = Vec::new();
                while let Some(frame) = seq.next_element()? {
                    out.push((None, frame));
                }
                Ok(Frames(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Frames, A::Error> {
                let mut out = Vec::new();
                while let Some((name, frame)) = map.next_entry::<String, Value>()? {
                    out.push((Some(name), frame));
                }
                Ok(Frames(out))
            }
        }
        d.deserialize_any(FramesVisitor)
    }
}

/// Decode a JSON sheet or an Aseprite document. `name` is the document's file
/// name, which its frames are named after as an export names them.
pub(super) fn decode(
    bytes: &[u8],
    base: Option<&Path>,
    name: Option<&str>,
    options: AsepriteLoadOptions,
) -> Result<SpriteSheetData, IoError> {
    if raw::is_document(bytes) {
        return Ok(from_document(&raw::read(bytes)?, name, options));
    }
    let document: Document = serde_json::from_slice(bytes)
        .map_err(|e| IoError::Parse(format!("sprite sheet JSON: {e}")))?;
    let frames = document
        .frames
        .ok_or_else(|| IoError::Parse("sprite sheet JSON has no `frames`".into()))?;
    let empty = Map::new();
    let meta = document.meta.as_object().unwrap_or(&empty);

    let mut frames: Vec<SpriteFrame> = frames
        .iter()
        .map(|(name, value)| frame(name.as_deref(), value))
        .collect();

    let size = meta
        .get("size")
        .and_then(Value::as_object)
        .map(|s| [int_of(s, "w").max(0) as u32, int_of(s, "h").max(0) as u32]);
    let size = size.filter(|s| s[0] > 0 && s[1] > 0).unwrap_or_else(|| {
        frames.iter().fold([0, 0], |acc, f| {
            [
                acc[0].max(f.rect.x + f.rect.width),
                acc[1].max(f.rect.y + f.rect.height),
            ]
        })
    });
    for f in &mut frames {
        f.uv = f.rect.uv(size);
    }

    Ok(SpriteSheetData {
        image: meta
            .get("image")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .map(|p| TextureSource::File(base.map_or_else(|| p.into(), |b| b.join(p)))),
        size,
        scale: match meta.get("scale") {
            Some(Value::String(s)) => s.trim().parse().unwrap_or(1.0),
            Some(v) => v.as_f64().unwrap_or(1.0) as f32,
            None => 1.0,
        },
        app: str_of(meta, "app").to_string(),
        animations: array(meta, "frameTags")
            .iter()
            .filter_map(Value::as_object)
            .map(animation)
            .collect(),
        slices: array(meta, "slices")
            .iter()
            .filter_map(Value::as_object)
            .map(slice)
            .collect(),
        frames,
        ..SpriteSheetData::default()
    })
}

/// Place rectangles of `sizes` on shelves, left to right and top to bottom, in
/// order, `gutter` pixels apart. Returns each top-left corner and the atlas
/// size.
///
/// A shelf is as wide as `ceil(sqrt(n))` rectangles of the average width, or
/// the widest one if that is wider, so a run of equal frames packs into a
/// near-square grid.
pub(super) fn pack(sizes: &[[u32; 2]], gutter: u32) -> (Vec<[u32; 2]>, [u32; 2]) {
    let n = sizes.len() as u64;
    let widest = sizes.iter().map(|s| s[0]).max().unwrap_or(0);
    let cell_sum: u64 = sizes.iter().map(|s| (s[0] + gutter) as u64).sum();
    let columns = (n as f64).sqrt().ceil() as u64;
    let shelf_width = if n == 0 {
        0
    } else {
        ((columns * cell_sum).div_ceil(n) as u32).saturating_sub(gutter)
    }
    .max(widest);

    let mut corners = Vec::with_capacity(sizes.len());
    let (mut x, mut y, mut shelf_height) = (0u32, 0u32, 0u32);
    let mut atlas = [0u32, 0u32];
    for size in sizes {
        if x > 0 && x + size[0] > shelf_width {
            x = 0;
            y += shelf_height + gutter;
            shelf_height = 0;
        }
        corners.push([x, y]);
        atlas = [atlas[0].max(x + size[0]), atlas[1].max(y + size[1])];
        x += size[0] + gutter;
        shelf_height = shelf_height.max(size[1]);
    }
    (corners, atlas)
}

/// A frame's name as an export writes it: `hero 3.aseprite`, or `hero.aseprite`
/// for a sprite of one frame. With no file name, the frame's index.
fn frame_name(name: Option<&str>, index: usize, count: usize) -> String {
    let Some(name) = name else {
        return index.to_string();
    };
    if count == 1 {
        return name.to_string();
    }
    match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem} {index}.{ext}"),
        None => format!("{name} {index}"),
    }
}

fn rgba_colour(c: [u8; 4]) -> Colour {
    Colour::srgb_u8(c[0], c[1], c[2], c[3])
}

/// Copy `image` (`size` pixels) into `atlas` (`atlas_width` wide) at `at`.
fn blit(atlas: &mut [u8], atlas_width: u32, image: &[u8], size: [u32; 2], at: [u32; 2]) {
    for row in 0..size[1] as usize {
        let src = row * size[0] as usize * 4;
        let dst = ((at[1] as usize + row) * atlas_width as usize + at[0] as usize) * 4;
        let len = size[0] as usize * 4;
        atlas[dst..dst + len].copy_from_slice(&image[src..src + len]);
    }
}

/// The bounds `[x, y, width, height]` of an image's pixels with any alpha, or
/// `None` when every pixel is transparent.
pub(super) fn opaque_bounds(rgba: &[u8], size: [u32; 2]) -> Option<[u32; 4]> {
    let (mut lo, mut hi) = ([u32::MAX; 2], [0u32; 2]);
    for y in 0..size[1] {
        for x in 0..size[0] {
            if rgba[((y * size[0] + x) * 4 + 3) as usize] != 0 {
                lo = [lo[0].min(x), lo[1].min(y)];
                hi = [hi[0].max(x + 1), hi[1].max(y + 1)];
            }
        }
    }
    (lo[0] != u32::MAX).then(|| [lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]])
}

/// The part of `rgba` (`size` pixels) inside `rect`.
fn crop(rgba: &[u8], size: [u32; 2], rect: [u32; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rect[2] as usize * rect[3] as usize * 4);
    for y in rect[1]..rect[1] + rect[3] {
        let start = ((y * size[0] + rect[0]) * 4) as usize;
        out.extend_from_slice(&rgba[start..start + rect[2] as usize * 4]);
    }
    out
}

/// The sheet an export of `doc` with `options` describes: every frame
/// composited at the canvas size, trimmed and merged as asked, packed into one
/// decoded atlas.
pub(super) fn from_document(
    doc: &raw::Document,
    name: Option<&str>,
    options: AsepriteLoadOptions,
) -> SpriteSheetData {
    let canvas = doc.size;

    // Each frame's pixels and where they sit in the untrimmed frame.
    let cut: Vec<(Vec<u8>, [u32; 4])> = composite::frame_images(doc)
        .into_iter()
        .map(|image| {
            if !options.trim {
                return (image, [0, 0, canvas[0], canvas[1]]);
            }
            match opaque_bounds(&image, canvas) {
                Some(bounds) => (crop(&image, canvas, bounds), bounds),
                None => (Vec::new(), [0, 0, 0, 0]),
            }
        })
        .collect();

    // Which frames get a cell of their own: all of them, or the first of each
    // set of identical ones. Empty frames take no space.
    let mut cell_of: Vec<Option<usize>> = Vec::with_capacity(cut.len());
    let mut cells: Vec<usize> = Vec::new();
    for (i, (pixels, bounds)) in cut.iter().enumerate() {
        if bounds[2] == 0 || bounds[3] == 0 {
            cell_of.push(None);
            continue;
        }
        let same = options.merge_duplicates.then(|| {
            cells
                .iter()
                .position(|c| cut[*c].0 == *pixels && cut[*c].1[2..] == bounds[2..])
        });
        match same.flatten() {
            Some(cell) => cell_of.push(Some(cell)),
            None => {
                cells.push(i);
                cell_of.push(Some(cells.len() - 1));
            }
        }
    }
    let sizes: Vec<[u32; 2]> = cells.iter().map(|c| [cut[*c].1[2], cut[*c].1[3]]).collect();
    let (corners, size) = pack(&sizes, options.gutter);

    let mut atlas = vec![0u8; size[0] as usize * size[1] as usize * 4];
    for (cell, frame) in cells.iter().enumerate() {
        blit(
            &mut atlas,
            size[0],
            &cut[*frame].0,
            sizes[cell],
            corners[cell],
        );
    }

    let mut frames = Vec::with_capacity(cut.len());
    for (index, (_, bounds)) in cut.iter().enumerate() {
        let rect = match cell_of[index] {
            Some(cell) => PixelRect::new(
                corners[cell][0],
                corners[cell][1],
                sizes[cell][0],
                sizes[cell][1],
            ),
            None => PixelRect::default(),
        };
        frames.push(SpriteFrame {
            name: frame_name(name, index, cut.len()),
            rect,
            uv: rect.uv(size),
            rotated: false,
            trimmed: bounds[2] != canvas[0] || bounds[3] != canvas[1],
            source_size: canvas,
            source_offset: [bounds[0] as i32, bounds[1] as i32],
            pivot: None,
            duration_ms: Some(doc.frames[index].duration_ms),
        });
    }

    SpriteSheetData {
        image: Some(TextureSource::Decoded(TextureData::srgb(
            size[0], size[1], atlas,
        ))),
        size,
        scale: 1.0,
        app: "aseprite".into(),
        frames,
        animations: doc
            .tags
            .iter()
            .map(|t| SpriteAnimation {
                name: t.name.clone(),
                first: t.from,
                last: t.to,
                direction: match t.direction {
                    1 => AnimationDirection::Reverse,
                    2 => AnimationDirection::PingPong,
                    3 => AnimationDirection::PingPongReverse,
                    _ => AnimationDirection::Forward,
                },
                repeat: (t.repeat > 0).then_some(t.repeat as u32),
                colour: Some(rgba_colour(
                    t.user.colour.unwrap_or([t.rgb[0], t.rgb[1], t.rgb[2], 255]),
                )),
                data: t.user.text.clone(),
            })
            .collect(),
        slices: doc
            .slices
            .iter()
            .map(|s| {
                let rect = |r: [i64; 4]| {
                    let at = |v: i64| v.clamp(0, u32::MAX as i64) as u32;
                    PixelRect::new(at(r[0]), at(r[1]), at(r[2]), at(r[3]))
                };
                SpriteSlice {
                    name: s.name.clone(),
                    colour: s.user.colour.map(rgba_colour),
                    data: s.user.text.clone(),
                    keys: s
                        .keys
                        .iter()
                        .map(|k| SpriteSliceKey {
                            frame: k.frame,
                            bounds: rect(k.bounds),
                            centre: k.centre.map(rect),
                            pivot: k.pivot,
                        })
                        .collect(),
                }
            })
            .collect(),
        pixel_ratio: doc.pixel_ratio,
        unsupported: doc.unsupported.clone(),
    }
}

fn array<'a>(object: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    object
        .get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn str_of<'a>(object: &'a Map<String, Value>, key: &str) -> &'a str {
    object.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn int_of(object: &Map<String, Value>, key: &str) -> i64 {
    object
        .get(key)
        .and_then(Value::as_f64)
        .map_or(0, |v| v as i64)
}

fn rect_of(object: Option<&Map<String, Value>>) -> Option<PixelRect> {
    let r = object?;
    let at = |k| int_of(r, k).max(0) as u32;
    Some(PixelRect::new(at("x"), at("y"), at("w"), at("h")))
}

fn object<'a>(object: &'a Map<String, Value>, key: &str) -> Option<&'a Map<String, Value>> {
    object.get(key).and_then(Value::as_object)
}

/// An Aseprite colour, `#RRGGBBAA` or `#RRGGBB`. sRGB.
fn parse_colour(text: &str) -> Option<Colour> {
    let hex = text.trim().trim_start_matches('#');
    let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
    match hex.len() {
        6 => Some(Colour::srgb_u8(byte(0)?, byte(2)?, byte(4)?, 255)),
        8 => Some(Colour::srgb_u8(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
        _ => None,
    }
}

/// One frame, named by its hash key or, in the array layout, its `filename`.
fn frame(name: Option<&str>, value: &Value) -> SpriteFrame {
    let empty = Map::new();
    let f = value.as_object().unwrap_or(&empty);
    let rotated = f.get("rotated").and_then(Value::as_bool).unwrap_or(false);
    let mut rect = rect_of(object(f, "frame")).unwrap_or_default();
    // A rotated frame's `w` and `h` are the upright sprite's; in the atlas it
    // occupies them the other way round.
    if rotated {
        std::mem::swap(&mut rect.width, &mut rect.height);
    }
    let trimmed_at = rect_of(object(f, "spriteSourceSize"));
    let source_size = object(f, "sourceSize").map_or(
        if rotated {
            [rect.height, rect.width]
        } else {
            [rect.width, rect.height]
        },
        |s| [int_of(s, "w").max(0) as u32, int_of(s, "h").max(0) as u32],
    );
    let pivot = object(f, "pivot").or_else(|| object(f, "anchor")).map(|p| {
        let at = |k| p.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        [at("x"), at("y")]
    });

    SpriteFrame {
        name: name
            .map(str::to_string)
            .unwrap_or_else(|| str_of(f, "filename").to_string()),
        rect,
        uv: [0.0; 4],
        rotated,
        trimmed: f.get("trimmed").and_then(Value::as_bool).unwrap_or(false),
        source_size,
        source_offset: trimmed_at.map_or([0, 0], |r| [r.x as i32, r.y as i32]),
        pivot,
        duration_ms: f.get("duration").and_then(Value::as_u64).map(|d| d as u32),
    }
}

fn animation(tag: &Map<String, Value>) -> SpriteAnimation {
    SpriteAnimation {
        name: str_of(tag, "name").to_string(),
        first: int_of(tag, "from").max(0) as u32,
        last: int_of(tag, "to").max(0) as u32,
        direction: match str_of(tag, "direction") {
            "reverse" => AnimationDirection::Reverse,
            "pingpong" => AnimationDirection::PingPong,
            "pingpong_reverse" => AnimationDirection::PingPongReverse,
            _ => AnimationDirection::Forward,
        },
        // Aseprite writes the count as a string, and leaves it out (or writes
        // 0) for an animation that loops.
        repeat: match tag.get("repeat") {
            Some(Value::String(s)) => s.trim().parse().ok(),
            Some(v) => v.as_u64().map(|r| r as u32),
            None => None,
        }
        .filter(|r| *r > 0),
        colour: tag
            .get("color")
            .and_then(Value::as_str)
            .and_then(parse_colour),
        data: str_of(tag, "data").to_string(),
    }
}

fn slice(slice: &Map<String, Value>) -> SpriteSlice {
    SpriteSlice {
        name: str_of(slice, "name").to_string(),
        colour: slice
            .get("color")
            .and_then(Value::as_str)
            .and_then(parse_colour),
        data: str_of(slice, "data").to_string(),
        keys: array(slice, "keys")
            .iter()
            .filter_map(Value::as_object)
            .map(|k| SpriteSliceKey {
                frame: int_of(k, "frame").max(0) as u32,
                bounds: rect_of(object(k, "bounds")).unwrap_or_default(),
                centre: rect_of(object(k, "center")),
                pivot: object(k, "pivot").map(|p| [int_of(p, "x") as i32, int_of(p, "y") as i32]),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::aseprite::sprite_sheet_from_bytes;

    #[test]
    fn hash_frames_keep_the_file_order() {
        let sheet = sprite_sheet_from_bytes(
            br#"{"frames": {
                "walk 2": {"frame": {"x": 0, "y": 0, "w": 1, "h": 1}},
                "walk 10": {"frame": {"x": 1, "y": 0, "w": 1, "h": 1}},
                "a": {"frame": {"x": 2, "y": 0, "w": 1, "h": 1}}
            }, "meta": {}}"#,
            None,
        )
        .unwrap();
        let names: Vec<_> = sheet.frames.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["walk 2", "walk 10", "a"]);
        assert_eq!(
            sheet.size,
            [3, 1],
            "the frames' extent when meta has no size"
        );
    }

    #[test]
    fn a_rotated_frame_occupies_its_turned_rectangle() {
        let f = frame(
            Some("r"),
            &serde_json::json!({"frame": {"x": 4, "y": 8, "w": 10, "h": 20}, "rotated": true,
                                "trimmed": true, "spriteSourceSize": {"x": 1, "y": 2, "w": 10, "h": 20},
                                "sourceSize": {"w": 12, "h": 24}, "pivot": {"x": 0.5, "y": 1.0}}),
        );
        assert_eq!(f.rect, PixelRect::new(4, 8, 20, 10));
        assert!(f.rotated && f.trimmed);
        assert_eq!((f.source_size, f.source_offset), ([12, 24], [1, 2]));
        assert_eq!(f.pivot, Some([0.5, 1.0]));
        assert_eq!(f.duration_ms, None);
    }

    #[test]
    fn repeat_reads_as_a_string_or_a_number_and_zero_loops() {
        let tag = |repeat: Value| {
            let mut m = Map::new();
            m.insert("repeat".into(), repeat);
            animation(&m).repeat
        };
        assert_eq!(tag(Value::String("3".into())), Some(3));
        assert_eq!(tag(serde_json::json!(2)), Some(2));
        assert_eq!(tag(Value::String("0".into())), None);
        assert_eq!(animation(&Map::new()).repeat, None);
    }

    #[test]
    fn a_truncated_document_is_a_parse_error() {
        let mut header = vec![0u8; 20];
        header[4] = 0xE0;
        header[5] = 0xA5;
        let err = sprite_sheet_from_bytes(&header, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("Aseprite")),
            "{err:?}"
        );
    }

    #[test]
    fn equal_frames_pack_into_a_near_square_grid() {
        let (corners, atlas) = pack(&[[16, 16]; 5], 0);
        assert_eq!(atlas, [48, 32]);
        assert_eq!(corners, [[0, 0], [16, 0], [32, 0], [0, 16], [16, 16]]);

        let (corners, atlas) = pack(&[[4, 4]; 4], 1);
        assert_eq!(corners, [[0, 0], [5, 0], [0, 5], [5, 5]]);
        assert_eq!(atlas, [9, 9], "gutters between frames, not after the last");

        assert_eq!(pack(&[], 0), (vec![], [0, 0]));
    }

    #[test]
    fn frames_are_named_as_an_export_names_them() {
        assert_eq!(frame_name(Some("hero.aseprite"), 3, 4), "hero 3.aseprite");
        assert_eq!(frame_name(Some("hero.ase"), 0, 1), "hero.ase");
        assert_eq!(frame_name(Some("hero"), 1, 2), "hero 1");
        assert_eq!(frame_name(None, 2, 4), "2");
    }

    #[test]
    fn a_document_decodes_to_the_sheet_its_export_describes() {
        use viewport_lib_io_testkit::synth::{
            AseCel, AseDocument, AseFrame, AseImage, AseLayer, AseSlice, AseSliceKey, AseTag,
        };

        let pixel = |rgba: [u8; 4]| AseImage::Raw {
            size: [1, 1],
            pixels: rgba.to_vec(),
        };
        let mut doc = AseDocument::new([2, 1], 32);
        doc.pixel_ratio = [2, 1];
        doc.layers = vec![
            AseLayer::image("base"),
            AseLayer {
                kind: 2,
                ..AseLayer::image("Ground")
            },
        ];
        doc.frames = vec![
            AseFrame {
                duration_ms: 100,
                cels: vec![AseCel::new(0, [0, 0], pixel([255, 0, 0, 255]))],
            },
            AseFrame {
                duration_ms: 150,
                cels: vec![AseCel::new(0, [1, 0], pixel([0, 0, 255, 255]))],
            },
        ];
        doc.tags = vec![AseTag {
            from: 0,
            to: 1,
            direction: 2,
            repeat: 2,
            name: "blink".into(),
            user: Some(("eyes".into(), [255, 0, 0, 255])),
            ..AseTag::default()
        }];
        doc.slices = vec![AseSlice {
            name: "hit".into(),
            keys: vec![AseSliceKey {
                frame: 0,
                bounds: [0, 0, 2, 1],
                centre: None,
                pivot: Some([1, 1]),
            }],
            user: None,
        }];

        let sheet = decode(
            &doc.to_bytes(),
            None,
            Some("hero.aseprite"),
            AsepriteLoadOptions::default(),
        )
        .unwrap();

        assert_eq!(sheet.app, "aseprite");
        assert_eq!((sheet.pixel_ratio, sheet.scale), ([2, 1], 1.0));
        assert_eq!(sheet.unsupported, ["tilemap layer \"Ground\""]);
        assert_eq!(sheet.size, [4, 1], "two 2x1 frames side by side");
        let names: Vec<_> = sheet.frames.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["hero 0.aseprite", "hero 1.aseprite"]);
        let second = &sheet.frames[1];
        assert_eq!(second.rect, PixelRect::new(2, 0, 2, 1));
        assert_eq!((second.source_size, second.source_offset), ([2, 1], [0, 0]));
        assert!(!second.trimmed && !second.rotated);
        assert_eq!(second.duration_ms, Some(150));
        assert_eq!(second.uv, [0.5, 0.0, 1.0, 1.0]);

        let Some(TextureSource::Decoded(atlas)) = &sheet.image else {
            panic!("expected a decoded atlas, got {:?}", sheet.image);
        };
        let viewport_lib_types::data::texture::TexturePayload::Rgba8(pixels) = atlas.payload()
        else {
            panic!("expected RGBA8");
        };
        assert_eq!(
            pixels,
            &[255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
            "each frame composited in its own cell"
        );

        let blink = &sheet.animations[0];
        assert_eq!((blink.first, blink.last), (0, 1));
        assert_eq!(
            (blink.direction, blink.repeat),
            (AnimationDirection::PingPong, Some(2))
        );
        assert_eq!(blink.data, "eyes");
        assert_eq!(blink.colour, Some(Colour::srgb_u8(255, 0, 0, 255)));
        assert_eq!(sheet.slices[0].keys[0].pivot, Some([1, 1]));
        assert_eq!(sheet.slices[0].keys[0].bounds, PixelRect::new(0, 0, 2, 1));
    }

    #[test]
    fn json_without_frames_is_an_error() {
        let err = sprite_sheet_from_bytes(br#"{"meta": {}}"#, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("frames")),
            "{err:?}"
        );
    }

    /// Three 4x4 frames: a 2x2 dot at (1, 2), the same dot again, and nothing.
    fn dotted() -> Vec<u8> {
        use viewport_lib_io_testkit::synth::{AseCel, AseDocument, AseFrame, AseImage, AseLayer};
        let dot = || AseImage::Raw {
            size: [2, 2],
            pixels: [9, 9, 9, 255].repeat(4),
        };
        let mut doc = AseDocument::new([4, 4], 32);
        doc.layers = vec![AseLayer::image("a")];
        doc.frames = vec![
            AseFrame {
                duration_ms: 10,
                cels: vec![AseCel::new(0, [1, 2], dot())],
            },
            AseFrame {
                duration_ms: 20,
                cels: vec![AseCel::new(0, [1, 2], dot())],
            },
            AseFrame {
                duration_ms: 30,
                cels: vec![],
            },
        ];
        doc.to_bytes()
    }

    #[test]
    fn trim_crops_to_the_pixels_and_records_the_offset() {
        let options = AsepriteLoadOptions::default().with_trim(true);
        let sheet = decode(&dotted(), None, None, options).unwrap();
        let f = &sheet.frames[0];
        assert!(f.trimmed);
        assert_eq!((f.rect.width, f.rect.height), (2, 2));
        assert_eq!((f.source_size, f.source_offset), ([4, 4], [1, 2]));

        let empty = &sheet.frames[2];
        assert_eq!(
            empty.rect,
            PixelRect::default(),
            "an empty frame takes no space"
        );
        assert_eq!((empty.source_offset, empty.duration_ms), ([0, 0], Some(30)));
        assert!(empty.trimmed);
        assert_eq!(sheet.size, [4, 2], "two 2x2 cells side by side");
    }

    #[test]
    fn merged_duplicates_share_a_rect_and_keep_their_durations() {
        let options = AsepriteLoadOptions::default()
            .with_trim(true)
            .with_merge_duplicates(true);
        let sheet = decode(&dotted(), None, None, options).unwrap();
        assert_eq!(sheet.frames[0].rect, sheet.frames[1].rect);
        assert_eq!(
            (sheet.frames[0].duration_ms, sheet.frames[1].duration_ms),
            (Some(10), Some(20))
        );
        assert_eq!(sheet.size, [2, 2], "one cell for both");

        // Untrimmed, the empty frame is a cell of its own.
        let untrimmed = AsepriteLoadOptions::default().with_merge_duplicates(true);
        let sheet = decode(&dotted(), None, None, untrimmed).unwrap();
        assert_eq!(sheet.frames[0].rect, sheet.frames[1].rect);
        assert_ne!(sheet.frames[0].rect, sheet.frames[2].rect);
    }

    #[test]
    fn the_gutter_spaces_frames_apart() {
        let options = AsepriteLoadOptions::default().with_gutter(2);
        let sheet = decode(&dotted(), None, None, options).unwrap();
        let x: Vec<u32> = sheet.frames.iter().map(|f| f.rect.x).collect();
        assert_eq!(x, [0, 6, 0]);
        assert_eq!(sheet.frames[2].rect.y, 6);
        assert_eq!(sheet.size, [10, 10]);
    }

    #[test]
    fn options_leave_a_json_sheet_as_it_is() {
        let json = br#"{"frames": {"a": {"frame": {"x": 0, "y": 0, "w": 3, "h": 3}}}, "meta": {}}"#;
        let plain = sprite_sheet_from_bytes(json, None).unwrap();
        let options = AsepriteLoadOptions::default()
            .with_trim(true)
            .with_merge_duplicates(true)
            .with_gutter(4);
        assert_eq!(decode(json, None, None, options).unwrap(), plain);
    }

    #[test]
    fn opaque_bounds_find_the_drawn_pixels() {
        let mut rgba = vec![0u8; 3 * 2 * 4];
        assert_eq!(opaque_bounds(&rgba, [3, 2]), None);
        rgba[(3 + 2) * 4 + 3] = 1;
        assert_eq!(opaque_bounds(&rgba, [3, 2]), Some([2, 1, 1, 1]));
    }
}
