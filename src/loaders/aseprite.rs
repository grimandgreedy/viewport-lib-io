//! Aseprite sprite sheet decoding: the JSON Aseprite writes beside the atlas
//! image when it exports a sprite sheet.
//!
//! TexturePacker exports the same shape, so its JSON reads through here too, as
//! does any other JSON with a `frames` member in that shape. Both the hash and
//! the array layout are read.
//! Frame tags and slices are read where the sheet has them, which in practice
//! means Aseprite. Aseprite's binary `.ase` and `.aseprite` files are refused:
//! export a JSON sheet from them instead. Aseprite's layer list and
//! TexturePacker's multi-pack references are not carried.

use std::fmt;
use std::path::Path;

use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use viewport_lib_types::colour::Colour;

use crate::error::IoError;
use crate::types::{
    AnimationDirection, PixelRect, SpriteAnimation, SpriteFrame, SpriteSheetData, SpriteSlice,
    SpriteSliceKey, TextureSource,
};

/// Decode a sprite sheet's JSON metadata into a [`SpriteSheetData`].
///
/// The atlas image path is resolved against the sheet's directory; the image
/// itself is not read.
pub fn sprite_sheet_from_path(path: &Path) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        let bytes = std::fs::read(path)?;
        sprite_sheet_from_bytes(&bytes, Some(path.parent().unwrap_or(Path::new("."))))
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "sprite sheet decoding",
        })
    }
}

/// Decode sprite sheet metadata from bytes already read. `base` is the
/// directory the atlas image path is relative to; without one the path is
/// left as the file wrote it.
pub fn sprite_sheet_from_bytes(
    bytes: &[u8],
    base: Option<&Path>,
) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        decode(bytes, base)
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = (bytes, base);
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "sprite sheet decoding",
        })
    }
}

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

/// Whether the bytes are an Aseprite document rather than an export: its
/// header has the magic number `0xA5E0` at byte 4.
fn is_aseprite_binary(bytes: &[u8]) -> bool {
    bytes.len() >= 6 && bytes[4..6] == [0xE0, 0xA5]
}

fn decode(bytes: &[u8], base: Option<&Path>) -> Result<SpriteSheetData, IoError> {
    if is_aseprite_binary(bytes) {
        return Err(IoError::Parse(
            "this is an Aseprite document, which is not read; export a JSON sprite sheet from it"
                .into(),
        ));
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
    })
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
    fn an_aseprite_document_is_refused_by_name() {
        let mut header = vec![0u8; 128];
        header[4] = 0xE0;
        header[5] = 0xA5;
        let err = sprite_sheet_from_bytes(&header, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("Aseprite")),
            "{err:?}"
        );
    }

    #[test]
    fn json_without_frames_is_an_error() {
        let err = sprite_sheet_from_bytes(br#"{"meta": {}}"#, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("frames")),
            "{err:?}"
        );
    }
}
