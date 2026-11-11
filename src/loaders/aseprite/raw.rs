//! The Aseprite document format: header, frames and chunks read into plain
//! structs, with no compositing.
//!
//! Every value is little-endian. A frame is walked chunk by chunk using each
//! chunk's size, so a chunk this reader does not know is skipped.

use std::io::Read;

use crate::error::IoError;

/// The magic number at byte 4 of every document.
pub(super) const MAGIC: u16 = 0xA5E0;
const FRAME_MAGIC: u16 = 0xF1FA;

/// Whether the bytes are an Aseprite document rather than an export.
pub(super) fn is_document(bytes: &[u8]) -> bool {
    bytes.len() >= 6 && u16::from_le_bytes([bytes[4], bytes[5]]) == MAGIC
}

/// How the document stores a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ColourDepth {
    /// Four bytes, red, green, blue, alpha.
    Rgba,
    /// Two bytes, value and alpha.
    Greyscale,
    /// One byte, a palette index.
    Indexed,
}

impl ColourDepth {
    pub(super) fn bytes_per_pixel(self) -> usize {
        match self {
            ColourDepth::Rgba => 4,
            ColourDepth::Greyscale => 2,
            ColourDepth::Indexed => 1,
        }
    }
}

/// What a layer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LayerKind {
    Image,
    Group,
    Tilemap,
}

/// Text and colour attached to a layer, cel, tag or slice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct UserData {
    pub(super) text: String,
    pub(super) colour: Option<[u8; 4]>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Layer {
    pub(super) name: String,
    pub(super) kind: LayerKind,
    pub(super) visible: bool,
    pub(super) background: bool,
    pub(super) reference: bool,
    /// Nesting depth; a layer's parent is the nearest earlier layer one level
    /// up.
    pub(super) child_level: u16,
    /// Blend mode id, `0` normal to `18` divide.
    pub(super) blend: u16,
    /// `255` when the header says layer opacity is not valid.
    pub(super) opacity: u8,
    pub(super) user: UserData,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Cel {
    /// Index into [`Document::layers`].
    pub(super) layer: usize,
    pub(super) position: [i32; 2],
    pub(super) opacity: u8,
    pub(super) z_index: i16,
    pub(super) size: [u32; 2],
    /// Pixels in the document's [`ColourDepth`], row by row.
    pub(super) pixels: Vec<u8>,
    pub(super) user: UserData,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Frame {
    pub(super) duration_ms: u32,
    pub(super) cels: Vec<Cel>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Tag {
    pub(super) name: String,
    pub(super) from: u32,
    pub(super) to: u32,
    /// `0` forward, `1` reverse, `2` ping-pong, `3` ping-pong reverse.
    pub(super) direction: u8,
    /// Times to play, `0` for ever.
    pub(super) repeat: u16,
    /// The colour in the chunk itself, which newer files leave to user data.
    pub(super) rgb: [u8; 3],
    pub(super) user: UserData,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SliceKey {
    pub(super) frame: u32,
    /// x, y, width, height.
    pub(super) bounds: [i64; 4],
    /// Relative to the bounds.
    pub(super) centre: Option<[i64; 4]>,
    /// Relative to the bounds.
    pub(super) pivot: Option<[i32; 2]>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Slice {
    pub(super) name: String,
    pub(super) keys: Vec<SliceKey>,
    pub(super) user: UserData,
}

/// A whole document, linked cels already resolved to copies.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Document {
    pub(super) size: [u32; 2],
    pub(super) depth: ColourDepth,
    /// Whether a group composites its children on their own and then blends
    /// the result with its own mode and opacity.
    pub(super) groups_isolated: bool,
    pub(super) transparent_index: u8,
    /// Width and height of one pixel.
    pub(super) pixel_ratio: [u32; 2],
    pub(super) layers: Vec<Layer>,
    pub(super) frames: Vec<Frame>,
    /// RGBA entries.
    pub(super) palette: Vec<[u8; 4]>,
    pub(super) tags: Vec<Tag>,
    pub(super) slices: Vec<Slice>,
    /// What the document holds that is not decoded, one entry per kind.
    pub(super) unsupported: Vec<String>,
}

fn truncated() -> IoError {
    IoError::Parse("Aseprite document is truncated".into())
}

/// A little-endian reader over a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let end = self.pos.checked_add(n).ok_or_else(truncated)?;
        let out = self.bytes.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(out)
    }

    fn rest(&mut self) -> &'a [u8] {
        let out = &self.bytes[self.pos.min(self.bytes.len())..];
        self.pos = self.bytes.len();
        out
    }

    fn skip(&mut self, n: usize) -> Result<(), IoError> {
        self.take(n).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, IoError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, IoError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i16(&mut self) -> Result<i16, IoError> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32, IoError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, IoError> {
        Ok(self.u32()? as i32)
    }

    fn string(&mut self) -> Result<String, IoError> {
        let len = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
    }
}

/// The number of chunks in a frame: the 32-bit field, unless it is zero (an
/// older file), in which case the 16-bit one.
pub(super) fn chunk_count(old: u16, new: u32) -> u32 {
    if new == 0 { old as u32 } else { new }
}

/// What a user data chunk attaches to: the chunk it follows.
enum Target {
    None,
    Layer(usize),
    Cel(usize),
    /// The tags chunk; user data chunks after it go to each tag in turn.
    Tags(usize),
    Slice(usize),
}

/// A cel as read, before linked cels are resolved.
enum CelImage {
    Pixels([u32; 2], Vec<u8>),
    Linked(u32),
}

struct ReadCel {
    cel: Cel,
    image: CelImage,
}

/// Read a whole document.
pub(super) fn read(bytes: &[u8]) -> Result<Document, IoError> {
    let mut c = Cursor::new(bytes);
    c.skip(4)?;
    if c.u16()? != MAGIC {
        return Err(IoError::Parse("not an Aseprite document".into()));
    }
    let frame_count = c.u16()? as usize;
    let size = [c.u16()? as u32, c.u16()? as u32];
    let depth = match c.u16()? {
        32 => ColourDepth::Rgba,
        16 => ColourDepth::Greyscale,
        8 => ColourDepth::Indexed,
        other => {
            return Err(IoError::Parse(format!(
                "Aseprite document has an unknown colour depth of {other} bits"
            )));
        }
    };
    let flags = c.u32()?;
    c.skip(2 + 8)?;
    let transparent_index = c.u8()?;
    c.skip(3 + 2)?;
    let (pw, ph) = (c.u8()? as u32, c.u8()? as u32);
    let pixel_ratio = if pw == 0 || ph == 0 { [1, 1] } else { [pw, ph] };
    c.skip(8 + 84)?;

    let mut doc = Document {
        size,
        depth,
        groups_isolated: flags & 2 != 0,
        transparent_index,
        pixel_ratio,
        layers: Vec::new(),
        frames: Vec::new(),
        palette: Vec::new(),
        tags: Vec::new(),
        slices: Vec::new(),
        unsupported: Vec::new(),
    };
    let opacity_valid = flags & 1 != 0;
    let mut new_palette = false;
    let mut read_frames: Vec<(u32, Vec<ReadCel>)> = Vec::with_capacity(frame_count);

    for _ in 0..frame_count {
        let frame_start = c.pos;
        let frame_bytes = c.u32()? as usize;
        if c.u16()? != FRAME_MAGIC {
            return Err(IoError::Parse(
                "Aseprite document has a frame without its magic number".into(),
            ));
        }
        let old_count = c.u16()?;
        let duration = c.u16()? as u32;
        c.skip(2)?;
        let count = chunk_count(old_count, c.u32()?);
        let mut cels = Vec::new();
        let mut target = Target::None;

        for _ in 0..count {
            let chunk_size = c.u32()? as usize;
            let kind = c.u16()?;
            if chunk_size < 6 {
                return Err(IoError::Parse(format!(
                    "Aseprite chunk {kind:#06x} is too short"
                )));
            }
            let mut d = Cursor::new(c.take(chunk_size - 6)?);
            match kind {
                0x2004 => {
                    let layer = read_layer(&mut d, opacity_valid, &mut doc.unsupported)?;
                    doc.layers.push(layer);
                    target = Target::Layer(doc.layers.len() - 1);
                }
                0x2005 => {
                    if let Some(cel) = read_cel(&mut d, depth, &mut doc.unsupported)? {
                        cels.push(cel);
                        target = Target::Cel(cels.len() - 1);
                    } else {
                        target = Target::None;
                    }
                }
                0x2018 => {
                    let first = doc.tags.len();
                    read_tags(&mut d, &mut doc.tags)?;
                    target = Target::Tags(first);
                }
                0x2022 => {
                    doc.slices.push(read_slice(&mut d)?);
                    target = Target::Slice(doc.slices.len() - 1);
                }
                0x2019 => {
                    read_new_palette(&mut d, &mut doc.palette)?;
                    new_palette = true;
                    target = Target::None;
                }
                0x0004 | 0x0011 if !new_palette => {
                    read_old_palette(&mut d, kind == 0x0011, &mut doc.palette)?;
                    target = Target::None;
                }
                0x2020 => {
                    let user = read_user_data(&mut d, &mut doc.unsupported)?;
                    match &mut target {
                        Target::Layer(i) => doc.layers[*i].user = user,
                        Target::Cel(i) => cels[*i].cel.user = user,
                        Target::Slice(i) => doc.slices[*i].user = user,
                        Target::Tags(next) => {
                            if let Some(tag) = doc.tags.get_mut(*next) {
                                tag.user = user;
                            }
                            *next += 1;
                        }
                        Target::None => {}
                    }
                    if !matches!(target, Target::Tags(_)) {
                        target = Target::None;
                    }
                }
                0x2007 => {
                    // 0 none, 1 sRGB, 2 an embedded ICC profile.
                    if d.u16()? == 2 {
                        note(&mut doc.unsupported, "colour profile: ICC");
                    }
                    target = Target::None;
                }
                0x2008 => {
                    note(&mut doc.unsupported, "external files");
                    target = Target::None;
                }
                0x2023 => {
                    note(&mut doc.unsupported, "tilesets");
                    target = Target::None;
                }
                // Cel extra, the old mask and path chunks, and anything newer.
                _ => target = Target::None,
            }
        }

        if frame_bytes >= 16 {
            c.pos = frame_start + frame_bytes;
        }
        read_frames.push((duration, cels));
    }

    // A linked cel is a copy of the same layer's cel in an earlier frame.
    for (index, (duration_ms, read_cels)) in read_frames.into_iter().enumerate() {
        let mut cels = Vec::with_capacity(read_cels.len());
        for read in read_cels {
            let mut cel = read.cel;
            match read.image {
                CelImage::Pixels(size, pixels) => {
                    cel.size = size;
                    cel.pixels = pixels;
                }
                CelImage::Linked(target) => {
                    let source = doc
                        .frames
                        .get(target as usize)
                        .and_then(|f| f.cels.iter().find(|c| c.layer == cel.layer))
                        .ok_or_else(|| {
                            IoError::Parse(format!(
                                "Aseprite cel in frame {index} links to frame {target}, which \
                                 has no cel on its layer"
                            ))
                        })?;
                    let user = std::mem::take(&mut cel.user);
                    cel = source.clone();
                    if !user.text.is_empty() || user.colour.is_some() {
                        cel.user = user;
                    }
                }
            }
            if cel.layer >= doc.layers.len() {
                return Err(IoError::Parse(format!(
                    "Aseprite cel in frame {index} is on layer {}, of {}",
                    cel.layer,
                    doc.layers.len()
                )));
            }
            cels.push(cel);
        }
        doc.frames.push(Frame { duration_ms, cels });
    }
    Ok(doc)
}

/// Record something the document holds that is not decoded, once.
fn note(unsupported: &mut Vec<String>, what: impl Into<String>) {
    let what = what.into();
    if !unsupported.contains(&what) {
        unsupported.push(what);
    }
}

fn read_layer(
    d: &mut Cursor,
    opacity_valid: bool,
    unsupported: &mut Vec<String>,
) -> Result<Layer, IoError> {
    let flags = d.u16()?;
    let kind = match d.u16()? {
        1 => LayerKind::Group,
        2 => LayerKind::Tilemap,
        _ => LayerKind::Image,
    };
    let child_level = d.u16()?;
    d.skip(4)?;
    let blend = d.u16()?;
    let opacity = d.u8()?;
    d.skip(3)?;
    let name = d.string()?;
    if kind == LayerKind::Tilemap {
        note(unsupported, format!("tilemap layer \"{name}\""));
    }
    Ok(Layer {
        name,
        kind,
        visible: flags & 1 != 0,
        background: flags & 8 != 0,
        reference: flags & 64 != 0,
        child_level,
        blend,
        opacity: if opacity_valid { opacity } else { 255 },
        user: UserData::default(),
    })
}

fn read_cel(
    d: &mut Cursor,
    depth: ColourDepth,
    unsupported: &mut Vec<String>,
) -> Result<Option<ReadCel>, IoError> {
    let layer = d.u16()? as usize;
    let position = [d.i16()? as i32, d.i16()? as i32];
    let opacity = d.u8()?;
    let kind = d.u16()?;
    let z_index = d.i16()?;
    d.skip(5)?;
    let image = match kind {
        0 | 2 => {
            let size = [d.u16()? as u32, d.u16()? as u32];
            let want = size[0] as usize * size[1] as usize * depth.bytes_per_pixel();
            let pixels = if kind == 0 {
                d.take(want)?.to_vec()
            } else {
                let mut out = Vec::with_capacity(want);
                flate2::read::ZlibDecoder::new(d.rest())
                    .take(want as u64)
                    .read_to_end(&mut out)
                    .map_err(|e| IoError::Parse(format!("Aseprite compressed cel: {e}")))?;
                out
            };
            if pixels.len() != want {
                return Err(IoError::Parse(format!(
                    "Aseprite cel holds {} bytes for a {}x{} image",
                    pixels.len(),
                    size[0],
                    size[1]
                )));
            }
            CelImage::Pixels(size, pixels)
        }
        1 => CelImage::Linked(d.u16()? as u32),
        3 => {
            note(unsupported, "tilemap cels");
            return Ok(None);
        }
        other => {
            note(unsupported, format!("cel type {other}"));
            return Ok(None);
        }
    };
    Ok(Some(ReadCel {
        cel: Cel {
            layer,
            position,
            opacity,
            z_index,
            size: [0, 0],
            pixels: Vec::new(),
            user: UserData::default(),
        },
        image,
    }))
}

fn read_tags(d: &mut Cursor, tags: &mut Vec<Tag>) -> Result<(), IoError> {
    let count = d.u16()?;
    d.skip(8)?;
    for _ in 0..count {
        let from = d.u16()? as u32;
        let to = d.u16()? as u32;
        let direction = d.u8()?;
        let repeat = d.u16()?;
        d.skip(6)?;
        let rgb = [d.u8()?, d.u8()?, d.u8()?];
        d.skip(1)?;
        tags.push(Tag {
            name: d.string()?,
            from,
            to,
            direction,
            repeat,
            rgb,
            user: UserData::default(),
        });
    }
    Ok(())
}

fn read_slice(d: &mut Cursor) -> Result<Slice, IoError> {
    let count = d.u32()?;
    let flags = d.u32()?;
    d.skip(4)?;
    let name = d.string()?;
    let mut keys = Vec::new();
    for _ in 0..count {
        let frame = d.u32()?;
        let bounds = [
            d.i32()? as i64,
            d.i32()? as i64,
            d.u32()? as i64,
            d.u32()? as i64,
        ];
        let centre = if flags & 1 != 0 {
            Some([
                d.i32()? as i64,
                d.i32()? as i64,
                d.u32()? as i64,
                d.u32()? as i64,
            ])
        } else {
            None
        };
        let pivot = if flags & 2 != 0 {
            Some([d.i32()?, d.i32()?])
        } else {
            None
        };
        keys.push(SliceKey {
            frame,
            bounds,
            centre,
            pivot,
        });
    }
    Ok(Slice {
        name,
        keys,
        user: UserData::default(),
    })
}

fn read_new_palette(d: &mut Cursor, palette: &mut Vec<[u8; 4]>) -> Result<(), IoError> {
    let size = d.u32()? as usize;
    let first = d.u32()? as usize;
    let last = d.u32()? as usize;
    d.skip(8)?;
    palette.resize(size.max(palette.len()), [0, 0, 0, 255]);
    for index in first..=last {
        let flags = d.u16()?;
        let colour = [d.u8()?, d.u8()?, d.u8()?, d.u8()?];
        if flags & 1 != 0 {
            d.string()?;
        }
        if index >= palette.len() {
            palette.resize(index + 1, [0, 0, 0, 255]);
        }
        palette[index] = colour;
    }
    Ok(())
}

/// The old palette chunks: packets of colours, each after a run of skipped
/// entries. The `0x0011` form stores each channel in `0..=63`.
fn read_old_palette(
    d: &mut Cursor,
    six_bit: bool,
    palette: &mut Vec<[u8; 4]>,
) -> Result<(), IoError> {
    let packets = d.u16()?;
    let mut index = 0usize;
    for _ in 0..packets {
        index += d.u8()? as usize;
        let count = match d.u8()? {
            0 => 256,
            n => n as usize,
        };
        for _ in 0..count {
            let mut rgb = [d.u8()?, d.u8()?, d.u8()?];
            if six_bit {
                rgb = rgb.map(|v| ((v as u32 * 255 + 31) / 63) as u8);
            }
            if index >= palette.len() {
                palette.resize(index + 1, [0, 0, 0, 255]);
            }
            palette[index] = [rgb[0], rgb[1], rgb[2], 255];
            index += 1;
        }
    }
    Ok(())
}

fn read_user_data(d: &mut Cursor, unsupported: &mut Vec<String>) -> Result<UserData, IoError> {
    let flags = d.u32()?;
    let text = if flags & 1 != 0 {
        d.string()?
    } else {
        String::new()
    };
    let colour = if flags & 2 != 0 {
        Some([d.u8()?, d.u8()?, d.u8()?, d.u8()?])
    } else {
        None
    };
    if flags & 4 != 0 {
        note(unsupported, "user data properties");
    }
    Ok(UserData { text, colour })
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_io_testkit::synth::{
        AseCel, AseDocument, AseFrame, AseImage, AseLayer, AseSlice, AseSliceKey, AseTag,
    };

    fn one_pixel(rgba: [u8; 4]) -> AseImage {
        AseImage::Raw {
            size: [1, 1],
            pixels: rgba.to_vec(),
        }
    }

    #[test]
    fn the_header_and_layers_read() {
        let mut doc = AseDocument::new([4, 3], 32);
        doc.flags = 1 | 2;
        doc.pixel_ratio = [2, 1];
        doc.layers = vec![
            AseLayer {
                kind: 1,
                ..AseLayer::image("group")
            },
            AseLayer {
                child_level: 1,
                blend: 1,
                opacity: 128,
                user_text: Some("shade".into()),
                ..AseLayer::image("inner")
            },
            AseLayer {
                flags: 64,
                ..AseLayer::image("guide")
            },
        ];
        doc.frames = vec![AseFrame {
            duration_ms: 90,
            cels: vec![],
        }];

        let read = read(&doc.to_bytes()).unwrap();
        assert_eq!(read.size, [4, 3]);
        assert_eq!(read.depth, ColourDepth::Rgba);
        assert!(read.groups_isolated);
        assert_eq!(read.pixel_ratio, [2, 1]);
        assert_eq!(read.frames[0].duration_ms, 90);
        let names: Vec<_> = read.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["group", "inner", "guide"]);
        assert_eq!(read.layers[0].kind, LayerKind::Group);
        assert_eq!((read.layers[1].child_level, read.layers[1].blend), (1, 1));
        assert_eq!(read.layers[1].opacity, 128);
        assert_eq!(read.layers[1].user.text, "shade");
        assert!(read.layers[2].reference && !read.layers[2].visible);
    }

    #[test]
    fn layer_opacity_is_ignored_unless_the_header_says_it_is_valid() {
        let mut doc = AseDocument::new([1, 1], 32);
        doc.flags = 0;
        doc.layers = vec![AseLayer {
            opacity: 10,
            ..AseLayer::image("a")
        }];
        doc.frames = vec![AseFrame::default()];
        assert_eq!(read(&doc.to_bytes()).unwrap().layers[0].opacity, 255);
    }

    #[test]
    fn raw_compressed_and_linked_cels_read_to_the_same_pixels() {
        let mut doc = AseDocument::new([2, 1], 32);
        doc.layers = vec![AseLayer::image("a")];
        let pixels = vec![1, 2, 3, 4, 5, 6, 7, 8];
        doc.frames = vec![
            AseFrame {
                duration_ms: 100,
                cels: vec![AseCel::new(
                    0,
                    [0, 0],
                    AseImage::Raw {
                        size: [2, 1],
                        pixels: pixels.clone(),
                    },
                )],
            },
            AseFrame {
                duration_ms: 100,
                cels: vec![AseCel::new(
                    0,
                    [0, 0],
                    AseImage::Compressed {
                        size: [2, 1],
                        pixels: pixels.clone(),
                    },
                )],
            },
            AseFrame {
                duration_ms: 100,
                cels: vec![AseCel::new(0, [5, 5], AseImage::Linked(1))],
            },
        ];
        let read = read(&doc.to_bytes()).unwrap();
        for frame in &read.frames {
            assert_eq!(frame.cels[0].pixels, pixels);
            assert_eq!(frame.cels[0].size, [2, 1]);
        }
        assert_eq!(
            read.frames[2].cels[0].position,
            [0, 0],
            "a link copies its source"
        );
    }

    #[test]
    fn a_link_to_a_frame_without_the_cel_is_an_error() {
        let mut doc = AseDocument::new([1, 1], 32);
        doc.layers = vec![AseLayer::image("a")];
        doc.frames = vec![
            AseFrame::default(),
            AseFrame {
                duration_ms: 100,
                cels: vec![AseCel::new(0, [0, 0], AseImage::Linked(0))],
            },
        ];
        assert!(matches!(read(&doc.to_bytes()), Err(IoError::Parse(_))));
    }

    #[test]
    fn tags_take_their_user_data_in_order() {
        let mut doc = AseDocument::new([1, 1], 32);
        doc.layers = vec![AseLayer::image("a")];
        doc.frames = vec![AseFrame::default(); 4];
        doc.tags = vec![
            AseTag {
                from: 0,
                to: 1,
                direction: 2,
                repeat: 3,
                name: "walk".into(),
                rgb: [1, 2, 3],
                user: Some(("feet".into(), [9, 8, 7, 255])),
            },
            AseTag {
                from: 2,
                to: 3,
                direction: 3,
                name: "idle".into(),
                ..AseTag::default()
            },
        ];
        let read = read(&doc.to_bytes()).unwrap();
        let walk = &read.tags[0];
        assert_eq!(
            (walk.from, walk.to, walk.direction, walk.repeat),
            (0, 1, 2, 3)
        );
        assert_eq!(walk.rgb, [1, 2, 3]);
        assert_eq!(walk.user.text, "feet");
        assert_eq!(walk.user.colour, Some([9, 8, 7, 255]));
        assert_eq!(
            (read.tags[1].name.as_str(), read.tags[1].direction),
            ("idle", 3)
        );
        assert_eq!(read.tags[1].user, UserData::default());
    }

    #[test]
    fn slices_read_with_centre_and_pivot() {
        let mut doc = AseDocument::new([16, 16], 32);
        doc.layers = vec![AseLayer::image("a")];
        doc.frames = vec![AseFrame::default()];
        doc.slices = vec![AseSlice {
            name: "button".into(),
            keys: vec![AseSliceKey {
                frame: 0,
                bounds: [1, 2, 12, 10],
                centre: Some([3, 3, 6, 4]),
                pivot: Some([6, 10]),
            }],
            user: Some(("ui".into(), [0, 0, 255, 255])),
        }];
        let read = read(&doc.to_bytes()).unwrap();
        let slice = &read.slices[0];
        assert_eq!(slice.name, "button");
        assert_eq!(slice.user.text, "ui");
        assert_eq!(slice.keys[0].bounds, [1, 2, 12, 10]);
        assert_eq!(slice.keys[0].centre, Some([3, 3, 6, 4]));
        assert_eq!(slice.keys[0].pivot, Some([6, 10]));
    }

    #[test]
    fn both_palette_forms_read_and_the_new_one_wins() {
        let mut old = AseDocument::new([1, 1], 8);
        old.layers = vec![AseLayer::image("a")];
        old.frames = vec![AseFrame::default()];
        // One packet: skip 1, then two colours, the second in 6-bit form.
        old.extra_chunks = vec![
            (0x0004, vec![1, 0, 1, 2, 10, 20, 30, 40, 50, 60]),
            (0x0011, vec![1, 0, 0, 1, 63, 0, 63]),
        ];
        let read_old = read(&old.to_bytes()).unwrap();
        assert_eq!(
            read_old.palette,
            [[255, 0, 255, 255], [10, 20, 30, 255], [40, 50, 60, 255]]
        );

        // A new palette chunk ahead of the old ones: they are then ignored.
        let mut both = old.clone();
        both.extra_chunks.insert(
            0,
            (
                0x2019,
                [
                    &2u32.to_le_bytes()[..],
                    &0u32.to_le_bytes(),
                    &1u32.to_le_bytes(),
                    &[0; 8],
                    &[0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8],
                ]
                .concat(),
            ),
        );
        let read_both = read(&both.to_bytes()).unwrap();
        assert_eq!(read_both.palette, [[1, 2, 3, 4], [5, 6, 7, 8]]);
    }

    #[test]
    fn unknown_chunks_are_skipped_and_unsupported_ones_named() {
        let mut doc = AseDocument::new([1, 1], 32);
        doc.layers = vec![
            AseLayer::image("a"),
            AseLayer {
                kind: 2,
                ..AseLayer::image("Ground")
            },
        ];
        doc.extra_chunks = vec![
            (0x7777, vec![1, 2, 3]),
            (0x2007, [2u16.to_le_bytes(), 0u16.to_le_bytes()].concat()),
            (0x2008, vec![0; 12]),
        ];
        doc.frames = vec![AseFrame {
            duration_ms: 100,
            cels: vec![AseCel::new(0, [0, 0], one_pixel([1, 2, 3, 4]))],
        }];
        let read = read(&doc.to_bytes()).unwrap();
        assert_eq!(read.frames[0].cels.len(), 1);
        assert_eq!(
            read.unsupported,
            [
                "colour profile: ICC",
                "external files",
                "tilemap layer \"Ground\""
            ]
        );
    }

    #[test]
    fn the_chunk_count_prefers_the_wide_field() {
        assert_eq!(
            chunk_count(3, 0),
            3,
            "older files leave the wide field zero"
        );
        assert_eq!(chunk_count(0xFFFF, 70000), 70000);
        assert_eq!(chunk_count(5, 5), 5);
    }

    #[test]
    fn a_truncated_document_is_a_parse_error() {
        let mut doc = AseDocument::new([2, 2], 32);
        doc.layers = vec![AseLayer::image("a")];
        doc.frames = vec![AseFrame {
            duration_ms: 100,
            cels: vec![AseCel::new(
                0,
                [0, 0],
                AseImage::Raw {
                    size: [2, 2],
                    pixels: vec![0; 16],
                },
            )],
        }];
        let bytes = doc.to_bytes();
        for cut in [10, 130, bytes.len() - 3] {
            assert!(
                matches!(read(&bytes[..cut]), Err(IoError::Parse(_))),
                "cut at {cut}"
            );
        }
        assert!(is_document(&bytes));
        assert!(!is_document(b"{\"frames\": []}"));
    }
}
