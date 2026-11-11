//! BMFont bitmap font decoding: the descriptor AngelCode's generator, Hiero,
//! and most bitmap font tools write beside their page images.
//!
//! Reads the text and XML descriptors, which carry the same records under the
//! same names. The binary descriptor is refused: save the font as text or XML
//! instead. What each page channel holds in a packed font (glyph, outline,
//! zero or one) is not carried.

use std::path::Path;

use crate::error::IoError;
use crate::types::{BitmapFontData, BitmapGlyph, GlyphChannels, PixelRect, TextureSource};

/// Decode a BMFont descriptor (`.fnt`, text or XML) into a [`BitmapFontData`].
///
/// Page image paths are resolved against the descriptor's directory; the
/// images themselves are not read.
pub fn bitmap_font_from_path(path: &Path) -> Result<BitmapFontData, IoError> {
    #[cfg(feature = "bmfont")]
    {
        let bytes = std::fs::read(path)?;
        bitmap_font_from_bytes(&bytes, Some(path.parent().unwrap_or(Path::new("."))))
    }

    #[cfg(not(feature = "bmfont"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "bmfont",
            context: "BMFont decoding",
        })
    }
}

/// Decode a BMFont descriptor from bytes already read. `base` is the
/// directory page paths are relative to; without one they are left as the
/// file wrote them.
pub fn bitmap_font_from_bytes(
    bytes: &[u8],
    base: Option<&Path>,
) -> Result<BitmapFontData, IoError> {
    #[cfg(feature = "bmfont")]
    {
        if bytes.starts_with(b"BMF") {
            return Err(IoError::Parse(
                "BMFont binary descriptors are not read; save the font as text or XML".into(),
            ));
        }
        let text = std::str::from_utf8(bytes)
            .map_err(|e| IoError::Parse(format!("BMFont descriptor is not UTF-8: {e}")))?;
        let records = if text.trim_start().starts_with('<') {
            xml_records(text)?
        } else {
            text.lines().filter_map(text_record).collect()
        };
        Ok(build(&records, base))
    }

    #[cfg(not(feature = "bmfont"))]
    {
        let _ = (bytes, base);
        Err(IoError::MissingFeature {
            feature: "bmfont",
            context: "BMFont decoding",
        })
    }
}

/// One descriptor record: a tag such as `char` and its `key=value` pairs.
type Record = (String, Vec<(String, String)>);

/// Split a text descriptor line into its tag and pairs. Values may be quoted,
/// and a quoted value may hold spaces.
fn text_record(line: &str) -> Option<Record> {
    let line = line.trim();
    let (tag, mut rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
    if tag.is_empty() {
        return None;
    }
    let mut pairs = Vec::new();
    loop {
        rest = rest.trim_start();
        let Some(eq) = rest.find('=') else { break };
        let key = rest[..eq].trim().to_string();
        rest = &rest[eq + 1..];
        let value = if let Some(quoted) = rest.strip_prefix('"') {
            let end = quoted.find('"').unwrap_or(quoted.len());
            let value = quoted[..end].to_string();
            rest = quoted.get(end + 1..).unwrap_or("");
            value
        } else {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let value = rest[..end].to_string();
            rest = &rest[end..];
            value
        };
        pairs.push((key, value));
    }
    Some((tag.to_string(), pairs))
}

/// Every element of an XML descriptor as a record. The nesting (`<chars>`
/// around `<char>`) adds nothing the text form does not have, so it is
/// flattened.
fn xml_records(text: &str) -> Result<Vec<Record>, IoError> {
    use quick_xml::events::Event;

    let err = |e: &dyn std::fmt::Display| IoError::Parse(format!("BMFont XML: {e}"));
    let mut reader = quick_xml::Reader::from_str(text);
    let mut out = Vec::new();
    loop {
        match reader.read_event().map_err(|e| err(&e))? {
            Event::Start(e) | Event::Empty(e) => {
                let mut pairs = Vec::new();
                for attr in e.attributes() {
                    let attr = attr.map_err(|e| err(&e))?;
                    let value = attr.unescape_value().map_err(|e| err(&e))?;
                    pairs.push((
                        String::from_utf8_lossy(attr.key.as_ref()).into_owned(),
                        value.into_owned(),
                    ));
                }
                out.push((
                    String::from_utf8_lossy(e.name().as_ref()).into_owned(),
                    pairs,
                ));
            }
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(out)
}

fn get<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

fn int(pairs: &[(String, String)], key: &str) -> i64 {
    get(pairs, key)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// A comma-separated list of integers, such as `padding=1,2,3,4`.
fn ints<const N: usize>(pairs: &[(String, String)], key: &str) -> [i32; N] {
    let mut out = [0; N];
    if let Some(v) = get(pairs, key) {
        for (slot, part) in out.iter_mut().zip(v.split(',')) {
            *slot = part.trim().parse().unwrap_or(0);
        }
    }
    out
}

/// The channels a glyph's `chnl` bits name: 1 blue, 2 green, 4 red, 8 alpha.
fn channels(bits: i64) -> GlyphChannels {
    if bits == 0 {
        return GlyphChannels::ALL;
    }
    GlyphChannels {
        blue: bits & 1 != 0,
        green: bits & 2 != 0,
        red: bits & 4 != 0,
        alpha: bits & 8 != 0,
    }
}

fn build(records: &[Record], base: Option<&Path>) -> BitmapFontData {
    let mut font = BitmapFontData::default();
    let mut pages: Vec<(i64, TextureSource)> = Vec::new();
    for (tag, p) in records {
        match tag.as_str() {
            "info" => {
                font.face = get(p, "face").unwrap_or_default().to_string();
                font.size = int(p, "size") as i32;
                font.bold = int(p, "bold") != 0;
                font.italic = int(p, "italic") != 0;
                font.padding = ints(p, "padding");
                font.spacing = ints(p, "spacing");
                font.outline = int(p, "outline").max(0) as u32;
            }
            "common" => {
                font.line_height = int(p, "lineHeight").max(0) as u32;
                font.base = int(p, "base").max(0) as u32;
                font.page_size = [
                    int(p, "scaleW").max(0) as u32,
                    int(p, "scaleH").max(0) as u32,
                ];
                font.packed = int(p, "packed") != 0;
            }
            "page" => {
                let file = get(p, "file").unwrap_or_default();
                let path = base.map_or_else(|| file.into(), |b| b.join(file));
                pages.push((int(p, "id"), TextureSource::File(path)));
            }
            "char" => {
                let at = |k| int(p, k).max(0) as u32;
                font.glyphs.insert(
                    at("id"),
                    BitmapGlyph {
                        page: at("page"),
                        rect: PixelRect::new(at("x"), at("y"), at("width"), at("height")),
                        offset: [int(p, "xoffset") as i32, int(p, "yoffset") as i32],
                        advance: int(p, "xadvance") as i32,
                        channels: channels(int(p, "chnl")),
                    },
                );
            }
            "kerning" => {
                font.kernings.insert(
                    (
                        int(p, "first").max(0) as u32,
                        int(p, "second").max(0) as u32,
                    ),
                    int(p, "amount") as i32,
                );
            }
            _ => {}
        }
    }
    pages.sort_by_key(|(id, _)| *id);
    font.pages = pages.into_iter().map(|(_, source)| source).collect();
    font
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(record: &Record) -> Vec<(&str, &str)> {
        record
            .1
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    #[test]
    fn a_text_line_splits_into_tag_and_pairs_with_quoted_spaces() {
        let record =
            text_record(r#"info face="Open Sans" size=-32 padding=1,2,3,4 charset="""#).unwrap();
        assert_eq!(record.0, "info");
        assert_eq!(
            pairs(&record),
            [
                ("face", "Open Sans"),
                ("size", "-32"),
                ("padding", "1,2,3,4"),
                ("charset", "")
            ]
        );
        assert!(text_record("   ").is_none());
    }

    #[test]
    fn channel_bits_name_their_channels() {
        assert_eq!(channels(15), GlyphChannels::ALL);
        assert_eq!(
            channels(0),
            GlyphChannels::ALL,
            "unpacked fonts may write 0"
        );
        assert_eq!(
            channels(4),
            GlyphChannels {
                red: true,
                ..GlyphChannels::default()
            }
        );
        assert_eq!(
            channels(1),
            GlyphChannels {
                blue: true,
                ..GlyphChannels::default()
            }
        );
    }

    #[test]
    fn a_binary_descriptor_is_refused_by_name() {
        let err = bitmap_font_from_bytes(b"BMF\x03\x01", None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("binary")),
            "{err:?}"
        );
    }
}
