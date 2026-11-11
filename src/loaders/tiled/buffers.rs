//! Tile layer data: the global tile ids behind a layer or chunk, in whichever
//! encoding the file used.

use std::io::Read;

use super::raw::Element;
use crate::error::IoError;

/// The global tile ids of a `<data>` or `<chunk>` element, row by row.
///
/// `encoding` and `compression` come from the enclosing `<data>`, since a
/// chunk inherits them.
pub(super) fn tile_ids(
    element: &Element,
    encoding: Option<&str>,
    compression: Option<&str>,
) -> Result<Vec<u32>, IoError> {
    if let Some(gids) = &element.gids {
        return Ok(gids.clone());
    }
    match encoding {
        None | Some("") => Ok(element
            .children_named("tile")
            .map(|t| t.attr("gid").and_then(|g| g.parse().ok()).unwrap_or(0))
            .collect()),
        Some("csv") => csv(&element.text),
        Some("base64") => {
            let bytes = base64(element.text.trim())?;
            let bytes = decompress(bytes, compression)?;
            if bytes.len() % 4 != 0 {
                return Err(IoError::Parse(format!(
                    "Tiled layer data: {} bytes is not a whole number of tile ids",
                    bytes.len()
                )));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        Some(other) => Err(IoError::Parse(format!(
            "Tiled layer data: unknown encoding `{other}`"
        ))),
    }
}

fn csv(text: &str) -> Result<Vec<u32>, IoError> {
    text.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .map_err(|_| IoError::Parse(format!("Tiled layer data: `{s}` is not a tile id")))
        })
        .collect()
}

fn decompress(bytes: Vec<u8>, compression: Option<&str>) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::new();
    let result = match compression {
        None | Some("") => return Ok(bytes),
        Some("zlib") => flate2::read::ZlibDecoder::new(bytes.as_slice()).read_to_end(&mut out),
        Some("gzip") => flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut out),
        Some("zstd") => {
            return Err(IoError::Parse(
                "Tiled layer data: zstd compression is not supported; save the map with zlib, \
                 gzip or no compression"
                    .into(),
            ));
        }
        Some(other) => {
            return Err(IoError::Parse(format!(
                "Tiled layer data: unknown compression `{other}`"
            )));
        }
    };
    result.map_err(|e| IoError::Parse(format!("Tiled layer data: {e}")))?;
    Ok(out)
}

/// Decode standard base64, ignoring whitespace. Padding is optional.
fn base64(text: &str) -> Result<Vec<u8>, IoError> {
    fn value(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    }

    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        if c.is_ascii_whitespace() || c == b'=' {
            continue;
        }
        let v = value(c).ok_or_else(|| {
            IoError::Parse(format!(
                "Tiled layer data: `{}` is not base64",
                char::from(c)
            ))
        })?;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn data(text: &str) -> Element {
        Element {
            name: "data".into(),
            text: text.into(),
            ..Element::default()
        }
    }

    fn le_bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter().flat_map(|id| id.to_le_bytes()).collect()
    }

    fn encode_base64(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk
                .iter()
                .enumerate()
                .fold(0u32, |acc, (i, b)| acc | (*b as u32) << (16 - 8 * i));
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    const IDS: [u32; 6] = [1, 0, 0x8000_0002, 7, 0, 65536];

    #[test]
    fn csv_reads_ids_across_lines() {
        let ids = tile_ids(&data("\n1,0,2147483650,\n7,0,65536\n"), Some("csv"), None).unwrap();
        assert_eq!(ids, IDS);
    }

    #[test]
    fn base64_reads_with_and_without_compression() {
        let raw = le_bytes(&IDS);

        let plain = encode_base64(&raw);
        assert_eq!(tile_ids(&data(&plain), Some("base64"), None).unwrap(), IDS);

        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&raw).unwrap();
        let zlib = encode_base64(&z.finish().unwrap());
        assert_eq!(
            tile_ids(&data(&zlib), Some("base64"), Some("zlib")).unwrap(),
            IDS
        );

        let mut g = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        g.write_all(&raw).unwrap();
        let gzip = encode_base64(&g.finish().unwrap());
        assert_eq!(
            tile_ids(
                &data(&format!("\n   {gzip}\n  ")),
                Some("base64"),
                Some("gzip")
            )
            .unwrap(),
            IDS
        );
    }

    #[test]
    fn xml_tiles_default_a_missing_gid_to_empty() {
        let mut element = data("");
        for gid in [Some("3"), None, Some("9")] {
            let mut tile = Element {
                name: "tile".into(),
                ..Element::default()
            };
            if let Some(gid) = gid {
                tile.attrs.push(("gid".into(), gid.into()));
            }
            element.children.push(tile);
        }
        assert_eq!(tile_ids(&element, None, None).unwrap(), [3, 0, 9]);
    }

    #[test]
    fn zstd_is_refused_by_name() {
        let err = tile_ids(&data("AAAA"), Some("base64"), Some("zstd")).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("zstd")),
            "{err:?}"
        );
    }

    #[test]
    fn a_truncated_id_is_an_error() {
        let text = encode_base64(&[1, 0, 0]);
        assert!(tile_ids(&data(&text), Some("base64"), None).is_err());
    }
}
