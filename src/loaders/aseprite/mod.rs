//! Aseprite sprite sheets: the document an artist saves (`.aseprite`,
//! `.ase`) and the JSON sheet an export writes beside its atlas image.
//!
//! A document decodes to the sheet a default export of it would describe:
//! each frame composited from its visible, non-reference layers with their
//! blend modes and opacity, at the full canvas size and untrimmed, packed into
//! one atlas carried as decoded pixels. Frame durations, tags and slices come
//! through, with their user data text and colour. Tilemap layers, external
//! files and embedded colour profiles are not decoded; each is named in
//! [`unsupported`](crate::types::SpriteSheetData::unsupported).
//!
//! TexturePacker writes the same JSON shape as Aseprite, so its sheets read
//! here too, as does any other JSON with a `frames` member in that shape, in
//! the hash or the array layout. TexturePacker's multi-pack references are not
//! carried.

use std::path::Path;

use crate::error::IoError;
use crate::types::SpriteSheetData;

mod composite;
mod raw;
mod sheet;

/// Decode an Aseprite document or a JSON sprite sheet into a
/// [`SpriteSheetData`].
///
/// A JSON sheet's atlas image path is resolved against the sheet's directory
/// and the image itself is not read. A document's frames are named after the
/// file, as an export names them (`hero 0.aseprite`).
pub fn sprite_sheet_from_path(path: &Path) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        let bytes = std::fs::read(path)?;
        sheet::decode(
            &bytes,
            Some(path.parent().unwrap_or(Path::new("."))),
            path.file_name().and_then(|n| n.to_str()),
        )
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

/// Decode an Aseprite document or a JSON sprite sheet from bytes already read.
///
/// `base` is the directory a JSON sheet's atlas image path is relative to;
/// without one the path is left as the file wrote it. With no file name, a
/// document's frames are named by their index.
pub fn sprite_sheet_from_bytes(
    bytes: &[u8],
    base: Option<&Path>,
) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        sheet::decode(bytes, base, None)
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
