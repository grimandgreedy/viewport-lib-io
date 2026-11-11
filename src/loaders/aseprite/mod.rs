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
//! [`layered_sprite_from_path`] reads a document without flattening it: the
//! canvas, the layers and each frame's cels, for a consumer that draws a
//! sprite in parts or hides a layer.
//!
//! TexturePacker writes the same JSON shape as Aseprite, so its sheets read
//! here too, as does any other JSON with a `frames` member in that shape, in
//! the hash or the array layout. TexturePacker's multi-pack references are not
//! carried.

use std::path::Path;

use crate::error::IoError;
use crate::types::{LayeredSpriteData, SpriteSheetData};

mod composite;
mod options;
mod raw;
mod sheet;

pub use options::AsepriteLoadOptions;

/// Decode an Aseprite document or a JSON sprite sheet into a
/// [`SpriteSheetData`].
///
/// Equivalent to [`sprite_sheet_from_path_with_options`] with the default
/// [`AsepriteLoadOptions`]. A JSON sheet's atlas image path is resolved
/// against the sheet's directory and the image itself is not read. A
/// document's frames are named after the file, as an export names them
/// (`hero 0.aseprite`).
pub fn sprite_sheet_from_path(path: &Path) -> Result<SpriteSheetData, IoError> {
    sprite_sheet_from_path_with_options(path, AsepriteLoadOptions::default())
}

/// Decode an Aseprite document or a JSON sprite sheet with per-call options,
/// which shape how a document's frames are packed.
pub fn sprite_sheet_from_path_with_options(
    path: &Path,
    options: AsepriteLoadOptions,
) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        let bytes = std::fs::read(path)?;
        sheet::decode(
            &bytes,
            Some(path.parent().unwrap_or(Path::new("."))),
            path.file_name().and_then(|n| n.to_str()),
            options,
        )
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = (path, options);
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "sprite sheet decoding",
        })
    }
}

/// Decode an Aseprite document or a JSON sprite sheet from bytes already read.
///
/// Equivalent to [`sprite_sheet_from_bytes_with_options`] with the default
/// [`AsepriteLoadOptions`]. `base` is the directory a JSON sheet's atlas image
/// path is relative to; without one the path is left as the file wrote it.
/// With no file name, a document's frames are named by their index.
pub fn sprite_sheet_from_bytes(
    bytes: &[u8],
    base: Option<&Path>,
) -> Result<SpriteSheetData, IoError> {
    sprite_sheet_from_bytes_with_options(bytes, base, AsepriteLoadOptions::default())
}

/// Decode an Aseprite document or a JSON sprite sheet from bytes already read,
/// with per-call options.
pub fn sprite_sheet_from_bytes_with_options(
    bytes: &[u8],
    base: Option<&Path>,
    options: AsepriteLoadOptions,
) -> Result<SpriteSheetData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        sheet::decode(bytes, base, None, options)
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = (bytes, base, options);
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "sprite sheet decoding",
        })
    }
}

/// Decode an Aseprite document into its layers and cels, without flattening
/// them into frames.
///
/// Flattening the result with each layer's blend mode and opacity gives the
/// frames [`sprite_sheet_from_path`] returns for the same file. A JSON sheet
/// has no layers, so it is an [`IoError::Parse`].
pub fn layered_sprite_from_path(path: &Path) -> Result<LayeredSpriteData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        layered_sprite_from_bytes(&std::fs::read(path)?)
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "Aseprite layer decoding",
        })
    }
}

/// Decode an Aseprite document already read into its layers and cels.
pub fn layered_sprite_from_bytes(bytes: &[u8]) -> Result<LayeredSpriteData, IoError> {
    #[cfg(feature = "aseprite")]
    {
        if !raw::is_document(bytes) {
            return Err(IoError::Parse(
                "not an Aseprite document; only a document has layers".into(),
            ));
        }
        Ok(sheet::layered_from_document(&raw::read(bytes)?))
    }

    #[cfg(not(feature = "aseprite"))]
    {
        let _ = bytes;
        Err(IoError::MissingFeature {
            feature: "aseprite",
            context: "Aseprite layer decoding",
        })
    }
}
