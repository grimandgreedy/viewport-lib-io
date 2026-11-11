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

use std::path::Path;

use crate::error::IoError;
use crate::types::SpriteSheetData;

// Read only by the tests until the document path is wired into the sheet.
#[allow(dead_code)]
mod raw;
mod sheet;

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
        sheet::decode(bytes, base)
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
