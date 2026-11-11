//! Tiled map and tileset decoding.
//!
//! Reads TMX and TSX (XML) and TMJ and TSJ (JSON), picking the format from the
//! first byte rather than the extension. `raw` parses either into one element
//! tree, `tilemap` walks it into the neutral types, `buffers` decodes tile
//! layer data and `attributes` reads values and custom properties.
//!
//! Tile layer data may be XML `<tile>` elements, CSV, or base64 with no
//! compression, zlib or gzip. Zstandard-compressed data is refused with an
//! error naming it. External tilesets and object templates are read from
//! beside the map. Wang sets, terrain and editor settings are not carried.

use std::path::Path;

use crate::error::IoError;
use crate::types::{TileMapData, TileSetData};

mod attributes;
mod buffers;
mod raw;
mod tilemap;

/// Decode a Tiled map (`.tmx` or `.tmj`) into a [`TileMapData`].
///
/// External tilesets, object templates and image paths are resolved against
/// the map's directory. A tileset or template that cannot be read is an
/// [`IoError::MissingDependency`] naming its path. Images are not read: each
/// is a [`TextureSource::File`](crate::types::TextureSource::File) for the
/// consumer to decode.
pub fn tilemap_from_path(path: &Path) -> Result<TileMapData, IoError> {
    #[cfg(feature = "tiled")]
    {
        let bytes = std::fs::read(path)?;
        tilemap_from_bytes(&bytes, Some(path.parent().unwrap_or(Path::new("."))))
    }

    #[cfg(not(feature = "tiled"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "tiled",
            context: "Tiled map decoding",
        })
    }
}

/// Decode a Tiled map from bytes already read.
///
/// `base` is the directory the map's references are relative to. Without one,
/// a map that uses an external tileset or an object template is an error, and
/// image paths are left as the file wrote them.
pub fn tilemap_from_bytes(bytes: &[u8], base: Option<&Path>) -> Result<TileMapData, IoError> {
    #[cfg(feature = "tiled")]
    {
        tilemap::map(&raw::parse(bytes)?, base)
    }

    #[cfg(not(feature = "tiled"))]
    {
        let _ = (bytes, base);
        Err(IoError::MissingFeature {
            feature: "tiled",
            context: "Tiled map decoding",
        })
    }
}

/// Decode a Tiled tileset file (`.tsx` or `.tsj`) on its own.
///
/// Its [`TileSetData::first_gid`] is `0`, since no map gives it a range, and
/// its image paths are resolved against its own directory.
pub fn tileset_from_path(path: &Path) -> Result<TileSetData, IoError> {
    #[cfg(feature = "tiled")]
    {
        let bytes = std::fs::read(path)?;
        tilemap::standalone_tileset(&raw::parse(&bytes)?, path)
    }

    #[cfg(not(feature = "tiled"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "tiled",
            context: "Tiled tileset decoding",
        })
    }
}
