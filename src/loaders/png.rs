use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;

/// Decode a PNG file into RGBA8 pixels.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    #[cfg(feature = "png")]
    {
        let image = image::open(path)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData {
            width,
            height,
            rgba: image.into_raw(),
        })
    }

    #[cfg(not(feature = "png"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "png",
            context: "PNG texture decoding",
        })
    }
}

/// Decode a PNG from in-memory bytes into RGBA8 pixels.
///
/// The in-memory sibling of [`texture_from_path`], for a texture served from a
/// cooked bundle (or fetched over the network for a `wasm32` target) rather than
/// read from a file.
pub fn texture_from_bytes(bytes: &[u8]) -> Result<TextureData, IoError> {
    #[cfg(feature = "png")]
    {
        let image = image::load_from_memory(bytes)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData {
            width,
            height,
            rgba: image.into_raw(),
        })
    }

    #[cfg(not(feature = "png"))]
    {
        let _ = bytes;
        Err(IoError::MissingFeature {
            feature: "png",
            context: "PNG texture decoding",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "viewport_lib_io_{name}_{}_{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[cfg(feature = "png")]
    #[test]
    fn decodes_png_rgba_pixels() {
        let path = temp_path("png_decode.png");
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];

        std::fs::write(&path, png_bytes).unwrap();
        let image = texture_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(image.width, 1);
        assert_eq!(image.height, 1);
        assert_eq!(image.rgba, vec![255, 0, 0, 255]);
    }

    #[cfg(feature = "png")]
    #[test]
    fn from_bytes_matches_from_path() {
        // The in-memory decode must produce exactly what the file decode does, so a
        // cooked-bundle asset renders identically to the loose one.
        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let path = temp_path("png_bytes.png");
        std::fs::write(&path, png_bytes).unwrap();
        let from_path = texture_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let from_bytes = texture_from_bytes(png_bytes).unwrap();
        assert_eq!(from_bytes.width, from_path.width);
        assert_eq!(from_bytes.height, from_path.height);
        assert_eq!(from_bytes.rgba, from_path.rgba);
    }

    #[cfg(feature = "png")]
    #[test]
    fn missing_png_file_returns_parse_error() {
        let path = temp_path("missing.png");

        let err = texture_from_path(&path).unwrap_err();
        match err {
            IoError::Parse(message) => {
                assert!(message.contains("No such file or directory"));
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
