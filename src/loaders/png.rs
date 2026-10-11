use std::path::Path;

use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::TextureSlot;

use crate::error::IoError;

/// Decode a PNG file into an 8-bit RGBA texture labelled sRGB colour.
///
/// Right for base colour, emissive, and images that are not material maps. For
/// a map whose slot is known, use [`texture_from_path_for_slot`] so a normal or
/// data map is labelled linear; anything else can be relabelled with
/// [`TextureData::with_colour_space`].
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    texture_from_path_for_slot(path, TextureSlot::Albedo)
}

/// Decode a PNG file into an 8-bit RGBA texture labelled for the material slot
/// it fills: sRGB for albedo and emissive, a normal map for the normal slot,
/// linear for the rest.
pub fn texture_from_path_for_slot(path: &Path, slot: TextureSlot) -> Result<TextureData, IoError> {
    #[cfg(feature = "png")]
    {
        let image = image::open(path)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData::for_slot(width, height, image.into_raw(), slot))
    }

    #[cfg(not(feature = "png"))]
    {
        let _ = (path, slot);
        Err(IoError::MissingFeature {
            feature: "png",
            context: "PNG texture decoding",
        })
    }
}

/// Decode a PNG from in-memory bytes into an 8-bit RGBA texture labelled sRGB
/// colour.
///
/// The in-memory sibling of [`texture_from_path`], for a texture served from a
/// cooked bundle (or fetched over the network for a `wasm32` target) rather than
/// read from a file.
pub fn texture_from_bytes(bytes: &[u8]) -> Result<TextureData, IoError> {
    texture_from_bytes_for_slot(bytes, TextureSlot::Albedo)
}

/// Decode a PNG from in-memory bytes into an 8-bit RGBA texture labelled for
/// the material slot it fills. The in-memory sibling of
/// [`texture_from_path_for_slot`].
pub fn texture_from_bytes_for_slot(
    bytes: &[u8],
    slot: TextureSlot,
) -> Result<TextureData, IoError> {
    #[cfg(feature = "png")]
    {
        let image = image::load_from_memory(bytes)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData::for_slot(width, height, image.into_raw(), slot))
    }

    #[cfg(not(feature = "png"))]
    {
        let _ = (bytes, slot);
        Err(IoError::MissingFeature {
            feature: "png",
            context: "PNG texture decoding",
        })
    }
}

#[cfg(test)]
mod tests {
    use viewport_lib_types::colour::ColourSpace;
    use viewport_lib_types::data::texture::{TexturePayload, TextureRole};

    use super::*;

    fn rgba8(texture: &TextureData) -> &[u8] {
        match texture.payload() {
            TexturePayload::Rgba8(bytes) => bytes,
            _ => panic!("expected an 8-bit payload"),
        }
    }

    const RED_PIXEL: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8,
        0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99, 0x3D, 0x1D, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

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

    #[test]
    fn decodes_png_rgba_pixels() {
        let path = temp_path("png_decode.png");
        let png_bytes = RED_PIXEL;

        std::fs::write(&path, png_bytes).unwrap();
        let image = texture_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(image.width(), 1);
        assert_eq!(image.height(), 1);
        assert_eq!(rgba8(&image), &[255, 0, 0, 255]);
        assert_eq!(image.colour_space(), ColourSpace::Srgb);
    }

    #[test]
    fn from_bytes_matches_from_path() {
        // The in-memory decode must produce exactly what the file decode does, so a
        // cooked-bundle asset renders identically to the loose one.
        let png_bytes = RED_PIXEL;
        let path = temp_path("png_bytes.png");
        std::fs::write(&path, png_bytes).unwrap();
        let from_path = texture_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let from_bytes = texture_from_bytes(png_bytes).unwrap();
        assert!(from_bytes == from_path);
    }

    #[test]
    fn a_slot_labels_the_texture_for_that_slot() {
        let normal = texture_from_bytes_for_slot(RED_PIXEL, TextureSlot::Normal).unwrap();
        assert_eq!(normal.colour_space(), ColourSpace::Linear);
        assert_eq!(normal.role(), TextureRole::NormalMap);

        let roughness =
            texture_from_bytes_for_slot(RED_PIXEL, TextureSlot::MetallicRoughness).unwrap();
        assert_eq!(roughness.colour_space(), ColourSpace::Linear);
        assert_eq!(roughness.role(), TextureRole::Image);

        let emissive = texture_from_bytes_for_slot(RED_PIXEL, TextureSlot::Emissive).unwrap();
        assert_eq!(emissive.colour_space(), ColourSpace::Srgb);

        // The pixels are passed through untouched whatever the label.
        assert_eq!(rgba8(&normal), rgba8(&emissive));
    }

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
