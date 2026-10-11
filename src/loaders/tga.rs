use std::path::Path;

use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::TextureSlot;

use crate::error::IoError;

/// Decode a TGA (Truevision Targa) file into an 8-bit RGBA texture labelled sRGB colour.
///
/// Right for base colour, emissive, and images that are not material maps. For
/// a map whose slot is known, use [`texture_from_path_for_slot`] so a normal or
/// data map is labelled linear; anything else can be relabelled with
/// [`TextureData::with_colour_space`].
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    texture_from_path_for_slot(path, TextureSlot::Albedo)
}

/// Decode a TGA (Truevision Targa) file into an 8-bit RGBA texture labelled for the material slot
/// it fills: sRGB for albedo and emissive, a normal map for the normal slot,
/// linear for the rest.
pub fn texture_from_path_for_slot(path: &Path, slot: TextureSlot) -> Result<TextureData, IoError> {
    #[cfg(feature = "tga")]
    {
        let image = image::open(path)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData::for_slot(width, height, image.into_raw(), slot))
    }

    #[cfg(not(feature = "tga"))]
    {
        let _ = (path, slot);
        Err(IoError::MissingFeature {
            feature: "tga",
            context: "TGA texture decoding",
        })
    }
}

/// Decode a TGA from in-memory bytes into an 8-bit RGBA texture labelled sRGB
/// colour.
///
/// The in-memory sibling of [`texture_from_path`], for a texture served from a
/// cooked bundle or fetched over the network rather than read from a file.
pub fn texture_from_bytes(bytes: &[u8]) -> Result<TextureData, IoError> {
    texture_from_bytes_for_slot(bytes, TextureSlot::Albedo)
}

/// Decode a TGA from in-memory bytes into an 8-bit RGBA texture labelled for
/// the material slot it fills. The in-memory sibling of
/// [`texture_from_path_for_slot`].
pub fn texture_from_bytes_for_slot(
    bytes: &[u8],
    slot: TextureSlot,
) -> Result<TextureData, IoError> {
    #[cfg(feature = "tga")]
    {
        let image = image::load_from_memory(bytes)
            .map_err(|error| IoError::Parse(format!("failed to decode image: {error}")))?;
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        Ok(TextureData::for_slot(width, height, image.into_raw(), slot))
    }

    #[cfg(not(feature = "tga"))]
    {
        let _ = (bytes, slot);
        Err(IoError::MissingFeature {
            feature: "tga",
            context: "TGA texture decoding",
        })
    }
}
