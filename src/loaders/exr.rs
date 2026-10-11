use std::path::Path;

use crate::error::IoError;
use viewport_lib_types::data::texture::TextureData;

/// Decode an OpenEXR file into an 8-bit RGBA texture labelled linear.
///
/// EXR is an HDR float format. For consumers that drive a non-HDR
/// pipeline (the typical case for an albedo texture), values are
/// clamped into `0..=1` and quantised to 8-bit. Authored HDRP albedos
/// usually already sit in that range, so the clamp is rarely active in
/// practice.
///
/// EXR stores linear values and the quantisation applies no transfer curve,
/// so the bytes are linear and the texture says so, whatever slot it is for.
/// Sampling it as sRGB would darken it.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    #[cfg(feature = "exr")]
    {
        use image::ImageReader;

        let reader = ImageReader::open(path)?
            .with_guessed_format()
            .map_err(|error| {
                IoError::Parse(format!("failed to probe EXR image format: {error}"))
            })?;
        let image = reader
            .decode()
            .map_err(|error| IoError::Parse(format!("failed to decode EXR image: {error}")))?;

        // `to_rgba8` performs the float-to-8-bit conversion the `image`
        // crate calls "linear", which clamps + scales each channel.
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(TextureData::linear(width, height, rgba.into_raw()))
    }

    #[cfg(not(feature = "exr"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "exr",
            context: "EXR texture decoding",
        })
    }
}

/// Decode an EXR from in-memory bytes into an 8-bit RGBA texture labelled linear. The in-memory sibling of
/// [`texture_from_path`], for an image served from a cooked bundle or fetched over
/// the network rather than read from a file.
pub fn texture_from_bytes(bytes: &[u8]) -> Result<TextureData, IoError> {
    #[cfg(feature = "exr")]
    {
        use image::ImageReader;

        let reader = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| {
                IoError::Parse(format!("failed to probe EXR image format: {error}"))
            })?;
        let image = reader
            .decode()
            .map_err(|error| IoError::Parse(format!("failed to decode EXR image: {error}")))?;

        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(TextureData::linear(width, height, rgba.into_raw()))
    }

    #[cfg(not(feature = "exr"))]
    {
        let _ = bytes;
        Err(IoError::MissingFeature {
            feature: "exr",
            context: "EXR texture decoding",
        })
    }
}
