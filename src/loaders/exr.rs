use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;

/// Decode an OpenEXR file into RGBA8 pixels.
///
/// EXR is an HDR float format. For consumers that drive a non-HDR
/// pipeline (the typical case for an albedo texture), values are
/// clamped into `0..=1` and quantised to 8-bit. Authored HDRP albedos
/// usually already sit in that range, so the clamp is rarely active in
/// practice.
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
        Ok(TextureData {
            width,
            height,
            rgba: rgba.into_raw(),
        })
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
