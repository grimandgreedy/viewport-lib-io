use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;

/// Decode a BMP (Windows bitmap) file into RGBA8 pixels.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    #[cfg(feature = "bmp")]
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

    #[cfg(not(feature = "bmp"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "bmp",
            context: "BMP texture decoding",
        })
    }
}
