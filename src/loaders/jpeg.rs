use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;

/// Decode a JPEG file into RGBA8 pixels.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    #[cfg(feature = "jpeg")]
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

    #[cfg(not(feature = "jpeg"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "jpeg",
            context: "JPEG texture decoding",
        })
    }
}
