use std::path::Path;

use crate::error::IoError;
use crate::scene::TextureData;

/// Decode an image file into RGBA8 pixels.
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
            context: "texture decoding",
        })
    }
}
