use std::path::Path;

use crate::error::IoError;
use crate::types::HdrTextureData;

/// Decode a Radiance HDR file into RGBA32F pixel data.
pub fn texture_from_path(path: &Path) -> Result<HdrTextureData, IoError> {
    #[cfg(feature = "hdr")]
    {
        let image = image::open(path)
            .map_err(|error| IoError::Parse(format!("failed to decode HDR image: {error}")))?;
        let image = image.to_rgb32f();
        let (width, height) = image.dimensions();

        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for pixel in image.pixels() {
            rgba.push(pixel.0[0]);
            rgba.push(pixel.0[1]);
            rgba.push(pixel.0[2]);
            rgba.push(1.0);
        }

        Ok(HdrTextureData {
            width,
            height,
            rgba,
        })
    }

    #[cfg(not(feature = "hdr"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "hdr",
            context: "HDR environment decoding",
        })
    }
}
