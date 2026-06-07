use std::path::Path;

use crate::error::IoError;
use crate::types::HdrTextureData;

/// Caps applied to the `image` decoder when loading an HDR file.
///
/// `viewport-lib-io`'s default behaviour for [`texture_from_path`] is to
/// decode trusted local assets without a size limit, since production
/// HDRIs (8K, 10K, 16K) routinely exceed the `image` crate's 512 MB
/// default budget. Embedders that ingest untrusted HDRs can opt back
/// into a cap by calling [`texture_from_path_with_limits`] with a
/// populated `DecodeLimits`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeLimits {
    /// Maximum total intermediate allocation, in bytes. `None` removes
    /// the cap entirely.
    pub max_alloc_bytes: Option<u64>,
    /// Maximum image dimension (width or height), in pixels. `None`
    /// removes the cap entirely.
    pub max_image_dimension: Option<u32>,
}

impl DecodeLimits {
    /// No caps: decode whatever the file claims. Matches the default
    /// behaviour of [`texture_from_path`].
    pub fn unlimited() -> Self {
        Self {
            max_alloc_bytes: None,
            max_image_dimension: None,
        }
    }

    /// Mirrors the `image` crate's historical defaults (~512 MB
    /// intermediate budget, no dimension cap). Suitable for embedders
    /// that want to refuse pathological inputs.
    pub fn image_crate_defaults() -> Self {
        Self {
            max_alloc_bytes: Some(512 * 1024 * 1024),
            max_image_dimension: None,
        }
    }
}

/// Decode a Radiance HDR file into RGBA32F pixel data.
///
/// Decodes without an allocation cap so production-sized HDR panoramas
/// (8K and above) load successfully. Use
/// [`texture_from_path_with_limits`] to apply a custom cap.
pub fn texture_from_path(path: &Path) -> Result<HdrTextureData, IoError> {
    texture_from_path_with_limits(path, DecodeLimits::unlimited())
}

/// Decode a Radiance HDR file into RGBA32F pixel data with explicit
/// decoder caps.
///
/// Pass [`DecodeLimits::unlimited`] (the default for
/// [`texture_from_path`]) for trusted local assets, or
/// [`DecodeLimits::image_crate_defaults`] to refuse anything the
/// `image` crate would have refused historically.
pub fn texture_from_path_with_limits(
    path: &Path,
    limits: DecodeLimits,
) -> Result<HdrTextureData, IoError> {
    #[cfg(feature = "hdr")]
    {
        use image::ImageReader;

        let mut reader = ImageReader::open(path)?
            .with_guessed_format()
            .map_err(|error| {
                IoError::Parse(format!("failed to probe HDR image format: {error}"))
            })?;

        let mut image_limits = image::Limits::default();
        image_limits.max_alloc = limits.max_alloc_bytes;
        image_limits.max_image_width = limits.max_image_dimension;
        image_limits.max_image_height = limits.max_image_dimension;
        reader.limits(image_limits);

        let image = reader
            .decode()
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
        let _ = (path, limits);
        Err(IoError::MissingFeature {
            feature: "hdr",
            context: "HDR environment decoding",
        })
    }
}
