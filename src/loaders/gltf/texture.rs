//! The images behind material textures, converted to `TextureData` for the slot
//! that samples them.

use std::path::Path;

use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::TextureSlot;

use crate::types::TextureSource;

/// Resolve the image behind a texture to a [`TextureSource`].
///
/// An image the document embedded (a GLB chunk or a data URI) comes back as
/// decoded pixels labelled for `slot`; one stored beside the file comes back as
/// a path resolved against `parent_dir`, so the caller can decide when to read
/// it.
pub(super) fn image_to_texture_source(
    texture: &gltf::Texture,
    slot: TextureSlot,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> Option<TextureSource> {
    let source = texture.source();
    let index = source.index();

    if let Some(data) = images.get(index) {
        return Some(TextureSource::Decoded(TextureData::for_slot(
            data.width,
            data.height,
            to_rgba8(data),
            slot,
        )));
    }

    match source.source() {
        gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => {
            Some(TextureSource::File(parent_dir.join(uri)))
        }
        _ => None,
    }
}

/// Expand any glTF image format to RGBA8.
///
/// Narrower formats are widened rather than rejected: a missing alpha channel
/// becomes opaque, and a single channel is broadcast to RGB.
pub(super) fn to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;

    match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 3 * 4);
            for rgb in data.pixels.chunks_exact(3) {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            rgba
        }
        Format::R8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
            for &r in &data.pixels {
                rgba.extend_from_slice(&[r, r, r, 255]);
            }
            rgba
        }
        Format::R8G8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 2 * 4);
            for rg in data.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            rgba
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let bytes_per_pixel = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * bytes_per_pixel * 2;
                let mut channels = [0u8; 4];
                for c in 0..bytes_per_pixel {
                    let lo = data.pixels.get(base + c * 2).copied().unwrap_or(0);
                    let hi = data.pixels.get(base + c * 2 + 1).copied().unwrap_or(0);
                    channels[c] = (u16::from_le_bytes([lo, hi]) >> 8) as u8;
                }
                match bytes_per_pixel {
                    1 => {
                        channels[1] = channels[0];
                        channels[2] = channels[0];
                        channels[3] = 255;
                    }
                    2 => {
                        channels[2] = 0;
                        channels[3] = 255;
                    }
                    3 => channels[3] = 255,
                    _ => {}
                }
                rgba.extend_from_slice(&channels);
            }
            rgba
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * channels * 4;
                let mut out = [0u8; 4];
                for c in 0..channels.min(4) {
                    let bytes = [
                        data.pixels.get(base + c * 4).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 1).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 2).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 3).copied().unwrap_or(0),
                    ];
                    let value = f32::from_le_bytes(bytes);
                    out[c] = (value.clamp(0.0, 1.0) * 255.0) as u8;
                }
                if channels < 4 {
                    out[3] = 255;
                }
                rgba.extend_from_slice(&out);
            }
            rgba
        }
    }
}
