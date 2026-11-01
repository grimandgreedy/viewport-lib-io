//! The images behind material textures, converted to `TextureData` for the slot
//! that samples them.

use std::path::{Path, PathBuf};

use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::TextureSlot;

use crate::types::TextureSource;

/// Every image in the document, decoded once, for the materials to draw on.
///
/// Each image decodes on its own, so one the loader cannot read (an unsupported
/// encoding, a missing file) leaves the others intact.
pub(super) struct ImageTable {
    /// Decoded pixels per document image, `None` where the image did not decode.
    decoded: Vec<Option<gltf::image::Data>>,
    /// Directory that external image URIs resolve against.
    parent_dir: PathBuf,
}

impl ImageTable {
    /// Decode every image in `document`. External URIs resolve against `base`;
    /// with no base only embedded images (GLB chunks) decode.
    pub(super) fn load(
        document: &gltf::Document,
        base: Option<&Path>,
        buffers: &[gltf::buffer::Data],
    ) -> Self {
        let decoded = document
            .images()
            .map(|image| gltf::image::Data::from_source(image.source(), base, buffers).ok())
            .collect();
        Self {
            decoded,
            // With no base, a path is still returned relative to the working
            // directory, as a last resort.
            parent_dir: base.unwrap_or(Path::new(".")).to_path_buf(),
        }
    }

    /// Resolve the image behind a texture to a [`TextureSource`], labelled for
    /// `slot`. `None` for a texture with no image.
    ///
    /// An image the document embedded (a GLB chunk or a data URI) comes back as
    /// decoded pixels; one stored beside the file that did not decode comes back
    /// as a path resolved against the base directory, so the caller can decide
    /// when to read it.
    pub(super) fn texture_source(
        &self,
        texture: &gltf::Texture,
        slot: TextureSlot,
    ) -> Option<TextureSource> {
        let image = texture.source()?;
        if let Some(Some(data)) = self.decoded.get(image.index()) {
            return Some(TextureSource::Decoded(TextureData::for_slot(
                data.width,
                data.height,
                to_rgba8(data),
                slot,
            )));
        }

        match image.source() {
            gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => {
                Some(TextureSource::File(self.parent_dir.join(uri)))
            }
            _ => None,
        }
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

#[cfg(test)]
mod tests {
    use super::super::scene_from_slice;
    use crate::types::TextureSource;
    use viewport_lib_io_testkit::synth::glb;

    /// A 1x1 RGBA PNG.
    const PNG_1X1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8,
        0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99, 0x3D, 0x1D, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// One triangle and three materials: one on an image that does not decode,
    /// one on a good PNG, and one on a texture with no image at all.
    fn three_texture_glb() -> Vec<u8> {
        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        let garbage_offset = bin.len();
        bin.extend_from_slice(b"not a png at all");
        let png_offset = bin.len();
        bin.extend_from_slice(PNG_1X1);
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,
            "scenes":[{{"nodes":[0,1,2]}}],
            "nodes":[{{"mesh":0}},{{"mesh":1}},{{"mesh":2}}],
            "meshes":[
              {{"primitives":[{{"attributes":{{"POSITION":0}},"material":0}}]}},
              {{"primitives":[{{"attributes":{{"POSITION":0}},"material":1}}]}},
              {{"primitives":[{{"attributes":{{"POSITION":0}},"material":2}}]}}],
            "materials":[
              {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}}}}}},
              {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":1}}}}}},
              {{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":2}}}}}}],
            "textures":[{{"source":0}},{{"source":1}},{{}}],
            "images":[
              {{"bufferView":1,"mimeType":"image/png"}},
              {{"bufferView":2,"mimeType":"image/png"}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
              "min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}}],
            "bufferViews":[
              {{"buffer":0,"byteOffset":0,"byteLength":36}},
              {{"buffer":0,"byteOffset":{garbage_offset},"byteLength":16}},
              {{"buffer":0,"byteOffset":{png_offset},"byteLength":{png_len}}}],
            "buffers":[{{"byteLength":{bin_len}}}]}}"#,
            png_len = PNG_1X1.len(),
            bin_len = bin.len(),
        );
        glb(json.as_bytes(), &bin)
    }

    #[test]
    fn one_bad_image_leaves_the_others_and_a_sourceless_texture_is_skipped() {
        let scene = scene_from_slice(&three_texture_glb(), None).unwrap();
        assert_eq!(scene.materials.len(), 3);

        assert!(
            scene.materials[0].base_color_texture.is_none(),
            "an embedded image that does not decode has nothing to fall back to"
        );
        match &scene.materials[1].base_color_texture {
            Some(TextureSource::Decoded(image)) => {
                assert_eq!((image.width(), image.height()), (1, 1));
            }
            other => panic!("expected the good PNG to decode, got {other:?}"),
        }
        assert!(scene.materials[2].base_color_texture.is_none());
    }
}
