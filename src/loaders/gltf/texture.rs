//! The images behind material textures, converted to `TextureData` for the slot
//! that samples them.

use std::path::{Path, PathBuf};

use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::TextureSlot;

use super::options::GltfLoadOptions;
use crate::types::TextureSource;

/// Every image in the document, decoded once, for the materials to draw on.
///
/// Each image decodes on its own, so one the loader cannot read (an unsupported
/// encoding, a missing file) leaves the others intact. KTX2 images are kept as
/// their file bytes and transcoded per texture slot, since the slot decides the
/// target format.
pub(super) struct ImageTable {
    /// Decoded pixels per document image, `None` where the image did not decode.
    decoded: Vec<Option<gltf::image::Data>>,
    /// KTX2 file bytes per document image, `None` for any other image.
    #[cfg(feature = "gltf-ktx2")]
    ktx2: Vec<Option<Vec<u8>>>,
    #[cfg(feature = "gltf-ktx2")]
    target: super::options::GltfTextureTarget,
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
        options: &GltfLoadOptions,
    ) -> Self {
        #[cfg(feature = "gltf-ktx2")]
        let ktx2: Vec<Option<Vec<u8>>> = document
            .images()
            .map(|image| ktx2::image_bytes(&image, base, buffers))
            .collect();
        let decoded = document
            .images()
            .map(|image| {
                #[cfg(feature = "gltf-ktx2")]
                if ktx2[image.index()].is_some() {
                    return None;
                }
                gltf::image::Data::from_source(image.source(), base, buffers).ok()
            })
            .collect();
        #[cfg(not(feature = "gltf-ktx2"))]
        let _ = options;
        Self {
            decoded,
            #[cfg(feature = "gltf-ktx2")]
            ktx2,
            #[cfg(feature = "gltf-ktx2")]
            target: options.texture_target,
            // With no base, a path is still returned relative to the working
            // directory, as a last resort.
            parent_dir: base.unwrap_or(Path::new(".")).to_path_buf(),
        }
    }

    /// Resolve the image behind a texture to a [`TextureSource`], labelled for
    /// `slot`. `None` for a texture with no image.
    ///
    /// A texture with a `KHR_texture_basisu` image is transcoded from it, and
    /// falls back to its plain `source` only when the KTX2 image does not
    /// transcode. An image the document embedded (a GLB chunk or a data URI)
    /// comes back as decoded pixels; one stored beside the file that did not
    /// decode comes back as a path resolved against the base directory, so the
    /// caller can decide when to read it.
    pub(super) fn texture_source(
        &self,
        texture: &gltf::Texture,
        slot: TextureSlot,
    ) -> Option<TextureSource> {
        #[cfg(feature = "gltf-ktx2")]
        {
            let basisu_source = texture
                .extension_value(ktx2::EXTENSION)
                .and_then(|extension| extension.get("source"))
                .and_then(serde_json::Value::as_u64)
                .map(|index| index as usize);
            for index in basisu_source
                .into_iter()
                .chain(texture.source().map(|i| i.index()))
            {
                if let Some(Some(bytes)) = self.ktx2.get(index) {
                    if let Some(data) = ktx2::transcode(bytes, slot, self.target) {
                        return Some(TextureSource::Decoded(data));
                    }
                }
            }
        }

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

#[cfg(feature = "gltf-ktx2")]
mod ktx2 {
    use std::path::Path;

    use basisu::{DecodeFlags, SourceFormat, TargetFormat, Transcoder};
    use viewport_lib_types::data::texture::{AstcBlock, CompressedFormat, TextureData};
    use viewport_lib_types::material::TextureSlot;

    use super::super::options::GltfTextureTarget;

    pub(super) const EXTENSION: &str = "KHR_texture_basisu";

    const KTX2_MAGIC: [u8; 12] = [
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ];

    /// The file bytes of `image` when it is a KTX2 image (by MIME type, file
    /// extension or magic number), `None` for any other image or one that
    /// cannot be read.
    pub(super) fn image_bytes(
        image: &gltf::Image,
        base: Option<&Path>,
        buffers: &[gltf::buffer::Data],
    ) -> Option<Vec<u8>> {
        let bytes = match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let bytes = buffers
                    .get(view.buffer().index())?
                    .0
                    .get(view.offset()..view.offset() + view.length())?;
                if mime_type != "image/ktx2" && !bytes.starts_with(&KTX2_MAGIC) {
                    return None;
                }
                bytes.to_vec()
            }
            gltf::image::Source::Uri { uri, mime_type } => {
                let named = mime_type == Some("image/ktx2")
                    || uri.starts_with("data:image/ktx2")
                    || uri.to_ascii_lowercase().ends_with(".ktx2");
                if !named {
                    return None;
                }
                // The buffer reader resolves data, file and relative URIs alike.
                gltf::buffer::Data::from_source(gltf::buffer::Source::Uri(uri), base)
                    .ok()?
                    .0
            }
        };
        bytes.starts_with(&KTX2_MAGIC).then_some(bytes)
    }

    /// What a slot samples, which decides the format within a family.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Role {
        /// Base colour and emissive: sRGB colour, alpha kept.
        Colour,
        /// Metallic-roughness: linear, green and blue read.
        Data,
        /// Occlusion: linear, red read.
        Occlusion,
        /// Tangent-space normals.
        Normal,
    }

    impl Role {
        fn of(slot: TextureSlot) -> Self {
            match slot {
                TextureSlot::Albedo | TextureSlot::Emissive => Role::Colour,
                TextureSlot::MetallicRoughness => Role::Data,
                TextureSlot::Ao => Role::Occlusion,
                TextureSlot::Normal => Role::Normal,
            }
        }
    }

    /// Transcode a KTX2 file for `slot` into the format `target` names for it,
    /// with every mip level the file holds. `None` when the file does not
    /// transcode (not Basis Universal, a cubemap, array or video, or corrupt).
    ///
    /// Falls back to RGBA8 (level 0 only) when the target family cannot carry
    /// the image: dimensions that are not a multiple of the 4x4 block, or a
    /// two-channel normal map under ASTC, whose X and Y sit in red and alpha
    /// rather than red and green.
    pub(super) fn transcode(
        bytes: &[u8],
        slot: TextureSlot,
        target: GltfTextureTarget,
    ) -> Option<TextureData> {
        let transcoder = Transcoder::new(bytes).ok()?;
        if transcoder.is_video() || transcoder.face_count() != 1 || transcoder.layer_count() > 1 {
            return None;
        }
        let (width, height) = transcoder.base_dimensions();
        let role = Role::of(slot);

        let hdr = matches!(
            transcoder.source_format(),
            SourceFormat::UastcHdr4x4 | SourceFormat::UastcHdr6x6 | SourceFormat::AstcHdr6x6
        );
        if hdr {
            return transcode_hdr(&transcoder, width, height, target);
        }

        // A two-channel normal map stores X in RGB and Y in alpha, the layout
        // Basis Universal's normal-map mode writes. Grey RGB with alpha is the
        // tell; a three-channel normal map never has R = G = B throughout.
        let base_rgba = || {
            transcoder
                .transcode(0, TargetFormat::Rgba32, DecodeFlags::NONE)
                .ok()
        };
        // Only a normal map with alpha needs its texels looked at.
        let mut rgba = None;
        let two_channel_normal = role == Role::Normal && transcoder.has_alpha() && {
            rgba = base_rgba();
            rgba.as_deref().is_some_and(is_grey)
        };

        let chosen = match target {
            GltfTextureTarget::Rgba8 => None,
            GltfTextureTarget::Bc => Some(match role {
                Role::Colour | Role::Data => (TargetFormat::Bc7Rgba, CompressedFormat::Bc7Rgba),
                Role::Occlusion => (TargetFormat::Bc4R, CompressedFormat::Bc4R),
                Role::Normal if two_channel_normal => {
                    (TargetFormat::Bc5Rg, CompressedFormat::Bc5Rg)
                }
                Role::Normal => (TargetFormat::Bc7Rgba, CompressedFormat::Bc7Rgba),
            }),
            GltfTextureTarget::Etc2Eac => Some(match role {
                Role::Colour if transcoder.has_alpha() => {
                    (TargetFormat::Etc2Rgba, CompressedFormat::Etc2Rgba8)
                }
                // ETC1 blocks are valid ETC2 RGB blocks.
                Role::Colour | Role::Data => (TargetFormat::Etc1Rgb, CompressedFormat::Etc2Rgb8),
                Role::Occlusion => (TargetFormat::EacR11, CompressedFormat::EacR11),
                Role::Normal if two_channel_normal => {
                    (TargetFormat::EacRg11, CompressedFormat::EacRg11)
                }
                Role::Normal => (TargetFormat::Etc1Rgb, CompressedFormat::Etc2Rgb8),
            }),
            GltfTextureTarget::Astc4x4 if two_channel_normal => None,
            GltfTextureTarget::Astc4x4 => Some((
                TargetFormat::Astc4x4Rgba,
                CompressedFormat::Astc {
                    block: AstcBlock::B4x4,
                    hdr: false,
                },
            )),
        };

        if let Some((source_target, format)) = chosen {
            if width % 4 == 0 && height % 4 == 0 && transcoder.supports(source_target) {
                if let Some(levels) = all_levels(&transcoder, source_target) {
                    return Some(match role {
                        Role::Normal => {
                            TextureData::compressed_normal_map(width, height, format, levels)
                        }
                        _ => TextureData::compressed(
                            width,
                            height,
                            format,
                            slot.colour_space(),
                            levels,
                        ),
                    });
                }
            }
        }

        let rgba = match rgba {
            Some(rgba) => rgba,
            None => base_rgba()?,
        };
        let rgba = if two_channel_normal {
            rebuild_normals(&rgba)
        } else {
            rgba
        };
        Some(TextureData::for_slot(width, height, rgba, slot))
    }

    /// An HDR image: BC6H or ASTC HDR when the family has one, otherwise float
    /// RGBA. Always linear, since HDR formats hold data rather than encoded
    /// colour.
    fn transcode_hdr(
        transcoder: &Transcoder,
        width: u32,
        height: u32,
        target: GltfTextureTarget,
    ) -> Option<TextureData> {
        use viewport_lib_types::colour::ColourSpace;

        let chosen = match target {
            GltfTextureTarget::Bc => Some((TargetFormat::Bc6h, CompressedFormat::Bc6hRgb)),
            GltfTextureTarget::Astc4x4 => Some((
                TargetFormat::AstcHdr4x4Rgba,
                CompressedFormat::Astc {
                    block: AstcBlock::B4x4,
                    hdr: true,
                },
            )),
            GltfTextureTarget::Rgba8 | GltfTextureTarget::Etc2Eac => None,
        };
        if let Some((source_target, format)) = chosen {
            if width % 4 == 0 && height % 4 == 0 && transcoder.supports(source_target) {
                if let Some(levels) = all_levels(transcoder, source_target) {
                    return Some(TextureData::compressed(
                        width,
                        height,
                        format,
                        ColourSpace::Linear,
                        levels,
                    ));
                }
            }
        }
        let half = transcoder
            .transcode(0, TargetFormat::RgbaHalf, DecodeFlags::NONE)
            .ok()?;
        let rgba = half
            .chunks_exact(2)
            .map(|bits| half_to_f32(u16::from_le_bytes([bits[0], bits[1]])))
            .collect();
        Some(TextureData::hdr(width, height, rgba))
    }

    /// Every level of the file, transcoded to `target`, level 0 first.
    fn all_levels(transcoder: &Transcoder, target: TargetFormat) -> Option<Vec<Vec<u8>>> {
        (0..transcoder.level_count())
            .map(|level| transcoder.transcode(level, target, DecodeFlags::NONE).ok())
            .collect()
    }

    /// Whether every texel's red, green and blue agree, within the error of a
    /// lossy encode.
    fn is_grey(rgba: &[u8]) -> bool {
        rgba.chunks_exact(4).all(|texel| {
            let (r, g, b) = (texel[0] as i16, texel[1] as i16, texel[2] as i16);
            (r - g).abs() <= 8 && (r - b).abs() <= 8
        })
    }

    /// Expand a two-channel normal map (X in red, Y in alpha) to three
    /// channels, rebuilding Z from the unit length.
    fn rebuild_normals(rgba: &[u8]) -> Vec<u8> {
        rgba.chunks_exact(4)
            .flat_map(|texel| {
                let x = texel[0] as f32 / 255.0 * 2.0 - 1.0;
                let y = texel[3] as f32 / 255.0 * 2.0 - 1.0;
                let z = (1.0 - x * x - y * y).max(0.0).sqrt();
                let z = ((z * 0.5 + 0.5) * 255.0).round() as u8;
                [texel[0], texel[3], z, 255]
            })
            .collect()
    }

    /// IEEE 754 binary16 to binary32.
    fn half_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = (bits & 0x3FF) as f32;
        match exponent {
            0 => sign * mantissa * 2f32.powi(-24),
            0x1F if mantissa == 0.0 => sign * f32::INFINITY,
            0x1F => f32::NAN,
            _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent as i32 - 15),
        }
    }
    #[cfg(test)]
    mod tests {
        use viewport_lib_io_testkit::synth::ktx2_uastc;
        use viewport_lib_types::colour::ColourSpace;
        use viewport_lib_types::data::texture::{
            AstcBlock, CompressedFormat, TextureData, TexturePayload, TextureRole,
        };
        use viewport_lib_types::material::TextureSlot;

        use super::super::super::options::GltfTextureTarget;
        use super::transcode;

        /// A 16x16 RGB image with its five levels, each block a different colour.
        fn colour_map(alpha: bool) -> Vec<u8> {
            ktx2_uastc(16, 16, true, alpha, true, |level, bx, by| {
                [(level * 40) as u8, (bx * 60) as u8, (by * 60) as u8, 200]
            })
        }

        /// A two-channel normal map: X in grey RGB, Y in alpha.
        fn two_channel_normals() -> Vec<u8> {
            ktx2_uastc(8, 8, true, true, false, |_, bx, _| {
                let x = if bx == 0 { 100 } else { 160 };
                [x, x, x, 90]
            })
        }

        fn compressed(data: &TextureData) -> (CompressedFormat, usize) {
            match data.payload() {
                TexturePayload::Compressed { format, mip_levels } => (*format, mip_levels.len()),
                _ => panic!("expected a compressed payload"),
            }
        }

        fn check(data: &TextureData) {
            data.validate().expect("a payload the renderer accepts");
        }

        #[test]
        fn rgba8_is_the_base_level_uncompressed_and_labelled_for_the_slot() {
            let data = transcode(
                &colour_map(false),
                TextureSlot::Albedo,
                GltfTextureTarget::Rgba8,
            )
            .unwrap();
            check(&data);
            assert!(matches!(data.payload(), TexturePayload::Rgba8(_)));
            assert_eq!((data.width(), data.height()), (16, 16));
            assert_eq!(data.colour_space(), ColourSpace::Srgb);
            let TexturePayload::Rgba8(rgba) = data.payload() else {
                unreachable!()
            };
            // Block (1, 0) of level 0 is (0, 60, 0, 200).
            assert_eq!(&rgba[4 * 4..4 * 4 + 4], &[0, 60, 0, 200]);
        }

        #[test]
        fn bc_picks_per_slot_and_keeps_every_level() {
            let colour = colour_map(true);
            let albedo = transcode(&colour, TextureSlot::Albedo, GltfTextureTarget::Bc).unwrap();
            check(&albedo);
            assert_eq!(compressed(&albedo), (CompressedFormat::Bc7Rgba, 5));
            assert_eq!(albedo.colour_space(), ColourSpace::Srgb);

            let ao = transcode(&colour, TextureSlot::Ao, GltfTextureTarget::Bc).unwrap();
            check(&ao);
            assert_eq!(compressed(&ao), (CompressedFormat::Bc4R, 5));
            assert_eq!(ao.colour_space(), ColourSpace::Linear);

            let mr = transcode(
                &colour,
                TextureSlot::MetallicRoughness,
                GltfTextureTarget::Bc,
            )
            .unwrap();
            check(&mr);
            assert_eq!(compressed(&mr), (CompressedFormat::Bc7Rgba, 5));
            assert_eq!(mr.colour_space(), ColourSpace::Linear);

            let rgb_normal =
                transcode(&colour, TextureSlot::Normal, GltfTextureTarget::Bc).unwrap();
            check(&rgb_normal);
            assert_eq!(compressed(&rgb_normal).0, CompressedFormat::Bc7Rgba);
            assert_eq!(rgb_normal.role(), TextureRole::NormalMap);

            let normal = transcode(
                &two_channel_normals(),
                TextureSlot::Normal,
                GltfTextureTarget::Bc,
            )
            .unwrap();
            check(&normal);
            assert_eq!(compressed(&normal), (CompressedFormat::Bc5Rg, 4));
            assert_eq!(normal.role(), TextureRole::NormalMap);
        }

        #[test]
        fn etc2_eac_picks_per_slot_and_alpha() {
            let with_alpha = transcode(
                &colour_map(true),
                TextureSlot::Albedo,
                GltfTextureTarget::Etc2Eac,
            )
            .unwrap();
            check(&with_alpha);
            assert_eq!(compressed(&with_alpha), (CompressedFormat::Etc2Rgba8, 5));

            let opaque = transcode(
                &colour_map(false),
                TextureSlot::Albedo,
                GltfTextureTarget::Etc2Eac,
            )
            .unwrap();
            check(&opaque);
            assert_eq!(compressed(&opaque), (CompressedFormat::Etc2Rgb8, 5));

            let ao = transcode(
                &colour_map(false),
                TextureSlot::Ao,
                GltfTextureTarget::Etc2Eac,
            )
            .unwrap();
            check(&ao);
            assert_eq!(compressed(&ao), (CompressedFormat::EacR11, 5));

            let normal = transcode(
                &two_channel_normals(),
                TextureSlot::Normal,
                GltfTextureTarget::Etc2Eac,
            )
            .unwrap();
            check(&normal);
            assert_eq!(compressed(&normal), (CompressedFormat::EacRg11, 4));
        }

        #[test]
        fn astc_compresses_colour_and_expands_a_two_channel_normal_map() {
            let albedo = transcode(
                &colour_map(false),
                TextureSlot::Albedo,
                GltfTextureTarget::Astc4x4,
            )
            .unwrap();
            check(&albedo);
            assert_eq!(
                compressed(&albedo),
                (
                    CompressedFormat::Astc {
                        block: AstcBlock::B4x4,
                        hdr: false
                    },
                    5
                )
            );

            let normal = transcode(
                &two_channel_normals(),
                TextureSlot::Normal,
                GltfTextureTarget::Astc4x4,
            )
            .unwrap();
            check(&normal);
            assert_eq!(normal.role(), TextureRole::NormalMap);
            let TexturePayload::Rgba8(rgba) = normal.payload() else {
                panic!("a two-channel normal map under ASTC falls back to RGBA8");
            };
            // Y moved from alpha to green; Z rebuilt and opaque alpha.
            assert_eq!(rgba[1], 90);
            assert_eq!(rgba[3], 255);
        }

        #[test]
        fn a_size_off_the_block_grid_falls_back_to_rgba8() {
            let odd = ktx2_uastc(6, 6, false, false, true, |_, _, _| [10, 20, 30, 255]);
            let data = transcode(&odd, TextureSlot::Albedo, GltfTextureTarget::Bc).unwrap();
            check(&data);
            assert!(matches!(data.payload(), TexturePayload::Rgba8(_)));
            assert_eq!((data.width(), data.height()), (6, 6));
        }

        #[test]
        fn bytes_that_are_not_ktx2_do_not_transcode() {
            assert!(
                transcode(b"not a texture", TextureSlot::Albedo, GltfTextureTarget::Bc).is_none()
            );
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
