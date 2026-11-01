//! Public load options.

/// The texture encoding a KTX2 (`KHR_texture_basisu`) image is transcoded to.
///
/// A Basis Universal image transcodes to any of these at load. io cannot see
/// the GPU, so the caller picks the family the device samples: check the wgpu
/// device's `Features` for `TEXTURE_COMPRESSION_BC`, `TEXTURE_COMPRESSION_ETC2`
/// or `TEXTURE_COMPRESSION_ASTC`, and take [`Rgba8`](Self::Rgba8) when it has
/// none. Within a family the loader picks the format per texture slot.
///
/// Only images in KTX2 are affected: PNG and JPEG images always decode to
/// RGBA8. Without the `gltf-ktx2` feature the setting has no effect.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GltfTextureTarget {
    /// Uncompressed 8-bit RGBA, which every device samples. The default.
    #[default]
    Rgba8,
    /// BC7 for colour and metallic-roughness, BC4 for occlusion, BC5 for a
    /// two-channel normal map and BC7 for a three-channel one, BC6H for an HDR
    /// image.
    Bc,
    /// ETC2 RGBA for colour with alpha and ETC2 RGB for the rest, EAC R11 for
    /// occlusion, EAC RG11 for a two-channel normal map.
    Etc2Eac,
    /// ASTC 4x4 for every slot, the HDR profile for an HDR image.
    Astc4x4,
}

/// Per-call options for [`super::scene_from_path_with_options`] and
/// [`super::scene_from_slice_with_options`].
///
/// The default decodes exactly as [`super::scene_from_path`] does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GltfLoadOptions {
    /// What KTX2 images transcode to.
    pub texture_target: GltfTextureTarget,
}

impl GltfLoadOptions {
    /// Set what KTX2 images transcode to.
    pub fn with_texture_target(mut self, target: GltfTextureTarget) -> Self {
        self.texture_target = target;
        self
    }
}
