//! Materials and the textures they reference.

use std::path::Path;

use crate::types::{
    AlphaMode, IoMaterial, MATERIAL_TEXTURE_SLOTS, MaterialTextureSlot, TextureData, TextureFilter,
    TextureSampler, TextureSource, UvTransform, WrapMode,
};

/// The per-slot sampling state a material is built up with: how each slot
/// transforms its UVs and what sampler its texture names.
struct SlotState {
    uv_transforms: [Option<UvTransform>; MATERIAL_TEXTURE_SLOTS],
    samplers: [Option<TextureSampler>; MATERIAL_TEXTURE_SLOTS],
}

/// Convert one glTF material into the neutral [`IoMaterial`].
///
/// Covers the metallic-roughness set, emissive, alpha mode and cutoff, and the
/// texture slots. A specular-glossiness material is converted to
/// metallic-roughness first, since that is the only model the neutral type
/// carries.
pub(super) fn convert_material(
    material: &gltf::Material,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> IoMaterial {
    let mut slots = SlotState {
        uv_transforms: [None; MATERIAL_TEXTURE_SLOTS],
        samplers: [None; MATERIAL_TEXTURE_SLOTS],
    };

    // KHR_materials_pbrSpecularGlossiness assets get the reference lossy
    // conversion to metallic-roughness; everything else reads the standard
    // metallic-roughness properties.
    let (base_color, metallic, roughness, opacity, base_color_texture, metallic_roughness_texture) =
        if let Some(sg) = material.pbr_specular_glossiness() {
            let diffuse = sg.diffuse_factor();
            let (rgb, metallic) = spec_gloss_to_metal_rough(
                [diffuse[0], diffuse[1], diffuse[2]],
                sg.specular_factor(),
            );
            // The diffuse texture stands in for the base colour texture. This
            // is part of the lossy factor-level conversion: the specular-
            // glossiness texture is not converted, so per-texel specular
            // variation is dropped and only the factors steer metallic and
            // roughness. There is no metallic-roughness texture on this path.
            let texture = sg.diffuse_texture().and_then(|info| {
                record_info(&mut slots, MaterialTextureSlot::BaseColour, &info);
                image_to_texture_source(&info.texture(), images, parent_dir)
            });
            (
                rgb,
                metallic,
                1.0 - sg.glossiness_factor(),
                diffuse[3],
                texture,
                None,
            )
        } else {
            let pbr = material.pbr_metallic_roughness();
            let base_color_factor = pbr.base_color_factor();
            let texture = pbr.base_color_texture().and_then(|info| {
                record_info(&mut slots, MaterialTextureSlot::BaseColour, &info);
                image_to_texture_source(&info.texture(), images, parent_dir)
            });
            let mr_texture = pbr.metallic_roughness_texture().and_then(|info| {
                record_info(&mut slots, MaterialTextureSlot::MetallicRoughness, &info);
                image_to_texture_source(&info.texture(), images, parent_dir)
            });
            (
                [
                    base_color_factor[0],
                    base_color_factor[1],
                    base_color_factor[2],
                ],
                pbr.metallic_factor(),
                pbr.roughness_factor(),
                base_color_factor[3],
                texture,
                mr_texture,
            )
        };

    let emissive = material.emissive_factor();
    // KHR_materials_emissive_strength multiplies the factor. Absent, the glTF
    // default of 1.0 leaves the emission at the authored factor.
    let emissive_strength = material.emissive_strength().unwrap_or(1.0);
    let emissive_texture = material.emissive_texture().and_then(|info| {
        record_info(&mut slots, MaterialTextureSlot::Emissive, &info);
        image_to_texture_source(&info.texture(), images, parent_dir)
    });

    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(material.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };
    let double_sided = material.double_sided();

    // Read the strength scalars before consuming the texture references. Absent
    // textures leave the scalars at the glTF defaults of 1.0.
    let normal_texture = material.normal_texture();
    let normal_scale = normal_texture.as_ref().map_or(1.0, |t| t.scale());
    let normal_map_texture = normal_texture.and_then(|info| {
        record_slot(
            &mut slots,
            MaterialTextureSlot::Normal,
            info.tex_coord(),
            info.extension_value(TEXTURE_TRANSFORM),
            &info.texture(),
        );
        image_to_texture_source(&info.texture(), images, parent_dir)
    });

    let occlusion_texture = material.occlusion_texture();
    let occlusion_strength = occlusion_texture.as_ref().map_or(1.0, |t| t.strength());
    let ao_texture = occlusion_texture.and_then(|info| {
        record_slot(
            &mut slots,
            MaterialTextureSlot::Occlusion,
            info.tex_coord(),
            info.extension_value(TEXTURE_TRANSFORM),
            &info.texture(),
        );
        image_to_texture_source(&info.texture(), images, parent_dir)
    });

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("material_{}", material.index().unwrap_or(0))),
        base_color,
        metallic,
        roughness,
        emissive,
        emissive_strength,
        opacity,
        alpha_mode,
        double_sided,
        base_color_texture,
        metallic_roughness_texture,
        normal_map_texture,
        normal_scale,
        ao_texture,
        occlusion_strength,
        emissive_texture,
        uv_transforms: slots.uv_transforms,
        samplers: slots.samplers,
    }
}

/// The extension a `textureInfo` carries its UV transform in.
const TEXTURE_TRANSFORM: &str = "KHR_texture_transform";

/// Record how a `textureInfo` slot samples: its `texCoord` set and its
/// KHR_texture_transform, plus the sampler its texture names.
fn record_info(slots: &mut SlotState, slot: MaterialTextureSlot, info: &gltf::texture::Info) {
    let mut transform = UvTransform {
        uv_set: info.tex_coord(),
        ..UvTransform::IDENTITY
    };
    if let Some(source) = info.texture_transform() {
        transform.offset = source.offset();
        transform.scale = source.scale();
        transform.rotation = source.rotation();
        // The extension's own texCoord overrides the textureInfo one.
        if let Some(uv_set) = source.tex_coord() {
            transform.uv_set = uv_set;
        }
    }
    store(slots, slot, transform, &info.texture());
}

/// The same for the normal and occlusion slots. The `gltf` crate does not model
/// those as `textureInfo`, so they have no typed `texture_transform()` and the
/// extension is read from the raw extension map instead.
fn record_slot(
    slots: &mut SlotState,
    slot: MaterialTextureSlot,
    tex_coord: u32,
    transform: Option<&serde_json::Value>,
    texture: &gltf::Texture,
) {
    let mut uv = UvTransform {
        uv_set: tex_coord,
        ..UvTransform::IDENTITY
    };
    if let Some(source) = transform {
        if let Some(offset) = source.get("offset").and_then(vec2_from_json) {
            uv.offset = offset;
        }
        if let Some(scale) = source.get("scale").and_then(vec2_from_json) {
            uv.scale = scale;
        }
        if let Some(rotation) = source.get("rotation").and_then(serde_json::Value::as_f64) {
            uv.rotation = rotation as f32;
        }
        if let Some(uv_set) = source.get("texCoord").and_then(serde_json::Value::as_u64) {
            uv.uv_set = uv_set as u32;
        }
    }
    store(slots, slot, uv, texture);
}

/// Keep a slot's transform and sampler, dropping either when it says nothing
/// beyond the neutral default: a consumer reads `None` as "sample UV0 plainly"
/// and "use your default sampler", so recording the default would only cost it
/// per-slot work.
fn store(
    slots: &mut SlotState,
    slot: MaterialTextureSlot,
    transform: UvTransform,
    texture: &gltf::Texture,
) {
    if !transform.is_identity() {
        slots.uv_transforms[slot.index()] = Some(transform);
    }
    slots.samplers[slot.index()] = sampler_from_texture(texture);
}

fn vec2_from_json(value: &serde_json::Value) -> Option<[f32; 2]> {
    let pair = value.as_array()?;
    let x = pair.first()?.as_f64()? as f32;
    let y = pair.get(1)?.as_f64()? as f32;
    Some([x, y])
}

/// The sampler state a texture asks for. `None` when it names no sampler, or
/// names one that resolves to the neutral default.
fn sampler_from_texture(texture: &gltf::Texture) -> Option<TextureSampler> {
    let sampler = texture.sampler();
    // No `sampler` on the texture: the file expresses no preference.
    sampler.index()?;

    let wrap = |mode| match mode {
        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => WrapMode::MirrorRepeat,
        gltf::texture::WrappingMode::Repeat => WrapMode::Repeat,
    };
    // Magnification decides the filter. Minification only adds mip selection,
    // which is the consumer's to make, so it is read for its base filter and
    // only when there is no magnification filter to read.
    let filter = match (sampler.mag_filter(), sampler.min_filter()) {
        (Some(gltf::texture::MagFilter::Nearest), _) => TextureFilter::Nearest,
        (Some(gltf::texture::MagFilter::Linear), _) => TextureFilter::Linear,
        (
            None,
            Some(
                gltf::texture::MinFilter::Nearest
                | gltf::texture::MinFilter::NearestMipmapNearest
                | gltf::texture::MinFilter::NearestMipmapLinear,
            ),
        ) => TextureFilter::Nearest,
        _ => TextureFilter::Linear,
    };

    let state = TextureSampler {
        wrap_u: wrap(sampler.wrap_s()),
        wrap_v: wrap(sampler.wrap_t()),
        filter,
    };
    (state != TextureSampler::default()).then_some(state)
}

/// Reference lossy conversion of specular-glossiness factors to
/// metallic-roughness, following the KHR_materials_pbrSpecularGlossiness
/// conversion published with the glTF spec: solve metallic from the specular
/// brightness (a dielectric F0 of 0.04), then reconstruct the base colour by
/// blending the diffuse- and specular-derived candidates by metallic^2.
pub(super) fn spec_gloss_to_metal_rough(diffuse: [f32; 3], specular: [f32; 3]) -> ([f32; 3], f32) {
    const DIELECTRIC_SPECULAR: f32 = 0.04;
    const EPS: f32 = 1e-4;

    // Perceived brightness per the reference implementation.
    let brightness = |c: [f32; 3]| -> f32 {
        (0.299 * c[0] * c[0] + 0.587 * c[1] * c[1] + 0.114 * c[2] * c[2]).sqrt()
    };
    let max_component = |c: [f32; 3]| -> f32 { c[0].max(c[1]).max(c[2]) };

    let one_minus_specular_strength = 1.0 - max_component(specular);
    let diffuse_brightness = brightness(diffuse);
    let specular_brightness = brightness(specular);

    let metallic = if specular_brightness < DIELECTRIC_SPECULAR {
        0.0
    } else {
        let a = DIELECTRIC_SPECULAR;
        let b = diffuse_brightness * one_minus_specular_strength / (1.0 - DIELECTRIC_SPECULAR)
            + specular_brightness
            - 2.0 * DIELECTRIC_SPECULAR;
        let c = DIELECTRIC_SPECULAR - specular_brightness;
        let d = (b * b - 4.0 * a * c).max(0.0);
        ((-b + d.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
    };

    let mut base = [0.0f32; 3];
    for i in 0..3 {
        let from_diffuse = diffuse[i] * one_minus_specular_strength
            / (1.0 - DIELECTRIC_SPECULAR)
            / (1.0 - metallic).max(EPS);
        let from_specular =
            (specular[i] - DIELECTRIC_SPECULAR * (1.0 - metallic)) / metallic.max(EPS);
        base[i] =
            (from_diffuse + (from_specular - from_diffuse) * metallic * metallic).clamp(0.0, 1.0);
    }
    (base, metallic)
}

/// Resolve the image behind a texture to a [`TextureSource`].
///
/// An image the document embedded (a GLB chunk or a data URI) comes back as
/// decoded pixels; one stored beside the file comes back as a path resolved
/// against `parent_dir`, so the caller can decide when to read it.
pub(super) fn image_to_texture_source(
    texture: &gltf::Texture,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> Option<TextureSource> {
    let source = texture.source();
    let index = source.index();

    if let Some(data) = images.get(index) {
        return Some(TextureSource::Decoded(TextureData {
            width: data.width,
            height: data.height,
            rgba: to_rgba8(data),
        }));
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

// ---------------------------------------------------------------------------
// Y-up to Z-up reorientation
//
// glTF stores everything in right-handed Y-up. viewport-lib-io exposes a
// right-handed Z-up scene. The conversion is a +90 degree rotation about the
// X-axis: Y -> Z, Z -> -Y, X unchanged. Every orientation-bearing piece of
// data (positions, normals, tangents, mesh transforms, inverse-bind matrices,
// animation samples) is rotated once at load.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::testkit::synth::temp_dir;

    #[cfg(feature = "gltf")]
    #[test]
    fn loads_external_base_color_texture() {
        let dir = temp_dir("gltf_external_texture");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");
        let png_path = dir.join("albedo.png");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&png_path, png_bytes).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [{
    "pbrMetallicRoughness": {
      "baseColorTexture": { "index": 0 }
    }
  }],
  "textures": [{ "sampler": 0, "source": 0 }],
  "samplers": [{}],
  "images": [{ "uri": "albedo.png" }],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let material = scene.materials.first().unwrap();
        match material.base_color_texture.as_ref() {
            Some(TextureSource::Decoded(image)) => {
                assert_eq!(image.width, 1);
                assert_eq!(image.height, 1);
                assert_eq!(image.rgba, vec![255, 0, 0, 255]);
            }
            other => panic!("expected decoded external texture, got {other:?}"),
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_file(png_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn reads_emissive_alpha_mode_and_double_sided() {
        let dir = temp_dir("gltf_material_render_state");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [
    {
      "name": "emissive_cutout",
      "pbrMetallicRoughness": { "baseColorFactor": [1.0, 1.0, 1.0, 1.0] },
      "emissiveFactor": [0.1, 0.2, 0.3],
      "alphaMode": "MASK",
      "alphaCutoff": 0.7,
      "doubleSided": true
    },
    {
      "name": "plain_opaque",
      "pbrMetallicRoughness": { "baseColorFactor": [0.5, 0.5, 0.5, 1.0] }
    }
  ],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.materials.len(), 2);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;

        let m = &scene.materials[0];
        assert!(
            close(m.emissive[0], 0.1) && close(m.emissive[1], 0.2) && close(m.emissive[2], 0.3)
        );
        assert_eq!(m.alpha_mode, crate::types::AlphaMode::Mask(0.7));
        assert!(m.double_sided);

        // A plain material keeps the neutral defaults.
        let p = &scene.materials[1];
        assert_eq!(p.emissive, [0.0, 0.0, 0.0]);
        assert_eq!(p.alpha_mode, crate::types::AlphaMode::Opaque);
        assert!(!p.double_sided);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn reads_emissive_strength_extension() {
        let dir = temp_dir("gltf_emissive_strength");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [
    {
      "name": "sign",
      "emissiveFactor": [1.0, 0.8, 0.2],
      "extensions": {
        "KHR_materials_emissive_strength": { "emissiveStrength": 40.0 }
      }
    },
    {
      "name": "no_extension",
      "emissiveFactor": [1.0, 0.8, 0.2]
    }
  ],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.materials.len(), 2);

        // The extension multiplies the factor, so the emissive factor itself is
        // unchanged and the strength carries the brightness.
        let lit = &scene.materials[0];
        assert!((lit.emissive_strength - 40.0).abs() < 1e-4);
        assert!((lit.emissive[0] - 1.0).abs() < 1e-4);

        // Without the extension the glTF default of 1.0 applies.
        assert!((scene.materials[1].emissive_strength - 1.0).abs() < 1e-4);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn reads_texture_transforms_and_sampler_state() {
        use crate::types::{MaterialTextureSlot, TextureFilter, TextureSampler, WrapMode};

        let dir = temp_dir("gltf_texture_transform");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");
        let png_path = dir.join("albedo.png");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let png_bytes: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0xF8, 0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99,
            0x3D, 0x1D, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        std::fs::write(&png_path, png_bytes).unwrap();

        // Material 0 tiles its albedo out of an atlas and puts its occlusion on
        // the second UV set; material 1 asks for nothing beyond the defaults.
        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [
    {
      "name": "atlas",
      "pbrMetallicRoughness": {
        "baseColorTexture": {
          "index": 0,
          "extensions": {
            "KHR_texture_transform": {
              "offset": [0.5, 0.25],
              "scale": [4.0, 4.0],
              "rotation": 1.5707963
            }
          }
        }
      },
      "normalTexture": {
        "index": 0,
        "extensions": {
          "KHR_texture_transform": {
            "offset": [0.5, 0.25],
            "scale": [4.0, 4.0]
          }
        }
      },
      "occlusionTexture": { "index": 1, "texCoord": 1 }
    },
    {
      "name": "plain",
      "pbrMetallicRoughness": { "baseColorTexture": { "index": 1 } }
    }
  ],
  "textures": [
    { "sampler": 0, "source": 0 },
    { "sampler": 1, "source": 0 }
  ],
  "samplers": [
    { "wrapS": 33071, "wrapT": 33648, "magFilter": 9728 },
    { "wrapS": 10497, "wrapT": 10497 }
  ],
  "images": [{ "uri": "albedo.png" }],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let atlas = &scene.materials[0];

        let base = atlas
            .uv_transform(MaterialTextureSlot::BaseColour)
            .expect("base colour carries KHR_texture_transform");
        assert_eq!(base.offset, [0.5, 0.25]);
        assert_eq!(base.scale, [4.0, 4.0]);
        assert!((base.rotation - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        assert_eq!(base.uv_set, 0);

        // The normal slot is not a `textureInfo` in the gltf crate's model, so
        // its transform comes out of the raw extension map.
        let normal = atlas
            .uv_transform(MaterialTextureSlot::Normal)
            .expect("normal map carries KHR_texture_transform");
        assert_eq!(normal.offset, [0.5, 0.25]);
        assert_eq!(normal.scale, [4.0, 4.0]);
        assert_eq!(normal.rotation, 0.0);

        // A plain texCoord with no transform still has to survive: it is the
        // only thing telling a consumer which UV set to sample.
        let occlusion = atlas
            .uv_transform(MaterialTextureSlot::Occlusion)
            .expect("occlusion samples the second UV set");
        assert_eq!(occlusion.uv_set, 1);
        assert_eq!(occlusion.scale, [1.0, 1.0]);

        assert_eq!(
            atlas.sampler(MaterialTextureSlot::BaseColour),
            Some(TextureSampler {
                wrap_u: WrapMode::ClampToEdge,
                wrap_v: WrapMode::MirrorRepeat,
                filter: TextureFilter::Nearest,
            })
        );
        // The occlusion texture names a sampler, but one that says exactly what
        // the consumer would have done anyway.
        assert_eq!(atlas.sampler(MaterialTextureSlot::Occlusion), None);

        let plain = &scene.materials[1];
        assert_eq!(
            plain.uv_transforms,
            [None; crate::types::MATERIAL_TEXTURE_SLOTS]
        );
        assert_eq!(plain.samplers, [None; crate::types::MATERIAL_TEXTURE_SLOTS]);

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_file(png_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn converts_spec_gloss_materials_to_metal_rough() {
        let dir = temp_dir("gltf_spec_gloss");
        let gltf_path = dir.join("scene.gltf");
        let bin_path = dir.join("mesh.bin");

        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 1, 2] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&bin_path, bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_pbrSpecularGlossiness"],
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "material": 0
    }]
  }],
  "materials": [
    {
      "name": "dielectric_sg",
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {
          "diffuseFactor": [0.8, 0.2, 0.2, 0.75],
          "specularFactor": [0.0, 0.0, 0.0],
          "glossinessFactor": 0.3
        }
      }
    },
    {
      "name": "metal_sg",
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {
          "diffuseFactor": [0.0, 0.0, 0.0, 1.0],
          "specularFactor": [1.0, 1.0, 1.0],
          "glossinessFactor": 0.9
        }
      }
    },
    {
      "name": "plain_mr",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.1, 0.2, 0.3, 1.0],
        "metallicFactor": 0.5,
        "roughnessFactor": 0.25
      }
    }
  ],
  "buffers": [{ "uri": "mesh.bin", "byteLength": 48 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "min": [0, 0, 0],
      "max": [1, 1, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5125,
      "count": 3,
      "type": "SCALAR"
    }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        assert_eq!(scene.materials.len(), 3);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;

        // Zero specular: metallic 0, roughness = 1 - glossiness, base colour
        // is diffuse / (1 - 0.04), opacity from the diffuse alpha.
        let d = &scene.materials[0];
        assert!(close(d.metallic, 0.0), "dielectric metallic {}", d.metallic);
        assert!(
            close(d.roughness, 0.7),
            "dielectric roughness {}",
            d.roughness
        );
        assert!(close(d.base_color[0], 0.8 / 0.96));
        assert!(close(d.base_color[1], 0.2 / 0.96));
        assert!(close(d.opacity, 0.75));

        // Full white specular, black diffuse: solves to metallic 1 with the
        // specular colour as base.
        let m = &scene.materials[1];
        assert!(close(m.metallic, 1.0), "metal metallic {}", m.metallic);
        assert!(close(m.roughness, 0.1), "metal roughness {}", m.roughness);
        assert!(close(m.base_color[0], 1.0));

        // The plain metallic-roughness material is untouched by the branch.
        let p = &scene.materials[2];
        assert!(close(p.metallic, 0.5));
        assert!(close(p.roughness, 0.25));
        assert!(close(p.base_color[2], 0.3));

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }
}
