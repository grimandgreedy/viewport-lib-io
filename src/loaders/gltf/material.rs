//! Materials and the textures they reference.

use std::path::Path;

use crate::types::{AlphaMode, IoMaterial, TextureData, TextureSource};

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
            let texture = sg
                .diffuse_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
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
            let texture = pbr
                .base_color_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
            let mr_texture = pbr
                .metallic_roughness_texture()
                .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
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
    let emissive_texture = material
        .emissive_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

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
    let normal_map_texture = normal_texture
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    let occlusion_texture = material.occlusion_texture();
    let occlusion_strength = occlusion_texture.as_ref().map_or(1.0, |t| t.strength());
    let ao_texture = occlusion_texture
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("material_{}", material.index().unwrap_or(0))),
        base_color,
        metallic,
        roughness,
        emissive,
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
    }
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
    use crate::testkit::synth::temp_dir;
    use super::super::*;
    use super::*;

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
