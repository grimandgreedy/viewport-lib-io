//! The material extensions that refine the metallic-roughness set: index of
//! refraction, specular strength and tint, clear coat, sheen and anisotropy.
//!
//! `KHR_materials_ior` and `KHR_materials_specular` are read through the `gltf`
//! crate's typed accessors. Clear coat, sheen and anisotropy have no typed
//! support in the crate and are read from the raw extension map. Only the
//! factors are carried; the extensions' textures are not read.

use crate::types::IoMaterial;

const CLEARCOAT: &str = "KHR_materials_clearcoat";
const SHEEN: &str = "KHR_materials_sheen";
const ANISOTROPY: &str = "KHR_materials_anisotropy";

/// Copy the extension factors of `material` into `out`. Fields whose extension
/// is absent keep their defaults, which are the glTF defaults.
pub(super) fn read_material_extensions(material: &gltf::Material, out: &mut IoMaterial) {
    if let Some(ior) = material.ior() {
        out.ior = ior;
    }
    if let Some(specular) = material.specular() {
        out.specular_factor = specular.specular_factor();
        out.specular_colour = specular.specular_color_factor();
    }
    if let Some(ext) = material.extension_value(CLEARCOAT) {
        if let Some(value) = number(ext, "clearcoatFactor") {
            out.clearcoat = value;
        }
        if let Some(value) = number(ext, "clearcoatRoughnessFactor") {
            out.clearcoat_roughness = value;
        }
    }
    if let Some(ext) = material.extension_value(SHEEN) {
        if let Some(value) = vec3(ext, "sheenColorFactor") {
            out.sheen_colour = value;
        }
        if let Some(value) = number(ext, "sheenRoughnessFactor") {
            out.sheen_roughness = value;
        }
    }
    if let Some(ext) = material.extension_value(ANISOTROPY) {
        if let Some(value) = number(ext, "anisotropyStrength") {
            out.anisotropy = value;
        }
        if let Some(value) = number(ext, "anisotropyRotation") {
            out.anisotropy_rotation = value;
        }
    }
}

fn number(ext: &serde_json::Value, key: &str) -> Option<f32> {
    ext.get(key)?.as_f64().map(|value| value as f32)
}

fn vec3(ext: &serde_json::Value, key: &str) -> Option<[f32; 3]> {
    let array = ext.get(key)?.as_array()?;
    if array.len() != 3 {
        return None;
    }
    let mut out = [0.0; 3];
    for (slot, value) in out.iter_mut().zip(array) {
        *slot = value.as_f64()? as f32;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::super::scene_from_slice;
    use viewport_lib_types::colour::Colour;
    use viewport_lib_types::material::{AlphaMode, MATERIAL_TEXTURE_SLOTS, Material};

    fn materials(json: &str) -> Vec<crate::types::MaterialData> {
        scene_from_slice(json.as_bytes(), None).unwrap().materials
    }

    #[test]
    fn reads_ior_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_ior"],
  "materials": [
    { "name": "glass", "extensions": { "KHR_materials_ior": { "ior": 1.5 } } },
    { "name": "diamond", "extensions": { "KHR_materials_ior": { "ior": 2.4 } } },
    { "name": "plain" }
  ]
}"#,
        );
        assert_eq!(materials[0].ior, 1.5);
        assert_eq!(materials[1].ior, 2.4);
        assert_eq!(materials[2].ior, 1.5);

        // IOR 1.5 is the renderer's default reflectance, bit for bit.
        let glass = materials[0].to_material(|_, _| None);
        assert_eq!(glass.reflectance, 0.5);
        assert_eq!(glass.ior, 1.5);

        let diamond = materials[1].to_material(|_, _| None);
        assert_eq!(diamond.ior, 2.4);
        let expected = ((2.4f32 - 1.0) / (2.4 + 1.0)).abs() / 0.4;
        assert_eq!(diamond.reflectance, expected);
    }

    #[test]
    fn reads_specular_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_specular"],
  "materials": [
    {
      "extensions": {
        "KHR_materials_specular": {
          "specularFactor": 0.25,
          "specularColorFactor": [1.0, 0.5, 0.25]
        }
      }
    },
    { "extensions": { "KHR_materials_specular": {} } }
  ]
}"#,
        );
        assert_eq!(materials[0].specular_factor, 0.25);
        assert_eq!(materials[0].specular_colour, [1.0, 0.5, 0.25]);
        // An empty extension takes the glTF defaults.
        assert_eq!(materials[1].specular_factor, 1.0);
        assert_eq!(materials[1].specular_colour, [1.0, 1.0, 1.0]);

        let material = materials[0].to_material(|_, _| None);
        assert_eq!(material.specular_factor, 0.25);
        assert_eq!(material.specular_colour.to_linear_rgb(), [1.0, 0.5, 0.25]);
        // Blinn-Phong specular is a different control and stays put.
        assert_eq!(material.specular, Material::default().specular);
    }

    #[test]
    fn reads_clearcoat_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_clearcoat"],
  "materials": [
    {
      "extensions": {
        "KHR_materials_clearcoat": {
          "clearcoatFactor": 0.75,
          "clearcoatRoughnessFactor": 0.125
        }
      }
    }
  ]
}"#,
        );
        assert_eq!(materials[0].clearcoat, 0.75);
        assert_eq!(materials[0].clearcoat_roughness, 0.125);

        let material = materials[0].to_material(|_, _| None);
        assert_eq!(material.clearcoat, 0.75);
        assert_eq!(material.clearcoat_roughness, 0.125);
    }

    #[test]
    fn reads_sheen_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_sheen"],
  "materials": [
    {
      "extensions": {
        "KHR_materials_sheen": {
          "sheenColorFactor": [0.5, 0.25, 1.0],
          "sheenRoughnessFactor": 0.625
        }
      }
    }
  ]
}"#,
        );
        assert_eq!(materials[0].sheen_colour, [0.5, 0.25, 1.0]);
        assert_eq!(materials[0].sheen_roughness, 0.625);

        let material = materials[0].to_material(|_, _| None);
        assert_eq!(material.sheen_colour, Colour::linear_rgb(0.5, 0.25, 1.0));
        assert_eq!(material.sheen_roughness, 0.625);
    }

    #[test]
    fn reads_anisotropy_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_materials_anisotropy"],
  "materials": [
    {
      "extensions": {
        "KHR_materials_anisotropy": {
          "anisotropyStrength": 0.5,
          "anisotropyRotation": 1.25
        }
      }
    }
  ]
}"#,
        );
        assert_eq!(materials[0].anisotropy, 0.5);
        assert_eq!(materials[0].anisotropy_rotation, 1.25);

        let material = materials[0].to_material(|_, _| None);
        assert_eq!(material.anisotropy, 0.5);
        assert_eq!(material.anisotropy_rotation, 1.25);
    }

    #[test]
    fn a_material_without_extensions_converts_as_before() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "materials": [
    {
      "pbrMetallicRoughness": {
        "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
        "metallicFactor": 0.3,
        "roughnessFactor": 0.7
      }
    }
  ]
}"#,
        );
        let material = materials[0].to_material(|_, _| None);

        // The conversion as it stood before the extension fields existed.
        let mut expected = Material::pbr(Colour::from_linear_rgb_array([0.2, 0.4, 0.6]), 0.3, 0.7);
        expected.emissive = Colour::from_linear_rgb_array([0.0, 0.0, 0.0]);
        expected.emissive_strength = 1.0;
        expected.normal_strength = 1.0;
        expected.ao_range = [0.0, 1.0];
        expected.alpha_mode = AlphaMode::Opaque;
        expected.texture_transforms = [None; MATERIAL_TEXTURE_SLOTS];
        expected.sampler = [None; MATERIAL_TEXTURE_SLOTS];
        assert_eq!(material, expected);
        assert_eq!(format!("{material:?}"), format!("{expected:?}"));
    }

    #[test]
    fn ignores_an_unknown_used_extension() {
        let materials = materials(
            r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["EXT_materials_imaginary"],
  "materials": [
    {
      "pbrMetallicRoughness": { "baseColorFactor": [0.2, 0.4, 0.6, 1.0] },
      "extensions": { "EXT_materials_imaginary": { "glow": 3.0 } }
    },
    { "pbrMetallicRoughness": { "baseColorFactor": [0.2, 0.4, 0.6, 1.0] } }
  ]
}"#,
        );
        assert_eq!(materials.len(), 2);
        let with = materials[0].to_material(|_, _| None);
        let without = materials[1].to_material(|_, _| None);
        assert_eq!(with, without);
    }
}
