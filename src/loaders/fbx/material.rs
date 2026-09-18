//! Materials and their texture slots, including the UV placement each slot samples
//! with.

use crate::types::{
    AlphaMode, IoMaterial, MATERIAL_TEXTURE_SLOTS, MaterialTextureSlot, TextureData, TextureFilter,
    TextureSampler, TextureSource, UvTransform, WrapMode,
};
use std::path::Path;

/// Interpret an FBX `TransparencyFactor` into an opacity and alpha mode.
///
/// FBX transparency is exporter-ambiguous: the spec treats the factor as transparency (0 = opaque),
/// but many common writers (Unity's FBX export, Blender) store it as an *opacity* (1.0 = fully
/// opaque), and a value of exactly 1.0 is never a deliberately fully-transparent base material. So
/// both extremes map to opaque and only a strict in-between is real blending. Without this guard an
/// ordinary opaque material comes back at opacity 0 and renders invisible while still casting a
/// shadow (the shadow pass ignores alpha): the tell-tale symptom.
pub(super) fn opacity_from_transparency(transparency: f32) -> (f32, AlphaMode) {
    if transparency > 0.0 && transparency < 1.0 {
        (1.0 - transparency, AlphaMode::Blend)
    } else {
        (1.0, AlphaMode::Opaque)
    }
}

/// One FBX material converted, with the UV-set *name* each textured slot asks
/// for. FBX names the layer a texture samples rather than indexing it, and the
/// layer names belong to a mesh, so resolving a name to a set index waits until
/// the meshes are decoded.
pub(super) struct ConvertedMaterial {
    pub(super) material: IoMaterial,
    pub(super) uv_set_names: Vec<(MaterialTextureSlot, String)>,
}

pub(super) fn convert_material(
    material: &fbxcel_dom::v7400::object::material::MaterialHandle<'_>,
    parent_dir: &Path,
) -> ConvertedMaterial {
    let props = material.properties();

    let diffuse_color = props
        .diffuse_color_or_default()
        .ok()
        .map(|color| [color.r as f32, color.g as f32, color.b as f32])
        .unwrap_or([0.7, 0.7, 0.7]);
    let diffuse_factor = props.diffuse_factor_or_default().ok().unwrap_or(1.0) as f32;
    let base_color = [
        diffuse_color[0] * diffuse_factor,
        diffuse_color[1] * diffuse_factor,
        diffuse_color[2] * diffuse_factor,
    ];

    // FBX `TransparencyFactor` is exporter-ambiguous. The spec treats it as transparency (0 =
    // opaque), but many common writers (Unity's FBX export, Blender) store it as an *opacity*
    // (1.0 = fully opaque), and a value of exactly 1.0 is never a deliberately fully-transparent
    // base material. So treat both extremes as opaque and only a strict in-between as real blending;
    // otherwise an ordinary opaque material comes back at opacity 0 and renders invisible (it still
    // casts a shadow, since the shadow pass ignores alpha, which is the tell-tale symptom).
    let transparency = props.transparency_factor_or_default().ok().unwrap_or(0.0) as f32;
    let (opacity, alpha_mode) = opacity_from_transparency(transparency);
    let shininess = props.shininess_or_default().ok().unwrap_or(20.0) as f32;
    let roughness = (1.0 - (shininess / 100.0).sqrt()).clamp(0.1, 1.0);

    let mut uv_transforms = [None; MATERIAL_TEXTURE_SLOTS];
    let mut samplers = [None; MATERIAL_TEXTURE_SLOTS];
    let mut uv_set_names = Vec::new();

    // Each textured slot carries its own placement and wrap state on the FBX
    // texture node, read here into the neutral per-slot records.
    let mut read_slot = |texture: &fbxcel_dom::v7400::object::texture::TextureHandle<'_>,
                         slot: MaterialTextureSlot| {
        let (transform, sampler, uv_set_name) = texture_slot_state(texture);
        uv_transforms[slot.index()] = transform;
        samplers[slot.index()] = sampler;
        if let Some(name) = uv_set_name {
            uv_set_names.push((slot, name));
        }
    };

    let base_color_texture = material.diffuse_texture().and_then(|texture| {
        read_slot(&texture, MaterialTextureSlot::BaseColour);
        extract_texture(&texture, parent_dir)
    });
    let normal_map_texture = material.normal_map_texture().and_then(|texture| {
        read_slot(&texture, MaterialTextureSlot::Normal);
        extract_texture(&texture, parent_dir)
    });

    ConvertedMaterial {
        material: IoMaterial {
            name: material
                .name()
                .map(std::borrow::ToOwned::to_owned)
                .unwrap_or_else(|| "fbx_material".into()),
            base_color,
            metallic: 0.0,
            roughness,
            emissive: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            opacity,
            alpha_mode,
            double_sided: false,
            base_color_texture,
            metallic_roughness_texture: None,
            normal_map_texture,
            normal_scale: 1.0,
            ao_texture: None,
            occlusion_strength: 1.0,
            emissive_texture: None,
            uv_transforms,
            samplers,
            ..IoMaterial::default()
        },
        uv_set_names,
    }
}

/// FBX texture placement re-expressed against the flipped V the loader emits.
///
/// In the file's own frame the placement maps `uv` to
/// `translation + rotate(rotation) * (uv * scaling)`, rotating about the UV
/// origin. The loader hands out `(u, 1 - v)`, so the transform a consumer
/// applies has to be the same mapping conjugated by that flip: the rotation
/// negates, the scale is unchanged, and the V offset reflects through the flip
/// along with the rotated scale. With no rotation this is just
/// `v_offset = 1 - translation.v - scaling.v`, and a transform that only tiles
/// (`translation` zero, `scaling` one) comes out unchanged.
#[cfg(feature = "fbx")]
pub(super) fn uv_transform_from_placement(
    translation: [f32; 2],
    scaling: [f32; 2],
    rotation: f32,
) -> UvTransform {
    let (sin, cos) = rotation.sin_cos();
    UvTransform {
        offset: [
            translation[0] + sin * scaling[1],
            1.0 - translation[1] - cos * scaling[1],
        ],
        scale: scaling,
        rotation: -rotation,
        uv_set: 0,
    }
}

/// How one FBX texture samples: its UV placement, its wrap modes, and the name
/// of the UV layer it reads.
///
/// FBX composes placement as translation, rotation and scaling about the
/// texture's rotation and scaling pivots. With the pivots at their default
/// origin, that is translate after rotate after scale about the UV origin,
/// which is [`UvTransform`]'s own convention, so the three properties map
/// across directly. A file setting a non-default pivot is not represented:
/// there is nowhere to put it, and approximating it silently would be worse
/// than leaving the placement where the pivot-free reading puts it. `UVSwap` is
/// not carried either.
///
/// The one conversion is V. The loader flips V on the UV sets it emits, so the
/// placement has to be re-expressed against flipped coordinates or an offset
/// texture lands mirrored: the rotation negates and the V offset reflects. A
/// pure tiling transform is unaffected, which is the common case.
pub(super) fn texture_slot_state(
    texture: &fbxcel_dom::v7400::object::texture::TextureHandle<'_>,
) -> (Option<UvTransform>, Option<TextureSampler>, Option<String>) {
    let props = texture.properties();

    let translation = props
        .translation_or_default()
        .map(|t| [t.x as f32, t.y as f32])
        .unwrap_or([0.0, 0.0]);
    let scaling = props
        .scaling_or_default()
        .map(|s| [s.x as f32, s.y as f32])
        .unwrap_or([1.0, 1.0]);
    // FBX Euler angles are degrees; only the Z angle turns a 2D UV plane.
    let rotation = props
        .rotation_or_default()
        .map(|r| (r[2] as f32).to_radians())
        .unwrap_or(0.0);

    let transform = uv_transform_from_placement(translation, scaling, rotation);

    let wrap = |mode| match mode {
        fbxcel_dom::v7400::data::texture::WrapMode::Clamp => WrapMode::ClampToEdge,
        fbxcel_dom::v7400::data::texture::WrapMode::Repeat => WrapMode::Repeat,
    };
    // FBX has no filter concept to map, so filtering stays at the neutral
    // default and only the wrap modes come from the file.
    let sampler = TextureSampler {
        wrap_u: props.wrap_mode_u_or_default().map(wrap).unwrap_or_default(),
        wrap_v: props.wrap_mode_v_or_default().map(wrap).unwrap_or_default(),
        filter: TextureFilter::default(),
    };

    // "default" is what fbxcel-dom reports for a texture that names no UV set,
    // and is not a layer name worth resolving.
    let uv_set_name = props
        .uv_set_or_default()
        .ok()
        .filter(|name| !name.is_empty() && *name != "default")
        .map(std::borrow::ToOwned::to_owned);

    (
        (!transform.is_identity()).then_some(transform),
        (sampler != TextureSampler::default()).then_some(sampler),
        uv_set_name,
    )
}

pub(super) fn extract_texture(
    texture: &fbxcel_dom::v7400::object::texture::TextureHandle<'_>,
    parent_dir: &Path,
) -> Option<TextureSource> {
    if let Some(clip) = texture.video_clip() {
        if let Some(content) = clip.content() {
            if !content.is_empty() {
                if let Ok(image) = image::load_from_memory(content) {
                    let rgba = image.to_rgba8();
                    let (width, height) = rgba.dimensions();
                    return Some(TextureSource::Decoded(TextureData {
                        width,
                        height,
                        rgba: rgba.into_raw(),
                    }));
                }
            }
        }

        if let Ok(relative_path) = clip.relative_filename() {
            let relative_path = relative_path.replace('\\', "/");
            let texture_path = parent_dir.join(&relative_path);
            if texture_path.exists() {
                return Some(TextureSource::File(texture_path));
            }
            if let Some(filename) = Path::new(&relative_path).file_name() {
                let fallback = parent_dir.join(filename);
                if fallback.exists() {
                    return Some(TextureSource::File(fallback));
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FBX `TransparencyFactor` is exporter-ambiguous; a plain opaque material (factor 0.0 *or* the
    /// opacity-convention 1.0) must come back opaque, and only a strict in-between blends. Guards the
    /// regression where opaque characters rendered invisible (opacity 0) but still cast shadows.
    #[test]
    fn opacity_from_transparency_treats_both_extremes_as_opaque() {
        assert_eq!(opacity_from_transparency(0.0), (1.0, AlphaMode::Opaque));
        assert_eq!(opacity_from_transparency(1.0), (1.0, AlphaMode::Opaque));
        let (o, mode) = opacity_from_transparency(0.25);
        assert!((o - 0.75).abs() < 1e-6);
        assert_eq!(mode, AlphaMode::Blend);
    }

    /// The FBX placement and the transform we hand out must sample the same
    /// texel, once the loader's V flip is accounted for on both the input UV and
    /// the resulting coordinate. Checked against a placement that translates,
    /// scales and rotates at once, which is where a wrong conjugation shows.
    #[test]
    fn texture_placement_survives_the_v_flip() {
        let translation = [0.3_f32, 0.15];
        let scaling = [2.0_f32, 3.0];
        let rotation = 0.6_f32;

        // The file's own mapping: scale, rotate about the UV origin, translate.
        let in_file = |uv: [f32; 2]| {
            let (sin, cos) = rotation.sin_cos();
            let p = [uv[0] * scaling[0], uv[1] * scaling[1]];
            [
                translation[0] + cos * p[0] + sin * p[1],
                translation[1] - sin * p[0] + cos * p[1],
            ]
        };
        // What a consumer applies to the UVs this loader emits.
        let converted = uv_transform_from_placement(translation, scaling, rotation);
        let ours = |uv: [f32; 2]| {
            let (sin, cos) = converted.rotation.sin_cos();
            let p = [uv[0] * converted.scale[0], uv[1] * converted.scale[1]];
            [
                converted.offset[0] + cos * p[0] + sin * p[1],
                converted.offset[1] - sin * p[0] + cos * p[1],
            ]
        };

        for uv in [[0.0, 0.0], [1.0, 1.0], [0.25, 0.8], [0.6, 0.1]] {
            let want = in_file(uv);
            // Same vertex, as the loader hands it out, and the same texel, in
            // the flipped frame the texture is sampled in.
            let got = ours([uv[0], 1.0 - uv[1]]);
            assert!(
                (want[0] - got[0]).abs() < 1e-5 && (1.0 - want[1] - got[1]).abs() < 1e-5,
                "uv {uv:?}: file {want:?} vs ours {got:?}"
            );
        }
    }

    /// A texture that only tiles is the common case and must come through
    /// untouched by the flip conversion.
    #[test]
    fn pure_tiling_placement_is_unchanged() {
        let transform = uv_transform_from_placement([0.0, 0.0], [4.0, 4.0], 0.0);
        assert_eq!(transform.scale, [4.0, 4.0]);
        assert_eq!(transform.rotation, 0.0);
        assert!((transform.offset[0]).abs() < 1e-6);
        // v' = 1 - 0 - 4: tiling four times up a flipped axis starts three
        // tiles below the origin, which is the same strip of texture.
        assert!((transform.offset[1] - (1.0 - 4.0)).abs() < 1e-6);
    }
}
