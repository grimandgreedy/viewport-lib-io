use std::collections::HashMap;
use std::path::PathBuf;

use viewport_lib_plugins::item_types::gaussian_splat::{GaussianSplatData, ShDegree};
use viewport_lib_plugins::item_types::point_cloud::PointCloudItem;
use viewport_lib_types::colour::Colour;
use viewport_lib_types::data::mesh::MeshData;
use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::data::volume_mesh::VolumeMeshData;
use viewport_lib_types::effects::lighting::{Candela, LightSource, Lux};
use viewport_lib_types::encoding::ColourSource;
use viewport_lib_types::ids::TextureId;
use viewport_lib_types::material::{
    AlphaMode, BackfacePolicy, ItemSettings, MATERIAL_TEXTURE_SLOTS, Material, SamplerKey,
    TextureFilter, TextureSlot, UvTransform, WrapMode,
};
use viewport_lib_types::overlay::vector::{FillRule, SubPath};
use viewport_lib_types::overlay::{
    GradientStop, LineCap, LineJoin, OverlayFill, OverlayStroke, StrokePattern,
};

/// Maximum number of joints permitted in a single skeleton.
///
/// Skinning palettes in real-time renderers are fixed-size; 256 matches the
/// conventional palette limit used across the viewport-lib stack. Loaders
/// reject any source skeleton that exceeds this bound rather than silently
/// truncating, and per-vertex `[u8; 4]` joint indices fit exactly into this
/// range.
pub const MAX_JOINTS: usize = 256;

/// Where a scene material's texture content comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum TextureSource {
    /// A file beside the scene, for the consumer to read when it chooses.
    /// Decode it for the slot it fills (each image loader has a `_for_slot`
    /// entrypoint) so it carries the right colour space and role.
    File(PathBuf),
    /// Pixels the scene embedded, already decoded and labelled for the slot
    /// they fill: sRGB for base colour and emissive, a normal map for the
    /// normal slot, linear for the rest.
    Decoded(TextureData),
}

/// Sampler state for one texture slot: wrap modes and filtering, as the source
/// file specifies them.
///
/// Narrower than the renderer's [`SamplerKey`], whose remaining fields
/// (anisotropy, LOD bias) are renderer policy rather than anything a scene file
/// carries. Convert with `SamplerKey::from`, which leaves those at their
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TextureSampler {
    /// Wrap mode for the U (glTF `wrapS`) axis.
    pub wrap_u: WrapMode,
    /// Wrap mode for the V (glTF `wrapT`) axis.
    pub wrap_v: WrapMode,
    /// Min/mag filtering. glTF mipmap modes collapse to the base filter, since
    /// mip selection is the consumer's.
    pub filter: TextureFilter,
}

impl From<TextureSampler> for SamplerKey {
    fn from(sampler: TextureSampler) -> Self {
        SamplerKey {
            wrap_u: sampler.wrap_u,
            wrap_v: sampler.wrap_v,
            filter: sampler.filter,
            ..SamplerKey::default()
        }
    }
}

/// A UV transform authored the way the file formats define it, rotating about
/// the UV origin:
///
/// ```text
/// uv' = offset + rotate(rotation) * (uv * scale)
/// ```
///
/// with positive `rotation` turning the UV axes anticlockwise (glTF
/// `KHR_texture_transform`, and an FBX placement with its pivots at the
/// origin), re-expressed in the convention [`UvTransform`] uses, which scales
/// and offsets first and then rotates about the texture centre:
///
/// ```text
/// uv' = rotate(rotation') * (uv * scale + offset' - 0.5) + 0.5
/// ```
///
/// `scale` and `uv_set` carry across; the rotation negates and the offset moves
/// with it. With no rotation the two forms are the same.
#[allow(dead_code)]
pub(crate) fn uv_transform_from_origin_rotation(
    offset: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    uv_set: u32,
) -> UvTransform {
    let (sin, cos) = (-rotation).sin_cos();
    let centred = [offset[0] - 0.5, offset[1] - 0.5];
    UvTransform {
        // Undo the centre rotation the renderer applies to the offset, so the
        // composed mapping lands where the source asked.
        offset: [
            cos * centred[0] + sin * centred[1] + 0.5,
            -sin * centred[0] + cos * centred[1] + 0.5,
        ],
        scale,
        rotation: -rotation,
        uv_set,
    }
}

/// Area-weighted smooth vertex normals, for a mesh whose source carries none.
///
/// Each triangle adds its unnormalised face normal to its three vertices, so
/// larger triangles pull harder, and each sum is then normalised. A vertex no
/// triangle touches, or one whose faces cancel or are all degenerate, gets +Z,
/// which is up in this crate's frame, so every normal is unit length.
#[allow(dead_code)]
pub(crate) fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![glam::Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [triangle[0], triangle[1], triangle[2]].map(|i| i as usize);
        if a >= positions.len() || b >= positions.len() || c >= positions.len() {
            continue;
        }
        let pa = glam::Vec3::from(positions[a]);
        let face = (glam::Vec3::from(positions[b]) - pa).cross(glam::Vec3::from(positions[c]) - pa);
        sums[a] += face;
        sums[b] += face;
        sums[c] += face;
    }
    sums.into_iter()
        .map(|n| n.try_normalize().unwrap_or(glam::Vec3::Z).to_array())
        .collect()
}

/// Material data extracted from a scene file.
///
/// [`to_material`](Self::to_material) builds the renderer's `Material` from it,
/// given ids for its textures. `occlusion_strength` becomes
/// `ao_range = [1.0 - occlusion_strength, 1.0]` there, which reproduces glTF
/// `occlusionStrength`, since `mix(1.0, sample, s) == mix(1 - s, 1, sample)`.
///
/// Materials are always metallic-roughness. glTF assets using
/// `KHR_materials_pbrSpecularGlossiness` are converted at import with the
/// reference lossy conversion (metallic solved from specular brightness,
/// `roughness = 1 - glossiness`, diffuse texture reused as the base colour
/// texture); the specular-glossiness texture itself is not converted, so
/// per-texel specular variation is dropped.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MaterialData {
    /// Material name from the source scene.
    pub name: String,
    /// Base colour in linear space (glTF `baseColorFactor` and equivalents are
    /// linear). Map it to a renderer material through the linear path, e.g.
    /// `Colour::linear_rgb`, not an sRGB constructor, so it is not decoded again.
    pub base_color: [f32; 3],
    /// Metallic factor.
    pub metallic: f32,
    /// Roughness factor.
    pub roughness: f32,
    /// Emissive colour in linear space, matching glTF `emissiveFactor`. Black
    /// `[0.0, 0.0, 0.0]` when the source declares no emission. Linear like
    /// [`base_color`](Self::base_color): map through the linear path. Scaled by
    /// [`emissive_strength`](Self::emissive_strength).
    pub emissive: [f32; 3],
    /// Multiplies [`emissive`](Self::emissive) and the emissive texture,
    /// matching glTF `KHR_materials_emissive_strength`. 1.0 leaves the emission
    /// at the authored factor, and formats with no equivalent leave it there.
    ///
    /// Carry it: a consumer whose exposure is photometric reads
    /// `emissive * emissive_strength` as a luminance in nits, so a surface
    /// authored to read as a light source emits one nit without it, which is
    /// indistinguishable from black.
    pub emissive_strength: f32,
    /// Opacity factor from the source file.
    pub opacity: f32,
    /// How the alpha channel is interpreted, matching glTF `alphaMode` +
    /// `alphaCutoff`. Defaults to [`AlphaMode::Opaque`] for formats without the
    /// concept.
    pub alpha_mode: AlphaMode,
    /// Whether back faces are drawn, matching glTF `doubleSided`. `false` for
    /// formats without the concept.
    pub double_sided: bool,
    /// Base colour texture, if present. sRGB colour ([`ColourSpace::Srgb`](viewport_lib_types::colour::ColourSpace::Srgb)).
    pub base_color_texture: Option<TextureSource>,
    /// Combined metallic-roughness (ORM) texture, if present. Matches the glTF
    /// `metallicRoughnessTexture`: G channel is roughness, B channel is metallic.
    /// Linear data ([`ColourSpace::Linear`](viewport_lib_types::colour::ColourSpace::Linear)).
    pub metallic_roughness_texture: Option<TextureSource>,
    /// Normal map, if present. Linear data with the normal-map role.
    pub normal_map_texture: Option<TextureSource>,
    /// Scales the tangent-space normal read from `normal_map_texture`, matching
    /// glTF `normalScale`. 1.0 leaves the map at authored strength. Files with no
    /// equivalent (OBJ, FBX) leave this at 1.0.
    pub normal_scale: f32,
    /// Ambient-occlusion texture, if present. Linear data
    /// ([`ColourSpace::Linear`](viewport_lib_types::colour::ColourSpace::Linear)).
    pub ao_texture: Option<TextureSource>,
    /// Strength of the ambient-occlusion contribution, matching glTF
    /// `occlusionStrength`. 1.0 applies the map fully, 0.0 disables it. Files with
    /// no equivalent leave this at 1.0.
    pub occlusion_strength: f32,
    /// Emissive texture, multiplied by `emissive`. Matches glTF `emissiveTexture`.
    /// sRGB colour ([`ColourSpace::Srgb`](viewport_lib_types::colour::ColourSpace::Srgb)).
    pub emissive_texture: Option<TextureSource>,
    /// How each slot samples its UVs, indexed by [`TextureSlot::index`]. `None`
    /// where the source asks for plain UV0 sampling, which is most materials.
    /// Read one with [`uv_transform`](Self::uv_transform).
    ///
    /// Already in the renderer's convention (scale and offset, then rotation
    /// about the texture centre), converted from the file's at decode, so it
    /// copies straight into viewport-lib's `Material::texture_transforms`.
    pub uv_transforms: [Option<UvTransform>; MATERIAL_TEXTURE_SLOTS],
    /// Sampler state per slot, indexed by [`TextureSlot::index`].
    /// `None` where the source specifies none, or specifies exactly the neutral
    /// default (repeat on both axes, linear filtering), so a consumer reads
    /// `None` as "your default sampler". Read one with
    /// [`sampler`](Self::sampler); converts to the renderer's [`SamplerKey`].
    pub samplers: [Option<TextureSampler>; MATERIAL_TEXTURE_SLOTS],
    /// Whether the surface skips lighting and shows its base colour as is,
    /// matching glTF `KHR_materials_unlit`. `false` for formats without the
    /// concept.
    ///
    /// The renderer carries this on the item, not the material: set
    /// `ItemSettings::unlit` from it, or start from
    /// [`item_settings`](Self::item_settings).
    pub unlit: bool,
    /// Index of refraction, matching glTF `KHR_materials_ior`. Default 1.5,
    /// which gives the renderer's default dielectric reflectance.
    pub ior: f32,
    /// Strength of the dielectric specular, matching glTF
    /// `KHR_materials_specular`'s `specularFactor`. Default 1.0. The
    /// extension's textures are not read.
    pub specular_factor: f32,
    /// Linear tint of the dielectric specular, matching glTF
    /// `KHR_materials_specular`'s `specularColorFactor`. Default white.
    pub specular_colour: [f32; 3],
    /// Clear coat strength, matching glTF `KHR_materials_clearcoat`'s
    /// `clearcoatFactor`. Default 0 (no coat). The extension's textures are
    /// not read.
    pub clearcoat: f32,
    /// Clear coat perceptual roughness (`clearcoatRoughnessFactor`). Default 0.
    pub clearcoat_roughness: f32,
    /// Linear sheen colour, matching glTF `KHR_materials_sheen`'s
    /// `sheenColorFactor`. Default black (no sheen). The extension's textures
    /// are not read.
    pub sheen_colour: [f32; 3],
    /// Sheen perceptual roughness (`sheenRoughnessFactor`). Default 0.
    pub sheen_roughness: f32,
    /// Anisotropy of the specular lobe, matching glTF
    /// `KHR_materials_anisotropy`'s `anisotropyStrength`. Default 0
    /// (isotropic). The extension's texture is not read.
    pub anisotropy: f32,
    /// Rotation of the anisotropy direction from the tangent, in radians
    /// (`anisotropyRotation`). Default 0.
    pub anisotropy_rotation: f32,
}

impl MaterialData {
    /// Every present texture with its slot. A [`TextureSource::Decoded`] is
    /// already labelled for its slot and uploads as-is; a
    /// [`TextureSource::File`] is decoded for the slot it fills:
    ///
    /// ```ignore
    /// for (slot, source) in material.textures() {
    ///     let data = match source {
    ///         TextureSource::Decoded(data) => data.clone(),
    ///         TextureSource::File(path) => png::texture_from_path_for_slot(path, slot)?,
    ///     };
    ///     let id = res.upload_texture(&device, &queue, data)?;
    /// }
    /// ```
    pub fn textures(&self) -> Vec<(TextureSlot, &TextureSource)> {
        [
            (TextureSlot::Albedo, &self.base_color_texture),
            (
                TextureSlot::MetallicRoughness,
                &self.metallic_roughness_texture,
            ),
            (TextureSlot::Normal, &self.normal_map_texture),
            (TextureSlot::Ao, &self.ao_texture),
            (TextureSlot::Emissive, &self.emissive_texture),
        ]
        .into_iter()
        .filter_map(|(slot, src)| src.as_ref().map(|s| (slot, s)))
        .collect()
    }

    /// The renderer's PBR material for this one, with each present texture
    /// resolved to an id by `texture`.
    ///
    /// `texture` is called once per present texture with its slot and source,
    /// and returns the uploaded texture's id (uploading it, or looking it up in
    /// a cache), or `None` to leave the slot empty. A
    /// [`TextureSource::Decoded`] uploads as-is; a [`TextureSource::File`] is
    /// decoded for its slot first.
    ///
    /// Carried across: base colour and emissive through the linear path,
    /// metallic, roughness, `emissive_strength`, alpha mode, `double_sided` (as
    /// [`BackfacePolicy::Identical`]), `normal_scale` as `normal_strength`,
    /// `occlusion_strength` as `ao_range = [1 - strength, 1]`, and the per-slot
    /// UV transforms and samplers; `ior` through `Material::with_ior`, so the
    /// reflectance follows it, then the specular factor and colour, clear coat,
    /// sheen and anisotropy. [`opacity`](Self::opacity) is not part of a
    /// renderer material: set it on the item (`ItemSettings::opacity`).
    pub fn to_material(
        &self,
        mut texture: impl FnMut(TextureSlot, &TextureSource) -> Option<TextureId>,
    ) -> Material {
        let mut material = Material::pbr(
            Colour::from_linear_rgb_array(self.base_color),
            self.metallic,
            self.roughness,
        );
        material.emissive = Colour::from_linear_rgb_array(self.emissive);
        material.emissive_strength = self.emissive_strength;
        material.normal_strength = self.normal_scale;
        material.ao_range = [1.0 - self.occlusion_strength, 1.0];
        material.alpha_mode = self.alpha_mode;
        if self.double_sided {
            material.backface_policy = BackfacePolicy::Identical;
        }
        material.texture_transforms = self.uv_transforms;
        material.sampler = self.samplers.map(|sampler| sampler.map(SamplerKey::from));
        for (slot, source) in self.textures() {
            material.set_texture(slot, texture(slot, source));
        }
        let mut material = material.with_ior(self.ior);
        material.specular_factor = self.specular_factor;
        material.specular_colour = Colour::from_linear_rgb_array(self.specular_colour);
        material
            .with_clearcoat(self.clearcoat, self.clearcoat_roughness)
            .with_sheen(
                Colour::from_linear_rgb_array(self.sheen_colour),
                self.sheen_roughness,
            )
            .with_anisotropy(self.anisotropy, self.anisotropy_rotation)
    }

    /// Item settings carrying what of this material the renderer keeps on the
    /// item rather than on its `Material`: [`unlit`](Self::unlit) as
    /// `ItemSettings::unlit`. Everything else is left at its default.
    ///
    /// [`opacity`](Self::opacity) is not set, since it is only meaningful under
    /// [`AlphaMode::Blend`]; copy it to `ItemSettings::opacity` when blending.
    pub fn item_settings(&self) -> ItemSettings {
        let mut settings = ItemSettings::default();
        settings.unlit = self.unlit;
        settings
    }

    /// How one slot samples its UVs, or `None` when it samples UV0
    /// untransformed.
    pub fn uv_transform(&self, slot: TextureSlot) -> Option<UvTransform> {
        self.uv_transforms[slot.index()]
    }

    /// Sampler state for one slot, or `None` when the source specifies nothing
    /// beyond the neutral default.
    pub fn sampler(&self, slot: TextureSlot) -> Option<TextureSampler> {
        self.samplers[slot.index()]
    }
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [0.7, 0.7, 0.7],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            opacity: 1.0,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_map_texture: None,
            normal_scale: 1.0,
            ao_texture: None,
            occlusion_strength: 1.0,
            emissive_texture: None,
            uv_transforms: [None; MATERIAL_TEXTURE_SLOTS],
            samplers: [None; MATERIAL_TEXTURE_SLOTS],
            unlit: false,
            ior: 1.5,
            specular_factor: 1.0,
            specular_colour: [1.0, 1.0, 1.0],
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            sheen_colour: [0.0, 0.0, 0.0],
            sheen_roughness: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
        }
    }
}

/// Per-vertex joint influence data for CPU skinning workflows.
#[derive(Clone, Debug)]
pub struct SkinWeights {
    /// Joint indices for each vertex: 4 per vertex, parallel to positions.
    pub joint_indices: Vec<[u8; 4]>,
    /// Blend weights for each vertex: 4 per vertex.
    pub joint_weights: Vec<[f32; 4]>,
}

/// One blend shape (morph target): per-vertex displacements from the base
/// mesh, parallel to the mesh's `positions`. A deformer sums the base
/// geometry with each target's displacements scaled by that target's weight.
#[derive(Clone, Debug)]
pub struct MorphTarget {
    /// Target name. Source formats that do not carry per-target names (or
    /// whose names are not read) fall back to `target_<index>`.
    pub name: String,
    /// Position displacement per vertex, one entry per base-mesh vertex.
    pub position_deltas: Vec<[f32; 3]>,
    /// Optional normal displacement per vertex, same length as
    /// `position_deltas` when present.
    pub normal_deltas: Option<Vec<[f32; 3]>>,
    /// Optional tangent displacement per vertex (xyz only; the bitangent sign
    /// on the base tangent is unaffected). Same length as `position_deltas`
    /// when present.
    pub tangent_deltas: Option<Vec<[f32; 3]>>,
}

/// One joint in a skeleton hierarchy.
#[derive(Clone, Debug)]
pub struct Joint {
    /// Display name for the joint, copied from the source file when present.
    pub name: String,
    /// Index of the parent joint within the same skeleton, or `None` for the
    /// root. Always less than the joint's own index (topological order).
    pub parent: Option<u8>,
    /// Inverse of the joint's world-space transform in the bind pose.
    pub inverse_bind: glam::Mat4,
}

/// A joint hierarchy with bind-pose inverse matrices, source-agnostic.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    /// Skeleton name, when the source format provides one.
    pub name: String,
    /// Joints in topological order: each parent index is less than its own.
    pub joints: Vec<Joint>,
}

/// Which component of a joint's local transform an animation track drives.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AnimationChannel {
    /// Translation channel (Vec3 sampler values).
    Translation,
    /// Rotation channel (Quat sampler values).
    Rotation,
    /// Scale channel (Vec3 sampler values).
    Scale,
}

/// How an animation sampler blends between adjacent keyframes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AnimationInterpolation {
    /// Hold the value of the lower keyframe until the next one starts.
    Step,
    /// Vec3 channels lerp; Quat channels slerp.
    Linear,
    /// Cubic-spline interpolation as defined by glTF. Each keyframe stores an
    /// in-tangent, a value and an out-tangent, in that order.
    CubicSpline,
}

impl AnimationInterpolation {
    /// How many stored values one keyframe takes: 3 under `CubicSpline`
    /// (in-tangent, value, out-tangent), 1 otherwise.
    pub fn values_per_key(self) -> usize {
        match self {
            Self::CubicSpline => 3,
            Self::Step | Self::Linear => 1,
        }
    }
}

/// Per-keyframe values for an animation sampler. The variant must match the
/// channel of the parent track.
#[derive(Clone, Debug)]
pub enum AnimationTrackValues {
    /// Translation or scale keyframes.
    Vec3(Vec<glam::Vec3>),
    /// Rotation keyframes.
    Quat(Vec<glam::Quat>),
}

/// Keyframe times paired with values plus an interpolation mode.
#[derive(Clone, Debug)]
pub struct AnimationSampler {
    /// Interpolation mode between keyframes.
    pub interpolation: AnimationInterpolation,
    /// Keyframe times in seconds, non-empty and strictly increasing.
    pub times: Vec<f32>,
    /// Keyframe values, same length as `times`. For `CubicSpline`, each
    /// keyframe carries three values (in-tangent, value, out-tangent) so the
    /// inner length is `3 * times.len()`.
    pub values: AnimationTrackValues,
}

/// One animation track: a sampler bound to one channel on one joint.
#[derive(Clone, Debug)]
pub struct AnimationTrack {
    /// Index into the target [`Skeleton::joints`].
    pub joint: usize,
    /// Which component of the joint's local transform this track drives.
    pub channel: AnimationChannel,
    /// Keyframe sampler producing values for this channel.
    pub sampler: AnimationSampler,
}

/// A collection of tracks that together animate one or more joints.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Clip name, when the source format provides one.
    pub name: String,
    /// Length of the clip in seconds, derived from the maximum sampler time.
    pub duration: f32,
    /// Index of the [`Skeleton`] this clip targets within the parent
    /// [`SceneData::skeletons`].
    pub skeleton_index: usize,
    /// Per-channel tracks.
    pub tracks: Vec<AnimationTrack>,
}

/// A keyframed blend-shape weight animation for one mesh's morph targets.
///
/// Kept apart from [`AnimationClip`] because it drives a mesh's morph target
/// set, not a skeleton: weights are scalar and target the mesh's targets in
/// order, so the joint-indexed [`AnimationTrack`] does not fit. Scalar weights
/// carry no orientation, so no Z-up reorientation applies.
#[derive(Clone, Debug)]
pub struct MorphWeightClip {
    /// Clip name, from the source animation. Anonymous clips are named by the
    /// loader.
    pub name: String,
    /// Length in seconds: the maximum keyframe time.
    pub duration: f32,
    /// Index into [`SceneData::meshes`]' source mesh (the glTF `mesh.index()`)
    /// whose morph targets these weights drive. All primitives of that mesh
    /// share the same target order.
    pub mesh_index: usize,
    /// Number of morph targets, so [`weights`](Self::weights) can be
    /// de-interleaved per keyframe.
    pub target_count: usize,
    /// Interpolation between keyframes.
    pub interpolation: AnimationInterpolation,
    /// Keyframe times in seconds, strictly increasing.
    pub times: Vec<f32>,
    /// Weights, row-major `[keyframe][target]`: length is
    /// `times.len() * target_count`.
    ///
    /// Under `CubicSpline` each keyframe holds three rows, in-tangents, values
    /// and out-tangents, so the layout is `[keyframe][in | value | out][target]`
    /// and the length is `3 * times.len() * target_count`, as glTF stores it.
    /// [`key_weights`](Self::key_weights) reads the values alone.
    pub weights: Vec<f32>,
}

impl MorphWeightClip {
    /// The weights at keyframe `key`, one per target, without the cubic-spline
    /// tangents. `None` past the last keyframe.
    pub fn key_weights(&self, key: usize) -> Option<&[f32]> {
        if key >= self.times.len() {
            return None;
        }
        let per_key = self.interpolation.values_per_key();
        let start = (key * per_key + per_key / 2) * self.target_count;
        self.weights.get(start..start + self.target_count)
    }
}

/// One node animation track: a sampler bound to one channel of a scene node's
/// local transform.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct NodeAnimationTrack {
    /// Index into [`SceneData::nodes`].
    pub node: usize,
    /// Which component of the node's local transform this track drives.
    pub channel: AnimationChannel,
    /// Keyframe sampler producing values for this channel, Z-up like
    /// [`SceneNode::transform`].
    pub sampler: AnimationSampler,
}

/// Tracks that animate scene nodes which are not skeleton joints: a door
/// swinging, a turntable, a light on a rail.
///
/// Kept apart from [`AnimationClip`], which targets one skeleton's joints. A
/// sample replaces the matching component of the node's local
/// [`SceneNode::transform`]; the node's world transform is then its parent
/// chain composed with it, which moves the node's meshes and lights.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct NodeAnimationClip {
    /// Clip name, from the source animation. Anonymous clips are named by the
    /// loader.
    pub name: String,
    /// Length of the clip in seconds, derived from the maximum sampler time.
    pub duration: f32,
    /// Per-channel tracks.
    pub tracks: Vec<NodeAnimationTrack>,
}

/// One node of the source scene graph, kept so node animation and lights have
/// something to refer to.
///
/// [`SceneMesh::transform`] already holds each mesh's composed world transform,
/// so a static scene needs nothing from here.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SceneNode {
    /// Node name, when the source provides one.
    pub name: Option<String>,
    /// Index of the parent node in [`SceneData::nodes`], or `None` for a root.
    pub parent: Option<usize>,
    /// Local transform relative to the parent, reoriented to Z-up like every
    /// other transform in the scene. Composing a node's chain of local
    /// transforms gives the world transform its meshes carry.
    pub transform: glam::Mat4,
    /// Indices into [`SceneData::meshes`] decoded from this node, one per
    /// primitive. A mesh rigidly attached to a joint has its transform baked
    /// and follows the skeleton instead of this node.
    pub meshes: Vec<usize>,
}

/// Which kind of punctual light a [`SceneLight`] is, with its kind-specific
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SceneLightKind {
    /// Parallel rays along the light's direction; intensity in lux.
    Directional,
    /// Light in every direction from the light's position; intensity in
    /// candela.
    Point {
        /// Distance at which the light may be treated as reaching zero, in
        /// world units. `None` when the source leaves it unset, which glTF
        /// reads as "infinite" in principle.
        range: Option<f32>,
    },
    /// A cone along the light's direction; intensity in candela.
    Spot {
        /// As [`SceneLightKind::Point::range`].
        range: Option<f32>,
        /// Half-angle in radians inside which the light is at full intensity.
        /// glTF's default is 0.
        inner_angle: f32,
        /// Half-angle in radians at which the light falls to zero. glTF's
        /// default is pi / 4.
        outer_angle: f32,
    },
}

/// A punctual light placed in a decoded scene (glTF `KHR_lights_punctual`).
///
/// Units pass straight through from the source: lux for directional lights,
/// candela for point and spot lights, which are the units `LightSource` takes.
/// [`to_light_source`](Self::to_light_source) builds one.
///
/// A source may leave a point or spot light's range unset, meaning unbounded.
/// The renderer culls a light at its range and has no unbounded value, so the
/// consumer picks one. A common heuristic is the distance at which the light
/// has faded below a small threshold; Filament's gltfio, for example, guesses
/// `(intensity / 0.05).powf(0.25)`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SceneLight {
    /// Light name, when the source provides one.
    pub name: Option<String>,
    /// Kind and kind-specific parameters.
    pub kind: SceneLightKind,
    /// Linear RGB colour, white by default.
    pub colour: [f32; 3],
    /// Photometric intensity: lux for [`SceneLightKind::Directional`], candela
    /// for the others.
    pub intensity: f32,
    /// World transform of the light's node, reoriented to Z-up. In this frame
    /// the light shines along its local +Y, which is where glTF's local -Z
    /// lands; read the direction with [`direction`](Self::direction).
    /// Scale does not affect the light.
    pub world_transform: glam::Mat4,
    /// Index into [`SceneData::nodes`] of the node carrying the light.
    pub node: Option<usize>,
}

impl SceneLight {
    /// World-space position of the light. Meaningless for a directional
    /// light, which has none.
    pub fn position(&self) -> [f32; 3] {
        self.world_transform
            .transform_point3(glam::Vec3::ZERO)
            .to_array()
    }

    /// Unit world-space direction the light shines along (light to surface).
    /// Meaningless for a point light.
    pub fn direction(&self) -> [f32; 3] {
        self.world_transform
            .transform_vector3(glam::Vec3::Y)
            .try_normalize()
            .unwrap_or(glam::Vec3::NEG_Z)
            .to_array()
    }

    /// The renderer's light for this one, with position, direction, colour,
    /// intensity, range and cone angles carried across and the source radius
    /// left at 0.
    ///
    /// `range_when_unset` is the range given to a point or spot light whose
    /// source left it unset (see the type's docs). The renderer culls a light
    /// beyond its range, so pass a finite distance that covers the scene.
    pub fn to_light_source(&self, range_when_unset: f32) -> LightSource {
        let position = self.position();
        let direction = self.direction();
        let mut light = match self.kind {
            SceneLightKind::Directional => {
                // LightKind::Directional takes the surface-to-light direction.
                let towards = direction.map(|c| -c);
                LightSource::directional_lux(towards, Lux(self.intensity))
            }
            SceneLightKind::Point { range } => LightSource::point_candela(
                position,
                Candela(self.intensity),
                range.unwrap_or(range_when_unset),
                0.0,
            ),
            SceneLightKind::Spot {
                range,
                inner_angle,
                outer_angle,
            } => LightSource::spot_candela(
                position,
                direction,
                Candela(self.intensity),
                range.unwrap_or(range_when_unset),
                inner_angle,
                outer_angle,
                0.0,
            ),
        };
        light.colour = Colour::from_linear_rgb_array(self.colour);
        light
    }
}

/// A mesh entry inside a loaded scene: the geometry, plus the skinning and
/// blend-shape data the renderer's mesh payload does not carry.
#[derive(Clone, Debug)]
pub struct SceneMesh {
    /// Mesh name.
    pub name: String,
    /// Geometry and named attributes, as the renderer uploads them.
    ///
    /// `normals` is always parallel to `positions`: a source with no normals
    /// gets area-weighted smooth normals at decode. `vertex_colours` is linear
    /// RGBA, from a channel such as glTF `COLOR_0` or PLY vertex colours.
    /// `uvs1` is the second content UV set (glTF `TEXCOORD_1`, or an FBX
    /// mesh's next authored UV layer), not a lightmap bake.
    ///
    /// Attributes are keyed by name. Per-vertex scalars are
    /// `AttributeData::Vertex` and per-vertex vectors `VertexVector`. Scalars
    /// a scientific format stores per cell are `Cell`, expanded to one value
    /// per triangle (a quad's value on both its triangles, a hex's on its whole
    /// boundary), and per-halfedge scalars `Halfedge`, three per triangle.
    pub mesh: MeshData,
    /// Skinning weights, parallel to `mesh.positions`, when the mesh is skinned.
    pub skin_weights: Option<SkinWeights>,
    /// Blend shapes for this mesh, in authored order. Empty when the source
    /// carries none. A downstream deformer indexes these by position, so the
    /// order is the order a weight palette is expected to follow.
    pub morph_targets: Vec<MorphTarget>,
    /// Index into `SceneData::materials`.
    pub material_index: Option<usize>,
    /// Local transform carried through from the source format.
    pub transform: glam::Mat4,
    /// Whether the source format explicitly marked the mesh double-sided.
    pub two_sided: bool,
    /// Parent mesh index for scene hierarchy reconstruction.
    pub parent_index: Option<usize>,
    /// Human-readable names for imported vertex attributes.
    pub vertex_attribute_names: Vec<String>,
    /// Optional importer-specific tags.
    pub metadata: HashMap<String, String>,
    /// Index into `SceneData::skeletons`, if this mesh is skinned.
    pub skeleton_index: Option<usize>,
}

impl Default for SceneMesh {
    fn default() -> Self {
        Self {
            name: String::new(),
            mesh: MeshData::default(),
            skin_weights: None,
            morph_targets: Vec::new(),
            material_index: None,
            transform: glam::Mat4::IDENTITY,
            two_sided: false,
            parent_index: None,
            vertex_attribute_names: Vec::new(),
            metadata: HashMap::new(),
            skeleton_index: None,
        }
    }
}

/// A generic point set with optional colours and scalar attributes.
#[derive(Clone, Debug, Default)]
pub struct PointSet {
    /// Point-set name.
    pub name: String,
    /// Point positions.
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point colours: sRGB-encoded RGB in `0..=1` with linear
    /// alpha, as point formats store them (a PLY `uchar` channel divided by
    /// 255, with no transfer curve applied). Empty when the source has none.
    ///
    /// Decode with `Colour::from_srgb_array` before treating them as linear.
    pub colors: Vec<[f32; 4]>,
    /// Optional primary scalar values.
    pub scalars: Vec<f32>,
    /// Named scalar attributes carried with the points.
    pub scalar_attributes: HashMap<String, Vec<f32>>,
}

impl PointSet {
    /// The renderer's point cloud item for these points, coloured by the best
    /// channel the set carries.
    ///
    /// Colour comes from the per-point colours when there are any (decoded from
    /// sRGB), otherwise from the primary scalar through the item's default
    /// colourmap over the scalar's own range, otherwise the item's default
    /// white. Size, render mode, transform and settings are the item's defaults,
    /// for the caller to set. The named scalar attributes are not carried: to
    /// colour by one, set `colour` to a `ColourSource::Scalar` built from
    /// [`scalar_attributes`](Self::scalar_attributes).
    pub fn to_point_cloud_item(&self) -> PointCloudItem {
        let mut item = PointCloudItem::default();
        item.positions = self.positions.clone();
        if !self.colors.is_empty() {
            item.colour = ColourSource::PerSample(
                self.colors
                    .iter()
                    .map(|&c| Colour::from_srgb_array(c))
                    .collect(),
            );
        } else if !self.scalars.is_empty() {
            item.colour = ColourSource::Scalar {
                values: self.scalars.clone(),
                range: None,
                colourmap: None,
            };
        }
        item
    }

    /// Create a degree-0 gaussian splat set from points.
    ///
    /// The colours become the degree-0 SH coefficients unchanged, since both are
    /// sRGB-referred; points without colours are white.
    pub fn to_gaussian_splats(
        &self,
        default_scale: [f32; 3],
        default_opacity: f32,
    ) -> GaussianSplatData {
        let count = self.positions.len();
        let mut sh_coefficients = Vec::with_capacity(count * 3);
        for index in 0..count {
            let rgb = self
                .colors
                .get(index)
                .copied()
                .map(|color| [color[0], color[1], color[2]])
                .unwrap_or([1.0, 1.0, 1.0]);
            sh_coefficients.extend_from_slice(&rgb);
        }

        GaussianSplatData {
            positions: self.positions.clone(),
            scales: vec![default_scale; count],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]; count],
            opacities: vec![default_opacity; count],
            sh_coefficients,
            sh_degree: ShDegree::Zero,
        }
    }
}

/// Geometry model for a structured volume.
#[derive(Clone, Debug)]
pub enum VolumeGridGeometry {
    /// Uniform voxel spacing.
    Uniform {
        /// World-space origin of the first sample.
        origin: [f32; 3],
        /// Uniform spacing between samples on each axis.
        spacing: [f32; 3],
    },
    /// Axis-aligned grid with variable spacing along each axis.
    Rectilinear {
        /// Sample coordinates along X.
        xs: Vec<f32>,
        /// Sample coordinates along Y.
        ys: Vec<f32>,
        /// Sample coordinates along Z.
        zs: Vec<f32>,
    },
}

impl Default for VolumeGridGeometry {
    fn default() -> Self {
        Self::Uniform {
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }
}

/// Dense structured volume data and named fields.
#[derive(Clone, Debug, Default)]
pub struct StructuredVolume {
    /// Volume name or field-set label.
    pub name: String,
    /// Point dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Geometry description for the grid.
    pub geometry: VolumeGridGeometry,
    /// Named point-centred scalar fields.
    pub point_fields: HashMap<String, Vec<f32>>,
    /// Named cell-centred scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
}

impl StructuredVolume {
    /// World-space axis-aligned bounds of the volume.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        match &self.geometry {
            VolumeGridGeometry::Uniform { origin, spacing } => {
                let max = [
                    origin[0] + spacing[0] * self.dims[0].saturating_sub(1) as f32,
                    origin[1] + spacing[1] * self.dims[1].saturating_sub(1) as f32,
                    origin[2] + spacing[2] * self.dims[2].saturating_sub(1) as f32,
                ];
                (*origin, max)
            }
            VolumeGridGeometry::Rectilinear { xs, ys, zs } => {
                let min = [
                    *xs.first().unwrap_or(&0.0),
                    *ys.first().unwrap_or(&0.0),
                    *zs.first().unwrap_or(&0.0),
                ];
                let max = [
                    *xs.last().unwrap_or(&0.0),
                    *ys.last().unwrap_or(&0.0),
                    *zs.last().unwrap_or(&0.0),
                ];
                (min, max)
            }
        }
    }

    /// Return scalar values for a named field, checking point fields then cell fields.
    pub fn scalar_values(&self, name: &str) -> Option<&[f32]> {
        if let Some(values) = self.point_fields.get(name) {
            return Some(values);
        }
        self.cell_fields.get(name).map(Vec::as_slice)
    }
}

/// Sparse regular grid data.
#[derive(Clone, Debug, Default)]
pub struct SparseGrid {
    /// World-space position of the `[0, 0, 0]` corner of cell `[0, 0, 0]`.
    pub origin: [f32; 3],
    /// Side length of one cubic cell in world units.
    pub cell_size: f32,
    /// Grid indices `[i, j, k]` of occupied cells.
    pub active_cells: Vec<[u32; 3]>,
    /// Named per-cell scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
    /// Named per-node scalar fields.
    pub node_fields: HashMap<String, Vec<f32>>,
    /// Named per-cell RGBA colours.
    pub cell_colors: HashMap<String, Vec<[f32; 4]>>,
}

/// A decoded scientific/container dataset that may expose several representations.
#[derive(Clone, Debug, Default)]
pub struct DecodedDataSet {
    /// Dataset name.
    pub name: String,
    /// Optional extracted surface mesh.
    pub surface_mesh: Option<MeshData>,
    /// Optional point-set representation.
    pub point_set: Option<PointSet>,
    /// Optional dense structured volume.
    pub volume: Option<StructuredVolume>,
    /// Optional sparse regular grid.
    pub sparse_grid: Option<Box<SparseGrid>>,
    /// Optional unstructured volume mesh.
    ///
    /// Per-cell fields are in `cell_scalars`, per-node fields in `node_scalars`
    /// as the file stores them (not averaged onto cells).
    pub volume_mesh: Option<Box<VolumeMeshData>>,
}

impl DecodedDataSet {
    /// Return the dataset as a surface mesh when available.
    pub fn as_surface_mesh(&self) -> Option<&MeshData> {
        self.surface_mesh.as_ref()
    }

    /// Return the dataset as a point set when available.
    pub fn as_point_set(&self) -> Option<&PointSet> {
        self.point_set.as_ref()
    }

    /// Return the dataset as a structured volume when available.
    pub fn as_structured_volume(&self) -> Option<&StructuredVolume> {
        self.volume.as_ref()
    }

    /// Return the dataset as a sparse grid when available.
    pub fn as_sparse_grid(&self) -> Option<&SparseGrid> {
        self.sparse_grid.as_deref()
    }

    /// Return the dataset as a volume mesh when available.
    pub fn as_volume_mesh(&self) -> Option<&VolumeMeshData> {
        self.volume_mesh.as_deref()
    }
}

/// A decoded multi-object scene.
#[derive(Clone, Debug, Default)]
pub struct SceneData {
    /// Meshes in the scene.
    pub meshes: Vec<SceneMesh>,
    /// Materials referenced by meshes.
    pub materials: Vec<MaterialData>,
    /// Point sets carried by the scene.
    pub point_sets: Vec<PointSet>,
    /// Skeletons referenced by skinned meshes via `SceneMesh::skeleton_index`.
    pub skeletons: Vec<Skeleton>,
    /// Animation clips targeting the scene's skeletons.
    pub animations: Vec<AnimationClip>,
    /// Blend-shape weight animations targeting the scene's morphable meshes.
    pub morph_animations: Vec<MorphWeightClip>,
    /// The source scene graph, for formats that keep one. Empty otherwise.
    /// glTF fills it in document node order, so a glTF node index is an index
    /// here.
    pub nodes: Vec<SceneNode>,
    /// Punctual lights placed in the scene.
    pub lights: Vec<SceneLight>,
    /// Animation clips targeting nodes in [`nodes`](Self::nodes) that are not
    /// skeleton joints.
    pub node_animations: Vec<NodeAnimationClip>,
}

/// How a gradient carries on past its end stops.
///
/// Follows SVG's `spreadMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpreadMethod {
    /// The end colours extend outward. Default.
    #[default]
    Pad,
    /// The gradient runs back and forth: each repeat is the mirror of the last.
    Reflect,
    /// The gradient starts again from the first stop.
    Repeat,
}

/// A fill or stroke paint on a [`VectorShape`].
///
/// Gradients keep the source geometry rather than a renderer's approximation of
/// it. Each is defined in its own frame, which `transform` maps into the art's
/// coordinates (the frame of [`VectorShape::subpaths`]: X right, Y down). The
/// transform folds in everything between the two: the gradient's own transform,
/// the shape's bounding box for a gradient given in bounding-box units, and the
/// transforms baked into the shape's geometry.
///
/// Stop colours are linear, with alpha carrying the stop opacity multiplied by
/// the paint and group opacity.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorPaint {
    /// One colour, alpha carrying the paint opacity multiplied by any enclosing
    /// group opacity.
    Solid(Colour),
    /// A linear gradient: position `0` at `start`, position `1` at `end`, and
    /// constant along lines perpendicular to that axis in the gradient's frame.
    LinearGradient {
        /// Where position `0` sits, in the gradient's frame.
        start: [f32; 2],
        /// Where position `1` sits, in the gradient's frame.
        end: [f32; 2],
        /// Colour stops, sorted by position, positions in `0..=1`.
        stops: Vec<GradientStop>,
        /// What happens past the end stops.
        spread: SpreadMethod,
        /// Maps the gradient's frame into the art's coordinates.
        transform: glam::Affine2,
    },
    /// A radial gradient: position `0` at `focal`, position `1` on the circle
    /// of `radius` about `centre`, in the gradient's frame.
    RadialGradient {
        /// Centre of the end circle, in the gradient's frame.
        centre: [f32; 2],
        /// Radius of the end circle, in the gradient's frame.
        radius: f32,
        /// Where position `0` sits. Equal to `centre` unless the source moved
        /// the focal point.
        focal: [f32; 2],
        /// Colour stops, sorted by position, positions in `0..=1`.
        stops: Vec<GradientStop>,
        /// What happens past the end stops.
        spread: SpreadMethod,
        /// Maps the gradient's frame into the art's coordinates. Under a
        /// non-uniform scale or a skew the circle becomes an ellipse.
        transform: glam::Affine2,
    },
}

impl VectorPaint {
    /// Convert to the overlay fill a renderer draws, for an item covering
    /// `bounds` (`[min_x, min_y, max_x, max_y]`) in the art's coordinates.
    ///
    /// Overlay gradients are relative to the item's box: a linear gradient is
    /// an angle across it and a radial one runs from its centre to its larger
    /// half-extent. The stops are resampled so the colours land where the
    /// source puts them within `bounds`:
    ///
    /// - A linear gradient converts exactly, whatever its spread or transform,
    ///   provided the item is drawn at a uniform scale of the art. Repeating
    ///   spreads become one stop run per repeat.
    /// - A radial gradient converts exactly when its circle is centred on
    ///   `bounds`, its focal point is its centre, and its transform has no
    ///   skew or non-uniform scale, out to the larger half-extent of `bounds`.
    ///   The overlay stops there, so the corners beyond hold that ring's
    ///   colour. Otherwise it is approximated: an off-centre circle is drawn
    ///   about the box centre, an ellipse as a circle of the mean radius, and
    ///   the focal point is ignored.
    ///
    /// The overlay blends between stops in linear colour, where SVG blends in
    /// sRGB, so a wide two-colour step can look slightly different midway.
    pub fn to_overlay_fill(&self, bounds: [f32; 4]) -> OverlayFill {
        let centre = [(bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5];
        let half = [
            ((bounds[2] - bounds[0]) * 0.5).abs(),
            ((bounds[3] - bounds[1]) * 0.5).abs(),
        ];
        match self {
            VectorPaint::Solid(colour) => OverlayFill::Solid(*colour),
            VectorPaint::LinearGradient {
                start,
                end,
                stops,
                spread,
                transform,
            } => match linear_gradient_axis(*start, *end, *transform) {
                Some((origin, axis)) => {
                    let (angle, t0, t1) = linear_gradient_span(origin, axis, centre, half);
                    OverlayFill::LinearGradient {
                        stops: resample_stops(stops, *spread, t0, t1),
                        angle,
                    }
                }
                // SVG paints a gradient with no length in its last stop.
                None => OverlayFill::Solid(last_stop_colour(stops)),
            },
            VectorPaint::RadialGradient {
                radius,
                stops,
                spread,
                transform,
                ..
            } => {
                let det = transform.matrix2.determinant().abs();
                let radius = radius * det.sqrt();
                if radius.is_nan() || radius <= 0.0 {
                    return OverlayFill::Solid(last_stop_colour(stops));
                }
                // The overlay's position 1 sits at the larger half-extent.
                let t1 = half[0].max(half[1]) / radius;
                OverlayFill::RadialGradient {
                    stops: resample_stops(stops, *spread, 0.0, t1),
                }
            }
        }
    }

    /// One colour standing in for the paint: the solid colour, or a gradient's
    /// average over its stops.
    pub fn mean_colour(&self) -> Colour {
        match self {
            VectorPaint::Solid(colour) => *colour,
            VectorPaint::LinearGradient { stops, .. }
            | VectorPaint::RadialGradient { stops, .. } => mean_stop_colour(stops),
        }
    }
}

/// The dash pattern of a [`VectorStroke`], following SVG's `stroke-dasharray`
/// and `stroke-dashoffset`.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDashes {
    /// Alternating dash and gap lengths, starting with a dash, in the same user
    /// units as the stroke width. Always an even count: an odd list from the
    /// source is repeated once, as SVG specifies.
    pub array: Vec<f32>,
    /// How far into the pattern the stroke starts, in the same units.
    pub offset: f32,
}

/// An outline stroke on a [`VectorShape`]: the contours drawn at `width` in the
/// drawing's user units, in `paint`.
///
/// Open subpaths stroke as open lines and closed ones stroke their whole
/// boundary, which is what makes stroke-drawn art (most icon and cursor sets)
/// readable: those contours are lines, not the outlines of filled blobs.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorStroke {
    /// Stroke paint, alpha carrying the stroke opacity multiplied by any
    /// enclosing group opacity.
    pub paint: VectorPaint,
    /// Stroke width in the same user units as the contour coordinates.
    ///
    /// Scaled by the transform baked into the geometry, so it stays in step
    /// with the contours. A non-uniform scale cannot be one scalar width, so it
    /// becomes the average of the transform's two scale factors: under
    /// `scale(2, 4)` the width is 3x the authored one. Consumers that scale the
    /// art further (fitting it to a rect, say) must scale this by the same
    /// factor, or the stroke stays a fixed width while the drawing grows. Note
    /// that `viewport-lib-ui` measures its own stroke width in logical pixels,
    /// so that scale factor is exactly the conversion.
    pub width: f32,
    /// How open ends and dash ends are drawn.
    pub cap: LineCap,
    /// How corners between segments are drawn.
    pub join: LineJoin,
    /// The longest a mitre corner may run, as a multiple of the width, before
    /// it is drawn as a bevel instead.
    pub mitre_limit: f32,
    /// The dash pattern, or `None` for a continuous stroke. Scaled with the
    /// width.
    pub dashes: Option<VectorDashes>,
}

impl VectorStroke {
    /// Convert to the overlay stroke a renderer draws, with every length
    /// multiplied by `scale`: the factor from the art's user units to the
    /// overlay's logical pixels.
    ///
    /// Cap, join and mitre limit carry over exactly. A two-entry dash array
    /// does too; a longer one becomes a single dash and gap, each the average
    /// of its kind. The overlay strokes in one colour, so a gradient stroke
    /// paint becomes its [`VectorPaint::mean_colour`].
    pub fn to_overlay_stroke(&self, scale: f32) -> OverlayStroke {
        let mut stroke = OverlayStroke::new(self.width * scale, self.paint.mean_colour())
            .with_cap(self.cap)
            .with_join(self.join)
            .with_mitre_limit(self.mitre_limit);
        if let Some(dashes) = &self.dashes {
            stroke = stroke.with_pattern(dash_pattern(&dashes.array, dashes.offset, scale));
        }
        stroke
    }
}

/// The point at position `0` and the gradient vector of a linear gradient, both
/// in the art's coordinates: position at `p` is `dot(p - origin, axis)`.
///
/// Mapping the two end points is not enough under a skew or non-uniform scale,
/// because lines of equal position stay parallel but stop being perpendicular
/// to the mapped axis. The gradient is a linear function of the point, so it
/// pulls back through the inverse transform exactly. `None` when the gradient
/// or its transform has no extent.
fn linear_gradient_axis(
    start: [f32; 2],
    end: [f32; 2],
    transform: glam::Affine2,
) -> Option<(glam::Vec2, glam::Vec2)> {
    let d = glam::Vec2::from(end) - glam::Vec2::from(start);
    let len2 = d.length_squared();
    if len2.is_nan() || len2 <= 0.0 || transform.matrix2.determinant() == 0.0 {
        return None;
    }
    let axis = transform.matrix2.inverse().transpose() * d / len2;
    axis.is_finite()
        .then(|| (transform.transform_point2(start.into()), axis))
}

/// The overlay angle for a linear gradient, and the source positions the
/// overlay's `0` and `1` land on across a box of `half` extents about
/// `centre`.
fn linear_gradient_span(
    origin: glam::Vec2,
    axis: glam::Vec2,
    centre: [f32; 2],
    half: [f32; 2],
) -> (f32, f32, f32) {
    let dir = axis.normalize();
    let angle = dir.y.atan2(dir.x);
    // The overlay runs position 0 to 1 across the box's extent along `dir`.
    let reach = (half[0] * dir.x).abs() + (half[1] * dir.y).abs();
    let mid = (glam::Vec2::from(centre) - origin).dot(axis);
    let span = reach * axis.length();
    (angle, mid - span, mid + span)
}

/// Resample `stops` so the overlay's position `0..=1` shows the source
/// gradient's positions `t0..=t1` (with `t0 < t1`), spread as `spread` says
/// past its ends.
///
/// The output has a stop at each end and at every source stop and repeat
/// boundary in between, so it is exact for an overlay that blends linearly
/// between stops. A spread over a very long range is capped at a few hundred
/// repeats.
fn resample_stops(
    stops: &[GradientStop],
    spread: SpreadMethod,
    t0: f32,
    t1: f32,
) -> Vec<GradientStop> {
    const MAX_REPEATS: f32 = 256.0;
    if stops.is_empty() {
        return Vec::new();
    }
    if !(t1 - t0).is_finite() || t1 - t0 <= f32::EPSILON {
        return vec![GradientStop::new(0.0, sample_spread(stops, spread, t0))];
    }

    // Source positions where the colour can bend: each stop, in every repeat
    // the range crosses.
    let mut breaks = vec![t0, t1];
    match spread {
        SpreadMethod::Pad => breaks.extend(stops.iter().map(|s| s.position)),
        SpreadMethod::Repeat | SpreadMethod::Reflect => {
            let mut period = t0.floor().max(t1.floor() - MAX_REPEATS);
            while period <= t1.floor() {
                breaks.push(period);
                for stop in stops {
                    breaks.push(period + stop.position);
                    breaks.push(period + 1.0 - stop.position);
                }
                period += 1.0;
            }
        }
    }
    breaks.retain(|t| *t >= t0 && *t <= t1);
    breaks.sort_by(f32::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    // A repeat boundary, or two source stops at one position, is a hard edge.
    // The colour is read just either side of each inner break, and where the
    // two differ both are kept at the same position, which the overlay draws
    // as a step.
    let last = breaks.len() - 1;
    let mut out = Vec::with_capacity(breaks.len() * 2);
    for (i, &t) in breaks.iter().enumerate() {
        let position = (t - t0) / (t1 - t0);
        let eps = 1e-4_f32.max(t.abs() * 1e-6);
        let before = sample_spread(stops, spread, t - eps);
        let after = sample_spread(stops, spread, t + eps);
        if i == 0 {
            out.push(GradientStop::new(position, after));
        } else if i == last {
            out.push(GradientStop::new(position, before));
        } else if colour_close(before, after) {
            out.push(GradientStop::new(position, sample_spread(stops, spread, t)));
        } else {
            out.push(GradientStop::new(position, before));
            out.push(GradientStop::new(position, after));
        }
    }
    out
}

fn colour_close(a: Colour, b: Colour) -> bool {
    let (a, b) = (a.to_linear_rgba(), b.to_linear_rgba());
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

/// The colour at source position `t`, with the spread applied past the ends.
fn sample_spread(stops: &[GradientStop], spread: SpreadMethod, t: f32) -> Colour {
    let t = match spread {
        SpreadMethod::Pad => t,
        SpreadMethod::Repeat => t - t.floor(),
        SpreadMethod::Reflect => {
            let m = t.rem_euclid(2.0);
            if m > 1.0 { 2.0 - m } else { m }
        }
    };
    sample_stops(stops, t)
}

/// The colour at position `t` along sorted stops, clamped to the end stops.
fn sample_stops(stops: &[GradientStop], t: f32) -> Colour {
    let Some(first) = stops.first() else {
        return Colour::linear(0.0, 0.0, 0.0, 0.0);
    };
    if t <= first.position {
        return first.colour;
    }
    for pair in stops.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if t <= b.position {
            let span = b.position - a.position;
            if span <= 0.0 {
                return b.colour;
            }
            return a.colour.lerp(b.colour, (t - a.position) / span);
        }
    }
    last_stop_colour(stops)
}

fn last_stop_colour(stops: &[GradientStop]) -> Colour {
    stops
        .last()
        .map_or(Colour::linear(0.0, 0.0, 0.0, 0.0), |s| s.colour)
}

/// The average colour over positions `0..=1`, the end stops padding outward.
fn mean_stop_colour(stops: &[GradientStop]) -> Colour {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return Colour::linear(0.0, 0.0, 0.0, 0.0);
    };
    let mut sum = [0.0f32; 4];
    let mut add = |colour: [f32; 4], weight: f32| {
        for (s, c) in sum.iter_mut().zip(colour) {
            *s += c * weight;
        }
    };
    add(
        first.colour.to_linear_rgba(),
        first.position.clamp(0.0, 1.0),
    );
    for pair in stops.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let weight = (b.position.clamp(0.0, 1.0) - a.position.clamp(0.0, 1.0)).max(0.0);
        let mid = a.colour.lerp(b.colour, 0.5).to_linear_rgba();
        add(mid, weight);
    }
    add(
        last.colour.to_linear_rgba(),
        1.0 - last.position.clamp(0.0, 1.0),
    );
    Colour::from_linear_array(sum)
}

/// The overlay pattern for an SVG dash array. Two entries map exactly; longer
/// arrays average their dashes and their gaps. An array with no length is a
/// continuous stroke.
fn dash_pattern(array: &[f32], offset: f32, scale: f32) -> StrokePattern {
    let mean = |values: Vec<f32>| -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    };
    let dash = mean(array.iter().step_by(2).copied().collect());
    let gap = mean(array.iter().skip(1).step_by(2).copied().collect());
    if dash.is_nan() || gap.is_nan() || dash + gap <= 0.0 {
        return StrokePattern::Solid;
    }
    StrokePattern::Dashed {
        dash_length: dash * scale,
        gap_length: gap * scale,
        offset: offset * scale,
    }
}

/// One shape from a vector drawing: its contours, fill rule, and the resolved
/// fill and stroke paint.
///
/// `fill` and `stroke` are independent and either can be absent. A shape with a
/// stroke and no fill is stroke-only: draw the contours as lines, not as filled
/// area. The one case the two fields cannot tell apart is a shape whose only
/// paint is a pattern, which the loader does not resolve: it arrives with both
/// `None`, indistinguishable from a shape with no paint at all. The geometry is
/// emitted either way.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorShape {
    /// Contours making up the shape. Curves are preserved, not flattened.
    ///
    /// Coordinates are in the source drawing's user units, in the 2D canvas
    /// frame: X right, Y down from the top-left origin. The crate's Z-up
    /// convention covers 3D scene geometry and does not apply here.
    pub subpaths: Vec<SubPath>,
    /// How the subpaths combine into filled area.
    pub fill_rule: FillRule,
    /// Resolved fill paint. `None` when the source shape has no fill, or a
    /// pattern paint the loader does not resolve.
    pub fill: Option<VectorPaint>,
    /// Resolved outline stroke. `None` when the source shape has no stroke, or
    /// a pattern stroke paint the loader does not resolve.
    pub stroke: Option<VectorStroke>,
    /// The source element's own id, or `None` when it has none. Basic shapes
    /// (`<rect>`, `<circle>`, and so on) carry theirs as well as `<path>`.
    ///
    /// A shape drawn by `<use>` has no id of its own: the referenced element's
    /// id is not copied onto the clone. The `<use>` element's id is in
    /// [`VectorShape::groups`] instead. Geometry generated by markers has no id
    /// either.
    pub id: Option<String>,
    /// Ids of the enclosing groups, outermost first. Groups with no id are
    /// left out, so a shape in `<g id="map"><g><g id="NSW">` lists `["map",
    /// "NSW"]`. A `<use>` element counts as a group here.
    ///
    /// The output is flat, so this is the only record of the nesting. A region
    /// drawn as several paths is usually keyed by a group id rather than by
    /// its paths' own ids.
    pub groups: Vec<String>,
}

/// A decoded 2D vector drawing: filled shapes in draw order plus the source
/// canvas size. Produced by [`crate::loaders::svg::vector_from_path`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VectorArt {
    /// Filled shapes in the drawing's paint order (first painted first).
    pub shapes: Vec<VectorShape>,
    /// Source canvas size in user units (width, height).
    pub size: [f32; 2],
}

/// A rectangle of pixels in an image, in the image's own frame: X right and Y
/// down from the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PixelRect {
    /// Left edge, in pixels from the image's left.
    pub x: u32,
    /// Top edge, in pixels from the image's top.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl PixelRect {
    /// A rectangle at `(x, y)` of `width` by `height` pixels.
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// The rectangle as texture coordinates `[u_min, v_min, u_max, v_max]` in an
    /// image of `image_size` pixels.
    ///
    /// `v` follows the pixel rows: `0` at the image's top row. A texture
    /// uploaded with its first row at the bottom wants `1 - v` instead.
    pub fn uv(self, image_size: [u32; 2]) -> [f32; 4] {
        let (w, h) = (image_size[0].max(1) as f32, image_size[1].max(1) as f32);
        [
            self.x as f32 / w,
            self.y as f32 / h,
            (self.x + self.width) as f32 / w,
            (self.y + self.height) as f32 / h,
        ]
    }
}

/// Custom properties attached to a map, layer, tileset, tile or object, by
/// name. Sorted, so two decodes of one file compare equal.
pub type Properties = std::collections::BTreeMap<String, PropertyValue>;

/// One typed custom property value.
///
/// An enum-typed property arrives as the string or integer it is stored as.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    /// Text.
    String(String),
    /// A whole number.
    Int(i64),
    /// A real number.
    Float(f64),
    /// True or false.
    Bool(bool),
    /// A colour, or `None` when the property is set to no colour.
    Colour(Option<Colour>),
    /// A file, resolved against the directory of the file that names it when
    /// that is known.
    File(PathBuf),
    /// A reference to a [`MapObject`] by its id, `0` for none.
    Object(u32),
    /// A list of values, from a format whose fields can hold several.
    List(Vec<PropertyValue>),
    /// A value of a custom class: its class name and its members.
    Class {
        /// The class name.
        class: String,
        /// The members that are set.
        members: Properties,
    },
}

/// How a tile map lays its grid out on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileOrientation {
    /// Square-on rows and columns.
    #[default]
    Orthogonal,
    /// Diamond tiles, each row running down and to the right.
    Isometric,
    /// Diamond tiles in staggered rows or columns.
    Staggered {
        /// Which axis is staggered.
        axis: StaggerAxis,
        /// Whether the odd or the even rows (or columns) are shifted.
        index: StaggerIndex,
    },
    /// Hexagonal tiles in staggered rows or columns.
    Hexagonal {
        /// Which axis is staggered.
        axis: StaggerAxis,
        /// Whether the odd or the even rows (or columns) are shifted.
        index: StaggerIndex,
        /// Length in pixels of the tile's flat side.
        side_length: u32,
    },
}

/// The staggered axis of a staggered or hexagonal map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StaggerAxis {
    /// Columns are staggered.
    X,
    /// Rows are staggered.
    #[default]
    Y,
}

/// Which rows or columns of a staggered or hexagonal map are shifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StaggerIndex {
    /// The odd ones.
    #[default]
    Odd,
    /// The even ones.
    Even,
}

/// The order a map's editor draws its tiles in. It matters only where tiles
/// overlap their neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TileRenderOrder {
    /// Left to right, top to bottom.
    #[default]
    RightDown,
    /// Left to right, bottom to top.
    RightUp,
    /// Right to left, top to bottom.
    LeftDown,
    /// Right to left, bottom to top.
    LeftUp,
}

/// A decoded 2D tile map: its grid, its layers in paint order, and the tilesets
/// its cells refer to.
///
/// Everything is in the map's pixel frame: X right and Y down from the top-left
/// corner, as the editor shows it. The crate's Z-up convention covers 3D scene
/// geometry and does not apply here; a consumer places the map on a plane of
/// its choosing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileMapData {
    /// Size in tiles. For an infinite map this is the editor's nominal size;
    /// each tile layer records the area it actually covers.
    pub size: [u32; 2],
    /// Size of one grid cell in pixels.
    pub tile_size: [u32; 2],
    /// How the grid is laid out. Only orthogonal maps draw as square rows and
    /// columns; the others are decoded as stored and the projection is left to
    /// the consumer.
    pub orientation: TileOrientation,
    /// The order the editor draws tiles in.
    pub render_order: TileRenderOrder,
    /// Whether the map is unbounded, its tile layers stored as chunks.
    pub infinite: bool,
    /// Background colour, if the map sets one.
    pub background: Option<Colour>,
    /// The point, in pixels, that parallax scrolling is relative to.
    pub parallax_origin: [f32; 2],
    /// The map's class name, empty when unset.
    pub class: String,
    /// Layers in paint order, first painted first.
    pub layers: Vec<TileLayerData>,
    /// Tilesets the map's cells and tile objects index into.
    pub tilesets: Vec<TileSetData>,
    /// The map's custom properties.
    pub properties: Properties,
}

/// Several tile maps placed in one world, as an LDtk project holds its levels.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileWorldData {
    /// The world's name, empty for a project with a single unnamed world.
    pub name: String,
    /// How the levels are arranged.
    pub layout: WorldLayout,
    /// The levels, in the project's order.
    pub levels: Vec<TileWorldLevel>,
}

/// One level of a [`TileWorldData`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileWorldLevel {
    /// The level's name.
    pub name: String,
    /// The level's unique id in the project.
    pub iid: String,
    /// The level's top-left corner in world pixels, X right and Y down.
    pub position: [i32; 2],
    /// Its depth in the world, for levels stacked on top of each other.
    pub depth: i32,
    /// The level itself.
    pub map: TileMapData,
}

/// How the levels of a [`TileWorldData`] are arranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WorldLayout {
    /// Anywhere, at their own positions.
    #[default]
    Free,
    /// On a grid of cells this many pixels across, at their own positions.
    GridVania {
        /// Size of one world grid cell in pixels.
        cell_size: [u32; 2],
    },
    /// In a row, left to right in level order.
    LinearHorizontal,
    /// In a column, top to bottom in level order.
    LinearVertical,
}

/// One layer of a [`TileMapData`]: the settings every layer has, and what it
/// holds.
///
/// A layer inside a group is relative to it: offsets add, opacity and tint
/// multiply, parallax multiplies and visibility needs both.
#[derive(Debug, Clone, PartialEq)]
pub struct TileLayerData {
    /// The layer's id, unique within the map. `0` when the file has none.
    pub id: u32,
    /// The layer's name.
    pub name: String,
    /// The layer's class name, empty when unset.
    pub class: String,
    /// Offset in pixels, X right and Y down.
    pub offset: [f32; 2],
    /// Opacity, `0..=1`.
    pub opacity: f32,
    /// Whether the layer is shown.
    pub visible: bool,
    /// Scroll factor relative to the camera on each axis: `1` moves with the
    /// map, `0` stays fixed on screen.
    pub parallax: [f32; 2],
    /// Colour multiplied into everything the layer draws, if set.
    pub tint: Option<Colour>,
    /// The layer's custom properties.
    pub properties: Properties,
    /// What the layer holds.
    pub kind: TileLayerKind,
}

/// The contents of a [`TileLayerData`].
#[derive(Debug, Clone, PartialEq)]
pub enum TileLayerKind {
    /// A grid of tile cells.
    Tiles(TileGrid),
    /// Free-placed objects: spawn points, triggers, collision shapes, tiles.
    Objects {
        /// Colour the editor draws the objects in, if set.
        colour: Option<Colour>,
        /// Whether objects draw in the order listed or sorted by their Y.
        draw_order: ObjectDrawOrder,
        /// The objects.
        objects: Vec<MapObject>,
    },
    /// One image placed at the layer's offset.
    Image {
        /// The image, or `None` when the layer has none set.
        image: Option<TileImage>,
        /// Whether the image repeats across the X and Y axes.
        repeat: [bool; 2],
    },
    /// A grid of integer values rather than tiles, from LDtk: collision,
    /// terrain kinds, anything a game reads by cell.
    IntGrid(IntGridData),
    /// A group of layers, in paint order.
    Group(Vec<TileLayerData>),
}

/// The cells of an integer grid layer, in row-major order, top row first.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IntGridData {
    /// Width and height in cells.
    pub size: [u32; 2],
    /// Size of one cell in pixels.
    pub cell_size: [u32; 2],
    /// `size[0] * size[1]` values, row by row from the top. `0` is an empty
    /// cell.
    pub values: Vec<u32>,
    /// What each value means, as the project defines it.
    pub definitions: Vec<IntGridValue>,
}

/// The definition of one value an integer grid holds.
#[derive(Debug, Clone, PartialEq)]
pub struct IntGridValue {
    /// The value in the grid.
    pub value: u32,
    /// Its name, empty when unset.
    pub name: String,
    /// The colour the editor shows it in.
    pub colour: Colour,
}

/// The cells of a tile layer: a dense grid in row-major order, top row first.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileGrid {
    /// Map coordinates in tiles of the grid's top-left cell. `[0, 0]` for a
    /// bounded map; an infinite map's layer can start anywhere.
    pub origin: [i32; 2],
    /// Width and height in tiles.
    pub size: [u32; 2],
    /// Size of one cell in pixels: the map's tile size for Tiled, the layer's
    /// own grid size for LDtk.
    pub cell_size: [u32; 2],
    /// `size[0] * size[1]` cells, row by row from the top. A cell with no tile
    /// is [`TileCell::EMPTY`].
    pub cells: Vec<TileCell>,
}

impl TileGrid {
    /// The cell at map coordinates `(x, y)` in tiles, or `None` outside the
    /// grid.
    pub fn get(&self, x: i32, y: i32) -> Option<TileCell> {
        let (cx, cy) = (x - self.origin[0], y - self.origin[1]);
        if cx < 0 || cy < 0 || cx as u32 >= self.size[0] || cy as u32 >= self.size[1] {
            return None;
        }
        self.cells
            .get(cy as usize * self.size[0] as usize + cx as usize)
            .copied()
    }
}

/// One cell of a tile layer, or a tile object's image: which tile, from which
/// tileset, and how it is turned.
///
/// The flips follow the editor: the diagonal flip mirrors across the
/// top-left to bottom-right diagonal and applies first, then the horizontal
/// and vertical flips. Together they give the eight turns and mirrors of a
/// square tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCell {
    /// Index into [`TileMapData::tilesets`].
    pub tileset: u32,
    /// The tile's index within that tileset.
    pub index: u32,
    /// Mirrored left to right.
    pub flip_h: bool,
    /// Mirrored top to bottom.
    pub flip_v: bool,
    /// Mirrored across the top-left to bottom-right diagonal.
    pub flip_d: bool,
    /// Turned 120 degrees clockwise, on a hexagonal map only.
    pub rotate_120: bool,
}

impl TileCell {
    /// A cell with no tile.
    pub const EMPTY: Self = Self {
        tileset: u32::MAX,
        index: u32::MAX,
        flip_h: false,
        flip_v: false,
        flip_d: false,
        rotate_120: false,
    };

    /// Tile `index` of tileset `tileset`, unturned.
    pub const fn new(tileset: u32, index: u32) -> Self {
        Self {
            tileset,
            index,
            ..Self::EMPTY
        }
    }

    /// Whether the cell has no tile.
    pub const fn is_empty(self) -> bool {
        self.tileset == u32::MAX
    }
}

/// An image a tileset, tile or image layer refers to.
#[derive(Debug, Clone, PartialEq)]
pub struct TileImage {
    /// Where the pixels are. A file is resolved against the directory of the
    /// file that names it when that is known; decode it with an image loader.
    pub source: TextureSource,
    /// Size in pixels as the file records it, `[0, 0]` when it does not.
    pub size: [u32; 2],
    /// A colour drawn as transparent, if the file sets one.
    pub transparent: Option<Colour>,
}

/// A tileset: the tiles a map's cells index into, and what each tile carries.
///
/// Tiles come either from one atlas image cut into a grid or, for an image
/// collection, from one image per tile. [`TileSetData::tile_rect`] answers
/// where a tile's pixels are either way.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileSetData {
    /// The tileset's name.
    pub name: String,
    /// The tileset's class name, empty when unset.
    pub class: String,
    /// The global id of the tileset's first tile in the map that references it.
    /// A cell's [`TileCell::index`] already has it taken off. `0` for a tileset
    /// read on its own, or one only an object template uses.
    pub first_gid: u32,
    /// The external file the tileset came from, if it is not embedded.
    pub source: Option<PathBuf>,
    /// The size of a tile in pixels: for an atlas, the grid cell.
    pub tile_size: [u32; 2],
    /// Number of tiles.
    pub tile_count: u32,
    /// Atlas columns. `0` for an image collection.
    pub columns: u32,
    /// Pixels between neighbouring atlas tiles.
    pub spacing: u32,
    /// Pixels between the atlas edge and the first tiles.
    pub margin: u32,
    /// Offset in pixels to draw each tile at, X right and Y down.
    pub tile_offset: [i32; 2],
    /// The atlas image, or `None` for an image collection.
    pub image: Option<TileImage>,
    /// The tiles that carry more than their place in the grid, sorted by index:
    /// properties, animation, collision, or an image of their own.
    pub tiles: Vec<TileData>,
    /// The tileset's custom properties.
    pub properties: Properties,
}

impl TileSetData {
    /// The extra data of tile `index`, if it has any.
    pub fn tile(&self, index: u32) -> Option<&TileData> {
        self.tiles
            .binary_search_by_key(&index, |t| t.index)
            .ok()
            .map(|i| &self.tiles[i])
    }

    /// Where tile `index`'s pixels are: in the atlas image, or in the tile's own
    /// image when it has one. `None` for a tile the tileset does not have.
    pub fn tile_rect(&self, index: u32) -> Option<PixelRect> {
        if let Some(tile) = self.tile(index) {
            if let Some(rect) = tile.rect {
                return Some(rect);
            }
            if let Some(image) = &tile.image {
                return Some(PixelRect::new(0, 0, image.size[0], image.size[1]));
            }
        }
        if self.image.is_none() || self.columns == 0 || index >= self.tile_count {
            return None;
        }
        let (col, row) = (index % self.columns, index / self.columns);
        Some(PixelRect::new(
            self.margin + col * (self.tile_size[0] + self.spacing),
            self.margin + row * (self.tile_size[1] + self.spacing),
            self.tile_size[0],
            self.tile_size[1],
        ))
    }
}

/// The extra data of one tile in a [`TileSetData`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TileData {
    /// The tile's index within its tileset.
    pub index: u32,
    /// The tile's class name, empty when unset.
    pub class: String,
    /// The chance the editor's terrain tools pick this tile, relative to others.
    pub probability: f32,
    /// The tile's own image, in an image collection.
    pub image: Option<TileImage>,
    /// The part of the image the tile shows, when it is not the whole of its own
    /// image or its place in the atlas grid.
    pub rect: Option<PixelRect>,
    /// Animation frames, in order, empty for a still tile.
    pub animation: Vec<TileFrame>,
    /// Collision shapes, in the tile's pixel frame.
    pub collision: Vec<MapObject>,
    /// The tile's custom properties.
    pub properties: Properties,
}

/// One frame of a tile animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileFrame {
    /// The tile shown, an index into the same tileset.
    pub index: u32,
    /// How long it shows, in milliseconds.
    pub duration_ms: u32,
}

/// Whether an object layer draws in the order listed or sorted by position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ObjectDrawOrder {
    /// Sorted by Y, lower objects in front.
    #[default]
    TopDown,
    /// In the order listed.
    Index,
}

/// An object placed on a map or attached to a tile: a shape with a name, a
/// class and properties.
///
/// `position` is in pixels, X right and Y down. For a tile object it is the
/// bottom-left corner of the tile's image; otherwise the top-left corner, or
/// the first point of a polygon or polyline. `rotation` turns the object about
/// `position`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MapObject {
    /// The object's id, unique within the map. `0` when the file has none.
    pub id: u32,
    /// The object's name.
    pub name: String,
    /// The object's class name (its type), empty when unset.
    pub class: String,
    /// Where the object is, in pixels.
    pub position: [f32; 2],
    /// Width and height in pixels. Zero for a point, and for a polygon or
    /// polyline, whose extent is its points.
    pub size: [f32; 2],
    /// Rotation in degrees, clockwise.
    pub rotation: f32,
    /// Whether the object is shown.
    pub visible: bool,
    /// The object's shape.
    pub shape: MapObjectShape,
    /// The tile the object draws, for a tile object.
    pub tile: Option<TileCell>,
    /// The object's custom properties.
    pub properties: Properties,
}

/// The shape of a [`MapObject`].
#[derive(Debug, Clone, PartialEq, Default)]
pub enum MapObjectShape {
    /// A rectangle of the object's size. A tile object has this shape too.
    #[default]
    Rectangle,
    /// An ellipse filling the object's size.
    Ellipse,
    /// A single point.
    Point,
    /// A closed polygon, its points relative to the object's position.
    Polygon(Vec<[f32; 2]>),
    /// An open line, its points relative to the object's position.
    Polyline(Vec<[f32; 2]>),
    /// A block of text filling the object's size.
    Text(MapText),
}

/// The text of a text [`MapObject`] and how it is set.
#[derive(Debug, Clone, PartialEq)]
pub struct MapText {
    /// The text.
    pub text: String,
    /// Font family name.
    pub font_family: String,
    /// Font size in pixels.
    pub pixel_size: u32,
    /// Whether the text wraps to the object's width.
    pub wrap: bool,
    /// Text colour.
    pub colour: Colour,
    /// Bold.
    pub bold: bool,
    /// Italic.
    pub italic: bool,
    /// Underlined.
    pub underline: bool,
    /// Struck through.
    pub strikeout: bool,
    /// Whether kerning is applied.
    pub kerning: bool,
    /// Horizontal alignment within the object.
    pub h_align: TextAlign,
    /// Vertical alignment within the object.
    pub v_align: TextAlign,
}

impl Default for MapText {
    fn default() -> Self {
        Self {
            text: String::new(),
            font_family: "sans-serif".into(),
            pixel_size: 16,
            wrap: false,
            colour: Colour::srgb_u8(0, 0, 0, 255),
            bold: false,
            italic: false,
            underline: false,
            strikeout: false,
            kerning: true,
            h_align: TextAlign::Start,
            v_align: TextAlign::Start,
        }
    }
}

/// Alignment along one axis of a text block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TextAlign {
    /// Left, or top.
    #[default]
    Start,
    /// Centred.
    Centre,
    /// Right, or bottom.
    End,
    /// Spread to both edges. Horizontal only.
    Justify,
}

/// A decoded sprite sheet: an atlas image and the frames, animations and
/// slices an exporter wrote beside it.
///
/// Everything is in the image's pixel frame: X right and Y down from the
/// top-left corner.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteSheetData {
    /// The atlas image, resolved against the sheet's directory when that is
    /// known, or `None` when the sheet does not name one.
    pub image: Option<TextureSource>,
    /// The atlas size in pixels. When the sheet does not record it, the extent
    /// of its frames.
    pub size: [u32; 2],
    /// The scale the sheet was exported at, `1` when unset.
    pub scale: f32,
    /// The exporter that wrote the sheet, as it names itself, empty when unset.
    pub app: String,
    /// Frames in the order the sheet lists them. Animations index into this.
    pub frames: Vec<SpriteFrame>,
    /// Named runs of frames.
    pub animations: Vec<SpriteAnimation>,
    /// Named regions of the sprite, with nine-slice centres and pivots.
    pub slices: Vec<SpriteSlice>,
    /// The width and height of one pixel as the artist drew it: `[1, 1]` for
    /// square pixels, `[2, 1]` for pixels twice as wide as tall. Draw each
    /// frame stretched by this. JSON sheets do not record it and give `[1, 1]`.
    pub pixel_ratio: [u32; 2],
    /// What the source holds that is not decoded, one entry per kind, such as
    /// `tilemap layer "Ground"`. Empty when everything was read.
    pub unsupported: Vec<String>,
}

impl Default for SpriteSheetData {
    fn default() -> Self {
        Self {
            image: None,
            size: [0, 0],
            scale: 1.0,
            app: String::new(),
            frames: Vec::new(),
            animations: Vec::new(),
            slices: Vec::new(),
            pixel_ratio: [1, 1],
            unsupported: Vec::new(),
        }
    }
}

/// One frame of a [`SpriteSheetData`].
///
/// Exporters trim the transparent border off a frame and pack what is left,
/// so a frame has two sizes: the trimmed pixels in the atlas, `rect`, and the
/// whole untrimmed sprite, `source_size`, with `source_offset` saying where
/// the trimmed pixels sit within it. Draw the frame at `source_offset` inside
/// a `source_size` box and every frame of an animation lines up.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteFrame {
    /// The frame's name, usually the source file it was packed from.
    pub name: String,
    /// The pixels the frame occupies in the atlas. For a rotated frame this is
    /// the turned rectangle, so its width and height are the sprite's height
    /// and width.
    pub rect: PixelRect,
    /// `rect` as texture coordinates `[u_min, v_min, u_max, v_max]`, `v` `0`
    /// at the image's top row. A texture uploaded with its first row at the
    /// bottom wants `1 - v` instead.
    pub uv: [f32; 4],
    /// Whether the exporter turned the frame 90 degrees clockwise to pack it.
    /// Turn it back anticlockwise to draw it upright.
    pub rotated: bool,
    /// Whether the exporter trimmed the frame's transparent border.
    pub trimmed: bool,
    /// The untrimmed sprite's size in pixels.
    pub source_size: [u32; 2],
    /// Where the trimmed pixels' top-left corner sits in the untrimmed sprite.
    pub source_offset: [i32; 2],
    /// The frame's pivot as a fraction of the untrimmed sprite, `[0, 0]` its
    /// top-left and `[1, 1]` its bottom-right, when the sheet sets one.
    pub pivot: Option<[f32; 2]>,
    /// How long the frame shows in an animation, when the sheet says.
    pub duration_ms: Option<u32>,
}

/// A named run of frames in a [`SpriteSheetData`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteAnimation {
    /// The animation's name.
    pub name: String,
    /// Index of its first frame in [`SpriteSheetData::frames`].
    pub first: u32,
    /// Index of its last frame, inclusive.
    pub last: u32,
    /// The order the frames play in.
    pub direction: AnimationDirection,
    /// How many times it plays, or `None` to loop for ever.
    pub repeat: Option<u32>,
    /// The colour the editor shows the animation in, if set.
    pub colour: Option<Colour>,
    /// Free text the author attached, empty when unset.
    pub data: String,
}

/// The order an animation's frames play in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AnimationDirection {
    /// First to last.
    #[default]
    Forward,
    /// Last to first.
    Reverse,
    /// First to last and back, not repeating the end frames.
    PingPong,
    /// Last to first and back, not repeating the end frames.
    PingPongReverse,
}

/// A named region of a sprite, which can change from frame to frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteSlice {
    /// The slice's name.
    pub name: String,
    /// The colour the editor shows the slice in, if set.
    pub colour: Option<Colour>,
    /// Free text the author attached, empty when unset.
    pub data: String,
    /// The slice from each frame it changes on, in frame order. A key holds
    /// until the next one.
    pub keys: Vec<SpriteSliceKey>,
}

/// One key of a [`SpriteSlice`]: its bounds from a frame onward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpriteSliceKey {
    /// The frame the key starts on.
    pub frame: u32,
    /// The slice's bounds in the untrimmed sprite, in pixels.
    pub bounds: PixelRect,
    /// The nine-slice centre, relative to `bounds`, if the slice has one: the
    /// part that stretches, with the border around it drawn at its own size.
    pub centre: Option<PixelRect>,
    /// The pivot in pixels relative to `bounds`, if the slice has one.
    pub pivot: Option<[i32; 2]>,
}

/// A layered sprite: the canvas, its layers and each frame's cels, before
/// they are flattened into one image per frame.
///
/// Everything is in the canvas's pixel frame: X right and Y down from the
/// top-left corner. Flattening the visible cels with their blend modes and
/// opacity gives the frames [`SpriteSheetData`] holds for the same source.
#[derive(Debug, Clone, PartialEq)]
pub struct LayeredSpriteData {
    /// Canvas width and height in pixels.
    pub size: [u32; 2],
    /// The width and height of one pixel as the artist drew it; see
    /// [`SpriteSheetData::pixel_ratio`].
    pub pixel_ratio: [u32; 2],
    /// How the source stored its pixels. Every cel image here is RGBA
    /// whatever this says.
    pub colour_mode: SpriteColourMode,
    /// Whether a group composites its children on their own and then blends
    /// the result with its own mode and opacity. Without it a group's mode and
    /// opacity do not apply, and its children draw straight onto what is
    /// below.
    pub groups_isolated: bool,
    /// Layers, bottom first.
    pub layers: Vec<SpriteLayer>,
    /// Frames, in order.
    pub frames: Vec<SpriteLayerFrame>,
    /// The palette, for a consumer that recolours an indexed sprite. Empty
    /// when the source has none.
    pub palette: Vec<Colour>,
    /// The palette index an indexed sprite draws as transparent on every
    /// layer but the background. `None` for other colour modes.
    pub transparent_index: Option<u32>,
    /// Named runs of frames.
    pub animations: Vec<SpriteAnimation>,
    /// Named regions of the sprite.
    pub slices: Vec<SpriteSlice>,
    /// What the source holds that is not decoded; see
    /// [`SpriteSheetData::unsupported`].
    pub unsupported: Vec<String>,
}

/// How a layered sprite's source stored its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpriteColourMode {
    /// Red, green, blue and alpha.
    #[default]
    Rgba,
    /// A grey value and alpha.
    Greyscale,
    /// An index into the palette.
    Indexed,
}

/// One layer of a [`LayeredSpriteData`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteLayer {
    /// The layer's name.
    pub name: String,
    /// What the layer holds.
    pub kind: SpriteLayerKind,
    /// Index of the group the layer is in, or `None` at the top level.
    pub parent: Option<u32>,
    /// The layer's own visibility. It draws only when every group around it
    /// is visible too.
    pub visible: bool,
    /// A guide the artist traced over. It never draws.
    pub reference: bool,
    /// The background layer, which is opaque and draws the transparent index.
    pub background: bool,
    /// How the layer's cels combine with what is below.
    pub blend: LayerBlendMode,
    /// Opacity, `0..=1`.
    pub opacity: f32,
    /// Free text the artist attached, empty when unset.
    pub data: String,
    /// The colour the editor shows the layer in, if set.
    pub colour: Option<Colour>,
}

/// What a [`SpriteLayer`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpriteLayerKind {
    /// Cels of pixels.
    #[default]
    Image,
    /// Other layers, which name this one as their parent.
    Group,
    /// Tiles from a tileset. Not decoded: its cels are absent and it is listed
    /// in [`LayeredSpriteData::unsupported`].
    Tilemap,
}

/// How a layer's pixels combine with what is below them, following the
/// usual compositing definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LayerBlendMode {
    /// The layer covers what is below.
    #[default]
    Normal,
    /// Colours multiply, darkening.
    Multiply,
    /// The inverse of multiplying the inverses, lightening.
    Screen,
    /// Multiply or screen, by the colour below.
    Overlay,
    /// The darker of the two.
    Darken,
    /// The lighter of the two.
    Lighten,
    /// Brightens what is below to reflect the layer.
    ColourDodge,
    /// Darkens what is below to reflect the layer.
    ColourBurn,
    /// Multiply or screen, by the layer's colour.
    HardLight,
    /// A softer hard light.
    SoftLight,
    /// The absolute difference.
    Difference,
    /// Like difference, lower in contrast.
    Exclusion,
    /// The layer's hue with the saturation and luminosity below.
    Hue,
    /// The layer's saturation with the hue and luminosity below.
    Saturation,
    /// The layer's hue and saturation with the luminosity below.
    Colour,
    /// The layer's luminosity with the hue and saturation below.
    Luminosity,
    /// The sum, clamped.
    Addition,
    /// What is below minus the layer, clamped.
    Subtract,
    /// What is below divided by the layer, clamped.
    Divide,
}

/// One frame of a [`LayeredSpriteData`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteLayerFrame {
    /// How long the frame shows, in milliseconds.
    pub duration_ms: u32,
    /// The frame's cels, at most one per layer.
    pub cels: Vec<SpriteCel>,
}

/// One layer's pixels in one frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteCel {
    /// Index into [`LayeredSpriteData::layers`].
    pub layer: u32,
    /// The image's top-left corner on the canvas, in pixels.
    pub position: [i32; 2],
    /// Opacity, `0..=1`, multiplied by the layer's.
    pub opacity: f32,
    /// Moves the cel among the frame's other cels: it draws at its layer's
    /// index plus this, the lower z-index first on a tie.
    pub z_index: i32,
    /// The pixels, straight-alpha RGBA labelled sRGB. A cel the source linked
    /// to an earlier frame is a copy.
    pub image: TextureData,
    /// Free text the artist attached, empty when unset.
    pub data: String,
    /// The colour the editor shows the cel in, if set.
    pub colour: Option<Colour>,
}

/// A decoded bitmap font: glyph images packed into one or more page images,
/// and the metrics to lay them out.
///
/// Everything is in pixels, X right and Y down: glyph rectangles in their page
/// image's frame, offsets and the baseline from the top of a line.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BitmapFontData {
    /// The font face the glyphs were rendered from.
    pub face: String,
    /// The size they were rendered at. Negative when the generator matched the
    /// character height rather than the cell height.
    pub size: i32,
    /// Rendered bold.
    pub bold: bool,
    /// Rendered italic.
    pub italic: bool,
    /// Distance in pixels from one line to the next.
    pub line_height: u32,
    /// Distance in pixels from the top of a line to the baseline.
    pub base: u32,
    /// Size of each page image in pixels.
    pub page_size: [u32; 2],
    /// Padding around each glyph: up, right, down, left.
    pub padding: [i32; 4],
    /// Spacing between glyphs in the pages, horizontal and vertical.
    pub spacing: [i32; 2],
    /// Outline thickness in pixels, `0` for none.
    pub outline: u32,
    /// Whether glyphs are packed into the separate colour channels of the
    /// pages; see [`BitmapGlyph::channels`].
    pub packed: bool,
    /// The page images, in page id order, resolved against the descriptor's
    /// directory when that is known.
    pub pages: Vec<TextureSource>,
    /// Glyphs by Unicode code point (or by the font's character id, for a
    /// font generated without Unicode).
    pub glyphs: std::collections::BTreeMap<u32, BitmapGlyph>,
    /// Extra horizontal advance in pixels between a pair of characters, by
    /// (first, second) code point.
    pub kernings: std::collections::BTreeMap<(u32, u32), i32>,
}

/// One glyph of a [`BitmapFontData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitmapGlyph {
    /// Index of the page image the glyph is in.
    pub page: u32,
    /// The glyph's pixels in its page image.
    pub rect: PixelRect,
    /// Where to draw `rect` relative to the pen position at the top of the
    /// line, in pixels.
    pub offset: [i32; 2],
    /// How far to move the pen after the glyph, in pixels.
    pub advance: i32,
    /// The page channels the glyph occupies. All four unless the font is
    /// packed.
    pub channels: GlyphChannels,
}

impl BitmapGlyph {
    /// The glyph's rectangle as texture coordinates `[u_min, v_min, u_max,
    /// v_max]` in a page of `page_size`, `v` `0` at the image's top row.
    pub fn uv(&self, page_size: [u32; 2]) -> [f32; 4] {
        self.rect.uv(page_size)
    }
}

/// Which colour channels of a page hold a glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GlyphChannels {
    /// The red channel.
    pub red: bool,
    /// The green channel.
    pub green: bool,
    /// The blue channel.
    pub blue: bool,
    /// The alpha channel.
    pub alpha: bool,
}

impl GlyphChannels {
    /// Every channel.
    pub const ALL: Self = Self {
        red: true,
        green: true,
        blue: true,
        alpha: true,
    };
}

// Naming-compat aliases the loaders use for these payload types. Each loader is
// behind its own cargo feature, so a given alias is unused when its loaders are
// disabled (the default build enables only some); the allow keeps that from
// warning without gating each alias on the union of loader features.
#[allow(dead_code)]
pub(crate) type IoMaterial = MaterialData;
#[allow(dead_code)]
pub(crate) type IoMesh = SceneMesh;
#[allow(dead_code)]
pub(crate) type IoPointCloud = PointSet;
#[allow(dead_code)]
pub(crate) type IoVolumeGeometry = VolumeGridGeometry;
#[allow(dead_code)]
pub(crate) type IoVolume = StructuredVolume;
#[allow(dead_code)]
pub(crate) type IoSparseVolume = SparseGrid;
#[allow(dead_code)]
pub(crate) type IoVolumeMesh = VolumeMeshData;
#[allow(dead_code)]
pub(crate) type IoDataSet = DecodedDataSet;
#[allow(dead_code)]
pub(crate) type IoScene = SceneData;
#[allow(dead_code)]
pub(crate) type IoNode = SceneNode;

#[cfg(test)]
mod tests {
    use super::*;

    /// The file formats' mapping: scale, then rotate about the UV origin, then
    /// offset.
    fn sample_origin(offset: [f32; 2], scale: [f32; 2], rotation: f32, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = rotation.sin_cos();
        let p = [uv[0] * scale[0], uv[1] * scale[1]];
        [
            offset[0] + cos * p[0] + sin * p[1],
            offset[1] - sin * p[0] + cos * p[1],
        ]
    }

    /// The renderer's mapping: scale and offset, then rotate about the texture
    /// centre.
    fn sample_centre(t: &UvTransform, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = t.rotation.sin_cos();
        let p = [
            uv[0] * t.scale[0] + t.offset[0] - 0.5,
            uv[1] * t.scale[1] + t.offset[1] - 0.5,
        ];
        [cos * p[0] - sin * p[1] + 0.5, sin * p[0] + cos * p[1] + 0.5]
    }

    fn red_to_blue() -> Vec<GradientStop> {
        vec![
            GradientStop::new(0.0, Colour::linear(1.0, 0.0, 0.0, 1.0)),
            GradientStop::new(0.4, Colour::linear(0.0, 1.0, 0.0, 1.0)),
            GradientStop::new(1.0, Colour::linear(0.0, 0.0, 1.0, 1.0)),
        ]
    }

    /// The colour SVG puts at `p`, evaluated straight from the source geometry.
    fn svg_colour(paint: &VectorPaint, p: glam::Vec2) -> [f32; 4] {
        match paint {
            VectorPaint::Solid(c) => c.to_linear_rgba(),
            VectorPaint::LinearGradient {
                start,
                end,
                stops,
                spread,
                transform,
            } => {
                let q = transform.inverse().transform_point2(p);
                let d = glam::Vec2::from(*end) - glam::Vec2::from(*start);
                let t = (q - glam::Vec2::from(*start)).dot(d) / d.length_squared();
                sample_spread(stops, *spread, t).to_linear_rgba()
            }
            VectorPaint::RadialGradient {
                centre,
                radius,
                stops,
                spread,
                transform,
                ..
            } => {
                let q = transform.inverse().transform_point2(p);
                let t = (q - glam::Vec2::from(*centre)).length() / radius;
                sample_spread(stops, *spread, t).to_linear_rgba()
            }
        }
    }

    /// Check the overlay fill against the source at a grid of points in `bounds`,
    /// or only those within `reach` of its centre.
    fn assert_fill_matches(paint: &VectorPaint, bounds: [f32; 4], reach: Option<f32>) {
        let fill = paint.to_overlay_fill(bounds);
        let centre = glam::vec2((bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5);
        let half = [(bounds[2] - bounds[0]) * 0.5, (bounds[3] - bounds[1]) * 0.5];
        for i in 0..=8 {
            for j in 0..=8 {
                let p = glam::vec2(
                    bounds[0] + (bounds[2] - bounds[0]) * i as f32 / 8.0,
                    bounds[1] + (bounds[3] - bounds[1]) * j as f32 / 8.0,
                );
                if reach.is_some_and(|r| (p - centre).length() > r) {
                    continue;
                }
                let want = svg_colour(paint, p);
                let got = fill.sample_in_box((p - centre).into(), half);
                for k in 0..4 {
                    assert!(
                        (want[k] - got[k]).abs() < 2e-3,
                        "at {p:?}: source {want:?}, overlay {got:?}, fill {fill:?}"
                    );
                }
            }
        }
    }

    fn linear(
        start: [f32; 2],
        end: [f32; 2],
        spread: SpreadMethod,
        transform: glam::Affine2,
    ) -> VectorPaint {
        VectorPaint::LinearGradient {
            start,
            end,
            stops: red_to_blue(),
            spread,
            transform,
        }
    }

    #[test]
    fn a_bounding_box_linear_gradient_maps_to_an_angle_with_its_stops_unchanged() {
        // A gradient in bounding-box units arrives with the box folded into its
        // transform.
        let bbox = glam::Affine2::from_scale_angle_translation(
            glam::vec2(40.0, 10.0),
            0.0,
            glam::vec2(5.0, 5.0),
        );
        let paint = linear([0.0, 0.0], [1.0, 0.0], SpreadMethod::Pad, bbox);
        let OverlayFill::LinearGradient { stops, angle } =
            paint.to_overlay_fill([5.0, 5.0, 45.0, 15.0])
        else {
            panic!("expected a linear gradient");
        };
        assert!(angle.abs() < 1e-6, "left to right: {angle}");
        let positions: Vec<f32> = stops.iter().map(|s| s.position).collect();
        assert_eq!(positions.len(), 3, "{positions:?}");
        for (got, want) in positions.iter().zip([0.0, 0.4, 1.0]) {
            assert!((got - want).abs() < 1e-5, "{positions:?}");
        }

        let down = linear([0.0, 0.0], [0.0, 1.0], SpreadMethod::Pad, bbox);
        let OverlayFill::LinearGradient { angle, .. } =
            down.to_overlay_fill([5.0, 5.0, 45.0, 15.0])
        else {
            panic!("expected a linear gradient");
        };
        assert!(
            (angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6,
            "top to bottom: {angle}"
        );
    }

    #[test]
    fn a_linear_gradient_matches_the_source_under_every_spread_and_transform() {
        let transforms = [
            glam::Affine2::IDENTITY,
            glam::Affine2::from_scale_angle_translation(
                glam::vec2(2.0, 0.5),
                0.6,
                glam::vec2(3.0, -2.0),
            ),
            // A skew: lines of equal position stop being perpendicular to the
            // mapped axis.
            glam::Affine2::from_cols_array(&[1.0, 0.0, 0.8, 1.0, 0.0, 0.0]),
        ];
        for transform in transforms {
            for spread in [
                SpreadMethod::Pad,
                SpreadMethod::Repeat,
                SpreadMethod::Reflect,
            ] {
                let paint = linear([2.0, 3.0], [9.0, 5.0], spread, transform);
                assert_fill_matches(&paint, [-10.0, -6.0, 30.0, 18.0], None);
            }
        }
    }

    #[test]
    fn a_centred_radial_gradient_matches_the_source() {
        let transform = glam::Affine2::from_scale_angle_translation(
            glam::Vec2::splat(2.0),
            0.3,
            glam::vec2(10.0, 10.0),
        );
        for spread in [
            SpreadMethod::Pad,
            SpreadMethod::Repeat,
            SpreadMethod::Reflect,
        ] {
            let paint = VectorPaint::RadialGradient {
                centre: [0.0, 0.0],
                radius: 3.0,
                focal: [0.0, 0.0],
                stops: red_to_blue(),
                spread,
                transform,
            };
            // Centred on the circle, so the overlay's centre is the source's.
            // The overlay ends at the larger half-extent; the corners past it
            // hold that ring's colour.
            assert_fill_matches(&paint, [0.0, 2.0, 20.0, 18.0], Some(10.0));
        }
    }

    #[test]
    fn a_gradient_with_no_length_paints_its_last_stop() {
        let paint = linear(
            [1.0, 1.0],
            [1.0, 1.0],
            SpreadMethod::Pad,
            glam::Affine2::IDENTITY,
        );
        assert_eq!(
            paint.to_overlay_fill([0.0, 0.0, 1.0, 1.0]),
            OverlayFill::Solid(Colour::linear(0.0, 0.0, 1.0, 1.0))
        );
    }

    #[test]
    fn a_two_entry_dash_array_maps_exactly_and_longer_ones_average() {
        assert_eq!(
            dash_pattern(&[4.0, 2.0], 1.0, 2.0),
            StrokePattern::Dashed {
                dash_length: 8.0,
                gap_length: 4.0,
                offset: 2.0
            }
        );
        assert_eq!(
            dash_pattern(&[4.0, 1.0, 2.0, 3.0], 0.0, 1.0),
            StrokePattern::Dashed {
                dash_length: 3.0,
                gap_length: 2.0,
                offset: 0.0
            }
        );
        assert_eq!(dash_pattern(&[0.0, 0.0], 0.0, 1.0), StrokePattern::Solid);
    }

    #[test]
    fn a_stroke_converts_with_its_lengths_scaled_and_its_style_kept() {
        let stroke = VectorStroke {
            paint: VectorPaint::LinearGradient {
                start: [0.0, 0.0],
                end: [1.0, 0.0],
                stops: GradientStop::pair(
                    Colour::linear(1.0, 0.0, 0.0, 1.0),
                    Colour::linear(0.0, 0.0, 1.0, 1.0),
                ),
                spread: SpreadMethod::Pad,
                transform: glam::Affine2::IDENTITY,
            },
            width: 2.0,
            cap: LineCap::Round,
            join: LineJoin::Bevel,
            mitre_limit: 7.0,
            dashes: Some(VectorDashes {
                array: vec![3.0, 1.0],
                offset: 0.5,
            }),
        };
        let overlay = stroke.to_overlay_stroke(3.0);
        assert_eq!(overlay.width, 6.0);
        assert_eq!(overlay.cap, LineCap::Round);
        assert_eq!(overlay.join, LineJoin::Bevel);
        assert_eq!(overlay.mitre_limit, 7.0);
        assert_eq!(
            overlay.pattern,
            StrokePattern::Dashed {
                dash_length: 9.0,
                gap_length: 3.0,
                offset: 1.5
            }
        );
        // One colour stands in for the gradient: its average.
        let mean = overlay.colour.to_linear_rgba();
        assert!(
            (mean[0] - 0.5).abs() < 1e-6 && (mean[2] - 0.5).abs() < 1e-6,
            "{mean:?}"
        );
    }

    #[test]
    fn origin_rotation_converts_to_the_same_texels() {
        let (offset, scale, rotation) = ([0.25, -0.1], [2.0, 3.0], 0.7);
        let converted = uv_transform_from_origin_rotation(offset, scale, rotation, 1);
        assert_eq!(converted.scale, scale);
        assert_eq!(converted.uv_set, 1);

        for uv in [[0.0, 0.0], [1.0, 1.0], [0.3, 0.8], [-0.4, 0.2]] {
            let want = sample_origin(offset, scale, rotation, uv);
            let got = sample_centre(&converted, uv);
            assert!(
                (want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5,
                "uv {uv:?}: {want:?} vs {got:?}"
            );
        }
    }

    #[test]
    fn origin_rotation_is_unchanged_without_rotation() {
        let converted = uv_transform_from_origin_rotation([0.25, -0.1], [2.0, 3.0], 0.0, 0);
        assert!((converted.offset[0] - 0.25).abs() < 1e-6);
        assert!((converted.offset[1] + 0.1).abs() < 1e-6);
        assert_eq!(converted.rotation, 0.0);
    }

    #[test]
    fn smooth_normals_are_unit_and_face_the_winding() {
        // A quad in the XY plane, wound anticlockwise seen from +Z, plus one
        // vertex no triangle uses.
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 5.0],
        ];
        let normals = smooth_normals(&positions, &[0, 1, 2, 0, 2, 3]);
        assert_eq!(normals.len(), positions.len());
        for n in &normals {
            assert!((glam::Vec3::from(*n).length() - 1.0).abs() < 1e-6);
            assert!((n[2] - 1.0).abs() < 1e-6, "{n:?}");
        }
    }

    #[test]
    fn a_point_set_colours_its_item_by_the_best_channel_it_has() {
        let mut points = PointSet {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0]],
            scalars: vec![1.0, 2.0],
            ..PointSet::default()
        };

        // No colours: the primary scalar drives the colourmap.
        let item = points.to_point_cloud_item();
        assert_eq!(item.positions, points.positions);
        assert!(matches!(
            &item.colour,
            ColourSource::Scalar { values, range: None, colourmap: None } if values == &points.scalars
        ));

        // Colours win over the scalar, and are decoded from sRGB.
        points.colors = vec![[0.5, 0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 0.5]];
        let item = points.to_point_cloud_item();
        let ColourSource::PerSample(colours) = &item.colour else {
            panic!("expected per-point colours, got {:?}", item.colour);
        };
        let grey = colours[0].to_linear_rgba();
        assert!((grey[0] - 0.214).abs() < 1e-3, "{grey:?}");
        assert_eq!(colours[1].alpha(), 0.5);

        // Neither: the item's own default.
        let bare = PointSet {
            positions: vec![[0.0; 3]],
            ..PointSet::default()
        };
        assert_eq!(
            bare.to_point_cloud_item().colour,
            PointCloudItem::default().colour
        );
    }

    #[test]
    fn a_material_converts_with_its_textures_resolved() {
        use viewport_lib_types::material::ShadingModel;

        let mut data = MaterialData {
            base_color: [0.5, 0.25, 0.125],
            metallic: 0.8,
            roughness: 0.3,
            emissive: [1.0, 0.5, 0.0],
            emissive_strength: 200.0,
            alpha_mode: AlphaMode::Mask(0.4),
            double_sided: true,
            normal_scale: 0.5,
            occlusion_strength: 0.75,
            base_color_texture: Some(TextureSource::File("albedo.png".into())),
            normal_map_texture: Some(TextureSource::File("normal.png".into())),
            ..MaterialData::default()
        };
        data.uv_transforms[TextureSlot::Albedo.index()] = Some(UvTransform::from_rotation(0.5));
        data.samplers[TextureSlot::Albedo.index()] = Some(TextureSampler {
            wrap_u: WrapMode::ClampToEdge,
            ..TextureSampler::default()
        });

        let mut seen = Vec::new();
        let material = data.to_material(|slot, _source| {
            seen.push(slot);
            // Leave the normal map unresolved, as an upload that failed would.
            (slot == TextureSlot::Albedo).then(|| TextureId::from_raw(7))
        });

        assert_eq!(seen, vec![TextureSlot::Albedo, TextureSlot::Normal]);
        assert_eq!(material.texture_id, Some(TextureId::from_raw(7)));
        assert_eq!(material.normal_map_id, None);
        assert_eq!(material.shading_model, ShadingModel::Pbr);
        assert_eq!(material.base_colour.to_linear_rgb(), [0.5, 0.25, 0.125]);
        assert_eq!(material.emissive.to_linear_rgb(), [1.0, 0.5, 0.0]);
        assert_eq!(material.emissive_strength, 200.0);
        assert_eq!((material.metallic, material.roughness), (0.8, 0.3));
        assert_eq!(material.alpha_mode, AlphaMode::Mask(0.4));
        assert!(material.is_two_sided());
        assert_eq!(material.normal_strength, 0.5);
        assert_eq!(material.ao_range, [0.25, 1.0]);
        assert_eq!(
            material.texture_transforms[TextureSlot::Albedo.index()],
            Some(UvTransform::from_rotation(0.5))
        );
        assert_eq!(
            material.sampler[TextureSlot::Albedo.index()].map(|s| s.wrap_u),
            Some(WrapMode::ClampToEdge)
        );
        assert_eq!(material.sampler[TextureSlot::Normal.index()], None);
    }

    #[test]
    fn a_sampler_converts_with_renderer_defaults_for_the_rest() {
        let sampler = TextureSampler {
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::MirrorRepeat,
            filter: TextureFilter::Nearest,
        };
        let key = SamplerKey::from(sampler);
        assert_eq!(key.wrap_u, WrapMode::ClampToEdge);
        assert_eq!(key.wrap_v, WrapMode::MirrorRepeat);
        assert_eq!(key.filter, TextureFilter::Nearest);
        let default = SamplerKey::default();
        assert_eq!(key.anisotropy, default.anisotropy);
        assert_eq!(key.lod_bias, default.lod_bias);
    }
}
