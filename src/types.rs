use std::collections::HashMap;
use std::path::PathBuf;

use viewport_lib_types::colour::Colour;
use viewport_lib_types::data::mesh::MeshData;
use viewport_lib_types::data::texture::TextureData;
use viewport_lib_types::material::{
    AlphaMode, MATERIAL_TEXTURE_SLOTS, SamplerKey, TextureFilter, TextureSlot, UvTransform,
    WrapMode,
};
use viewport_lib_types::overlay::vector::{FillRule, SubPath};

/// Sentinel used to pad unused slots in fixed-width volumetric cell connectivity.
pub const CELL_SENTINEL: u32 = u32::MAX;

/// Maximum number of joints permitted in a single skeleton.
///
/// Skinning palettes in real-time renderers are fixed-size; 256 matches the
/// conventional palette limit used across the viewport-lib stack. Loaders
/// reject any source skeleton that exceeds this bound rather than silently
/// truncating, and per-vertex `[u8; 4]` joint indices fit exactly into this
/// range.
pub const MAX_JOINTS: usize = 256;

/// Where a scene material's texture content comes from.
#[derive(Clone, Debug)]
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
/// `normal_scale` and `occlusion_strength` map directly onto a viewport-lib
/// `Material`: set `normal_strength = normal_scale`, and set
/// `ao_range = [1.0 - occlusion_strength, 1.0]` (which reproduces glTF
/// `occlusionStrength`, since `mix(1.0, sample, s) == mix(1 - s, 1, sample)`).
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
    /// Cubic-spline interpolation as defined by glTF. Not yet consumed by
    /// downstream players; preserved on import for round-trip fidelity.
    CubicSpline,
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
    /// Interpolation between keyframes. `CubicSpline` is collapsed to its value
    /// component on read, so this is `Step` or `Linear` in practice.
    pub interpolation: AnimationInterpolation,
    /// Keyframe times in seconds, strictly increasing.
    pub times: Vec<f32>,
    /// Weights, row-major `[keyframe][target]`: length is
    /// `times.len() * target_count`.
    pub weights: Vec<f32>,
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
    /// a scientific format stores per cell are `Cell`, and per halfedge
    /// `Halfedge`, one value per source cell or halfedge: the same count the
    /// renderer expects when the surface is all triangles, but not when the
    /// source cells were quads or polygons split into triangles.
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
    /// Optional per-point colours.
    pub colors: Vec<[f32; 4]>,
    /// Optional primary scalar values.
    pub scalars: Vec<f32>,
    /// Named scalar attributes carried with the points.
    pub scalar_attributes: HashMap<String, Vec<f32>>,
}

/// Spherical-harmonic degree for gaussian splat colour data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShDegree {
    /// Degree-0 SH; three coefficients `[r, g, b]` per splat.
    Zero,
    /// Degree-1 SH.
    One,
    /// Degree-3 SH.
    Three,
}

impl ShDegree {
    /// Number of RGB coefficients per splat.
    pub fn coeff_count(self) -> usize {
        match self {
            Self::Zero => 3,
            Self::One => 12,
            Self::Three => 48,
        }
    }
}

/// Source-agnostic gaussian splat input data.
#[derive(Clone, Debug)]
pub struct GaussianSplatSet {
    /// Object-space center positions, one per splat.
    pub positions: Vec<[f32; 3]>,
    /// Scale per splat.
    pub scales: Vec<[f32; 3]>,
    /// Rotation per splat as `[x, y, z, w]`.
    pub rotations: Vec<[f32; 4]>,
    /// Opacity per splat.
    pub opacities: Vec<f32>,
    /// Packed SH coefficients.
    pub sh_coefficients: Vec<f32>,
    /// SH degree for this set.
    pub sh_degree: ShDegree,
}

impl Default for GaussianSplatSet {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: ShDegree::Zero,
        }
    }
}

impl PointSet {
    /// Create a degree-0 gaussian splat set from points.
    pub fn to_gaussian_splats(
        &self,
        default_scale: [f32; 3],
        default_opacity: f32,
    ) -> GaussianSplatSet {
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

        GaussianSplatSet {
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

/// Unstructured volumetric mesh data.
#[derive(Clone, Debug, Default)]
pub struct VolumeMesh {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Cell connectivity; unused slots are padded with [`CELL_SENTINEL`].
    pub cells: Vec<[u32; 8]>,
    /// Named per-cell scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
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
    pub volume_mesh: Option<Box<VolumeMesh>>,
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
    pub fn as_volume_mesh(&self) -> Option<&VolumeMesh> {
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
}

/// An outline stroke on a [`VectorShape`]: the contours drawn at `width` in the
/// drawing's user units, in `colour`.
///
/// Open subpaths stroke as open lines and closed ones stroke their whole
/// boundary, which is what makes stroke-drawn art (most icon and cursor sets)
/// readable: those contours are lines, not the outlines of filled blobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorStroke {
    /// Stroke colour, alpha carrying the stroke opacity multiplied by any
    /// enclosing group opacity.
    pub colour: Colour,
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
}

/// One shape from a vector drawing: its contours, fill rule, and the resolved
/// fill and stroke paint.
///
/// `fill` and `stroke` are independent and either can be absent. A shape with a
/// stroke and no fill is stroke-only: draw the contours as lines, not as filled
/// area. The one case the two fields cannot tell apart is a shape whose only
/// paint is a gradient or pattern the loader does not resolve: it arrives with
/// both `None`, indistinguishable from a shape with no paint at all. The
/// geometry is emitted either way.
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
    /// Resolved fill colour, alpha carrying the fill opacity multiplied by any
    /// enclosing group opacity. `None` when the
    /// source shape has no fill, or a gradient or pattern paint the loader does
    /// not resolve to a single colour.
    pub fill: Option<Colour>,
    /// Resolved outline stroke. `None` when the source shape has no stroke, or
    /// a gradient or pattern stroke paint the loader does not resolve to a
    /// single colour.
    ///
    /// Stroke cap, join, dash, and miter limit are not carried: the renderer
    /// draws a uniform outline of the contours and has no vocabulary for them.
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
pub(crate) type IoVolumeMesh = VolumeMesh;
#[allow(dead_code)]
pub(crate) type IoDataSet = DecodedDataSet;
#[allow(dead_code)]
pub(crate) type IoScene = SceneData;

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
