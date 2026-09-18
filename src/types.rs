use std::collections::HashMap;
use std::path::PathBuf;

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

/// CPU-side RGBA8 image data.
#[derive(Clone, Debug)]
pub struct RasterImageData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixel data.
    pub rgba: Vec<u8>,
}

/// CPU-side RGBA32F image data.
#[derive(Clone, Debug)]
pub struct HdrImageData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA32F pixel data.
    pub rgba: Vec<f32>,
}

/// Where a scene material's texture content comes from.
#[derive(Clone, Debug)]
pub enum TextureSource {
    /// Resolve texture bytes from a file path at the consumer layer.
    File(PathBuf),
    /// Use already-decoded pixels directly.
    Decoded(RasterImageData),
}

/// The colour space a texture's pixels are in, so a consumer routes the GPU
/// upload correctly: sRGB colour textures must decode to linear on sample;
/// linear data textures (normal, metallic-roughness, occlusion) must not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourSpace {
    /// sRGB-encoded colour (base colour, emissive). Upload so the sampler
    /// decodes to linear.
    Srgb,
    /// Linear data. Upload without any sRGB decode.
    Linear,
}

/// How many texture slots a material has, so the per-slot arrays on
/// [`MaterialData`] can be indexed by [`MaterialTextureSlot::index`].
pub const MATERIAL_TEXTURE_SLOTS: usize = 5;

/// Which material texture slot a [`TextureSource`] fills.
///
/// The discriminants match viewport-lib's `TextureSlot`, so
/// [`index`](Self::index) indexes the renderer's per-slot arrays as well as the
/// ones here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialTextureSlot {
    /// Base colour (albedo). sRGB.
    BaseColour = 0,
    /// Tangent-space normal map. Linear data.
    Normal = 1,
    /// Ambient occlusion. Linear data.
    Occlusion = 2,
    /// Combined metallic-roughness (ORM). Linear data.
    MetallicRoughness = 3,
    /// Emissive. sRGB.
    Emissive = 4,
}

impl MaterialTextureSlot {
    /// Every slot, in discriminant order.
    pub const ALL: [MaterialTextureSlot; MATERIAL_TEXTURE_SLOTS] = [
        MaterialTextureSlot::BaseColour,
        MaterialTextureSlot::Normal,
        MaterialTextureSlot::Occlusion,
        MaterialTextureSlot::MetallicRoughness,
        MaterialTextureSlot::Emissive,
    ];

    /// This slot's position in the per-slot arrays on [`MaterialData`].
    pub fn index(self) -> usize {
        self as usize
    }

    /// The colour space this slot's pixels are in.
    pub fn colour_space(self) -> ColourSpace {
        match self {
            MaterialTextureSlot::BaseColour | MaterialTextureSlot::Emissive => ColourSpace::Srgb,
            MaterialTextureSlot::MetallicRoughness
            | MaterialTextureSlot::Normal
            | MaterialTextureSlot::Occlusion => ColourSpace::Linear,
        }
    }
}

/// How one texture slot samples its UVs: tiling, offset, rotation, and which UV
/// set to read.
///
/// This carries glTF `KHR_texture_transform` (and the per-map `_ST` a Unity or
/// Unreal export writes) and maps onto a viewport-lib `UvTransform` field for
/// field. The mapping is
///
/// ```text
/// uv' = offset + rotate(rotation) * (uv * scale)
/// ```
///
/// with the rotation about the UV origin and positive `rotation` turning the UV
/// axes anticlockwise, which is what the glTF extension defines. A consumer that
/// instead rotates about the texture centre after applying scale and offset
/// (viewport-lib does) wants [`as_centre_rotation`](Self::as_centre_rotation),
/// which re-expresses the same mapping in that form. The two agree exactly when
/// `rotation` is 0.0, which covers every pure tile-and-offset material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    /// UV offset, added after scale and rotation.
    pub offset: [f32; 2],
    /// UV scale (tiling).
    pub scale: [f32; 2],
    /// Rotation in radians about the UV origin, anticlockwise.
    pub rotation: f32,
    /// Which UV set to sample, matching glTF `texCoord`: 0 is
    /// [`SurfaceMesh::uvs`] and 1 is [`SurfaceMesh::uvs1`]. What a consumer does
    /// with a slot naming a set the mesh does not carry is its own business, and
    /// is not always a fallback to UV0: viewport-lib samples `(0, 0)` there.
    /// Sets above 1 are not decoded, so the value is 0 or 1 today.
    pub uv_set: u32,
}

impl UvTransform {
    /// Passes UVs through unchanged, reading UV0.
    pub const IDENTITY: UvTransform = UvTransform {
        offset: [0.0, 0.0],
        scale: [1.0, 1.0],
        rotation: 0.0,
        uv_set: 0,
    };

    /// Whether this is [`IDENTITY`](Self::IDENTITY): the texture samples UV0
    /// untransformed, so a consumer can skip it.
    pub fn is_identity(&self) -> bool {
        *self == UvTransform::IDENTITY
    }

    /// The same mapping expressed for a consumer that rotates about the texture
    /// centre `(0.5, 0.5)` after scale and offset, and treats positive rotation
    /// as turning the sampled image anticlockwise:
    ///
    /// ```text
    /// uv' = rotate(rotation) * (uv * scale + offset - 0.5) + 0.5
    /// ```
    ///
    /// `scale` and `uv_set` are unchanged; the rotation flips sign and the
    /// offset moves with it. Identical to `self` when `rotation` is 0.0.
    ///
    /// **It is its own inverse**, so this one call converts in both directions:
    /// the rotation negates, and the offset rotates back through the opposite
    /// angle about the same centre. Call it on a centre-rotation transform to
    /// get one in this type's convention, which is what an exporter or anything
    /// else *writing* a [`MaterialData`] needs;
    /// [`from_centre_rotation`](Self::from_centre_rotation) is the same
    /// operation under the name that reads correctly at a write site.
    pub fn as_centre_rotation(&self) -> UvTransform {
        let (sin, cos) = (-self.rotation).sin_cos();
        let centred = [self.offset[0] - 0.5, self.offset[1] - 0.5];
        UvTransform {
            // Undo the centre rotation the consumer will apply to the offset, so
            // the composed mapping lands where the source asked.
            offset: [
                cos * centred[0] + sin * centred[1] + 0.5,
                -sin * centred[0] + cos * centred[1] + 0.5,
            ],
            scale: self.scale,
            rotation: -self.rotation,
            uv_set: self.uv_set,
        }
    }

    /// A transform authored for a consumer that rotates about the texture centre
    /// after scale and offset, brought into this type's convention (rotation
    /// about the UV origin, as the file formats define it).
    ///
    /// The same operation as [`as_centre_rotation`](Self::as_centre_rotation),
    /// which is its own inverse; this name exists so a write site does not read
    /// as though it converts the wrong way.
    pub fn from_centre_rotation(transform: &UvTransform) -> UvTransform {
        transform.as_centre_rotation()
    }
}

impl Default for UvTransform {
    fn default() -> Self {
        UvTransform::IDENTITY
    }
}

/// How a sampler treats UVs outside `[0, 1]`, matching the glTF sampler wrap
/// modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WrapMode {
    /// Tile the texture (glTF `REPEAT`). The default.
    #[default]
    Repeat,
    /// Clamp to the nearest edge texel (glTF `CLAMP_TO_EDGE`).
    ClampToEdge,
    /// Mirror on each repeat (glTF `MIRRORED_REPEAT`).
    MirrorRepeat,
}

/// How a sampler blends texels, matching the glTF sampler filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextureFilter {
    /// Nearest-neighbour: crisp and blocky under magnification.
    Nearest,
    /// Linear. The default.
    #[default]
    Linear,
}

/// Sampler state for one texture slot: wrap modes and filtering, as the source
/// file specifies them.
///
/// Maps onto a viewport-lib `SamplerKey`, whose remaining fields (anisotropy,
/// LOD bias) are renderer policy rather than anything a scene file carries, so
/// they keep their defaults.
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

/// How a material's alpha channel is interpreted, matching glTF `alphaMode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlphaMode {
    /// Fully opaque; the alpha channel is ignored.
    Opaque,
    /// Alpha-tested: fragments with alpha below the cutoff are discarded.
    Mask(f32),
    /// Alpha-blended over the background.
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
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
    /// Base colour texture, if present. sRGB colour ([`ColourSpace::Srgb`]):
    /// upload it so the sampler decodes to linear (build the payload with the
    /// renderer's `TextureData::srgb`).
    pub base_color_texture: Option<TextureSource>,
    /// Combined metallic-roughness (ORM) texture, if present. Matches the glTF
    /// `metallicRoughnessTexture`: G channel is roughness, B channel is metallic.
    /// Linear data ([`ColourSpace::Linear`]): upload without sRGB decode (build
    /// the payload with the renderer's `TextureData::linear`).
    pub metallic_roughness_texture: Option<TextureSource>,
    /// Normal map, if present. Linear data ([`ColourSpace::Linear`]): upload
    /// without sRGB decode (`TextureData::normal_map`, which also binds it into
    /// the normal slot rather than the albedo one).
    pub normal_map_texture: Option<TextureSource>,
    /// Scales the tangent-space normal read from `normal_map_texture`, matching
    /// glTF `normalScale`. 1.0 leaves the map at authored strength. Files with no
    /// equivalent (OBJ, FBX) leave this at 1.0.
    pub normal_scale: f32,
    /// Ambient-occlusion texture, if present. Linear data
    /// ([`ColourSpace::Linear`]): upload without sRGB decode (`TextureData::linear`).
    pub ao_texture: Option<TextureSource>,
    /// Strength of the ambient-occlusion contribution, matching glTF
    /// `occlusionStrength`. 1.0 applies the map fully, 0.0 disables it. Files with
    /// no equivalent leave this at 1.0.
    pub occlusion_strength: f32,
    /// Emissive texture, multiplied by `emissive`. Matches glTF `emissiveTexture`.
    /// sRGB colour ([`ColourSpace::Srgb`]): upload so the sampler decodes to
    /// linear (`TextureData::srgb`).
    pub emissive_texture: Option<TextureSource>,
    /// How each slot samples its UVs, indexed by
    /// [`MaterialTextureSlot::index`]. `None` where the source asks for plain
    /// UV0 sampling, which is most materials. Read one with
    /// [`uv_transform`](Self::uv_transform); maps onto viewport-lib
    /// `Material::texture_transforms`.
    pub uv_transforms: [Option<UvTransform>; MATERIAL_TEXTURE_SLOTS],
    /// Sampler state per slot, indexed by [`MaterialTextureSlot::index`].
    /// `None` where the source specifies none, or specifies exactly the neutral
    /// default (repeat on both axes, linear filtering), so a consumer reads
    /// `None` as "your default sampler". Read one with
    /// [`sampler`](Self::sampler); maps onto viewport-lib `Material::sampler`.
    pub samplers: [Option<TextureSampler>; MATERIAL_TEXTURE_SLOTS],
}

impl MaterialData {
    /// Every present texture with its slot, so a consumer can route each upload
    /// by colour space: `slot.colour_space()` is [`ColourSpace::Srgb`] for base
    /// colour and emissive (upload so the sampler decodes to linear) and
    /// [`ColourSpace::Linear`] for metallic-roughness, normal, and occlusion
    /// (upload without decode). Store each returned id back in the matching
    /// material texture field.
    ///
    /// Against viewport-lib that is a `TextureData` per texture:
    ///
    /// ```ignore
    /// for (slot, source) in material.textures() {
    ///     let (w, h, pixels) = decode(source);
    ///     let data = match slot.colour_space() {
    ///         ColourSpace::Srgb => TextureData::srgb(w, h, pixels),
    ///         ColourSpace::Linear if slot == MaterialTextureSlot::Normal => {
    ///             TextureData::normal_map(w, h, pixels)
    ///         }
    ///         ColourSpace::Linear => TextureData::linear(w, h, pixels),
    ///     };
    ///     let id = res.upload_texture(&device, &queue, data)?;
    /// }
    /// ```
    pub fn textures(&self) -> Vec<(MaterialTextureSlot, &TextureSource)> {
        [
            (MaterialTextureSlot::BaseColour, &self.base_color_texture),
            (
                MaterialTextureSlot::MetallicRoughness,
                &self.metallic_roughness_texture,
            ),
            (MaterialTextureSlot::Normal, &self.normal_map_texture),
            (MaterialTextureSlot::Occlusion, &self.ao_texture),
            (MaterialTextureSlot::Emissive, &self.emissive_texture),
        ]
        .into_iter()
        .filter_map(|(slot, src)| src.as_ref().map(|s| (slot, s)))
        .collect()
    }

    /// How one slot samples its UVs, or `None` when it samples UV0
    /// untransformed.
    pub fn uv_transform(&self, slot: MaterialTextureSlot) -> Option<UvTransform> {
        self.uv_transforms[slot.index()]
    }

    /// Sampler state for one slot, or `None` when the source specifies nothing
    /// beyond the neutral default.
    pub fn sampler(&self, slot: MaterialTextureSlot) -> Option<TextureSampler> {
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

/// Domain on which an attribute is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributeDomain {
    /// One value per point or vertex.
    Point,
    /// One value per cell.
    Cell,
    /// One value per face.
    Face,
    /// One value per directed edge.
    Edge,
    /// One value per halfedge/corner-edge.
    Halfedge,
    /// One value per face corner.
    Corner,
}

/// Attribute payload values.
#[derive(Clone, Debug)]
pub enum AttributeValues {
    /// Scalar values.
    Scalars(Vec<f32>),
    /// RGBA colours.
    Colors(Vec<[f32; 4]>),
    /// 3D vectors.
    Vectors(Vec<[f32; 3]>),
}

/// Named attribute data on a mesh-like topology.
#[derive(Clone, Debug)]
pub struct AttributeData {
    /// Topological domain the values are defined on.
    pub domain: AttributeDomain,
    /// Attribute payload.
    pub values: AttributeValues,
}

impl AttributeData {
    /// Construct scalar attribute data.
    pub fn scalars(domain: AttributeDomain, values: Vec<f32>) -> Self {
        Self {
            domain,
            values: AttributeValues::Scalars(values),
        }
    }

    /// Construct colour attribute data.
    pub fn colors(domain: AttributeDomain, values: Vec<[f32; 4]>) -> Self {
        Self {
            domain,
            values: AttributeValues::Colors(values),
        }
    }

    /// Construct vector attribute data.
    pub fn vectors(domain: AttributeDomain, values: Vec<[f32; 3]>) -> Self {
        Self {
            domain,
            values: AttributeValues::Vectors(values),
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
/// mesh, parallel to [`SurfaceMesh::positions`]. A deformer sums the base
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

/// Source-agnostic surface mesh data.
#[derive(Clone, Debug, Default)]
pub struct SurfaceMesh {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Triangle index list.
    pub indices: Vec<u32>,
    /// Optional per-vertex UV coordinates. The set a material samples by
    /// default, and what [`UvTransform::uv_set`] 0 means.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Optional second per-vertex UV set, from glTF `TEXCOORD_1` or an FBX
    /// mesh's next authored UV layer. `None` when the source carries one set,
    /// which is most meshes. What [`UvTransform::uv_set`] 1 means, and what a
    /// material slot transformed onto that set reads. Maps to
    /// `viewport_lib::MeshData::uvs1`.
    ///
    /// This is the content channel, not a lightmap bake: a baked lightmap set
    /// is sidecar data with its own owner.
    pub uvs1: Option<Vec<[f32; 2]>>,
    /// Optional per-vertex tangents.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Optional per-vertex RGBA colours (linear 0..1), from a format's
    /// per-vertex colour channel such as glTF `COLOR_0` or PLY vertex colours.
    /// Maps to `viewport_lib::MeshData::vertex_colours`.
    pub colours: Option<Vec<[f32; 4]>>,
    /// Named attributes on this mesh.
    pub attributes: HashMap<String, AttributeData>,
    /// Optional skinning weights.
    pub skin_weights: Option<SkinWeights>,
    /// Blend shapes for this mesh, in authored order. Empty when the source
    /// carries none. A downstream deformer indexes these by position, so the
    /// order is the order a weight palette is expected to follow.
    pub morph_targets: Vec<MorphTarget>,
}

/// A mesh entry inside a loaded scene.
#[derive(Clone, Debug)]
pub struct SceneMesh {
    /// Mesh name.
    pub name: String,
    /// Mesh geometry and attributes.
    pub mesh: SurfaceMesh,
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
            mesh: SurfaceMesh::default(),
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
    pub surface_mesh: Option<SurfaceMesh>,
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
    pub fn as_surface_mesh(&self) -> Option<&SurfaceMesh> {
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

/// One segment of a [`SubPath`]. The start point is implicit: it is the
/// subpath's `start` for the first segment, and the previous segment's end
/// point after that. Coordinates are in the source drawing's user units, in the
/// 2D canvas frame: X right, Y down from the top-left origin. The crate's Z-up
/// convention covers 3D scene geometry and does not apply here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathSegment {
    /// Straight line to `to`.
    Line {
        /// End point.
        to: [f32; 2],
    },
    /// Quadratic Bezier through control point `ctrl` to `to`.
    Quad {
        /// Control point.
        ctrl: [f32; 2],
        /// End point.
        to: [f32; 2],
    },
    /// Cubic Bezier through control points `ctrl1`, `ctrl2` to `to`.
    Cubic {
        /// First control point.
        ctrl1: [f32; 2],
        /// Second control point.
        ctrl2: [f32; 2],
        /// End point.
        to: [f32; 2],
    },
}

/// A single contour: a start point, a run of segments, and whether it closes
/// back to the start. Multiple subpaths in one [`VectorShape`] combine under
/// the shape's [`FillRule`], so the inner loop of a letter "O" is a hole.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubPath {
    /// Start point in the drawing's user units.
    pub start: [f32; 2],
    /// Segments in order from `start`.
    pub segments: Vec<PathSegment>,
    /// Whether the last point connects back to `start`.
    pub closed: bool,
}

/// How overlapping and nested subpaths combine into filled area. Matches the
/// SVG fill rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FillRule {
    /// Inside where the signed crossing count is non-zero. The SVG default.
    #[default]
    NonZero,
    /// Inside where the crossing count is odd.
    EvenOdd,
}

/// An outline stroke on a [`VectorShape`]: the contours drawn at `width` in the
/// drawing's user units, in `colour`. Field-for-field the shape of
/// `viewport_lib_ui::primitives::draw::VectorStroke` (the one on
/// `DrawCommand::Vector`, which names its paint `colour`; the document-level
/// `WidgetKind::Vector` stroke names it `paint` instead), so mapping across is a
/// copy.
///
/// Open subpaths stroke as open lines and closed ones stroke their whole
/// boundary, which is what makes stroke-drawn art (most icon and cursor sets)
/// readable: those contours are lines, not the outlines of filled blobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorStroke {
    /// Stroke colour as linear RGBA in `[0, 1]`, alpha carrying the stroke
    /// opacity multiplied by any enclosing group opacity.
    pub colour: [f32; 4],
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
    pub subpaths: Vec<SubPath>,
    /// How the subpaths combine into filled area.
    pub fill_rule: FillRule,
    /// Resolved fill colour as linear RGBA in `[0, 1]`, alpha carrying the fill
    /// opacity multiplied by any enclosing group opacity. `None` when the
    /// source shape has no fill, or a gradient or pattern paint the loader does
    /// not resolve to a single colour.
    pub fill: Option<[f32; 4]>,
    /// Resolved outline stroke. `None` when the source shape has no stroke, or
    /// a gradient or pattern stroke paint the loader does not resolve to a
    /// single colour.
    ///
    /// Stroke cap, join, dash, and miter limit are not carried: the renderer
    /// draws a uniform outline of the contours and has no vocabulary for them.
    pub stroke: Option<VectorStroke>,
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
pub(crate) type TextureData = RasterImageData;
#[allow(dead_code)]
pub(crate) type HdrTextureData = HdrImageData;
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

    /// The glTF mapping: scale, then rotate about the UV origin, then offset.
    fn sample_gltf(t: &UvTransform, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = t.rotation.sin_cos();
        let p = [uv[0] * t.scale[0], uv[1] * t.scale[1]];
        [
            t.offset[0] + cos * p[0] + sin * p[1],
            t.offset[1] - sin * p[0] + cos * p[1],
        ]
    }

    /// The consumer-side mapping: scale and offset, then rotate about the
    /// texture centre.
    fn sample_centre(t: &UvTransform, uv: [f32; 2]) -> [f32; 2] {
        let (sin, cos) = t.rotation.sin_cos();
        let p = [
            uv[0] * t.scale[0] + t.offset[0] - 0.5,
            uv[1] * t.scale[1] + t.offset[1] - 0.5,
        ];
        [cos * p[0] - sin * p[1] + 0.5, sin * p[0] + cos * p[1] + 0.5]
    }

    #[test]
    fn centre_rotation_form_samples_the_same_texels() {
        let source = UvTransform {
            offset: [0.25, -0.1],
            scale: [2.0, 3.0],
            rotation: 0.7,
            uv_set: 1,
        };
        let converted = source.as_centre_rotation();
        assert_eq!(converted.scale, source.scale);
        assert_eq!(converted.uv_set, source.uv_set);

        for uv in [[0.0, 0.0], [1.0, 1.0], [0.3, 0.8], [-0.4, 0.2]] {
            let want = sample_gltf(&source, uv);
            let got = sample_centre(&converted, uv);
            assert!(
                (want[0] - got[0]).abs() < 1e-5 && (want[1] - got[1]).abs() < 1e-5,
                "uv {uv:?}: {want:?} vs {got:?}"
            );
        }
    }

    /// The conversion is an involution, and consumers writing a transform rely
    /// on that: `from_centre_rotation` is the same call. Nothing else here would
    /// fail if a change to the rotation convention stopped the round trip
    /// closing, since the other tests only ever convert one way.
    #[test]
    fn centre_rotation_form_is_its_own_inverse() {
        for rotation in [0.0, 0.7, -1.9, std::f32::consts::FRAC_PI_2, 3.0] {
            for offset in [[0.0, 0.0], [0.25, -0.1], [1.3, 0.6]] {
                let source = UvTransform {
                    offset,
                    scale: [2.0, 0.5],
                    rotation,
                    uv_set: 1,
                };
                let round_trip = UvTransform::from_centre_rotation(&source.as_centre_rotation());
                assert!(
                    (round_trip.offset[0] - source.offset[0]).abs() < 1e-5
                        && (round_trip.offset[1] - source.offset[1]).abs() < 1e-5
                        && (round_trip.rotation - source.rotation).abs() < 1e-5,
                    "rotation {rotation}, offset {offset:?}: got {round_trip:?}"
                );
                assert_eq!(round_trip.scale, source.scale);
                assert_eq!(round_trip.uv_set, source.uv_set);
            }
        }
    }

    #[test]
    fn centre_rotation_form_is_a_no_op_without_rotation() {
        let source = UvTransform {
            offset: [0.25, -0.1],
            scale: [2.0, 3.0],
            rotation: 0.0,
            uv_set: 0,
        };
        let converted = source.as_centre_rotation();
        assert!((converted.offset[0] - source.offset[0]).abs() < 1e-6);
        assert!((converted.offset[1] - source.offset[1]).abs() < 1e-6);
        assert_eq!(converted.rotation, 0.0);
    }

    #[test]
    fn slot_indices_are_stable() {
        assert_eq!(MaterialTextureSlot::BaseColour.index(), 0);
        assert_eq!(MaterialTextureSlot::Normal.index(), 1);
        assert_eq!(MaterialTextureSlot::Occlusion.index(), 2);
        assert_eq!(MaterialTextureSlot::MetallicRoughness.index(), 3);
        assert_eq!(MaterialTextureSlot::Emissive.index(), 4);
        for (i, slot) in MaterialTextureSlot::ALL.iter().enumerate() {
            assert_eq!(slot.index(), i);
        }
    }
}
