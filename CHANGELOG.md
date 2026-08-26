# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Morph-target (blend-shape) geometry on `SurfaceMesh`. A new `MorphTarget`
  neutral type carries per-vertex position (and optional normal / tangent)
  displacements from the base mesh, and `SurfaceMesh::morph_targets` holds them
  in authored order (empty when a format carries none, so every other loader is
  unaffected). The glTF loader reads a primitive's morph targets and reorients
  the displacements into Z-up alongside the base attributes. Targets are named
  by index (`target_<n>`) for now: glTF stores names in `mesh.extras`, which
  needs the `extras` feature and raw-JSON parsing, so named targets are a
  follow-up. FBX blend-shape deformers are not yet parsed.
- Morph-weight animation import. A new `MorphWeightClip` type carries keyframed
  blend-shape weights for one mesh's targets (row-major `[keyframe][target]`),
  and `SceneData::morph_animations` collects them. The glTF loader reads
  `MorphTargetWeights` channels (previously skipped): one clip per morph
  channel, named after its animation, with `CubicSpline` output collapsed to its
  value component. Kept apart from `AnimationClip` because the weights drive a
  mesh's morph target set, not a skeleton.

### Tests
- Cross-format skinning parity test (`tests/fbx_gltf_skinning_parity.rs`): the
  same rigged Fox asset loaded from FBX and from glTF must produce matching
  geometry, both in bind pose and when posed by sampling the "Walk" clip. This
  locks in the FBX axis / unit conversion applied uniformly to mesh transforms,
  skeleton inverse-binds and animation curves: under `AxisPolicy::HonourHeader`
  the Z-up fixture converts by identity and must equal the glTF output directly;
  under `AxisPolicy::ForceYUpRaw` both outputs differ by exactly the same +90
  degree X rotation. It is the regression guard for the sampled-pose bug where
  curves stayed in raw file space and the posed geometry came out ~90 degrees
  rotated against the bind pose. Ships the Fox fixtures (`fox.glb`,
  `fox_rigged.fbx`, CC-BY 4.0) under `tests/fixtures/`.

### Internal
- Reformatted `build_fbx_scene` (its body was indented one level too deep),
  `lightmap.rs` and the `probe_fbx_uv` example to satisfy `cargo fmt`. No
  behavioural change.

## [0.6.0] - 2026-08-12

### Added
- Biovision Hierarchy (`.bvh`) loader (new `loaders::bvh` module, behind a
  `bvh` feature): `scene_from_path` / `scene_from_bytes` / `scene_from_str`
  decode a BVH motion-capture file into an `IoScene` carrying one `Skeleton`
  and one `AnimationClip`. The `HIERARCHY` block builds the joint tree with
  inverse-bind matrices from the accumulated `OFFSET`s; the `MOTION` block
  becomes per-joint rotation tracks (Euler channels composed in their listed
  order) plus translation tracks wherever position channels are present. BVH is
  authored Y-up, so samples and bind matrices are reoriented into the library's
  Z-up convention exactly as the glTF loader does. End Sites are consumed and
  dropped (they fix a leaf tip, which animation never drives). This opens the
  large free mocap libraries (CMU, Bandai Namco, SFU, ACCAD), which ship as BVH.
  A `probe-bvh` example prints a file's skeleton and clip shape.

## [0.5.0] - 2026-08-12

### Changed
- FBX UV-channel selection now follows Unity's rule: the base map samples the
  first authored UV layer (UV0), and UV value ranges are never a selection
  signal. It replaces the previous heuristic (reject channels whose values
  exceed a magnitude limit, then prefer the largest-area channel). A layer is
  only skipped when it is not a texture coordinate at all — a packed per-vertex
  scalar that holds one axis constant across the mesh (a wind phase) — in which
  case the next layer is used. `VIEWPORT_FBX_UV_CHANNEL=<n>` still forces a
  channel for A/B testing.

### Fixed
- Foliage whose albedo UV0 bakes an integer per-card offset into V (a wind
  convention: card index in the integer part, the real texture coordinate in
  the fraction, recovered by repeat-wrap sampling) no longer renders with dark /
  black branches. The old picker rejected that channel on its large raw V range
  and fell through to a secondary lightmap unwrap; the UV0-first rule keeps it.

## [0.4.0] - 2026-08-11

### Fixed
- FBX texture coordinates are V-flipped on import to match the renderer's
  top-left texture origin (the same convention the glTF loader reads to).
  Without this, FBX UV atlases sampled upside down: a character's face texture
  landed on the wrong geometry at the wrong scale. Symmetric or tiling textures
  hid the fault; an asymmetric atlas (a face) exposed it.
- FBX skeleton joint order is now deterministic, so repeated loads of the same
  file produce the same joint indexing.
- Ambiguous FBX transparency extremes are treated as opaque, so a material
  whose opacity is a degenerate 0 or 1 no longer renders unexpectedly
  see-through.

## [0.3.0] - 2026-08-10

### Added
- Lightmap loader (new `lightmap` module): `LightmapData` and
  `LightmapEncoding`, with `radiance()`, `direction()` (present for directional
  lightmaps), `texel_count()`, and `is_well_formed()`. The glTF loader reads
  baked lightmap data where present.
- OpenEXR image loader (`loaders::exr::texture_from_path`), decoding EXR into
  `TextureData`.
- In-memory bytes loaders alongside the path-based ones: `texture_from_bytes`
  for PNG, JPEG, BMP, HDR, and EXR (`hdr::texture_from_bytes_with_limits` for
  the decode-limit override), and `fbx::scene_from_bytes` /
  `scene_from_bytes_with_options`. These load assets from a byte buffer with no
  filesystem path.
- glTF specular-glossiness materials (`KHR_materials_pbrSpecularGlossiness`) are
  converted to metallic-roughness at import.
- glTF `COLOR_0` vertex colours are decoded into `MeshData::colours`
  (`Option<Vec<[f32; 4]>>`).
- `MaterialData` reaches full glTF material parity: new `emissive`,
  `alpha_mode` (`AlphaMode`), `double_sided`, `metallic_roughness_texture`, and
  `emissive_texture` fields.
- `MaterialData::normal_scale` and `MaterialData::occlusion_strength`, matching
  glTF `normalScale` and `occlusionStrength`. The glTF loader now reads both
  (they were previously dropped); OBJ and FBX leave them at 1.0. Apply them to a
  viewport-lib `Material` as `normal_strength = normal_scale` and
  `ao_range = [1.0 - occlusion_strength, 1.0]`.
- FBX load options: `FbxLoadOptions` with `AxisPolicy` and `CumulativeOrder`
  overrides, applied through `fbx::scene_from_path_with_options`.
- The `pvd` loader is now public (`loaders::pvd`), and `PvdSeries::timesteps` is
  documented.

### Changed
- `MaterialData` is now `#[non_exhaustive]`. Build it via `Default` and
  struct-update syntax; later field additions will not be breaking. Code that
  reads `MaterialData` is unaffected.
- A skinned mesh reuses its existing skeleton for animation instead of appending
  a duplicate.

### Fixed
- FBX Y-up to Z-up axis conversion for Unity asset packs: the axis conversion
  now walks Unity `Null` wrappers correctly (signed cumulative-Y veto check) and
  applies to FBX animation tracks, not just mesh transforms.
- FBX multi-layer iteration and texture-coordinate UV selection, so meshes with
  several UV layers pick the right set.

## [0.2.0] - 2026-06-07

### Added
- FBX animation extraction: `AnimStack` / `AnimLayer` / `AnimCurveNode` curves
  are decoded into `AnimationClip`s with translation, rotation, and scale
  tracks. Rotation curves compose `PreRotation` and `PostRotation` so samples
  agree with the bind pose at `t = 0`.
- FBX rig synthesis from `LimbNode` and `Null` models, with reuse of an
  existing skin-derived skeleton when bone names match.
- HDR decode limit overrides: new `DecodeLimits` struct and
  `texture_from_path_with_limits` entry point. The default
  `texture_from_path` now decodes without a size cap so production HDRIs
  (8K and larger) load successfully; embedders ingesting untrusted input
  can opt back into the historical `image`-crate caps.

### Changed
- FBX mesh world transforms now walk the full parent chain through every
  node type (`Mesh`, `Null`, `LimbNode`, `Light`, `Camera`) rather than
  only `Mesh -> Mesh` links. Fixes detached "floating" sub-meshes in
  Unity-exported FBX files where placement transforms live on `Null`
  ancestors (LODGroup containers, `DummyHelper` scaffolding).

### Removed
- Internal references to the upstream DRAKE/drake-assets project in
  loader comments. Behaviour is unchanged.

## [0.1.0] - 2026-05

Initial release.

- PNG, JPEG, HDR texture loaders.
- STL, OBJ, PLY, MSH mesh loaders.
- glTF and FBX scene loaders with skinned mesh, skeleton, and animation
  extraction (glTF) and skin cluster extraction (FBX).
- CSV, NumPy, raw, VTK, VTU, VTP, XDMF, Exodus, EnSight, Tecplot, CGNS,
  NetCDF volume / dataset loaders.
- Z-up scene convention with one-shot reorientation at load time.
