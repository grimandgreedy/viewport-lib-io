# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SVG vector paths** - `loaders::svg::vector_from_path` decodes an SVG into neutral vector paths instead of pixels, alongside the unchanged `texture_from_path`. New types `VectorArt`, `VectorShape`, `SubPath`, `PathSegment`, `FillRule`; feed them into `viewport_lib::OverlayShape::Vector`. Curves stay unflattened, each path's transform is baked in, and coordinates are the SVG canvas frame (X right, Y down), not the Z-up scene convention. Gradient and pattern fills leave the colour unset, and `<text>` and image nodes are skipped.
- **Stroke paint and width on vector art** - `VectorShape::stroke` carries a `VectorStroke { colour, width }`, so stroke-drawn icon and cursor sets import with their paint instead of as unpainted contours. Width scales with the baked transform (averaged under a non-uniform scale) and stroke opacity folds in group opacity. Cap, join, dash, and miter limit are not carried.
- **`loaders::svg::vector_from_bytes`** - the in-memory sibling of `vector_from_path`, for SVG art served from a bundle or fetched over the network. Output is identical to the path form. There is no base directory to resolve external references against, which costs nothing here: the nodes that reference external files (images, text) are the ones this loader skips.
- **Morph targets on `SurfaceMesh`** - `MorphTarget` carries per-vertex position, normal, and tangent displacements, and `SurfaceMesh::morph_targets` holds them in authored order (empty for formats that have none). The glTF loader reads them and reorients into Z-up with the base attributes. Targets are named by index for now: glTF keeps names in `mesh.extras`.
- **FBX blend shapes** - the FBX loader reads `Deformer`(BlendShape) -> `BlendShapeChannel` -> `Shape` chains, expands each shape's sparse deltas through the control-point map skinning uses, and splits them across material sub-meshes.
- **Legacy FBX blend shapes** - pre-7.5 exports nest `Shape` nodes inside `Geometry` with no deformer objects, which `fbxcel-dom` does not surface. When the modern walk finds nothing, the loader falls back to the raw geometry tree. Verified against a character carrying the full 52-name ARKit set.
- **Morph-weight animation** - `MorphWeightClip` carries keyframed blend-shape weights for one mesh (row-major `[keyframe][target]`) in `SceneData::morph_animations`. glTF reads `MorphTargetWeights` channels; FBX reads each channel's `DeformPercent` curve, merges a mesh's independently-keyed curves onto a union timeline, and converts percentages to `0..1`. The two formats are now at parity.

### Fixed
- **The scientific loader features did not build on their own** - selecting any one of `vtk`, `xdmf`, `exodus`, `ensight`, `tecplot`, `cgns`, or `netcdf` without the others failed to compile, which a default build hid because it turns most features on. Three causes: the shared `ReadError` imported `vtkio` (only the `vtk` feature provides it), the PVD time-series reader was gated on three features but used by seven and needs an XML parser only `xdmf` pulled in, and the internal dataset model derived serde, which only `raw` enables. The plumbing the scientific loaders share now sits behind one internal feature the format features pull in, the two `ReadError` variants that named `vtkio` types are gone (nothing constructed them), and the unused serde derives are dropped. Every feature now builds on its own, checked one at a time across all 24.
- **FBX mesh order varied between decodes of the same file** - `fbxcel-dom` yields a document's objects from a `HashMap`, so each walk came out in a different order and the loader appended meshes in that order. A positional index recorded against one decode (a material assignment, a saved selection, a retarget map) addressed a different mesh in the next, which showed up downstream as a character wearing the wrong textures after a scene reload. Every walk that builds ordered output now goes through the objects sorted by FBX object id, so decoding is reproducible across runs and machines. Mesh order will differ from what a given run produced before, which nothing can have depended on.
- **Hidden SVG paths were emitted** - a path with `visibility="hidden"` came through as a normal filled shape, disagreeing with what `texture_from_path` rasterizes from the same file. Hidden paths are now skipped.
- **Group opacity was dropped on SVG vector art** - a shape inside `<g opacity="0.5">` came back fully opaque. Group opacity now multiplies down the tree into the fill and stroke alpha, alongside the path's own `fill-opacity`.
- **Two probe examples were missing `required-features`** - `probe_fbx_orientation` and `probe_fbx_uv` failed to compile on any build without the `fbx` feature, including single-loader test runs.
- **glTF morph targets with no `POSITION` were dropped** - a normals-only face shape shifted every later target index and changed the count, so a 52-blendshape face imported 51 geometry targets against a 52-target animation. Absent position deltas are zero-filled.

### Tests
- **A conformance test per loader** (`tests/loader_<format>.rs`) - every loader now has one file that runs the same floor over its fixtures (decode, satisfy the invariants, decode identically a second time) before its format-specific cases. Nine so far: obj, stl, ply, csv, vtk, gltf, svg, numpy, and fbx. `testkit::conformance` holds the floor, so a new loader inherits it by calling one function and a new invariant reaches every loader the day it is written.
- **A `testkit` feature** (off by default, outside `all-formats`) - shared checks for loader output and in-memory fixture builders, usable from this crate's tests or by a consumer checking its own asset library. `check_surface_mesh`, `check_scene`, `check_skeleton`, and `check_point_set` collect every broken invariant into a `Report` (index bounds, per-vertex array lengths, unit normals, skin weights summing to one, cross-references that resolve, sampler values matching keyframe times); `decode_twice_identical` and `decode_twice_matches` assert a file decodes the same way twice. `testkit::synth` builds valid obj, stl (ASCII and binary), ply (mesh and point cloud, ASCII and binary), csv, npy, legacy vtk, GLB, and SVG input as the same two-triangle quad, so a test needs no committed asset and one format's decode can be compared against another's.
- **FBX axis-veto truth table** - the per-leaf veto that decides whether a mesh gets the file's Y-up to Z-up conversion now has tests for all three chain orientations (+Y ordinary, +Z already converted, -Z placed for a Y-up consumer), for the 0.9/0.5 threshold pair, and for the other three `AxisPolicy` values not vetoing at all. It was the loader's most expensive fix and had no test.
- **FBX decode determinism** (`tests/loader_fbx.rs`) - decoding one file twice must give the same scene: mesh order and content, material order, and joint order.
- **Cross-format skinning parity** (`tests/fbx_gltf_skinning_parity.rs`) - the same rigged Fox asset from FBX and from glTF must match in bind pose and when posed from the "Walk" clip, under both `AxisPolicy::HonourHeader` (identity, equal directly) and `AxisPolicy::ForceYUpRaw` (both offset by the same +90 degree X). Guards the sampled-pose bug where curves stayed in raw file space. Ships the Fox fixtures (CC-BY 4.0) under `tests/fixtures/`.

### Internal
- **The glTF loader is a module** - `loaders/gltf.rs` became `loaders/gltf/` with one file per concern: `node`, `primitive`, `material`, `skin`, `animation`, `axis` (the Y-up to Z-up conversion), and `mod.rs` holding the two entrypoints. Each module carries the tests for what it does, so the 1150-line test block that made up half the old file now sits beside the code it covers. `scene_from_path` and `scene_from_slice` are unchanged, and no decoding logic moved with it.
- **The FBX axis veto lives in one function** - `effective_axis_transform` is now the single implementation of the per-leaf veto rule, called by both the loader and the `chain_breakdown` diagnostic. The two carried copy-pasted thresholds, so the tool that exists to explain the loader's axis decision could have drifted from the decision actually applied.
- **Formatting** - reformatted `build_fbx_scene`, `lightmap.rs`, and the `probe_fbx_uv` example to satisfy `cargo fmt`. No behavioural change.

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
  only skipped when it is not a texture coordinate at all : a packed per-vertex
  scalar that holds one axis constant across the mesh (a wind phase) : in which
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
