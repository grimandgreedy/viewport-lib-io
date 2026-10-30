# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Trained gaussian splat `.ply` ingestion** - `loaders::ply_splats::splats_from_path` and `splats_from_bytes` decode a 3D Gaussian Splatting trainer's output into a `GaussianSplatData` (from `viewport-lib-plugins`, built without the renderer), which uploads as-is; the coefficients are left sRGB-referred, as trained. The parse is the point-set loader's; what this adds is the stored units, none of which the file announces: `scale_0..2` are logarithms, `opacity` is a logit, and `rot_0..3` is an unnormalised quaternion written `w` first. A reader that takes the numbers at face value produces a set that loads and draws and is wrong in a way that looks like a bad training run. The spherical-harmonic degree is read from the `f_rest_*` count (none, three per channel, or fifteen), and the file's channel-major layout is transposed into the coefficient-major packed form; a count matching no supported degree is refused rather than flattened to degree zero.
- **SVG element and group ids on vector art** - `VectorShape::id` carries the source element's id and `VectorShape::groups` the ids of its enclosing groups, outermost first with unnamed groups left out, so a keyed drawing (a states map, a seat map, a floor plan) joins to records by id instead of by paint order. A shape drawn by `<use>` has no id of its own; the `<use>` element's id is in `groups`. `class` and other attributes are not carried. Struct-literal construction of `VectorShape` needs the two new fields.
- **`loaders::ply_points::point_cloud_from_bytes`** - the in-memory sibling of `point_cloud_from_path`, for a PLY that arrived over a network or through a sandbox rather than from a filesystem. It was already the whole implementation of the path form; this makes it reachable.
- **In-memory decoding for the remaining loaders** - `stl::mesh_from_bytes`, `ply_scene::scene_from_bytes`, `csv::point_cloud_from_bytes`, `svg::texture_from_bytes` and `obj::scene_from_bytes(bytes, base)`, each decoding what its path form does. OBJ takes an optional base directory for its `mtllib` and texture paths, and reads no materials without one; the SVG raster leaves out `<image>` files it cannot resolve.
- **Slot-labelled image decoding** - `png`, `jpeg`, `tga` and `bmp` gain `texture_from_path_for_slot` and `texture_from_bytes_for_slot`, which take a material `TextureSlot` and return a texture labelled for it: sRGB for albedo and emissive, a normal map for the normal slot, linear for the rest.
- **`viewport_lib_types` re-export** - the crate re-exports `viewport-lib-types`, so the shared types its loaders return are nameable without a second dependency.
- **`MaterialData::to_material`** - builds the renderer's PBR `Material`, with a closure that turns each present texture into an id. Carries the colours, factors, alpha mode, double-sidedness, normal and occlusion strengths, and per-slot UV transforms and samplers; opacity belongs on the item's `ItemSettings` and is not carried.
- **`PointSet::to_point_cloud_item`** - builds the renderer's `PointCloudItem` from a decoded point set: per-point colours (decoded from sRGB) if present, otherwise the primary scalar through the default colourmap, with size, render mode and transform left at the item's defaults. Named scalars are not carried.
- **`From<TextureSampler> for SamplerKey`** - a decoded sampler converts to the renderer's, with anisotropy and LOD bias left at their defaults.
- **Unlit glTF materials** - `MaterialData::unlit` is set from `KHR_materials_unlit` (false elsewhere), and `MaterialData::item_settings()` returns `ItemSettings` with `unlit` set, since the renderer carries unlit on the item rather than the material. Unlit assets used to decode as lit.
- **glTF punctual lights** - `SceneData::lights` holds a `SceneLight` per node carrying a `KHR_lights_punctual` light: kind (`SceneLightKind::Directional`, `Point { range }`, `Spot { range, inner_angle, outer_angle }`), linear colour, intensity in lux or candela as the file gives it, and a Z-up world transform in which the light shines along local +Y. `position()`, `direction()` and `to_light_source(range_when_unset)` build the renderer's `LightSource`; an unset range stays `None` and the caller supplies one, since the renderer has no unbounded range.
- **Scene node table** - `SceneData::nodes` lists the source scene graph as `SceneNode { name, parent, transform, meshes }`, with local transforms in Z-up. glTF fills it in document order, so a glTF node index is an index into it; other formats leave it empty.
- **Node animation** - glTF translation, rotation and scale channels on nodes that are not skeleton joints are read into `SceneData::node_animations` as `NodeAnimationClip`s whose `NodeAnimationTrack::node` indexes `SceneData::nodes`, reoriented to Z-up like joint tracks. These channels used to be dropped.
- **`AnimationInterpolation::values_per_key` and `MorphWeightClip::key_weights`** - how many stored values a keyframe takes (3 under cubic-spline), and one keyframe's morph weights without its tangents.
- **glTF IOR and specular** - `MaterialData::ior`, `specular_factor` and `specular_colour` are read from `KHR_materials_ior` and `KHR_materials_specular` (defaults 1.5, 1.0 and white). `to_material` sets the IOR through `Material::with_ior`, so the reflectance follows it, and copies the specular factor and colour. The specular textures are not read.
- **glTF clear coat, sheen and anisotropy** - `MaterialData::clearcoat`, `clearcoat_roughness`, `sheen_colour`, `sheen_roughness`, `anisotropy` and `anisotropy_rotation` are read from `KHR_materials_clearcoat`, `KHR_materials_sheen` and `KHR_materials_anisotropy`, and `to_material` carries them onto the `Material`. Only the factors are read, not the extensions' textures. A material without these extensions converts exactly as before.

### Changed
- **No format is on by default** - `default` is empty, so a plain dependency no longer builds HDF5 (and needs `cmake`) and every other decoder. Name the formats you read, or depend with `features = ["all-formats"]` for the old behaviour. docs.rs builds with `all-formats`.
- **Depends on `viewport-lib-types` and `viewport-lib-plugins`** - loaders return the shared payload types wherever one says the same thing, instead of mirrors a consumer converted by hand. Neither dependency brings `viewport-lib` or `wgpu`; `viewport-lib-plugins` is taken with default features off, and enabling its `gpu` feature downstream is harmless.
- **Images are `TextureData`** - every `texture_from_path` / `texture_from_bytes` returns `viewport_lib_types::data::texture::TextureData` with its colour space set, replacing `RasterImageData` and `HdrImageData`. Width, height and pixels are now `width()`, `height()` and `payload()`. 8-bit images are labelled sRGB, HDR images are float, and EXR stays 8-bit and is labelled linear, since its quantised bytes are linear.
- **Embedded material textures are labelled for their slot** - `TextureSource::Decoded` holds a `TextureData` built for the slot it fills, so an embedded normal map arrives as a normal map and an embedded metallic-roughness map as linear. io's `ColourSpace` is replaced by the shared one.
- **Material leaf types are the shared ones** - `MaterialTextureSlot` is replaced by `viewport_lib_types::material::TextureSlot` (`BaseColour` is `Albedo`, `Occlusion` is `Ao`), and `UvTransform`, `WrapMode`, `TextureFilter`, `AlphaMode` and `MATERIAL_TEXTURE_SLOTS` by theirs. `MaterialData` and `TextureSampler` stay io's.
- **UV transforms are decoded in the renderer's convention** - `MaterialData::uv_transforms` now rotate about the texture centre after scale and offset, converted from the file's rotation about the UV origin at decode, so they copy straight into `Material::texture_transforms`. `UvTransform::as_centre_rotation` and `from_centre_rotation` are gone; a consumer still calling one should drop the call.
- **Meshes are `MeshData`** - `SurfaceMesh` is removed. `SceneMesh::mesh`, `DecodedDataSet::surface_mesh` and `mesh_from_path` carry `viewport_lib_types::data::mesh::MeshData`; skin weights and morph targets moved up to `SceneMesh::skin_weights` and `SceneMesh::morph_targets`, and `colours` is `vertex_colours`.
- **Attributes are the shared `AttributeData`** - io's `AttributeData`, `AttributeDomain` and `AttributeValues` are replaced by the renderer's enum: per-vertex scalars are `Vertex`, per-vertex vectors `VertexVector`, scientific per-cell and per-halfedge scalars `Cell` (one value per triangle) and `Halfedge` (three per triangle).
- **Every mesh has normals** - a source without them gets area-weighted smooth normals at decode, so a decoded mesh uploads without a normals pass. Degenerate and unreferenced vertices get +Z rather than a zero vector.
- **Volume meshes are `VolumeMeshData`** - `DecodedDataSet::volume_mesh` holds `viewport_lib_types::data::volume_mesh::VolumeMeshData`, replacing `VolumeMesh` (`cell_fields` is `cell_scalars`, `cell_colors` is `cell_colours`). `CELL_SENTINEL` is the shared one.
- **VTK point data on volume meshes stays per node** - point fields land in `VolumeMeshData::node_scalars` as the file stores them, instead of being averaged onto cells. A consumer reading a point field from the cell fields now finds it among the node fields.
- **Vector paths and paint are the shared types** - `VectorShape::subpaths` uses the renderer's `SubPath`, `PathSegment` and `FillRule`, and `VectorShape::fill` and `VectorStroke::colour` are `Colour` instead of linear `[f32; 4]`. Read the old arrays with `to_linear_rgba()`.
- **Cubic-spline morph weights keep their tangents** - a glTF `CUBICSPLINE` weights sampler now decodes as `AnimationInterpolation::CubicSpline` with `weights` laid out `[keyframe][in | value | out][target]`, three times the length of a linear clip, instead of collapsing to its values as `Linear`. A consumer that only wants the values reads them with `key_weights`.
- **`SceneData` gains `nodes`, `lights` and `node_animations`** - a struct literal naming every field needs the three new ones, or `..SceneData::default()`.

### Fixed
- **Scientific cell fields landed on the wrong triangles** - Exodus, CGNS, XDMF and EnSight attached one value per source cell to a surface that has several triangles per quad, polygon or hex face, and VTK dropped such fields outright. Each triangle now carries its source cell, so a cell value covers every triangle the cell became, and a field that cannot be placed is left off rather than misplaced. VTK structured, rectilinear and 2D image grids also keep cell fields they used to drop.
- **STL facet normals written as zero** - a facet stored as `facet normal 0 0 0`, as many exporters write it, came through as a zero normal and rendered unlit. It is now computed from the triangle's winding; a stored normal with a direction is kept and normalised, even where it disagrees with the winding.
- **PLY mesh vertex colours were dropped** - a PLY with faces read its `red` / `green` / `blue` properties and then discarded them. They now land in `MeshData::vertex_colours`, decoded from the file's sRGB bytes to linear.
- **`PointSet::colors` did not say its colour space** - the field is now documented as sRGB-encoded RGB in `0..=1` with linear alpha, which is what every loader already produced. No data changes.
- **OBJ files without normals got a constant +Y normal** - every vertex of a normal-less OBJ faced +Y whatever its geometry. Normals are now generated from the faces and follow the winding.

### Tests
- **Temp fixture directories no longer collide** - `synth::temp_dir` adds a per-process counter to the name, so two tests starting in the same clock tick no longer share a directory and read each other's half-written files.
- **Testkit checks every attribute length** - `check_mesh` now counts `Cell`, `Face` and `FaceColour` against the triangles and `Edge`, `Halfedge` and `Corner` against three per triangle, not only per-vertex attributes.
- **Testkit mesh checks follow `MeshData`** - `check_surface_mesh` is replaced by `check_mesh` (geometry, normals now required) and `check_scene_mesh` (adds skin and morph data), and `conformance::mesh` compares two decodes field for field.
- **Volume mesh coverage** - a VTK tetrahedron with point and cell data checks that the two land in `node_scalars` and `cell_scalars` apart.
- **UV transforms checked by the texel they sample** - the glTF and FBX transform tests now sample the decoded transform with the renderer's formula and compare against the file's own.

### Internal
- **One smooth-normals implementation** - nine per-loader copies of the area-weighted normal computation are now `types::smooth_normals`.

## [0.8.0] - 2026-09-19

### Added
- **SVG vector paths** - `loaders::svg::vector_from_path` decodes an SVG into neutral vector paths instead of pixels, alongside the unchanged `texture_from_path`. New types `VectorArt`, `VectorShape`, `SubPath`, `PathSegment`, `FillRule`; feed them into `viewport_lib::OverlayShape::Vector`. Curves stay unflattened, each path's transform is baked in, and coordinates are the SVG canvas frame (X right, Y down), not the Z-up scene convention. Gradient and pattern fills leave the colour unset, and `<text>` and image nodes are skipped.
- **Stroke paint and width on vector art** - `VectorShape::stroke` carries a `VectorStroke { colour, width }`, so stroke-drawn icon and cursor sets import with their paint instead of as unpainted contours. Width scales with the baked transform (averaged under a non-uniform scale) and stroke opacity folds in group opacity. Cap, join, dash, and miter limit are not carried.
- **`loaders::svg::vector_from_bytes`** - the in-memory sibling of `vector_from_path`, for SVG art served from a bundle or fetched over the network. Output is identical to the path form. There is no base directory to resolve external references against, which costs nothing here: the nodes that reference external files (images, text) are the ones this loader skips.
- **Morph targets on `SurfaceMesh`** - `MorphTarget` carries per-vertex position, normal, and tangent displacements, and `SurfaceMesh::morph_targets` holds them in authored order (empty for formats that have none). The glTF loader reads them and reorients into Z-up with the base attributes. Targets are named by index for now: glTF keeps names in `mesh.extras`.
- **FBX blend shapes** - the FBX loader reads `Deformer`(BlendShape) -> `BlendShapeChannel` -> `Shape` chains, expands each shape's sparse deltas through the control-point map skinning uses, and splits them across material sub-meshes.
- **Legacy FBX blend shapes** - pre-7.5 exports nest `Shape` nodes inside `Geometry` with no deformer objects, which `fbxcel-dom` does not surface. When the modern walk finds nothing, the loader falls back to the raw geometry tree. Verified against a character carrying the full 52-name ARKit set.
- **FBX per-texture UV placement and wrap modes** - the FBX loader reads each texture's `Translation`, `Scaling`, `Rotation`, `WrapModeU`/`WrapModeV` and `UVSet` into `MaterialData::uv_transforms` and `samplers`, so a tiled, offset or clamped FBX material imports with its placement instead of untransformed. The placement is re-expressed against the flipped V the loader emits. Non-default rotation and scaling pivots and `UVSwap` are not carried.
- **FBX data layers kept as named attributes** - a UV layer that is not a texture coordinate (a packed per-vertex scalar such as a foliage wind phase) and any third or later UV set now arrive in `SurfaceMesh::attributes` under the FBX layer name, on the `Point` domain, instead of being discarded. Values are as authored, with V unflipped, since they are not texture coordinates; the third vector component is padding.
- **A second UV set on meshes** - `SurfaceMesh::uvs1` carries glTF `TEXCOORD_1`, and the FBX loader emits the next authored texture-coordinate layer instead of discarding every layer but one. `None` when a mesh has one set. This is the content channel a material slot reaches through `UvTransform::uv_set`, not a lightmap bake. Layers the FBX UV pick rejects as packed per-vertex scalars stay rejected, and third and later sets are not carried.
- **Per-texture UV transforms** - `MaterialData::uv_transforms` carries a `UvTransform { offset, scale, rotation, uv_set }` per texture slot, indexed by `MaterialTextureSlot::index`, so a glTF material that tiles into an atlas, rotates a map, or samples occlusion from `TEXCOORD_1` decodes with the instruction for sampling it. The glTF loader reads `KHR_texture_transform` and `texCoord` on all five slots. Values follow the glTF convention (rotation about the UV origin); `UvTransform::as_centre_rotation` re-expresses one for a consumer that rotates about the texture centre, which is what viewport-lib's `UvTransform` does; it is its own inverse, and `UvTransform::from_centre_rotation` is the same call named for a write site.
- **Sampler state on materials** - `MaterialData::samplers` carries a `TextureSampler { wrap_u, wrap_v, filter }` per slot, read from the glTF sampler a texture names, so a `CLAMP_TO_EDGE` gradient or a `MIRRORED_REPEAT` sheet no longer imports as plain repeat. A slot is `None` when the file specifies nothing, or specifies exactly the neutral default, which a consumer reads as "use your own default sampler".
- **Emissive strength on materials** - `MaterialData::emissive_strength` multiplies `emissive` and the emissive texture, matching glTF `KHR_materials_emissive_strength`; the glTF loader reads it and every other format leaves it at 1.0. A consumer with photometric exposure reads `emissive * emissive_strength` as nits, so an authored light-source surface no longer imports at one nit and renders as black.
- **Morph-weight animation** - `MorphWeightClip` carries keyframed blend-shape weights for one mesh (row-major `[keyframe][target]`) in `SceneData::morph_animations`. glTF reads `MorphTargetWeights` channels; FBX reads each channel's `DeformPercent` curve, merges a mesh's independently-keyed curves onto a union timeline, and converts percentages to `0..1`. The two formats are now at parity.

### Changed
- **`MaterialTextureSlot` discriminants match viewport-lib's `TextureSlot`** - the variants are now numbered `BaseColour = 0`, `Normal = 1`, `Occlusion = 2`, `MetallicRoughness = 3`, `Emissive = 4`, and `MaterialTextureSlot::index` indexes both the per-slot arrays here and the renderer's. Only code casting the enum to an integer sees a difference; matching on variants is unaffected.

### Fixed
- **VTK surface extraction reordered its triangles between runs** - the boundary of an unstructured grid was collected by iterating a hash map, so the same file decoded to a different triangle list on every process, and a positional index recorded against one decode addressed a different triangle in the next. The surface now comes out in the order the file wrote its cells, which is what the OBJ, PLY and Gmsh loaders already produced for the same mesh. Triangle order will differ from what a given run produced before.
- **glTF animation clips came back in a random order** - an animation driving more than one skeleton had its clips pushed out of a hash map, so `SceneData::animations` was ordered differently on each run. Clips are now emitted in skeleton order. Files whose animations each drive a single skeleton are unaffected.
- **The scientific loader features did not build on their own** - selecting any one of `vtk`, `xdmf`, `exodus`, `ensight`, `tecplot`, `cgns`, or `netcdf` without the others failed to compile, which a default build hid because it turns most features on. Three causes: the shared `ReadError` imported `vtkio` (only the `vtk` feature provides it), the PVD time-series reader was gated on three features but used by seven and needs an XML parser only `xdmf` pulled in, and the internal dataset model derived serde, which only `raw` enables. The plumbing the scientific loaders share now sits behind one internal feature the format features pull in, the two `ReadError` variants that named `vtkio` types are gone (nothing constructed them), and the unused serde derives are dropped. Every feature now builds on its own, checked one at a time across all 24.
- **FBX mesh order varied between decodes of the same file** - `fbxcel-dom` yields a document's objects from a `HashMap`, so each walk came out in a different order and the loader appended meshes in that order. A positional index recorded against one decode (a material assignment, a saved selection, a retarget map) addressed a different mesh in the next, which showed up downstream as a character wearing the wrong textures after a scene reload. Every walk that builds ordered output now goes through the objects sorted by FBX object id, so decoding is reproducible across runs and machines. Mesh order will differ from what a given run produced before, which nothing can have depended on.
- **Hidden SVG paths were emitted** - a path with `visibility="hidden"` came through as a normal filled shape, disagreeing with what `texture_from_path` rasterizes from the same file. Hidden paths are now skipped.
- **Group opacity was dropped on SVG vector art** - a shape inside `<g opacity="0.5">` came back fully opaque. Group opacity now multiplies down the tree into the fill and stroke alpha, alongside the path's own `fill-opacity`.
- **Two probe examples were missing `required-features`** - `probe_fbx_orientation` and `probe_fbx_uv` failed to compile on any build without the `fbx` feature, including single-loader test runs.
- **glTF morph targets with no `POSITION` were dropped** - a normals-only face shape shifted every later target index and changed the count, so a 52-blendshape face imported 51 geometry targets against a 52-target animation. Absent position deltas are zero-filled.

### Tests
- **Cross-format parity** (`tests/cross_format_parity.rs`) - the same quad authored in OBJ, PLY, STL, Gmsh and legacy VTK, eight encodings through five loaders, must decode to the same triangles. STL's unindexed vertices are the only difference, and that is the format's.
- **Cross-process decode determinism** (`tests/determinism_across_processes.rs`) - the conformance floor decodes twice in one process, which cannot see a loader that walks a hash map to build ordered output, because the hasher seed is per process. This test re-runs itself as a child process and compares, over eleven loader entry points. It found both ordering bugs fixed above.
- **Unit tests for the loaders that had none** - `msh`, `common` (the scientific dataset and volume-grid model), `ply_scene`, `ply_points`, `numpy`, `raw`, and `pvd` gained 56 tests between them, covering format arithmetic that had no coverage at all: the Gmsh 2.2 and 4.1 parsers against each other, PLY scalar widths and byte orders, every NumPy and raw dtype, cell-to-point averaging, rectilinear resampling, and PVD path resolution and sorting. A Gmsh fixture builder joins the testkit, and `tests/loader_msh.rs` runs the conformance floor over both format versions.
- **VTK topology and cell-buffer unit tests** - the helpers the module split made reachable now have tests: a tetrahedron's four faces, a quad's diagonal, cells too short for their type being skipped rather than indexed out of bounds, the winding-independent face key, a face shared by two cells not being surface, the polygon fan, and the legacy and XML cell encodings unpacking to the same cells (including a truncated array stopping early). The loader had no unit tests at all.
- **A conformance test per loader** (`tests/loader_<format>.rs`) - every loader now has one file that runs the same floor over its fixtures (decode, satisfy the invariants, decode identically a second time) before its format-specific cases. Nine so far: obj, stl, ply, csv, vtk, gltf, svg, numpy, and fbx. `testkit::conformance` holds the floor, so a new loader inherits it by calling one function and a new invariant reaches every loader the day it is written.
- **A `viewport-lib-io-testkit` crate** (in `testkit/`, `publish = false`) - shared checks for loader output and in-memory fixture builders, pulled in as a dev-dependency, so a plain `cargo test` runs the conformance suite and a release build never compiles the scaffolding. A consumer can depend on it directly to check its own asset library. `check_surface_mesh`, `check_scene`, `check_skeleton`, and `check_point_set` collect every broken invariant into a `Report` (index bounds, per-vertex array lengths, unit normals, skin weights summing to one, cross-references that resolve, sampler values matching keyframe times); `decode_twice_identical` and `decode_twice_matches` assert a file decodes the same way twice. `testkit::synth` builds valid obj, stl (ASCII and binary), ply (mesh and point cloud, ASCII and binary), csv, npy, legacy vtk, GLB, and SVG input as the same two-triangle quad, so a test needs no committed asset and one format's decode can be compared against another's.
- **FBX axis-veto truth table** - the per-leaf veto that decides whether a mesh gets the file's Y-up to Z-up conversion now has tests for all three chain orientations (+Y ordinary, +Z already converted, -Z placed for a Y-up consumer), for the 0.9/0.5 threshold pair, and for the other three `AxisPolicy` values not vetoing at all. It was the loader's most expensive fix and had no test.
- **FBX decode determinism** (`tests/loader_fbx.rs`) - decoding one file twice must give the same scene: mesh order and content, material order, and joint order.
- **Cross-format skinning parity** (`tests/fbx_gltf_skinning_parity.rs`) - the same rigged Fox asset from FBX and from glTF must match in bind pose and when posed from the "Walk" clip, under both `AxisPolicy::HonourHeader` (identity, equal directly) and `AxisPolicy::ForceYUpRaw` (both offset by the same +90 degree X). Guards the sampled-pose bug where curves stayed in raw file space. Ships the Fox fixtures (CC-BY 4.0) under `tests/fixtures/`.

### Internal
- **The glTF loader is a module** - `loaders/gltf.rs` became `loaders/gltf/` with one file per concern: `node`, `primitive`, `material`, `skin`, `animation`, `axis` (the Y-up to Z-up conversion), and `mod.rs` holding the two entrypoints. Each module carries the tests for what it does, so the 1150-line test block that made up half the old file now sits beside the code it covers. `scene_from_path` and `scene_from_slice` are unchanged, and no decoding logic moved with it.
- **The FBX axis veto lives in one function** - `effective_axis_transform` is now the single implementation of the per-leaf veto rule, called by both the loader and the `chain_breakdown` diagnostic. The two carried copy-pasted thresholds, so the tool that exists to explain the loader's axis decision could have drifted from the decision actually applied.
- **The FBX loader is a module** - `loaders/fbx.rs` became `loaders/fbx/` with one file per concern: `options`, `raw` (node and property reads), `transform` (node transforms and the axis conversion), `geometry`, `material`, `blendshape`, `skin`, `animation`, `scene` (the decode walk), `chain` (the breakdown diagnostic), and `mod.rs` holding the four entrypoints. The tests moved to the modules they cover. The public paths, including `AxisPolicy`, `FbxLoadOptions`, and `chain_breakdown`, are unchanged, and no decoding logic moved with it.
- **The VTK loader is a module** - `loaders/vtk.rs` and its `vtu` / `vtp` forwards became `loaders/vtk/`, split into `dataset` (one conversion per dataset kind), `topology` (cell faces, surface extraction, triangulation, structured-grid layout), `attributes`, and `buffers`, with the private `mod imp` wrapper gone. `loaders::vtk`, `loaders::vtu`, and `loaders::vtp` are unchanged. The shared PVD time-series reader stays in `loaders::pvd`, where the other six scientific loaders reach it.
- **`build_fbx_scene` is seven functions** - the 545-line decode walk is now a short loop over mesh models, calling out to resolve the model's materials, read its vertex arrays, expand skin influences and blend-shape deltas to per-vertex, choose its UV sets, and emit it as one mesh or one per material. No decoding logic changed.
- **Plain ASCII through the FBX loader** - the em-dashes, arrows, degree signs, and set symbols in its comments and doc strings are gone, so the source reads the same everywhere it is opened.
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
