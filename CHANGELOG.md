# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `MaterialData::normal_scale` and `MaterialData::occlusion_strength`, matching
  glTF `normalScale` and `occlusionStrength`. The glTF loader now reads both
  (they were previously dropped); OBJ and FBX leave them at 1.0. Apply them to a
  viewport-lib `Material` as `normal_strength = normal_scale` and
  `ao_range = [1.0 - occlusion_strength, 1.0]`.

### Changed
- `MaterialData` is now `#[non_exhaustive]`. Build it via `Default` and
  struct-update syntax; later field additions will not be breaking. Code that
  reads `MaterialData` is unaffected.

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
