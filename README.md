# `viewport-lib-io`

Source-format loaders and neutral decoded data for applications built on
`viewport-lib`.

## Goal

`viewport-lib-io` owns file-format decoding and reusable CPU-side data
structures. It does not depend on `viewport-lib` runtime or upload structs
directly.

This crate should:
- load source files through `loaders/*`
- return neutral decoded data that can be reused in several downstream paths
- preserve format-specific fields, layers, geometry, and attributes
- let applications or adapter code decide whether loaded data becomes a surface
  mesh, structured volume, sparse grid, point set, or gaussian splat set

This crate should not:
- create `viewport-lib` scene items
- upload to GPU
- own renderer policy

## Public Layout

- `loaders/` contains source-format entrypoints
- `types/` contains neutral decoded data models

Examples of loader modules:
- `loaders::stl`
- `loaders::obj`
- `loaders::vtk`
- `loaders::png`
- `loaders::numpy`

Examples of decoded types:
- `SurfaceMesh`
- `SceneData`
- `PointSet`
- `StructuredVolume`
- `SparseGrid`
- `VolumeMesh`
- `DecodedDataSet`
- `GaussianSplatSet`

## Usage

Surface mesh:

```rust
let mesh = viewport_lib_io::loaders::stl::load_from_path("part.stl".as_ref())?;
```

Image:

```rust
let image = viewport_lib_io::loaders::png::load_from_path("albedo.png".as_ref())?;
```

Scene:

```rust
let scene = viewport_lib_io::loaders::obj::load_from_path("model.obj".as_ref())?;
```

Scientific dataset:

```rust
let datasets = viewport_lib_io::loaders::vtk::load_from_path("field.vtu".as_ref())?;
if let Some(volume) = datasets[0].as_structured_volume() {
    let density = volume.scalar_values("density");
}
```

Point set to gaussian splats:

```rust
let points = viewport_lib_io::loaders::csv::load_from_path("samples.csv".as_ref())?;
let splats = points.to_gaussian_splats([0.01, 0.01, 0.01], 1.0);
```
