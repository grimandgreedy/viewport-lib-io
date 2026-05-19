# `viewport-lib-io`

Decode-only IO helpers for `viewport-lib`.

## Goal

`viewport-lib-io` keeps format-decoding dependencies out of `viewport-lib` itself.
It translates file formats into the CPU-side data families that `viewport-lib`
already knows how to consume.

This crate should:
- decode files into `viewport-lib`-shaped CPU data
- stay independent from GPU upload and renderer policy
- keep one Rust file per source filetype
- group modules by `viewport-lib` input family

This crate should not:
- upload to GPU
- depend on `wgpu`
- create app objects
- own runtime or UI policy

## Current Layout

The source tree is organized around the data families that `viewport-lib`
accepts:

- `surface_mesh/`
- `scene/`
- `texture/`
- `volume/`
- `volume_mesh/`
- `sparse_volume/`
- `point_cloud/`
- `gaussian_splat/`
- `path/`
- `glyph/`
- `lighting/`

Inside each family, each source filetype gets its own `.rs` file.

Examples:
- `surface_mesh/stl.rs`
- `surface_mesh/msh.rs`
- `scene/obj.rs`
- `texture/png.rs`
- `point_cloud/csv.rs`
- `volume/raw.rs`
- `volume/numpy.rs`

## Usage

Surface mesh:

```rust
let mesh = viewport_lib_io::surface_mesh::stl::mesh_from_path("part.stl".as_ref())?;
let mesh_id = renderer.resources_mut().upload_mesh_data(&device, &mesh)?;
```

Texture:

```rust
let texture = viewport_lib_io::texture::png::texture_from_path("albedo.png".as_ref())?;
let texture_id = renderer
    .resources_mut()
    .upload_texture(&device, &queue, texture.width, texture.height, &texture.rgba)?;
```

Scene:

```rust
let scene = viewport_lib_io::scene::obj::scene_from_path("model.obj".as_ref())?;
```

Point cloud:

```rust
let points = viewport_lib_io::point_cloud::csv::point_cloud_from_path("samples.csv".as_ref())?;
```

Volume:

```rust
let volume = viewport_lib_io::volume::raw::volume_from_path("density.json".as_ref())?;
let density = &volume.scalar_fields["density"];
let volume_id = renderer
    .resources_mut()
    .upload_volume(&device, &queue, density, volume.dims);
```

## Shared Types

Shared decode-side types live in `src/types.rs`.

These currently include:
- `IoScene`
- `IoMesh`
- `IoMaterial`
- `IoPointCloud`
- `IoVolume`
- `TextureData`
- `TextureSource`
- `IoError`
