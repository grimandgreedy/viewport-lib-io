# `viewport-lib-io`

Source-format loaders and neutral decoded data for applications built on `viewport-lib`.

## Goal

`viewport-lib-io` owns file-format decoding and reusable CPU-side data structures. By design it does not depend on `viewport-lib` runtime or upload structs directly.

This crate should:
- load source files through `loaders/*`
- return neutral decoded data that can be reused in several downstream paths
- preserve format-specific fields, layers, geometry, and attributes
- let applications or adapter code decide whether loaded data becomes a surface mesh, structured volume, sparse grid, point set, or gaussian splat set

This crate should not:
- create `viewport-lib` scene items
- upload to GPU
- own renderer policy

## Public Layout

- `loaders/` contains source-format entrypoints
- `types.rs` contains neutral decoded data models

Supported loader modules are listed in `loaders/` with their extension, e.g., `loaders/png.rs`

## Usage

Surface mesh:

```rust
let mesh = viewport_lib_io::loaders::stl::mesh_from_path("part.stl".as_ref())?;
```

Image:

```rust
let image = viewport_lib_io::loaders::png::texture_from_path("albedo.png".as_ref())?;
let icon = viewport_lib_io::loaders::svg::texture_from_path("icon.svg".as_ref())?;
```

Scene (glTF, OBJ, FBX, PLY):

```rust
let scene = viewport_lib_io::loaders::gltf::scene_from_path("model.glb".as_ref())?;

// Geometry
for mesh in &scene.meshes {
    let positions = &mesh.mesh.positions;
    let indices   = &mesh.mesh.indices;

    // Skin weights (present on skinned meshes)
    if let Some(skin) = &mesh.mesh.skin_weights {
        let joint_indices = &skin.joint_indices; // Vec<[u8; 4]> per vertex
        let joint_weights = &skin.joint_weights; // Vec<[f32; 4]> per vertex, sum to 1
    }
}

// Animations
for clip in &scene.animations {
    println!("clip: {} ({:.2}s)", clip.name, clip.duration);
    for track in &clip.tracks {
        // track.joint is an index into scene.skeletons[clip.skeleton_index].joints
        // track.channel is Translation, Rotation, or Scale
        // track.sampler.times and track.sampler.values hold the keyframes
    }
}
```

Scientific dataset:

```rust
let datasets = viewport_lib_io::loaders::vtk::datasets_from_path("field.vtu".as_ref())?;
if let Some(volume) = datasets[0].as_structured_volume() {
    let density = volume.scalar_values("density");
}
```

Point set to gaussian splats:

```rust
let points = viewport_lib_io::loaders::csv::point_cloud_from_path("samples.csv".as_ref())?;
let splats = points.to_gaussian_splats([0.01, 0.01, 0.01], 1.0);
```
