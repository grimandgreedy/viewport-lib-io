# `viewport-lib-io`

Source-format loaders and neutral decoded data for applications built on `viewport-lib`.

## Goal

`viewport-lib-io` owns file-format decoding and reusable CPU-side data structures. It returns the shared payload types from `viewport-lib-types` (images, meshes, attributes, vector paths, material leaf types) wherever one fits, so decoded data reaches the renderer without a conversion. It does not depend on `viewport-lib` itself: no GPU stack, no renderer.

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
- `types.rs` contains the decoded data models this crate owns: scenes, materials, skeletons and animation, point sets, volumes, vector art, tile maps, sprite sheets, bitmap fonts
- `viewport_lib_types` is re-exported, so the shared types are reachable without a second dependency

Supported loader modules are listed in `loaders/` with their extension, e.g., `loaders/png.rs`

## Features

No format is enabled by default. Name the ones you read, or take them all:

```toml
viewport-lib-io = { version = "0.9", features = ["png", "gltf"] }
viewport-lib-io = { version = "0.9", features = ["all-formats"] }
```

One feature per format: `png`, `jpeg`, `tga`, `bmp`, `hdr`, `exr`, `svg`, `stl`,
`obj`, `msh`, `gltf`, `fbx`, `bvh`, `ply`, `csv`, `numpy`, `raw`, `vtk`, `xdmf`,
`exodus`, `ensight`, `tecplot`, `cgns`, `netcdf`, and for 2D assets `tiled`,
`ldtk`, `sprite-sheet` and `bmfont`. `xdmf`, `exodus` and `cgns`
build the HDF5 C library statically and need `cmake`.

Three more extend `gltf` to compressed assets: `gltf-draco`
(`KHR_draco_mesh_compression`), `gltf-meshopt` (`EXT_meshopt_compression`) and
`gltf-ktx2` (`KHR_texture_basisu`). KTX2 textures transcode to the
`GltfTextureTarget` passed in `GltfLoadOptions` (RGBA8 by default, or the BC,
ETC2/EAC or ASTC family the device supports).

## Usage

Surface mesh, as the renderer's `MeshData` (normals always present):

```rust
let mesh = viewport_lib_io::loaders::stl::mesh_from_path("part.stl".as_ref())?;
```

Image:

```rust
use viewport_lib_io::viewport_lib_types::material::TextureSlot;

// A `TextureData`, labelled sRGB colour.
let image = viewport_lib_io::loaders::png::texture_from_path("albedo.png".as_ref())?;
let icon = viewport_lib_io::loaders::svg::texture_from_path("icon.svg".as_ref())?;

// Labelled for the material slot it fills: a normal map, linear.
let normal = viewport_lib_io::loaders::png::texture_from_path_for_slot(
    "normal.png".as_ref(),
    TextureSlot::Normal,
)?;
```

Scene (glTF, OBJ, FBX, PLY):

```rust
let scene = viewport_lib_io::loaders::gltf::scene_from_path("model.glb".as_ref())?;

// Geometry
for mesh in &scene.meshes {
    let positions = &mesh.mesh.positions;
    let indices   = &mesh.mesh.indices;

    // Skin weights (present on skinned meshes)
    if let Some(skin) = &mesh.skin_weights {
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

Compressed glTF, with KTX2 textures transcoded for a device with
`TEXTURE_COMPRESSION_BC` (features `gltf-draco`, `gltf-meshopt`, `gltf-ktx2`):

```rust
use viewport_lib_io::loaders::gltf::{self, GltfLoadOptions, GltfTextureTarget};

let options = GltfLoadOptions::default().with_texture_target(GltfTextureTarget::Bc);
let scene = gltf::scene_from_path_with_options("model.glb".as_ref(), options)?;
```

Scientific dataset:

```rust
let datasets = viewport_lib_io::loaders::vtk::datasets_from_path("field.vtu".as_ref())?;
if let Some(volume) = datasets[0].as_structured_volume() {
    let density = volume.scalar_values("density");
}
```

Point set to gaussian splats or a point cloud:

```rust
let points = viewport_lib_io::loaders::csv::point_cloud_from_path("samples.csv".as_ref())?;
let splats = points.to_gaussian_splats([0.01, 0.01, 0.01], 1.0);

// Or the renderer's point cloud item, coloured by the best channel the set has.
let cloud = points.to_point_cloud_item();
```

Trained gaussian splats:

```rust
let splats = viewport_lib_io::loaders::ply_splats::splats_from_path("scene.ply".as_ref())?;
```

The stored units are undone on the way in: scales are logarithms, opacity is a
logit, and the quaternion is written `w` first and unnormalised. The
spherical-harmonic degree is read from how many `f_rest_*` properties the file
carries.

SVG as vector paths rather than pixels, with gradients and stroke style, in
the SVG canvas frame (Y down):

```rust
let art = viewport_lib_io::loaders::svg::vector_from_path("icon.svg".as_ref())?;
for shape in &art.shapes {
    // Gradients are relative to the overlay item's box, here the whole art.
    let fill = shape
        .fill
        .as_ref()
        .map(|paint| paint.to_overlay_fill([0.0, 0.0, art.size[0], art.size[1]]));
    // Lengths scale from the art's units to logical pixels.
    let stroke = shape.stroke.as_ref().map(|s| s.to_overlay_stroke(scale));
}
```

Tile maps from Tiled (`.tmx`, `.tmj`) or LDtk (`.ldtk`), in a Y-down pixel
frame. External tilesets, templates and level files are read from beside the
map; images are left as paths:

```rust
use viewport_lib_io::types::TileLayerKind;

let map = viewport_lib_io::loaders::tiled::tilemap_from_path("level.tmx".as_ref())?;
for layer in &map.layers {
    if let TileLayerKind::Tiles(grid) = &layer.kind {
        for cell in grid.cells.iter().filter(|c| !c.is_empty()) {
            let tileset = &map.tilesets[cell.tileset as usize];
            let pixels = tileset.tile_rect(cell.index); // where in the atlas
        }
    }
}

let worlds = viewport_lib_io::loaders::ldtk::worlds_from_path("game.ldtk".as_ref())?;
let first_level = &worlds[0].levels[0].map;
```

Sprite sheets (Aseprite or TexturePacker JSON) and bitmap fonts (BMFont):

```rust
let sheet = viewport_lib_io::loaders::sprite_sheet::sprite_sheet_from_path("hero.json".as_ref())?;
let walk = sheet.animations.iter().find(|a| a.name == "walk");
let uv = sheet.frames[0].uv; // [u_min, v_min, u_max, v_max], v down

let font = viewport_lib_io::loaders::bmfont::bitmap_font_from_path("ui.fnt".as_ref())?;
let a = font.glyphs[&('A' as u32)];
```

## Versions

| viewport-lib-io | viewport-lib | viewport-lib-types |
| --- | --- | --- |
| 0.9.0 | 0.23 | 0.3 |
| 0.8.0 | 0.23 | - |
| 0.7.0 | 0.21 - 0.22 | - |
| 0.4.0 | 0.20 | - |
| 0.3.0 | 0.20 | - |
| 0.2.0 | 0.17 - 0.19 | - |
| 0.1.0 | 0.15 - 0.16 | - |
