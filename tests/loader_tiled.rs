//! Tiled conformance: one map written as TMX in each layer encoding and as
//! TMJ in each of its own, every decode clearing the floor and all of them
//! decoding to the same map. Then what is peculiar to Tiled: global ids and
//! flips, external tilesets, templates, every object shape, typed properties.

#![cfg(feature = "tiled")]

use std::path::{Path, PathBuf};

use viewport_lib_io::IoError;
use viewport_lib_io::loaders::tiled;
use viewport_lib_io::types::{
    MapObjectShape, ObjectDrawOrder, PixelRect, PropertyValue, TextAlign, TextureSource, TileCell,
    TileFrame, TileLayerKind, TileMapData,
};
use viewport_lib_io::viewport_lib_types::colour::Colour;
use viewport_lib_io_testkit::{conformance, synth};

/// A directory holding the map as `name`, beside its tileset and template.
fn write_map(name: &str, contents: &str) -> PathBuf {
    let dir = synth::temp_dir("tiled");
    synth::write(&dir.join("props.tsx"), synth::tiled_tsx());
    synth::write(&dir.join("crate.tx"), synth::tiled_template());
    let path = dir.join(name);
    synth::write(&path, contents);
    path
}

fn decode(label: &str, name: &str, contents: &str) -> (PathBuf, TileMapData) {
    let path = write_map(name, contents);
    let map = conformance::tilemap(label, || tiled::tilemap_from_path(&path));
    (path, map)
}

/// The map with its directory taken off every path, so maps decoded from two
/// directories compare.
fn relative(mut map: TileMapData, dir: &Path) -> TileMapData {
    let strip = |p: &mut PathBuf| {
        if let Ok(rel) = p.strip_prefix(dir) {
            *p = rel.to_path_buf();
        }
    };
    let strip_image = |i: &mut Option<viewport_lib_io::types::TileImage>| {
        if let Some(TextureSource::File(p)) = i.as_mut().map(|i| &mut i.source) {
            strip(p);
        }
    };
    for set in &mut map.tilesets {
        if let Some(source) = &mut set.source {
            strip(source);
        }
        strip_image(&mut set.image);
        for tile in &mut set.tiles {
            strip_image(&mut tile.image);
        }
    }
    for layer in &mut map.layers {
        if let TileLayerKind::Image { image, .. } = &mut layer.kind {
            strip_image(image);
        }
    }
    if let Some(PropertyValue::File(p)) = map.properties.get_mut("music") {
        strip(p);
    }
    map
}

fn every_encoding() -> Vec<(String, TileMapData)> {
    let mut out = Vec::new();
    for encoding in ["xml", "csv", "base64"] {
        let (path, map) = decode(
            &format!("tmx {encoding}"),
            "level.tmx",
            &synth::tiled_tmx(encoding),
        );
        out.push((
            format!("tmx {encoding}"),
            relative(map, path.parent().unwrap()),
        ));
    }
    for encoding in ["array", "base64"] {
        let (path, map) = decode(
            &format!("tmj {encoding}"),
            "level.tmj",
            &synth::tiled_tmj(encoding),
        );
        out.push((
            format!("tmj {encoding}"),
            relative(map, path.parent().unwrap()),
        ));
    }
    out
}

#[test]
fn every_format_and_encoding_decodes_to_the_same_map() {
    let maps = every_encoding();
    let (first_label, first) = &maps[0];
    for (label, map) in &maps[1..] {
        assert_eq!(map, first, "{label} differs from {first_label}");
    }
}

#[test]
fn the_map_carries_its_grid_settings_and_typed_properties() {
    let (path, map) = decode("tmx", "level.tmx", &synth::tiled_tmx("csv"));
    let dir = path.parent().unwrap();

    assert_eq!(map.size, [3, 2]);
    assert_eq!(map.tile_size, [16, 16]);
    assert_eq!(map.class, "Level");
    assert_eq!(map.background, Some(Colour::srgb_u8(0x33, 0x66, 0x99, 255)));

    let p = &map.properties;
    assert_eq!(p["title"], PropertyValue::String("Cave".into()));
    assert_eq!(p["level"], PropertyValue::Int(3));
    assert_eq!(p["gravity"], PropertyValue::Float(9.5));
    assert_eq!(p["dark"], PropertyValue::Bool(true));
    assert_eq!(
        p["fog"],
        PropertyValue::Colour(Some(Colour::srgb_u8(255, 0, 0, 0x80)))
    );
    assert_eq!(p["music"], PropertyValue::File(dir.join("audio/cave.ogg")));
    assert_eq!(p["door"], PropertyValue::Object(2));
    let PropertyValue::Class { class, members } = &p["spawn"] else {
        panic!("expected a class, got {:?}", p["spawn"]);
    };
    assert_eq!(class, "Spawn");
    assert_eq!(members["count"], PropertyValue::Int(4));
}

#[test]
fn ground_cells_resolve_their_tileset_and_flips() {
    let (_, map) = decode("tmx", "level.tmx", &synth::tiled_tmx("base64"));
    let ground = &map.layers[0];
    assert_eq!(ground.name, "ground");
    assert_eq!(
        (ground.opacity, ground.offset, ground.parallax),
        (0.5, [4.0, -2.0], [0.5, 1.0])
    );
    assert_eq!(ground.tint, Some(Colour::srgb_u8(0, 255, 0, 255)));
    let TileLayerKind::Tiles(grid) = &ground.kind else {
        panic!("expected a tile layer");
    };

    let flipped = |cell: TileCell, h, v, d| TileCell {
        flip_h: h,
        flip_v: v,
        flip_d: d,
        ..cell
    };
    assert_eq!(
        grid.cells,
        [
            TileCell::new(0, 0),
            flipped(TileCell::new(0, 1), true, false, false),
            flipped(TileCell::new(0, 2), false, true, true),
            TileCell::EMPTY,
            TileCell::new(0, 3),
            TileCell::new(1, 0),
        ]
    );
    assert_eq!(grid.get(2, 1), Some(TileCell::new(1, 0)));
}

#[test]
fn tilesets_carry_animation_collision_and_their_images() {
    let (path, map) = decode("tmx", "level.tmx", &synth::tiled_tmx("xml"));
    let dir = path.parent().unwrap();

    let terrain = &map.tilesets[0];
    assert_eq!(
        (terrain.first_gid, terrain.columns, terrain.tile_count),
        (1, 2, 4)
    );
    assert_eq!(
        terrain.image.as_ref().map(|i| (&i.source, i.size)),
        Some((&TextureSource::File(dir.join("terrain.png")), [32, 32]))
    );
    assert_eq!(terrain.tile_rect(3), Some(PixelRect::new(16, 16, 16, 16)));
    assert_eq!(
        terrain.tile(1).unwrap().animation,
        [
            TileFrame {
                index: 1,
                duration_ms: 100
            },
            TileFrame {
                index: 2,
                duration_ms: 150
            }
        ]
    );
    let solid = terrain.tile(0).unwrap();
    assert_eq!(solid.properties["solid"], PropertyValue::Bool(true));
    assert_eq!(solid.collision.len(), 1);
    assert_eq!(
        (solid.collision[0].position, solid.collision[0].size),
        ([0.0, 8.0], [16.0, 8.0])
    );

    // The external tileset: its own file, its images beside it.
    let props = &map.tilesets[1];
    assert_eq!(props.first_gid, 5);
    assert_eq!(
        props.source.as_deref(),
        Some(dir.join("props.tsx").as_path())
    );
    assert!(props.image.is_none(), "an image collection");
    let barrel = props.tile(0).unwrap();
    assert_eq!(barrel.class, "Barrel");
    assert_eq!(
        barrel.image.as_ref().map(|i| &i.source),
        Some(&TextureSource::File(dir.join("art/barrel.png")))
    );
    assert_eq!(props.tile_rect(0), Some(PixelRect::new(0, 0, 16, 24)));
}

#[test]
fn objects_keep_every_shape_and_templates_fill_in() {
    let (_, map) = decode("tmj", "level.tmj", &synth::tiled_tmj("array"));
    let TileLayerKind::Objects {
        colour,
        draw_order,
        objects,
    } = &map.layers[1].kind
    else {
        panic!("expected an object layer");
    };
    assert_eq!(*draw_order, ObjectDrawOrder::Index);
    assert_eq!(*colour, Some(Colour::srgb_u8(0xa0, 0xa0, 0xa4, 255)));

    let by_name = |n: &str| objects.iter().find(|o| o.name == n).unwrap();
    let box_ = by_name("box");
    assert_eq!((box_.class.as_str(), box_.rotation), ("Trigger", 45.0));
    assert_eq!(box_.shape, MapObjectShape::Rectangle);
    assert_eq!(box_.properties["target"], PropertyValue::Object(3));
    assert_eq!(by_name("pond").shape, MapObjectShape::Ellipse);
    assert_eq!(by_name("spawn").shape, MapObjectShape::Point);
    assert_eq!(
        by_name("wall").shape,
        MapObjectShape::Polygon(vec![[0.0, 0.0], [16.0, 0.0], [8.0, 12.0]])
    );
    let path = by_name("path");
    assert!(!path.visible);
    assert_eq!(
        path.shape,
        MapObjectShape::Polyline(vec![[0.0, 0.0], [10.0, 5.5]])
    );
    let MapObjectShape::Text(text) = &by_name("sign").shape else {
        panic!("expected text");
    };
    assert_eq!(
        (
            text.text.as_str(),
            text.font_family.as_str(),
            text.pixel_size
        ),
        ("Hello, cave", "Mono", 10)
    );
    assert!(text.wrap && text.bold && !text.italic);
    assert_eq!(
        (text.h_align, text.v_align),
        (TextAlign::Centre, TextAlign::End)
    );
    assert_eq!(text.colour, Colour::srgb_u8(255, 255, 0, 255));

    let barrel = by_name("barrel");
    assert_eq!(
        barrel.tile,
        Some(TileCell {
            flip_h: true,
            ..TileCell::new(1, 0)
        })
    );

    // The template's crate, from the same external tileset the map uses, so it
    // resolves to that tileset rather than a second copy of it.
    let crate_ = objects.iter().find(|o| o.id == 8).unwrap();
    assert_eq!(
        (crate_.name.as_str(), crate_.class.as_str()),
        ("crate", "Prop")
    );
    assert_eq!((crate_.position, crate_.size), ([40.0, 32.0], [16.0, 16.0]));
    assert_eq!(crate_.tile, Some(TileCell::new(1, 1)));
    assert_eq!(crate_.properties["breakable"], PropertyValue::Bool(true));
    assert_eq!(map.tilesets.len(), 2);
}

#[test]
fn image_and_group_layers_decode() {
    let (_, map) = decode("tmx", "level.tmx", &synth::tiled_tmx("csv"));
    let TileLayerKind::Image { image, repeat } = &map.layers[2].kind else {
        panic!("expected an image layer");
    };
    assert_eq!(*repeat, [true, false]);
    assert_eq!(image.as_ref().map(|i| i.size), Some([64, 32]));

    let decor = &map.layers[3];
    assert!(!decor.visible);
    let TileLayerKind::Group(children) = &decor.kind else {
        panic!("expected a group");
    };
    let TileLayerKind::Tiles(grid) = &children[0].kind else {
        panic!("expected a tile layer in the group");
    };
    assert_eq!(grid.get(2, 1), Some(TileCell::new(0, 0)));
}

#[test]
fn a_missing_external_tileset_names_its_path() {
    let dir = synth::temp_dir("tiled_missing");
    let path = dir.join("level.tmx");
    synth::write(&path, synth::tiled_tmx("csv"));

    match tiled::tilemap_from_path(&path) {
        Err(IoError::MissingDependency(missing)) => assert_eq!(missing, dir.join("props.tsx")),
        other => panic!("expected a missing dependency, got {other:?}"),
    }
}

#[test]
fn bytes_without_a_directory_refuse_external_files() {
    let err = tiled::tilemap_from_bytes(synth::tiled_tmx("csv").as_bytes(), None).unwrap_err();
    assert!(
        matches!(&err, IoError::Parse(m) if m.contains("props.tsx")),
        "{err:?}"
    );
}

#[test]
fn bytes_with_a_directory_decode_like_the_path() {
    let path = write_map("level.tmx", &synth::tiled_tmx("csv"));
    let from_path = tiled::tilemap_from_path(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let from_bytes = tiled::tilemap_from_bytes(&bytes, path.parent()).unwrap();
    assert_eq!(from_bytes, from_path);
}

#[test]
fn a_tileset_reads_on_its_own() {
    let dir = synth::temp_dir("tiled_tsx");
    let path = dir.join("props.tsx");
    synth::write(&path, synth::tiled_tsx());

    let set = tiled::tileset_from_path(&path).unwrap();
    assert_eq!(
        (set.name.as_str(), set.first_gid, set.tile_count),
        ("props", 0, 2)
    );
    assert_eq!(set.source.as_deref(), Some(path.as_path()));
    assert_eq!(
        set.tile(1)
            .and_then(|t| t.image.as_ref())
            .map(|i| &i.source),
        Some(&TextureSource::File(dir.join("art/crate.png")))
    );
}
