//! LDtk conformance: one project with its levels inline and saved separately,
//! both clearing the floor and decoding the same. Then what is peculiar to
//! LDtk: world placement, each layer kind, stacked tiles, entities and fields.

#![cfg(feature = "ldtk")]

use std::path::PathBuf;

use viewport_lib_io::IoError;
use viewport_lib_io::loaders::ldtk;
use viewport_lib_io::types::{
    MapObjectShape, PropertyValue, TextureSource, TileCell, TileLayerKind, TileWorldData,
    WorldLayout,
};
use viewport_lib_io::viewport_lib_types::colour::Colour;
use viewport_lib_io_testkit::{conformance, synth};

fn write_project(external: bool) -> PathBuf {
    let dir = synth::temp_dir("ldtk");
    let (project, files) = synth::ldtk_project(external);
    for (rel, contents) in files {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        synth::write(&path, contents);
    }
    let path = dir.join("game.ldtk");
    synth::write(&path, project);
    path
}

fn decode(external: bool) -> (PathBuf, TileWorldData) {
    let path = write_project(external);
    let mut worlds = conformance::worlds(&format!("ldtk external={external}"), || {
        ldtk::worlds_from_path(&path)
    });
    assert_eq!(worlds.len(), 1, "one world for a single-world project");
    (path, worlds.remove(0))
}

#[test]
fn inline_and_external_levels_decode_the_same() {
    let (inline_path, inline) = decode(false);
    let (external_path, external) = decode(true);

    // Paths differ by directory only.
    let rebase = |world: &TileWorldData, from: &std::path::Path| {
        format!("{world:?}").replace(&from.parent().unwrap().display().to_string(), "<dir>")
    };
    assert_eq!(
        rebase(&inline, &inline_path),
        rebase(&external, &external_path)
    );
}

#[test]
fn levels_are_placed_in_their_world() {
    let (_, world) = decode(false);
    assert_eq!(
        world.layout,
        WorldLayout::GridVania {
            cell_size: [256, 256]
        }
    );
    let placed: Vec<_> = world
        .levels
        .iter()
        .map(|l| (l.name.as_str(), l.iid.as_str(), l.position, l.depth))
        .collect();
    assert_eq!(
        placed,
        [
            ("Level_0", "level-0", [0, 0], 0),
            ("Level_1", "level-1", [256, 0], 1)
        ]
    );
    let map = &world.levels[0].map;
    assert_eq!((map.size, map.tile_size), ([2, 1], [16, 16]));
    assert_eq!(map.background, Some(Colour::srgb_u8(0x11, 0x22, 0x33, 255)));
    assert_eq!(
        map.properties["music"],
        PropertyValue::String("cave".into())
    );
}

#[test]
fn layers_come_out_in_paint_order_with_stacks_split() {
    let (path, world) = decode(false);
    let dir = path.parent().unwrap();
    let map = &world.levels[0].map;
    let names: Vec<_> = map.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "background",
            "Shadows",
            "Ground",
            "Ground",
            "Walls",
            "Walls",
            "Entities"
        ]
    );

    let TileLayerKind::Image { image, .. } = &map.layers[0].kind else {
        panic!("expected the background image");
    };
    assert_eq!(map.layers[0].offset, [-4.0, 2.0]);
    assert_eq!(
        image.as_ref().map(|i| &i.source),
        Some(&TextureSource::File(dir.join("bg/sky.png")))
    );

    let shadows = &map.layers[1];
    assert!(!shadows.visible);
    let TileLayerKind::Tiles(grid) = &shadows.kind else {
        panic!("expected tiles");
    };
    assert_eq!(
        grid.cells[0],
        TileCell {
            flip_h: true,
            flip_v: true,
            ..TileCell::new(0, 7)
        }
    );

    let (ground, stacked) = (&map.layers[2], &map.layers[3]);
    assert_eq!(ground.parallax, [0.5, 1.0]);
    assert_eq!(ground.offset, [4.0, -2.0]);
    let (TileLayerKind::Tiles(lower), TileLayerKind::Tiles(upper)) = (&ground.kind, &stacked.kind)
    else {
        panic!("expected two tile grids");
    };
    assert_eq!(
        lower.cells,
        [
            TileCell::new(0, 0),
            TileCell {
                flip_h: true,
                ..TileCell::new(0, 1)
            }
        ]
    );
    assert_eq!(
        upper.cells,
        [
            TileCell::EMPTY,
            TileCell {
                flip_v: true,
                ..TileCell::new(0, 3)
            }
        ]
    );

    let TileLayerKind::IntGrid(walls) = &map.layers[4].kind else {
        panic!("expected an IntGrid");
    };
    assert_eq!(map.layers[4].opacity, 0.5);
    assert_eq!((walls.size, walls.cell_size), ([4, 2], [8, 8]));
    assert_eq!(walls.values, [1, 1, 0, 0, 0, 2, 2, 0]);
    let defs: Vec<_> = walls
        .definitions
        .iter()
        .map(|d| (d.value, d.name.as_str()))
        .collect();
    assert_eq!(defs, [(1, "wall"), (2, "water")]);
    let TileLayerKind::Tiles(auto) = &map.layers[5].kind else {
        panic!("expected the IntGrid's auto tiles");
    };
    assert_eq!(auto.cell_size, [8, 8]);
    assert_eq!(auto.cells[0], TileCell::new(0, 5));
}

#[test]
fn entities_become_objects_with_their_fields() {
    let (_, world) = decode(false);
    let TileLayerKind::Objects { objects, .. } = &world.levels[0].map.layers[6].kind else {
        panic!("expected the entity layer");
    };
    let player = &objects[0];
    assert_eq!(
        (player.class.as_str(), player.name.as_str()),
        ("Player", "p-1")
    );
    assert_eq!(
        player.position,
        [0.0, 0.0],
        "top-left, from a bottom-centre pivot"
    );
    assert_eq!(player.size, [16.0, 16.0]);
    assert_eq!(player.shape, MapObjectShape::Rectangle);
    assert_eq!(player.tile, Some(TileCell::new(0, 6)));
    assert_eq!(player.properties["hp"], PropertyValue::Int(3));
    assert_eq!(
        player.properties["target"],
        PropertyValue::String("c-1".into())
    );
    assert_eq!(objects[1].position, [16.0, 0.0]);
    assert_eq!(objects[1].tile, None);
}

#[test]
fn the_tileset_carries_its_grid_and_tile_data() {
    let (path, world) = decode(false);
    let set = &world.levels[0].map.tilesets[0];
    assert_eq!(set.name, "Terrain");
    assert_eq!(
        (set.tile_size, set.columns, set.tile_count),
        ([16, 16], 4, 8)
    );
    assert_eq!(
        set.image.as_ref().map(|i| (&i.source, i.size)),
        Some((
            &TextureSource::File(path.parent().unwrap().join("atlas/terrain.png")),
            [64, 32]
        ))
    );
    assert_eq!(
        set.tile(1).unwrap().properties["customData"],
        PropertyValue::String("grass".into())
    );
    assert_eq!(
        set.tile(2).unwrap().properties["enumTags"],
        PropertyValue::List(vec![PropertyValue::String("Solid".into())])
    );
}

#[test]
fn a_missing_level_file_names_its_path() {
    let path = write_project(true);
    let dir = path.parent().unwrap();
    std::fs::remove_file(dir.join("game/Level_1.ldtkl")).unwrap();
    match ldtk::worlds_from_path(&path) {
        Err(IoError::MissingDependency(missing)) => {
            assert_eq!(missing, dir.join("game/Level_1.ldtkl"))
        }
        other => panic!("expected a missing dependency, got {other:?}"),
    }
}

#[test]
fn external_levels_from_bytes_need_a_directory() {
    let (project, _) = synth::ldtk_project(true);
    let err = ldtk::worlds_from_bytes(project.as_bytes(), None).unwrap_err();
    assert!(
        matches!(&err, IoError::Parse(m) if m.contains("Level_0.ldtkl")),
        "{err:?}"
    );
}
