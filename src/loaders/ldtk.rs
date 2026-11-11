//! LDtk project decoding.
//!
//! An LDtk project is one JSON file holding every level, or a project file
//! plus one `.ldtkl` file per level when it saves levels separately. Each level
//! becomes a [`TileMapData`](crate::types::TileMapData) in its world, with the project's tilesets.
//!
//! Layers map onto the tile map types like this:
//!
//! - `Tiles` and `AutoLayer` layers become tile layers over their tileset's
//!   grid. LDtk can stack several tiles in one cell; each further tile in a
//!   cell goes to another tile layer of the same name painted just above.
//! - `IntGrid` layers become [`TileLayerKind::IntGrid`](crate::types::TileLayerKind::IntGrid), with the project's
//!   value names and colours. An IntGrid layer that also draws auto-layer tiles
//!   is followed by tile layers of the same name holding them.
//! - `Entities` layers become object layers. Each entity is a rectangle
//!   object whose class is the entity's identifier and whose name is its
//!   `iid`, at its top-left corner (LDtk stores the pivot point), with its
//!   fields as properties. An entity's tile is the top-left tile of its tile
//!   rectangle.
//! - A level's background image becomes an image layer painted first, at its
//!   top-left position; its scale and crop are not carried.
//!
//! LDtk lists layers top first; the output is in paint order, so they are
//! reversed. LDtk's parallax factor (`0` moves with the map) is converted to
//! the tile map's scroll factor (`1` moves with the map).
//!
//! Field values map to [`PropertyValue`](crate::types::PropertyValue): points
//! and tile rectangles as classes named `Point` and `Tile`, entity references
//! as the referenced entity's `iid`, enum values as their names, arrays as
//! lists. A field with no value is left out. Per-tile opacity and LDtk's rule
//! definitions are not carried.

use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};
use viewport_lib_types::colour::Colour;

use crate::error::IoError;
use crate::types::{
    IntGridData, IntGridValue, MapObject, MapObjectShape, ObjectDrawOrder, Properties,
    PropertyValue, TextureSource, TileCell, TileData, TileGrid, TileImage, TileLayerData,
    TileLayerKind, TileMapData, TileSetData, TileWorldData, TileWorldLevel, WorldLayout,
};

/// Decode an LDtk project (`.ldtk`) into its worlds.
///
/// A project without multiple worlds gives one world holding every level.
/// Levels saved in their own files and every image path are resolved against
/// the project's directory. A level file that cannot be read is an
/// [`IoError::MissingDependency`] naming its path. Images are not read.
pub fn worlds_from_path(path: &Path) -> Result<Vec<TileWorldData>, IoError> {
    #[cfg(feature = "ldtk")]
    {
        let bytes = std::fs::read(path)?;
        worlds_from_bytes(&bytes, Some(path.parent().unwrap_or(Path::new("."))))
    }

    #[cfg(not(feature = "ldtk"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "ldtk",
            context: "LDtk project decoding",
        })
    }
}

/// Decode an LDtk project from bytes already read.
///
/// `base` is the project's directory. Without one, a project that saves its
/// levels in separate files is an error, and image paths are left as the file
/// wrote them.
pub fn worlds_from_bytes(bytes: &[u8], base: Option<&Path>) -> Result<Vec<TileWorldData>, IoError> {
    #[cfg(feature = "ldtk")]
    {
        decode(bytes, base)
    }

    #[cfg(not(feature = "ldtk"))]
    {
        let _ = (bytes, base);
        Err(IoError::MissingFeature {
            feature: "ldtk",
            context: "LDtk project decoding",
        })
    }
}

fn decode(bytes: &[u8], base: Option<&Path>) -> Result<Vec<TileWorldData>, IoError> {
    let root: Value =
        serde_json::from_slice(bytes).map_err(|e| IoError::Parse(format!("LDtk JSON: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| IoError::Parse("LDtk JSON: the project is not an object".into()))?;
    let project = Project::new(root, base)?;

    let worlds = array(root, "worlds");
    if worlds.is_empty() {
        return Ok(vec![project.world("", root)?]);
    }
    worlds
        .iter()
        .filter_map(Value::as_object)
        .map(|w| project.world(str_of(w, "identifier"), w))
        .collect()
}

fn array<'a>(object: &'a Map<String, Value>, key: &str) -> &'a [Value] {
    object
        .get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn str_of<'a>(object: &'a Map<String, Value>, key: &str) -> &'a str {
    object.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn int_of(object: &Map<String, Value>, key: &str) -> i64 {
    object
        .get(key)
        .and_then(Value::as_f64)
        .map_or(0, |v| v as i64)
}

fn float_of(object: &Map<String, Value>, key: &str, default: f64) -> f64 {
    object.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn pair(object: &Map<String, Value>, key: &str) -> [f64; 2] {
    let values = array(object, key);
    let at = |i: usize| values.get(i).and_then(Value::as_f64).unwrap_or(0.0);
    [at(0), at(1)]
}

/// An LDtk colour, `#RRGGBB`, sRGB.
fn parse_colour(text: &str) -> Option<Colour> {
    let hex = text.trim().trim_start_matches('#');
    if hex.len() != 6 {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some(Colour::srgb_u8(byte(0)?, byte(2)?, byte(4)?, 255))
}

/// `path` resolved against `base` when there is one, with `.` and `..` taken
/// out.
fn resolve(base: Option<&Path>, path: &str) -> PathBuf {
    let joined = base.map_or_else(|| PathBuf::from(path), |b| b.join(path));
    let mut out = PathBuf::new();
    for part in joined.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir
                if matches!(out.components().next_back(), Some(Component::Normal(_))) =>
            {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// LDtk's tile flip bits: bit 0 mirrors X, bit 1 mirrors Y.
fn flipped(mut cell: TileCell, bits: i64) -> TileCell {
    cell.flip_h = bits & 1 != 0;
    cell.flip_v = bits & 2 != 0;
    cell
}

/// Convert LDtk's parallax factor, where `0` moves with the map and `1` stays
/// fixed, into a scroll factor where `1` moves with the map.
fn scroll_factor(ldtk: f64) -> f32 {
    (1.0 - ldtk) as f32
}

/// The project-wide definitions every level reads.
struct Project<'a> {
    base: Option<&'a Path>,
    grid_size: u32,
    tilesets: Vec<TileSetData>,
    /// Tileset uid to index in `tilesets`.
    tileset_uids: Vec<i64>,
    layer_defs: Vec<&'a Map<String, Value>>,
}

impl<'a> Project<'a> {
    fn new(root: &'a Map<String, Value>, base: Option<&'a Path>) -> Result<Self, IoError> {
        static EMPTY: std::sync::LazyLock<Map<String, Value>> = std::sync::LazyLock::new(Map::new);
        let defs = root
            .get("defs")
            .and_then(Value::as_object)
            .unwrap_or(&EMPTY);
        let mut tilesets = Vec::new();
        let mut tileset_uids = Vec::new();
        for set in array(defs, "tilesets").iter().filter_map(Value::as_object) {
            tileset_uids.push(int_of(set, "uid"));
            tilesets.push(tileset(set, base));
        }
        Ok(Project {
            base,
            grid_size: int_of(root, "defaultGridSize").max(0) as u32,
            tilesets,
            tileset_uids,
            layer_defs: array(defs, "layers")
                .iter()
                .filter_map(Value::as_object)
                .collect(),
        })
    }

    fn tileset_index(&self, uid: Option<i64>) -> Option<u32> {
        let uid = uid?;
        self.tileset_uids
            .iter()
            .position(|u| *u == uid)
            .map(|i| i as u32)
    }

    fn layer_def(&self, uid: i64) -> Option<&'a Map<String, Value>> {
        self.layer_defs
            .iter()
            .copied()
            .find(|d| int_of(d, "uid") == uid)
    }

    fn world(&self, name: &str, world: &Map<String, Value>) -> Result<TileWorldData, IoError> {
        let layout = match world.get("worldLayout").and_then(Value::as_str) {
            Some("GridVania") => WorldLayout::GridVania {
                cell_size: [
                    int_of(world, "worldGridWidth").max(0) as u32,
                    int_of(world, "worldGridHeight").max(0) as u32,
                ],
            },
            Some("LinearHorizontal") => WorldLayout::LinearHorizontal,
            Some("LinearVertical") => WorldLayout::LinearVertical,
            _ => WorldLayout::Free,
        };
        let levels = array(world, "levels")
            .iter()
            .filter_map(Value::as_object)
            .map(|l| self.level(l))
            .collect::<Result<_, _>>()?;
        Ok(TileWorldData {
            name: name.to_string(),
            layout,
            levels,
        })
    }

    fn level(&self, level: &Map<String, Value>) -> Result<TileWorldLevel, IoError> {
        // A level saved on its own keeps its layers in its own file.
        let external;
        let full = match level.get("externalRelPath").and_then(Value::as_str) {
            Some(rel) if level.get("layerInstances").is_none_or(Value::is_null) => {
                if self.base.is_none() {
                    return Err(IoError::Parse(format!(
                        "LDtk project: level `{}` is in `{rel}`, and there is no directory to \
                         find it in; read the project from a path or pass its directory",
                        str_of(level, "identifier")
                    )));
                }
                let path = resolve(self.base, rel);
                let bytes =
                    std::fs::read(&path).map_err(|_| IoError::MissingDependency(path.clone()))?;
                let value: Value = serde_json::from_slice(&bytes)
                    .map_err(|e| IoError::Parse(format!("LDtk level `{rel}`: {e}")))?;
                external = value;
                external
                    .as_object()
                    .ok_or_else(|| IoError::Parse(format!("LDtk level `{rel}` is not an object")))?
            }
            _ => level,
        };

        let size = [
            int_of(full, "pxWid").max(0) as u32,
            int_of(full, "pxHei").max(0) as u32,
        ];
        let grid = self.grid_size.max(1);

        let mut layers = Vec::new();
        if let Some(bg) = full.get("bgRelPath").and_then(Value::as_str) {
            let pos = full.get("__bgPos").and_then(Value::as_object);
            let top_left = pos.map_or([0.0; 2], |p| pair(p, "topLeftPx"));
            layers.push(TileLayerData {
                id: 0,
                name: "background".into(),
                class: String::new(),
                offset: [top_left[0] as f32, top_left[1] as f32],
                opacity: 1.0,
                visible: true,
                parallax: [1.0, 1.0],
                tint: None,
                properties: Properties::new(),
                kind: TileLayerKind::Image {
                    image: Some(TileImage {
                        source: TextureSource::File(resolve(self.base, bg)),
                        size: [0, 0],
                        transparent: None,
                    }),
                    repeat: [false, false],
                },
            });
        }
        for layer in array(full, "layerInstances")
            .iter()
            .rev()
            .filter_map(Value::as_object)
        {
            layers.extend(self.layer(layer)?);
        }

        Ok(TileWorldLevel {
            name: str_of(full, "identifier").to_string(),
            iid: str_of(full, "iid").to_string(),
            position: [int_of(full, "worldX") as i32, int_of(full, "worldY") as i32],
            depth: int_of(full, "worldDepth") as i32,
            map: TileMapData {
                size: [size[0] / grid, size[1] / grid],
                tile_size: [grid, grid],
                background: full
                    .get("__bgColor")
                    .and_then(Value::as_str)
                    .and_then(parse_colour),
                properties: fields(array(full, "fieldInstances"), self.base),
                layers,
                tilesets: self.tilesets.clone(),
                ..TileMapData::default()
            },
        })
    }

    /// One layer instance, as one layer or, when tiles stack, several.
    fn layer(&self, layer: &Map<String, Value>) -> Result<Vec<TileLayerData>, IoError> {
        let def = self.layer_def(int_of(layer, "layerDefUid"));
        let parallax = def.map_or([1.0, 1.0], |d| {
            [
                scroll_factor(float_of(d, "parallaxFactorX", 0.0)),
                scroll_factor(float_of(d, "parallaxFactorY", 0.0)),
            ]
        });
        let base = |kind| TileLayerData {
            id: int_of(layer, "layerDefUid").max(0) as u32,
            name: str_of(layer, "__identifier").to_string(),
            class: String::new(),
            offset: [
                int_of(layer, "__pxTotalOffsetX") as f32,
                int_of(layer, "__pxTotalOffsetY") as f32,
            ],
            opacity: float_of(layer, "__opacity", 1.0) as f32,
            visible: layer
                .get("visible")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            parallax,
            tint: None,
            properties: Properties::new(),
            kind,
        };
        let size = [
            int_of(layer, "__cWid").max(0) as u32,
            int_of(layer, "__cHei").max(0) as u32,
        ];
        let cell = int_of(layer, "__gridSize").max(1) as u32;
        let tileset = self.tileset_index(
            layer
                .get("overrideTilesetUid")
                .and_then(Value::as_i64)
                .or_else(|| layer.get("__tilesetDefUid").and_then(Value::as_i64)),
        );

        let mut out = Vec::new();
        match str_of(layer, "__type") {
            "Entities" => {
                let objects = array(layer, "entityInstances")
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|e| self.entity(e))
                    .collect();
                out.push(base(TileLayerKind::Objects {
                    colour: None,
                    draw_order: ObjectDrawOrder::Index,
                    objects,
                }));
            }
            "IntGrid" => {
                let values: Vec<u32> = array(layer, "intGridCsv")
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as u32)
                    .collect();
                if values.len() != size[0] as usize * size[1] as usize {
                    return Err(IoError::Parse(format!(
                        "LDtk layer `{}`: {} values for a {}x{} grid",
                        str_of(layer, "__identifier"),
                        values.len(),
                        size[0],
                        size[1]
                    )));
                }
                let definitions = def
                    .map(|d| array(d, "intGridValues"))
                    .unwrap_or_default()
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|v| IntGridValue {
                        value: int_of(v, "value").max(0) as u32,
                        name: str_of(v, "identifier").to_string(),
                        colour: parse_colour(str_of(v, "color"))
                            .unwrap_or(Colour::srgb_u8(0, 0, 0, 255)),
                    })
                    .collect();
                out.push(base(TileLayerKind::IntGrid(IntGridData {
                    size,
                    cell_size: [cell, cell],
                    values,
                    definitions,
                })));
                for grid in tile_stack(array(layer, "autoLayerTiles"), size, cell, tileset) {
                    out.push(base(TileLayerKind::Tiles(grid)));
                }
            }
            "Tiles" | "AutoLayer" => {
                let tiles = if str_of(layer, "__type") == "Tiles" {
                    array(layer, "gridTiles")
                } else {
                    array(layer, "autoLayerTiles")
                };
                let stack = tile_stack(tiles, size, cell, tileset);
                if stack.is_empty() {
                    out.push(base(TileLayerKind::Tiles(TileGrid {
                        origin: [0, 0],
                        size,
                        cell_size: [cell, cell],
                        cells: vec![TileCell::EMPTY; size[0] as usize * size[1] as usize],
                    })));
                }
                for grid in stack {
                    out.push(base(TileLayerKind::Tiles(grid)));
                }
            }
            _ => {}
        }
        Ok(out)
    }

    fn entity(&self, entity: &Map<String, Value>) -> MapObject {
        let size = [
            float_of(entity, "width", 0.0) as f32,
            float_of(entity, "height", 0.0) as f32,
        ];
        let px = pair(entity, "px");
        let pivot = pair(entity, "__pivot");
        let tile = entity
            .get("__tile")
            .and_then(Value::as_object)
            .and_then(|rect| self.tile_from_rect(rect));
        MapObject {
            id: 0,
            name: str_of(entity, "iid").to_string(),
            class: str_of(entity, "__identifier").to_string(),
            position: [
                (px[0] - pivot[0] * size[0] as f64) as f32,
                (px[1] - pivot[1] * size[1] as f64) as f32,
            ],
            size,
            rotation: 0.0,
            visible: true,
            shape: MapObjectShape::Rectangle,
            tile,
            properties: fields(array(entity, "fieldInstances"), self.base),
        }
    }

    /// The cell for the top-left tile of an LDtk tile rectangle.
    fn tile_from_rect(&self, rect: &Map<String, Value>) -> Option<TileCell> {
        let tileset = self.tileset_index(rect.get("tilesetUid").and_then(Value::as_i64))?;
        let set = &self.tilesets[tileset as usize];
        let step = |size: u32| (size + set.spacing).max(1) as i64;
        let col = (int_of(rect, "x") - set.margin as i64) / step(set.tile_size[0]);
        let row = (int_of(rect, "y") - set.margin as i64) / step(set.tile_size[1]);
        Some(TileCell::new(
            tileset,
            (row * set.columns as i64 + col).max(0) as u32,
        ))
    }
}

/// Place tile instances on a grid, a further grid for each tile stacked on a
/// cell already filled. Empty when there are no tiles.
fn tile_stack(tiles: &[Value], size: [u32; 2], cell: u32, tileset: Option<u32>) -> Vec<TileGrid> {
    let mut grids: Vec<TileGrid> = Vec::new();
    let Some(tileset) = tileset else {
        return grids;
    };
    for tile in tiles.iter().filter_map(Value::as_object) {
        let px = pair(tile, "px");
        let (cx, cy) = ((px[0] / cell as f64) as i64, (px[1] / cell as f64) as i64);
        if cx < 0 || cy < 0 || cx >= size[0] as i64 || cy >= size[1] as i64 {
            continue;
        }
        let i = cy as usize * size[0] as usize + cx as usize;
        let value = flipped(
            TileCell::new(tileset, int_of(tile, "t").max(0) as u32),
            int_of(tile, "f"),
        );
        let level = grids.iter().position(|g| g.cells[i].is_empty());
        let grid = match level {
            Some(level) => &mut grids[level],
            None => {
                grids.push(TileGrid {
                    origin: [0, 0],
                    size,
                    cell_size: [cell, cell],
                    cells: vec![TileCell::EMPTY; size[0] as usize * size[1] as usize],
                });
                grids.last_mut().unwrap()
            }
        };
        grid.cells[i] = value;
    }
    grids
}

fn tileset(set: &Map<String, Value>, base: Option<&Path>) -> TileSetData {
    let grid = int_of(set, "tileGridSize").max(0) as u32;
    let (columns, rows) = (
        int_of(set, "__cWid").max(0) as u32,
        int_of(set, "__cHei").max(0) as u32,
    );

    // Per-tile custom data and enum tags, gathered by tile.
    let mut tiles: std::collections::BTreeMap<u32, TileData> = Default::default();
    fn entry(tiles: &mut std::collections::BTreeMap<u32, TileData>, index: u32) -> &mut TileData {
        tiles.entry(index).or_insert_with(|| TileData {
            index,
            probability: 1.0,
            ..TileData::default()
        })
    }
    for data in array(set, "customData").iter().filter_map(Value::as_object) {
        let index = int_of(data, "tileId").max(0) as u32;
        entry(&mut tiles, index).properties.insert(
            "customData".into(),
            PropertyValue::String(str_of(data, "data").to_string()),
        );
    }
    for tag in array(set, "enumTags").iter().filter_map(Value::as_object) {
        let name = str_of(tag, "enumValueId").to_string();
        for id in array(tag, "tileIds") {
            let index = id.as_u64().unwrap_or(0) as u32;
            let tags = entry(&mut tiles, index)
                .properties
                .entry("enumTags".into())
                .or_insert_with(|| PropertyValue::List(Vec::new()));
            if let PropertyValue::List(list) = tags {
                list.push(PropertyValue::String(name.clone()));
            }
        }
    }

    TileSetData {
        name: str_of(set, "identifier").to_string(),
        first_gid: 0,
        tile_size: [grid, grid],
        tile_count: columns * rows,
        columns,
        spacing: int_of(set, "spacing").max(0) as u32,
        margin: int_of(set, "padding").max(0) as u32,
        image: set
            .get("relPath")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty())
            .map(|p| TileImage {
                source: TextureSource::File(resolve(base, p)),
                size: [
                    int_of(set, "pxWid").max(0) as u32,
                    int_of(set, "pxHei").max(0) as u32,
                ],
                transparent: None,
            }),
        tiles: tiles.into_values().collect(),
        ..TileSetData::default()
    }
}

/// Field instances as properties, leaving out fields with no value.
fn fields(list: &[Value], base: Option<&Path>) -> Properties {
    let mut out = Properties::new();
    for field in list.iter().filter_map(Value::as_object) {
        let kind = str_of(field, "__type");
        if let Some(value) = field
            .get("__value")
            .and_then(|v| field_value(kind, v, base))
        {
            out.insert(str_of(field, "__identifier").to_string(), value);
        }
    }
    out
}

/// One field value typed by its LDtk `__type`. `None` for a null.
fn field_value(kind: &str, value: &Value, base: Option<&Path>) -> Option<PropertyValue> {
    if value.is_null() {
        return None;
    }
    if let Some(inner) = kind
        .strip_prefix("Array<")
        .and_then(|k| k.strip_suffix('>'))
    {
        return Some(PropertyValue::List(
            value
                .as_array()?
                .iter()
                .filter_map(|v| field_value(inner, v, base))
                .collect(),
        ));
    }
    let object = |class: &str, keys: &[&str]| -> Option<PropertyValue> {
        let o = value.as_object()?;
        let members = keys
            .iter()
            .map(|k| (k.to_string(), PropertyValue::Int(int_of(o, k))))
            .collect();
        Some(PropertyValue::Class {
            class: class.into(),
            members,
        })
    };
    Some(match kind {
        "Int" => PropertyValue::Int(value.as_f64()? as i64),
        "Float" => PropertyValue::Float(value.as_f64()?),
        "Bool" => PropertyValue::Bool(value.as_bool()?),
        "Color" => PropertyValue::Colour(value.as_str().and_then(parse_colour)),
        "FilePath" => PropertyValue::File(resolve(base, value.as_str()?)),
        "Point" => object("Point", &["cx", "cy"])?,
        "Tile" => object("Tile", &["tilesetUid", "x", "y", "w", "h"])?,
        "EntityRef" => PropertyValue::String(str_of(value.as_object()?, "entityIid").to_string()),
        // String, Multilines, and enum values (`LocalEnum.Name`,
        // `ExternEnum.Name`), which are stored as the value's name.
        _ => PropertyValue::String(value.as_str()?.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parallax_factors_convert_to_scroll_factors() {
        assert_eq!(scroll_factor(0.0), 1.0, "moves with the map");
        assert_eq!(scroll_factor(1.0), 0.0, "fixed on screen");
        assert_eq!(scroll_factor(-0.5), 1.5, "faster than the map");
    }

    #[test]
    fn flip_bits_map_to_horizontal_and_vertical() {
        let cell = TileCell::new(0, 3);
        assert_eq!(flipped(cell, 0), cell);
        assert!(flipped(cell, 1).flip_h && !flipped(cell, 1).flip_v);
        assert!(!flipped(cell, 2).flip_h && flipped(cell, 2).flip_v);
        assert!(flipped(cell, 3).flip_h && flipped(cell, 3).flip_v);
        assert!(!flipped(cell, 3).flip_d);
    }

    #[test]
    fn stacked_tiles_spill_into_further_grids() {
        let tiles = json!([
            {"px": [0, 0], "t": 1, "f": 0},
            {"px": [8, 0], "t": 2, "f": 1},
            {"px": [0, 0], "t": 3, "f": 0},
            {"px": [0, 0], "t": 4, "f": 2},
            {"px": [99, 0], "t": 5, "f": 0}
        ]);
        let grids = tile_stack(tiles.as_array().unwrap(), [2, 1], 8, Some(0));
        assert_eq!(grids.len(), 3);
        assert_eq!(
            grids[0].cells,
            [TileCell::new(0, 1), flipped(TileCell::new(0, 2), 1)]
        );
        assert_eq!(grids[1].cells, [TileCell::new(0, 3), TileCell::EMPTY]);
        assert_eq!(
            grids[2].cells,
            [flipped(TileCell::new(0, 4), 2), TileCell::EMPTY]
        );
        assert!(tile_stack(tiles.as_array().unwrap(), [2, 1], 8, None).is_empty());
    }

    #[test]
    fn field_values_are_typed_and_nulls_left_out() {
        let list = json!([
            {"__identifier": "hp", "__type": "Int", "__value": 5},
            {"__identifier": "speed", "__type": "Float", "__value": 1.5},
            {"__identifier": "boss", "__type": "Bool", "__value": true},
            {"__identifier": "label", "__type": "Multilines", "__value": "a\nb"},
            {"__identifier": "tint", "__type": "Color", "__value": "#FF8000"},
            {"__identifier": "sound", "__type": "FilePath", "__value": "../sfx/hit.wav"},
            {"__identifier": "goal", "__type": "Point", "__value": {"cx": 3, "cy": 4}},
            {"__identifier": "target", "__type": "EntityRef", "__value": {"entityIid": "abc"}},
            {"__identifier": "kind", "__type": "LocalEnum.Kind", "__value": "Fire"},
            {"__identifier": "path", "__type": "Array<Point>", "__value": [{"cx": 1, "cy": 2}]},
            {"__identifier": "unset", "__type": "Int", "__value": null}
        ]);
        let props = fields(list.as_array().unwrap(), Some(Path::new("game/levels")));

        assert_eq!(props["hp"], PropertyValue::Int(5));
        assert_eq!(props["speed"], PropertyValue::Float(1.5));
        assert_eq!(props["boss"], PropertyValue::Bool(true));
        assert_eq!(props["label"], PropertyValue::String("a\nb".into()));
        assert_eq!(
            props["tint"],
            PropertyValue::Colour(Some(Colour::srgb_u8(255, 128, 0, 255)))
        );
        assert_eq!(
            props["sound"],
            PropertyValue::File("game/sfx/hit.wav".into())
        );
        let point = |cx, cy| PropertyValue::Class {
            class: "Point".into(),
            members: [
                ("cx".to_string(), PropertyValue::Int(cx)),
                ("cy".to_string(), PropertyValue::Int(cy)),
            ]
            .into(),
        };
        assert_eq!(props["goal"], point(3, 4));
        assert_eq!(props["target"], PropertyValue::String("abc".into()));
        assert_eq!(props["kind"], PropertyValue::String("Fire".into()));
        assert_eq!(props["path"], PropertyValue::List(vec![point(1, 2)]));
        assert!(!props.contains_key("unset"));
    }
}
