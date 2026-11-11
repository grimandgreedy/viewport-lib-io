//! The walk from a Tiled element tree to the neutral tile map types.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::attributes::{
    bool_attr, class_attr, colour_attr, f32_attr, i32_attr, properties, resolve, string_attr,
    u32_attr,
};
use super::buffers::tile_ids;
use super::raw::{self, Element};
use crate::error::IoError;
use crate::types::{
    MapObject, MapObjectShape, MapText, ObjectDrawOrder, PixelRect, StaggerAxis, StaggerIndex,
    TextAlign, TextureSource, TileCell, TileData, TileFrame, TileGrid, TileImage, TileLayerData,
    TileLayerKind, TileMapData, TileOrientation, TileRenderOrder, TileSetData,
};

const FLIP_H: u32 = 0x8000_0000;
const FLIP_V: u32 = 0x4000_0000;
const FLIP_D: u32 = 0x2000_0000;
const ROTATE_120: u32 = 0x1000_0000;
const FLAGS: u32 = FLIP_H | FLIP_V | FLIP_D | ROTATE_120;

/// Split a global tile id into a cell: the tileset whose range holds it, the
/// index within that tileset, and the flip bits.
///
/// `first_gids` lists each tileset's first global id in tileset order; a `0`
/// entry belongs to a tileset no gid in the map refers to. Id `0` is the empty
/// cell, whatever its flag bits. `None` when no tileset's range holds the id.
pub(super) fn cell_from_gid(gid: u32, first_gids: &[u32]) -> Option<TileCell> {
    let id = gid & !FLAGS;
    if id == 0 {
        return Some(TileCell::EMPTY);
    }
    let (tileset, first) = first_gids
        .iter()
        .enumerate()
        .filter(|(_, first)| **first != 0 && **first <= id)
        .max_by_key(|(_, first)| **first)?;
    Some(TileCell {
        tileset: tileset as u32,
        index: id - first,
        flip_h: gid & FLIP_H != 0,
        flip_v: gid & FLIP_V != 0,
        flip_d: gid & FLIP_D != 0,
        rotate_120: gid & ROTATE_120 != 0,
    })
}

/// The object a template instance stands for: the template's object with the
/// instance's own attributes and children laid over it.
///
/// Properties merge by name, the instance's winning. A shape child on the
/// instance replaces the template's shape, since an object has one.
pub(super) fn merge_template(template: &Element, instance: &Element) -> Element {
    const SHAPES: [&str; 5] = ["ellipse", "point", "polygon", "polyline", "text"];

    let mut merged = template.clone();
    merged.name = "object".into();
    for (key, value) in &instance.attrs {
        if key != "template" {
            merged.set_attr(key, value.clone());
        }
    }
    for child in &instance.children {
        if child.name == "properties" {
            let own = merged.children.iter_mut().find(|c| c.name == "properties");
            match own {
                Some(own) => {
                    for property in &child.children {
                        let name = property.attr("name");
                        own.children.retain(|p| p.attr("name") != name);
                        own.children.push(property.clone());
                    }
                }
                None => merged.children.push(child.clone()),
            }
        } else {
            if SHAPES.contains(&child.name.as_str()) {
                merged
                    .children
                    .retain(|c| !SHAPES.contains(&c.name.as_str()));
            } else {
                merged.children.retain(|c| c.name != child.name);
            }
            merged.children.push(child.clone());
        }
    }
    merged
}

/// A loaded template: its object, and the tileset its gid counts from.
struct Template {
    object: Element,
    /// The template's tileset as (its first gid in the template, its index in
    /// the map's tilesets).
    tileset: Option<(u32, u32)>,
}

/// State for one map: where its files are, and the tilesets and templates
/// read so far.
struct Walk {
    base: Option<PathBuf>,
    tilesets: Vec<TileSetData>,
    templates: HashMap<PathBuf, Template>,
}

impl Walk {
    fn first_gids(&self) -> Vec<u32> {
        self.tilesets.iter().map(|t| t.first_gid).collect()
    }

    fn cell(&self, gid: u32) -> Result<TileCell, IoError> {
        cell_from_gid(gid, &self.first_gids()).ok_or_else(|| {
            IoError::Parse(format!(
                "Tiled map: tile id {} is in no tileset's range",
                gid & !FLAGS
            ))
        })
    }

    /// Read a file another one names, relative to `base`.
    fn read(
        &self,
        base: Option<&Path>,
        path: &str,
        what: &str,
    ) -> Result<(PathBuf, Element), IoError> {
        if base.is_none() {
            return Err(IoError::Parse(format!(
                "Tiled map: the {what} `{path}` is in another file, and there is no directory to \
                 find it in; read the map from a path or pass its directory"
            )));
        }
        let full = resolve(base, path);
        let bytes = std::fs::read(&full).map_err(|_| IoError::MissingDependency(full.clone()))?;
        Ok((full, raw::parse(&bytes)?))
    }

    /// The index in the map's tilesets of the external tileset at `path`,
    /// reading it and adding it with no gid range if the map does not have it.
    fn tileset_by_source(&mut self, base: Option<&Path>, source: &str) -> Result<u32, IoError> {
        let full = resolve(base, source);
        if let Some(i) = self
            .tilesets
            .iter()
            .position(|t| t.source.as_deref() == Some(full.as_path()))
        {
            return Ok(i as u32);
        }
        let (full, element) = self.read(base, source, "tileset")?;
        let set = tileset(self, &element, 0, Some(full))?;
        self.tilesets.push(set);
        Ok(self.tilesets.len() as u32 - 1)
    }

    fn template(&mut self, path: &str) -> Result<&Template, IoError> {
        let full = resolve(self.base.as_deref(), path);
        if !self.templates.contains_key(&full) {
            let base = self.base.clone();
            let (full, element) = self.read(base.as_deref(), path, "object template")?;
            let dir = full.parent().map(Path::to_path_buf);
            let tileset = match element.child("tileset") {
                Some(set) => {
                    let first = u32_attr(set, "firstgid", 1);
                    let source = set.attr("source").ok_or_else(|| {
                        IoError::Parse(format!(
                            "Tiled template `{path}`: its tileset has no source"
                        ))
                    })?;
                    Some((first, self.tileset_by_source(dir.as_deref(), source)?))
                }
                None => None,
            };
            let object = element
                .child("object")
                .cloned()
                .ok_or_else(|| IoError::Parse(format!("Tiled template `{path}` has no object")))?;
            self.templates
                .insert(full.clone(), Template { object, tileset });
        }
        Ok(&self.templates[&full])
    }
}

/// Decode a map element. `base` is the map's directory, when known.
pub(super) fn map(root: &Element, base: Option<&Path>) -> Result<TileMapData, IoError> {
    if root.name != "map" {
        return Err(IoError::Parse(format!(
            "expected a Tiled map, found `{}`",
            root.name
        )));
    }
    let mut walk = Walk {
        base: base.map(Path::to_path_buf),
        tilesets: Vec::new(),
        templates: HashMap::new(),
    };

    for set in root.children_named("tileset") {
        let first_gid = u32_attr(set, "firstgid", 1);
        let tileset = match set.attr("source") {
            Some(source) => {
                let (full, element) = walk.read(base, source, "tileset")?;
                tileset(&mut walk, &element, first_gid, Some(full))?
            }
            None => tileset(&mut walk, set, first_gid, None)?,
        };
        walk.tilesets.push(tileset);
    }

    let layers = layers(&mut walk, root)?;
    let axis = match root.attr("staggeraxis") {
        Some("x") => StaggerAxis::X,
        _ => StaggerAxis::Y,
    };
    let index = match root.attr("staggerindex") {
        Some("even") => StaggerIndex::Even,
        _ => StaggerIndex::Odd,
    };

    Ok(TileMapData {
        size: [u32_attr(root, "width", 0), u32_attr(root, "height", 0)],
        tile_size: [
            u32_attr(root, "tilewidth", 0),
            u32_attr(root, "tileheight", 0),
        ],
        orientation: match root.attr("orientation") {
            Some("isometric") => TileOrientation::Isometric,
            Some("staggered") => TileOrientation::Staggered { axis, index },
            Some("hexagonal") => TileOrientation::Hexagonal {
                axis,
                index,
                side_length: u32_attr(root, "hexsidelength", 0),
            },
            _ => TileOrientation::Orthogonal,
        },
        render_order: match root.attr("renderorder") {
            Some("right-up") => TileRenderOrder::RightUp,
            Some("left-down") => TileRenderOrder::LeftDown,
            Some("left-up") => TileRenderOrder::LeftUp,
            _ => TileRenderOrder::RightDown,
        },
        infinite: bool_attr(root, "infinite", false),
        background: colour_attr(root, "backgroundcolor"),
        parallax_origin: [
            f32_attr(root, "parallaxoriginx", 0.0),
            f32_attr(root, "parallaxoriginy", 0.0),
        ],
        class: class_attr(root),
        properties: properties(root, base),
        layers,
        tilesets: walk.tilesets,
    })
}

/// Decode a tileset file read on its own.
pub(super) fn standalone_tileset(root: &Element, path: &Path) -> Result<TileSetData, IoError> {
    if root.name != "tileset" {
        return Err(IoError::Parse(format!(
            "expected a Tiled tileset, found `{}`",
            root.name
        )));
    }
    let mut walk = Walk {
        base: path.parent().map(Path::to_path_buf),
        tilesets: Vec::new(),
        templates: HashMap::new(),
    };
    tileset(&mut walk, root, 0, Some(path.to_path_buf()))
}

fn tileset(
    walk: &mut Walk,
    element: &Element,
    first_gid: u32,
    source: Option<PathBuf>,
) -> Result<TileSetData, IoError> {
    // An external tileset's paths are relative to its own file.
    let own_base = source
        .as_deref()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let base = own_base
        .as_deref()
        .or(walk.base.as_deref())
        .map(Path::to_path_buf);
    let base = base.as_deref();

    let mut tiles = Vec::new();
    for tile in element.children_named("tile") {
        let rect = tile.attr("width").map(|_| {
            PixelRect::new(
                u32_attr(tile, "x", 0),
                u32_attr(tile, "y", 0),
                u32_attr(tile, "width", 0),
                u32_attr(tile, "height", 0),
            )
        });
        let collision = match tile.child("objectgroup") {
            Some(group) => objects(walk, group, base)?,
            None => Vec::new(),
        };
        tiles.push(TileData {
            index: u32_attr(tile, "id", 0),
            class: class_attr(tile),
            probability: f32_attr(tile, "probability", 1.0),
            image: tile.child("image").and_then(|i| image(i, base)),
            rect,
            animation: tile
                .child("animation")
                .map(|a| {
                    a.children_named("frame")
                        .map(|f| TileFrame {
                            index: u32_attr(f, "tileid", 0),
                            duration_ms: u32_attr(f, "duration", 0),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            collision,
            properties: properties(tile, base),
        });
    }
    tiles.sort_by_key(|t| t.index);

    let offset = element.child("tileoffset");
    Ok(TileSetData {
        name: string_attr(element, "name"),
        class: class_attr(element),
        first_gid,
        source,
        tile_size: [
            u32_attr(element, "tilewidth", 0),
            u32_attr(element, "tileheight", 0),
        ],
        tile_count: u32_attr(element, "tilecount", 0),
        columns: u32_attr(element, "columns", 0),
        spacing: u32_attr(element, "spacing", 0),
        margin: u32_attr(element, "margin", 0),
        tile_offset: offset.map_or([0, 0], |o| [i32_attr(o, "x", 0), i32_attr(o, "y", 0)]),
        image: element.child("image").and_then(|i| image(i, base)),
        tiles,
        properties: properties(element, base),
    })
}

/// An `<image>`, or `None` for one with no file. Pixels embedded in the
/// element itself are not read.
fn image(element: &Element, base: Option<&Path>) -> Option<TileImage> {
    let source = element.attr("source").filter(|s| !s.is_empty())?;
    Some(TileImage {
        source: TextureSource::File(resolve(base, source)),
        size: [
            u32_attr(element, "width", 0),
            u32_attr(element, "height", 0),
        ],
        transparent: colour_attr(element, "trans"),
    })
}

fn layers(walk: &mut Walk, parent: &Element) -> Result<Vec<TileLayerData>, IoError> {
    let mut out = Vec::new();
    for child in &parent.children {
        let base = walk.base.clone();
        let base = base.as_deref();
        let kind = match child.name.as_str() {
            "layer" => TileLayerKind::Tiles(grid(walk, child)?),
            "objectgroup" => TileLayerKind::Objects {
                colour: colour_attr(child, "color"),
                draw_order: match child.attr("draworder") {
                    Some("index") => ObjectDrawOrder::Index,
                    _ => ObjectDrawOrder::TopDown,
                },
                objects: objects(walk, child, base)?,
            },
            "imagelayer" => TileLayerKind::Image {
                image: child.child("image").and_then(|i| image(i, base)),
                repeat: [
                    bool_attr(child, "repeatx", false),
                    bool_attr(child, "repeaty", false),
                ],
            },
            "group" => TileLayerKind::Group(layers(walk, child)?),
            _ => continue,
        };
        out.push(TileLayerData {
            id: u32_attr(child, "id", 0),
            name: string_attr(child, "name"),
            class: class_attr(child),
            offset: [
                f32_attr(child, "offsetx", 0.0),
                f32_attr(child, "offsety", 0.0),
            ],
            opacity: f32_attr(child, "opacity", 1.0),
            visible: bool_attr(child, "visible", true),
            parallax: [
                f32_attr(child, "parallaxx", 1.0),
                f32_attr(child, "parallaxy", 1.0),
            ],
            tint: colour_attr(child, "tintcolor"),
            properties: properties(child, base),
            kind,
        });
    }
    Ok(out)
}

/// A tile layer's cells. An infinite map stores them as chunks, which are
/// gathered into one grid over the area they cover.
fn grid(walk: &Walk, layer: &Element) -> Result<TileGrid, IoError> {
    let Some(data) = layer.child("data") else {
        return Ok(TileGrid::default());
    };
    let (encoding, compression) = (data.attr("encoding"), data.attr("compression"));
    let name = layer.attr("name").unwrap_or_default();

    let chunks: Vec<&Element> = data.children_named("chunk").collect();
    if chunks.is_empty() {
        let size = [u32_attr(layer, "width", 0), u32_attr(layer, "height", 0)];
        let ids = tile_ids(data, encoding, compression)?;
        let want = size[0] as usize * size[1] as usize;
        if ids.len() != want {
            return Err(IoError::Parse(format!(
                "Tiled layer `{name}`: {} tile ids for a {}x{} layer",
                ids.len(),
                size[0],
                size[1]
            )));
        }
        let cells = ids
            .iter()
            .map(|gid| walk.cell(*gid))
            .collect::<Result<_, _>>()?;
        return Ok(TileGrid {
            origin: [0, 0],
            size,
            cells,
        });
    }

    let rect = |c: &Element| {
        [
            i32_attr(c, "x", 0),
            i32_attr(c, "y", 0),
            i32_attr(c, "width", 0),
            i32_attr(c, "height", 0),
        ]
    };
    let (mut lo, mut hi) = ([i32::MAX; 2], [i32::MIN; 2]);
    for chunk in &chunks {
        let [x, y, w, h] = rect(chunk);
        lo = [lo[0].min(x), lo[1].min(y)];
        hi = [hi[0].max(x + w), hi[1].max(y + h)];
    }
    let size = [(hi[0] - lo[0]).max(0) as u32, (hi[1] - lo[1]).max(0) as u32];
    let mut cells = vec![TileCell::EMPTY; size[0] as usize * size[1] as usize];
    for chunk in chunks {
        let [x, y, w, h] = rect(chunk);
        let ids = tile_ids(chunk, encoding, compression)?;
        if ids.len() != (w.max(0) * h.max(0)) as usize {
            return Err(IoError::Parse(format!(
                "Tiled layer `{name}`: {} tile ids for a {w}x{h} chunk",
                ids.len()
            )));
        }
        for (i, gid) in ids.iter().enumerate() {
            let (cx, cy) = (x - lo[0] + i as i32 % w, y - lo[1] + i as i32 / w);
            cells[cy as usize * size[0] as usize + cx as usize] = walk.cell(*gid)?;
        }
    }
    Ok(TileGrid {
        origin: lo,
        size,
        cells,
    })
}

fn objects(
    walk: &mut Walk,
    group: &Element,
    base: Option<&Path>,
) -> Result<Vec<MapObject>, IoError> {
    group
        .children_named("object")
        .map(|o| object(walk, o, base))
        .collect()
}

fn object(walk: &mut Walk, element: &Element, base: Option<&Path>) -> Result<MapObject, IoError> {
    let mut template_tileset = None;
    let merged;
    let element = match element.attr("template") {
        Some(path) => {
            let template = walk.template(path)?;
            if element.attr("gid").is_none() && template.object.attr("gid").is_some() {
                template_tileset = template.tileset;
            }
            merged = merge_template(&template.object, element);
            &merged
        }
        None => element,
    };

    let tile = match element.attr("gid") {
        Some(gid) => {
            let gid = gid.trim().parse::<f64>().map_or(0, |g| g as u32);
            Some(match template_tileset {
                Some((first, tileset)) => {
                    let mut cell = cell_from_gid(gid, &[first]).unwrap_or(TileCell::EMPTY);
                    if !cell.is_empty() {
                        cell.tileset = tileset;
                    }
                    cell
                }
                None => walk.cell(gid)?,
            })
        }
        None => None,
    };

    let points = |e: &Element| -> Vec<[f32; 2]> {
        e.attr("points")
            .unwrap_or_default()
            .split_whitespace()
            .filter_map(|p| {
                let (x, y) = p.split_once(',')?;
                Some([x.trim().parse().ok()?, y.trim().parse().ok()?])
            })
            .collect()
    };
    let shape = if element.child("ellipse").is_some() {
        MapObjectShape::Ellipse
    } else if element.child("point").is_some() {
        MapObjectShape::Point
    } else if let Some(p) = element.child("polygon") {
        MapObjectShape::Polygon(points(p))
    } else if let Some(p) = element.child("polyline") {
        MapObjectShape::Polyline(points(p))
    } else if let Some(t) = element.child("text") {
        MapObjectShape::Text(text(t))
    } else {
        MapObjectShape::Rectangle
    };

    Ok(MapObject {
        id: u32_attr(element, "id", 0),
        name: string_attr(element, "name"),
        class: class_attr(element),
        position: [f32_attr(element, "x", 0.0), f32_attr(element, "y", 0.0)],
        size: [
            f32_attr(element, "width", 0.0),
            f32_attr(element, "height", 0.0),
        ],
        rotation: f32_attr(element, "rotation", 0.0),
        visible: bool_attr(element, "visible", true),
        shape,
        tile,
        properties: properties(element, base),
    })
}

fn text(element: &Element) -> MapText {
    let defaults = MapText::default();
    let align = |key: &str| match element.attr(key) {
        Some("center") => TextAlign::Centre,
        Some("right") | Some("bottom") => TextAlign::End,
        Some("justify") => TextAlign::Justify,
        _ => TextAlign::Start,
    };
    MapText {
        text: element.text.clone(),
        font_family: element
            .attr("fontfamily")
            .map_or(defaults.font_family, str::to_string),
        pixel_size: u32_attr(element, "pixelsize", defaults.pixel_size),
        wrap: bool_attr(element, "wrap", false),
        colour: colour_attr(element, "color").unwrap_or(defaults.colour),
        bold: bool_attr(element, "bold", false),
        italic: bool_attr(element, "italic", false),
        underline: bool_attr(element, "underline", false),
        strikeout: bool_attr(element, "strikeout", false),
        kerning: bool_attr(element, "kerning", true),
        h_align: align("halign"),
        v_align: align("valign"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::tiled::raw::parse;

    #[test]
    fn every_flip_combination_decodes_and_clears_from_the_id() {
        for bits in 0..16u32 {
            let gid = 5 | bits << 28;
            let cell = cell_from_gid(gid, &[1]).unwrap();
            assert_eq!(cell.index, 4, "gid {gid:#x}");
            assert_eq!(cell.rotate_120, bits & 1 != 0);
            assert_eq!(cell.flip_d, bits & 2 != 0);
            assert_eq!(cell.flip_v, bits & 4 != 0);
            assert_eq!(cell.flip_h, bits & 8 != 0);
        }
    }

    #[test]
    fn a_gid_resolves_to_the_tileset_whose_range_holds_it() {
        // Out of order on purpose, plus a template-only tileset with no range.
        let first = [33, 1, 0, 17];
        assert_eq!(cell_from_gid(1, &first), Some(TileCell::new(1, 0)));
        assert_eq!(cell_from_gid(16, &first), Some(TileCell::new(1, 15)));
        assert_eq!(cell_from_gid(17, &first), Some(TileCell::new(3, 0)));
        assert_eq!(cell_from_gid(40, &first), Some(TileCell::new(0, 7)));
        assert_eq!(cell_from_gid(0, &first), Some(TileCell::EMPTY));
        assert_eq!(cell_from_gid(FLIP_H, &first), Some(TileCell::EMPTY));
        assert_eq!(cell_from_gid(3, &[10]), None, "below every range");
    }

    #[test]
    fn a_template_instance_overrides_attributes_properties_and_shape() {
        let template = parse(
            br#"<object name="crate" type="Prop" width="16" height="16">
  <properties><property name="hp" type="int" value="3"/><property name="loot" value="coin"/></properties>
  <ellipse/>
</object>"#,
        )
        .unwrap();
        let instance = parse(
            br#"<object id="9" template="crate.tx" x="40" y="8" width="32">
  <properties><property name="hp" type="int" value="10"/></properties>
  <point/>
</object>"#,
        )
        .unwrap();

        let merged = merge_template(&template, &instance);

        assert_eq!(merged.attr("template"), None);
        assert_eq!(merged.attr("name"), Some("crate"), "kept from the template");
        assert_eq!(merged.attr("width"), Some("32"), "the instance wins");
        assert_eq!(merged.attr("height"), Some("16"));
        assert_eq!(merged.attr("id"), Some("9"));
        assert!(merged.child("point").is_some());
        assert!(merged.child("ellipse").is_none(), "one shape per object");
        let props = properties(&merged, None);
        assert_eq!(props["hp"], crate::types::PropertyValue::Int(10));
        assert_eq!(
            props["loot"],
            crate::types::PropertyValue::String("coin".into())
        );
    }

    #[test]
    fn infinite_chunks_gather_into_one_grid() {
        let root = parse(
            br#"<map width="4" height="4" tilewidth="8" tileheight="8" infinite="1">
  <tileset firstgid="1" name="t" tilewidth="8" tileheight="8" tilecount="4" columns="2"/>
  <layer id="1" name="ground" width="4" height="4">
    <data encoding="csv">
      <chunk x="-2" y="0" width="2" height="1">1,2</chunk>
      <chunk x="2" y="1" width="1" height="1">3</chunk>
    </data>
  </layer>
</map>"#,
        )
        .unwrap();
        let map = map(&root, None).unwrap();
        let TileLayerKind::Tiles(grid) = &map.layers[0].kind else {
            panic!("expected a tile layer");
        };
        assert!(map.infinite);
        assert_eq!(grid.origin, [-2, 0]);
        assert_eq!(grid.size, [5, 2]);
        assert_eq!(grid.get(-2, 0), Some(TileCell::new(0, 0)));
        assert_eq!(grid.get(-1, 0), Some(TileCell::new(0, 1)));
        assert_eq!(grid.get(2, 1), Some(TileCell::new(0, 2)));
        assert_eq!(grid.get(0, 0), Some(TileCell::EMPTY), "between chunks");
        assert_eq!(grid.get(3, 0), None, "outside the grid");
    }

    #[test]
    fn a_layer_with_the_wrong_cell_count_is_an_error() {
        let root = parse(
            br#"<map width="2" height="2"><tileset firstgid="1" name="t"/>
  <layer name="short" width="2" height="2"><data encoding="csv">1,1,1</data></layer></map>"#,
        )
        .unwrap();
        let err = map(&root, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("short")),
            "{err:?}"
        );
    }

    #[test]
    fn an_external_tileset_without_a_directory_is_an_error_naming_it() {
        let root = parse(br#"<map><tileset firstgid="1" source="tiles.tsx"/></map>"#).unwrap();
        let err = map(&root, None).unwrap_err();
        assert!(
            matches!(&err, IoError::Parse(m) if m.contains("tiles.tsx")),
            "{err:?}"
        );
    }

    #[test]
    fn atlas_tile_rects_follow_margin_spacing_and_columns() {
        let root = parse(
            br#"<tileset name="t" tilewidth="16" tileheight="8" tilecount="6" columns="3" spacing="2" margin="1">
  <image source="t.png" width="54" height="20"/>
  <tile id="4" x="0" y="0" width="5" height="5"/>
</tileset>"#,
        )
        .unwrap();
        let set = standalone_tileset(&root, Path::new("t.tsx")).unwrap();
        assert_eq!(set.tile_rect(0), Some(PixelRect::new(1, 1, 16, 8)));
        assert_eq!(set.tile_rect(5), Some(PixelRect::new(37, 11, 16, 8)));
        assert_eq!(
            set.tile_rect(4),
            Some(PixelRect::new(0, 0, 5, 5)),
            "its own rect"
        );
        assert_eq!(set.tile_rect(6), None, "past the tile count");
        assert_eq!(
            PixelRect::new(1, 1, 16, 8).uv([54, 20]),
            [1.0 / 54.0, 0.05, 17.0 / 54.0, 0.45]
        );
    }
}
