//! Reading values off elements: numbers, flags, colours, paths, and typed
//! custom properties.

use std::path::{Component, Path, PathBuf};

use viewport_lib_types::colour::Colour;

use super::raw::Element;
use crate::types::{Properties, PropertyValue};

pub(super) fn f32_attr(element: &Element, key: &str, default: f32) -> f32 {
    element
        .attr(key)
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// A whole number, read through `f64` because JSON writers sometimes give
/// `16.0` where TMX gives `16`.
pub(super) fn u32_attr(element: &Element, key: &str, default: u32) -> u32 {
    element
        .attr(key)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map_or(default, |v| v as u32)
}

pub(super) fn i32_attr(element: &Element, key: &str, default: i32) -> i32 {
    element
        .attr(key)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map_or(default, |v| v as i32)
}

/// A flag: `1` or `true` in TMX, a boolean in JSON.
pub(super) fn bool_attr(element: &Element, key: &str, default: bool) -> bool {
    match element.attr(key).map(str::trim) {
        Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        _ => default,
    }
}

pub(super) fn string_attr(element: &Element, key: &str) -> String {
    element.attr(key).unwrap_or_default().to_string()
}

/// The class name, which older files write as `type`.
pub(super) fn class_attr(element: &Element) -> String {
    element
        .attr("class")
        .or_else(|| element.attr("type"))
        .unwrap_or_default()
        .to_string()
}

pub(super) fn colour_attr(element: &Element, key: &str) -> Option<Colour> {
    element.attr(key).and_then(parse_colour)
}

/// A Tiled colour: `#AARRGGBB` or `#RRGGBB`, the `#` optional. sRGB.
pub(super) fn parse_colour(text: &str) -> Option<Colour> {
    let hex = text.trim().trim_start_matches('#');
    let byte = |i: usize| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok();
    match hex.len() {
        6 => Some(Colour::srgb_u8(byte(0)?, byte(2)?, byte(4)?, 255)),
        8 => Some(Colour::srgb_u8(byte(2)?, byte(4)?, byte(6)?, byte(0)?)),
        _ => None,
    }
}

/// `path` resolved against `base` when there is one, with `.` and `..` taken
/// out so one file named two ways resolves to one path.
pub(super) fn resolve(base: Option<&Path>, path: &str) -> PathBuf {
    let joined = match base {
        Some(base) => base.join(path),
        None => PathBuf::from(path),
    };
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

/// The custom properties under an element's `<properties>` child.
pub(super) fn properties(element: &Element, base: Option<&Path>) -> Properties {
    let mut out = Properties::new();
    if let Some(list) = element.child("properties") {
        for property in list.children_named("property") {
            let name = string_attr(property, "name");
            out.insert(name, property_value(property, base));
        }
    }
    out
}

/// One `<property>`'s value, typed by its `type`. A string property may hold
/// its value as element text when it spans lines.
fn property_value(property: &Element, base: Option<&Path>) -> PropertyValue {
    let raw = property
        .attr("value")
        .map(str::to_string)
        .unwrap_or_else(|| property.text.clone());
    let text = raw.trim();
    match property.attr("type").unwrap_or("string") {
        "int" => PropertyValue::Int(
            text.parse()
                .or_else(|_| text.parse::<f64>().map(|f| f as i64))
                .unwrap_or(0),
        ),
        "float" => PropertyValue::Float(text.parse().unwrap_or(0.0)),
        "bool" => PropertyValue::Bool(text == "true" || text == "1"),
        "color" => PropertyValue::Colour(parse_colour(text)),
        "file" => PropertyValue::File(resolve(base, text)),
        "object" => PropertyValue::Object(text.parse::<f64>().map_or(0, |v| v as u32)),
        "class" => PropertyValue::Class {
            class: string_attr(property, "propertytype"),
            members: properties(property, base),
        },
        _ => PropertyValue::String(raw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::tiled::raw::parse;

    #[test]
    fn colours_read_with_and_without_alpha() {
        assert_eq!(
            parse_colour("#80ff0000"),
            Some(Colour::srgb_u8(255, 0, 0, 128))
        );
        assert_eq!(
            parse_colour("00ff00"),
            Some(Colour::srgb_u8(0, 255, 0, 255))
        );
        assert_eq!(parse_colour(""), None);
        assert_eq!(parse_colour("#zz0000"), None);
    }

    #[test]
    fn paths_resolve_against_the_base_without_dot_segments() {
        let base = Path::new("maps/level1");
        assert_eq!(
            resolve(Some(base), "../tiles/./grass.png"),
            PathBuf::from("maps/tiles/grass.png")
        );
        assert_eq!(resolve(None, "grass.png"), PathBuf::from("grass.png"));
        assert_eq!(resolve(None, "../grass.png"), PathBuf::from("../grass.png"));
    }

    #[test]
    fn every_property_type_is_typed() {
        let root = parse(
            br##"<map><properties>
  <property name="title" value="Cave"/>
  <property name="notes">line one
line two</property>
  <property name="hp" type="int" value="12"/>
  <property name="speed" type="float" value="1.5"/>
  <property name="dark" type="bool" value="true"/>
  <property name="fog" type="color" value="#80ff0000"/>
  <property name="nofog" type="color" value=""/>
  <property name="music" type="file" value="../audio/cave.ogg"/>
  <property name="door" type="object" value="7"/>
  <property name="spawn" type="class" propertytype="Spawn">
    <properties><property name="count" type="int" value="3"/></properties>
  </property>
</properties></map>"##,
        )
        .unwrap();
        let props = properties(&root, Some(Path::new("maps")));

        assert_eq!(props["title"], PropertyValue::String("Cave".into()));
        assert_eq!(
            props["notes"],
            PropertyValue::String("line one\nline two".into())
        );
        assert_eq!(props["hp"], PropertyValue::Int(12));
        assert_eq!(props["speed"], PropertyValue::Float(1.5));
        assert_eq!(props["dark"], PropertyValue::Bool(true));
        assert_eq!(
            props["fog"],
            PropertyValue::Colour(Some(Colour::srgb_u8(255, 0, 0, 128)))
        );
        assert_eq!(props["nofog"], PropertyValue::Colour(None));
        assert_eq!(props["music"], PropertyValue::File("audio/cave.ogg".into()));
        assert_eq!(props["door"], PropertyValue::Object(7));
        let PropertyValue::Class { class, members } = &props["spawn"] else {
            panic!("expected a class value, got {:?}", props["spawn"]);
        };
        assert_eq!(class, "Spawn");
        assert_eq!(members["count"], PropertyValue::Int(3));
    }
}
