//! One element tree for both of Tiled's formats.
//!
//! TMX is XML and TMJ is JSON, but they describe the same document with the
//! same names, so the JSON is rewritten into the shape of the XML (`layers`
//! with `"type": "tilelayer"` becomes a `<layer>` element, an `image` path
//! becomes an `<image>` child, and so on) and a single walk reads both.

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use serde_json::Value;

use crate::error::IoError;

/// An element: its name, attributes, children and text.
#[derive(Debug, Clone, Default)]
pub(super) struct Element {
    pub(super) name: String,
    pub(super) attrs: Vec<(String, String)>,
    pub(super) children: Vec<Element>,
    pub(super) text: String,
    /// Tile ids already parsed, for a JSON layer that stores them as an array
    /// rather than as encoded text.
    pub(super) gids: Option<Vec<u32>>,
}

impl Element {
    fn named(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Self::default()
        }
    }

    pub(super) fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub(super) fn set_attr(&mut self, key: &str, value: String) {
        match self.attrs.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.attrs.push((key.to_string(), value)),
        }
    }

    pub(super) fn child(&self, name: &str) -> Option<&Element> {
        self.children.iter().find(|c| c.name == name)
    }

    pub(super) fn children_named<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = &'a Element> + 'a {
        self.children.iter().filter(move |c| c.name == name)
    }
}

/// Whether the bytes are JSON rather than XML, by their first non-blank byte.
pub(super) fn looks_like_json(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|b| *b == b'{')
}

/// Parse a Tiled document, XML or JSON, into its root element.
pub(super) fn parse(bytes: &[u8]) -> Result<Element, IoError> {
    if looks_like_json(bytes) {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|e| IoError::Parse(format!("Tiled JSON: {e}")))?;
        Ok(from_json(&value))
    } else {
        parse_xml(bytes)
    }
}

fn parse_xml(bytes: &[u8]) -> Result<Element, IoError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| IoError::Parse(format!("Tiled XML is not UTF-8: {e}")))?;
    let mut reader = Reader::from_str(text);
    let mut stack: Vec<Element> = vec![Element::named("")];
    let err = |e: &dyn std::fmt::Display| IoError::Parse(format!("Tiled XML: {e}"));

    loop {
        match reader.read_event().map_err(|e| err(&e))? {
            Event::Start(start) => stack.push(start_element(&start)?),
            Event::Empty(start) => {
                let element = start_element(&start)?;
                stack.last_mut().unwrap().children.push(element);
            }
            Event::End(_) => {
                let element = stack.pop().unwrap();
                match stack.last_mut() {
                    Some(parent) => parent.children.push(element),
                    None => return Err(err(&"unbalanced closing tag")),
                }
            }
            Event::Text(t) => {
                let t = t.unescape().map_err(|e| err(&e))?;
                stack.last_mut().unwrap().text.push_str(&t);
            }
            Event::CData(t) => {
                let t = std::str::from_utf8(&t).map_err(|e| err(&e))?.to_string();
                stack.last_mut().unwrap().text.push_str(&t);
            }
            Event::Eof => break,
            _ => {}
        }
    }

    let mut document = stack
        .pop()
        .filter(|_| stack.is_empty())
        .ok_or_else(|| err(&"unclosed element"))?;
    document
        .children
        .pop()
        .filter(|_| document.children.is_empty())
        .ok_or_else(|| err(&"expected one root element"))
}

fn start_element(start: &BytesStart) -> Result<Element, IoError> {
    let mut element = Element::named(&String::from_utf8_lossy(start.name().as_ref()));
    for attr in start.attributes() {
        let attr = attr.map_err(|e| IoError::Parse(format!("Tiled XML attribute: {e}")))?;
        let value = attr
            .unescape_value()
            .map_err(|e| IoError::Parse(format!("Tiled XML attribute: {e}")))?;
        element.attrs.push((
            String::from_utf8_lossy(attr.key.as_ref()).into_owned(),
            value.into_owned(),
        ));
    }
    Ok(element)
}

/// The text form of a JSON scalar, as TMX would write it.
fn scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Copy every scalar member of `object` onto `element` as an attribute.
fn scalar_attrs(element: &mut Element, object: &serde_json::Map<String, Value>) {
    for (key, value) in object {
        if let Some(text) = scalar(value) {
            element.attrs.push((key.clone(), text));
        }
    }
}

/// As [`scalar_attrs`], leaving out `type`: on a map, tileset or layer it
/// says which element this is, not the class it has, which an object or tile
/// in an older file writes under the same name.
fn scalar_attrs_but_type(element: &mut Element, object: &serde_json::Map<String, Value>) {
    scalar_attrs(element, object);
    element.attrs.retain(|(k, _)| k != "type");
}

/// Rewrite a TMJ, TSJ or template JSON document into the TMX element tree.
fn from_json(value: &Value) -> Element {
    let Some(object) = value.as_object() else {
        return Element::default();
    };
    match object.get("type").and_then(Value::as_str) {
        Some("tileset") => tileset(object),
        Some("template") => {
            let mut element = Element::named("template");
            if let Some(set) = object.get("tileset").and_then(Value::as_object) {
                element.children.push(tileset(set));
            }
            if let Some(obj) = object.get("object").and_then(Value::as_object) {
                element.children.push(map_object(obj));
            }
            element
        }
        // A tileset file written before the "type" member existed still has
        // its grid members.
        None if object.contains_key("tilecount") => tileset(object),
        _ => map(object),
    }
}

fn map(object: &serde_json::Map<String, Value>) -> Element {
    let mut element = Element::named("map");
    scalar_attrs_but_type(&mut element, object);
    push_properties(&mut element, object);
    for set in array(object, "tilesets") {
        if let Some(set) = set.as_object() {
            element.children.push(tileset(set));
        }
    }
    push_layers(&mut element, object);
    element
}

fn array<'a>(object: &'a serde_json::Map<String, Value>, key: &str) -> &'a [Value] {
    object
        .get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn push_layers(element: &mut Element, object: &serde_json::Map<String, Value>) {
    for layer in array(object, "layers") {
        if let Some(layer) = layer.as_object() {
            element.children.push(self::layer(layer));
        }
    }
}

fn layer(object: &serde_json::Map<String, Value>) -> Element {
    let kind = object.get("type").and_then(Value::as_str).unwrap_or("");
    let name = match kind {
        "tilelayer" => "layer",
        "objectgroup" => "objectgroup",
        "imagelayer" => "imagelayer",
        "group" => "group",
        other => other,
    };
    let mut element = Element::named(name);
    scalar_attrs_but_type(&mut element, object);
    push_properties(&mut element, object);

    match name {
        "layer" => {
            let mut data = Element::named("data");
            for key in ["encoding", "compression"] {
                if let Some(v) = object.get(key).and_then(scalar) {
                    data.attrs.push((key.to_string(), v));
                }
            }
            if let Some(chunks) = object.get("chunks").and_then(Value::as_array) {
                for chunk in chunks.iter().filter_map(Value::as_object) {
                    let mut c = Element::named("chunk");
                    scalar_attrs(&mut c, chunk);
                    fill_data(&mut c, chunk.get("data"));
                    data.children.push(c);
                }
            } else {
                fill_data(&mut data, object.get("data"));
            }
            element.children.push(data);
        }
        "objectgroup" => {
            for obj in array(object, "objects").iter().filter_map(Value::as_object) {
                element.children.push(map_object(obj));
            }
        }
        "imagelayer" => {
            if let Some(image) = image(object, "image", "imagewidth", "imageheight") {
                element.children.push(image);
            }
        }
        "group" => push_layers(&mut element, object),
        _ => {}
    }
    element
}

/// Tile ids stored either as a JSON array or as encoded text.
fn fill_data(element: &mut Element, data: Option<&Value>) {
    match data {
        Some(Value::Array(ids)) => {
            element.gids = Some(ids.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect());
        }
        Some(Value::String(text)) => element.text = text.clone(),
        _ => {}
    }
}

/// An `<image>` element from the image members of a tileset, tile or image
/// layer, which JSON spreads across the owner.
fn image(
    object: &serde_json::Map<String, Value>,
    source: &str,
    width: &str,
    height: &str,
) -> Option<Element> {
    let path = object.get(source).and_then(Value::as_str)?;
    if path.is_empty() {
        return None;
    }
    let mut element = Element::named("image");
    element.attrs.push(("source".into(), path.to_string()));
    for (from, to) in [
        (width, "width"),
        (height, "height"),
        ("transparentcolor", "trans"),
    ] {
        if let Some(v) = object.get(from).and_then(scalar) {
            element.attrs.push((to.into(), v));
        }
    }
    Some(element)
}

fn tileset(object: &serde_json::Map<String, Value>) -> Element {
    let mut element = Element::named("tileset");
    scalar_attrs_but_type(&mut element, object);
    push_properties(&mut element, object);
    if let Some(image) = image(object, "image", "imagewidth", "imageheight") {
        element.children.push(image);
    }
    if let Some(offset) = object.get("tileoffset").and_then(Value::as_object) {
        let mut o = Element::named("tileoffset");
        scalar_attrs(&mut o, offset);
        element.children.push(o);
    }
    for tile in array(object, "tiles").iter().filter_map(Value::as_object) {
        let mut t = Element::named("tile");
        scalar_attrs(&mut t, tile);
        push_properties(&mut t, tile);
        if let Some(image) = image(tile, "image", "imagewidth", "imageheight") {
            t.children.push(image);
        }
        if let Some(group) = tile.get("objectgroup").and_then(Value::as_object) {
            t.children.push(layer(group));
        }
        let frames = array(tile, "animation");
        if !frames.is_empty() {
            let mut animation = Element::named("animation");
            for frame in frames.iter().filter_map(Value::as_object) {
                let mut f = Element::named("frame");
                scalar_attrs(&mut f, frame);
                animation.children.push(f);
            }
            t.children.push(animation);
        }
        element.children.push(t);
    }
    element
}

fn map_object(object: &serde_json::Map<String, Value>) -> Element {
    let mut element = Element::named("object");
    for (key, value) in object {
        // `ellipse` and `point` are flags in JSON and child elements in TMX.
        if matches!(key.as_str(), "ellipse" | "point") {
            if value.as_bool() == Some(true) {
                element.children.push(Element::named(key));
            }
        } else if let Some(text) = scalar(value) {
            element.attrs.push((key.clone(), text));
        }
    }
    push_properties(&mut element, object);
    for kind in ["polygon", "polyline"] {
        if let Some(points) = object.get(kind).and_then(Value::as_array) {
            let points: Vec<String> = points
                .iter()
                .filter_map(Value::as_object)
                .map(|p| {
                    let coord = |k| p.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                    format!("{},{}", coord("x"), coord("y"))
                })
                .collect();
            let mut shape = Element::named(kind);
            shape.attrs.push(("points".into(), points.join(" ")));
            element.children.push(shape);
        }
    }
    if let Some(text) = object.get("text").and_then(Value::as_object) {
        let mut t = Element::named("text");
        for (key, value) in text {
            if key == "text" {
                t.text = value.as_str().unwrap_or_default().to_string();
            } else if let Some(v) = scalar(value) {
                t.attrs.push((key.clone(), v));
            }
        }
        element.children.push(t);
    }
    element
}

/// A `<properties>` child from a JSON `properties` array.
fn push_properties(element: &mut Element, object: &serde_json::Map<String, Value>) {
    let list = array(object, "properties");
    if list.is_empty() {
        return;
    }
    let mut properties = Element::named("properties");
    for property in list.iter().filter_map(Value::as_object) {
        let mut p = Element::named("property");
        for (key, value) in property {
            if key == "value" {
                continue;
            }
            if let Some(v) = scalar(value) {
                p.attrs.push((key.clone(), v));
            }
        }
        match property.get("value") {
            Some(Value::Object(members)) => p.children.push(class_members(members)),
            Some(value) => {
                if let Some(v) = scalar(value) {
                    p.attrs.push(("value".into(), v));
                }
            }
            None => {}
        }
        properties.children.push(p);
    }
    element.children.push(properties);
}

/// The members of a class-typed JSON property. JSON gives their values but
/// not their types, so the type is read off the value.
fn class_members(members: &serde_json::Map<String, Value>) -> Element {
    let mut properties = Element::named("properties");
    for (name, value) in members {
        let mut p = Element::named("property");
        p.attrs.push(("name".into(), name.clone()));
        let kind = match value {
            Value::Bool(_) => "bool",
            Value::Number(n) if n.is_i64() || n.is_u64() => "int",
            Value::Number(_) => "float",
            Value::Object(_) => "class",
            _ => "string",
        };
        p.attrs.push(("type".into(), kind.into()));
        match value {
            Value::Object(inner) => p.children.push(class_members(inner)),
            other => {
                if let Some(v) = scalar(other) {
                    p.attrs.push(("value".into(), v));
                }
            }
        }
        properties.children.push(p);
    }
    properties
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_parses_into_nested_elements_with_text() {
        let root = parse(
            br#"<?xml version="1.0"?>
<map width="2"><layer name="a &amp; b"><data encoding="csv">1,2</data></layer><tileset firstgid="1"/></map>"#,
        )
        .unwrap();
        assert_eq!(root.name, "map");
        assert_eq!(root.attr("width"), Some("2"));
        let layer = root.child("layer").unwrap();
        assert_eq!(layer.attr("name"), Some("a & b"));
        assert_eq!(layer.child("data").unwrap().text.trim(), "1,2");
        assert_eq!(root.children_named("tileset").count(), 1);
    }

    #[test]
    fn json_layers_and_objects_take_the_tmx_shape() {
        let root = parse(
            br#"{"type":"map","width":2,"layers":[
                {"type":"tilelayer","name":"ground","data":[1,0,2,3],"visible":true},
                {"type":"objectgroup","objects":[
                    {"id":4,"ellipse":true,"x":1.5},
                    {"id":5,"polygon":[{"x":0,"y":0},{"x":4,"y":2}]}
                ]}
            ]}"#,
        )
        .unwrap();
        let layer = root.child("layer").unwrap();
        assert_eq!(layer.attr("visible"), Some("true"));
        assert_eq!(layer.child("data").unwrap().gids, Some(vec![1, 0, 2, 3]));
        let objects: Vec<_> = root
            .child("objectgroup")
            .unwrap()
            .children_named("object")
            .collect();
        assert!(objects[0].child("ellipse").is_some());
        assert_eq!(objects[0].attr("x"), Some("1.5"));
        assert_eq!(
            objects[1].child("polygon").unwrap().attr("points"),
            Some("0,0 4,2")
        );
    }

    #[test]
    fn json_class_members_are_typed_by_their_values() {
        let root = parse(
            br#"{"type":"map","properties":[
                {"name":"spawn","type":"class","propertytype":"Spawn","value":{"hp":3,"speed":1.5,"boss":true}}
            ]}"#,
        )
        .unwrap();
        let property = root.child("properties").unwrap().child("property").unwrap();
        assert_eq!(property.attr("propertytype"), Some("Spawn"));
        let members: Vec<_> = property
            .child("properties")
            .unwrap()
            .children
            .iter()
            .map(|p| (p.attr("name").unwrap(), p.attr("type").unwrap()))
            .collect();
        assert_eq!(
            members,
            [("boss", "bool"), ("hp", "int"), ("speed", "float")]
        );
    }

    #[test]
    fn malformed_xml_is_a_parse_error() {
        assert!(matches!(
            parse(b"<map><layer></map>"),
            Err(IoError::Parse(_))
        ));
    }
}
