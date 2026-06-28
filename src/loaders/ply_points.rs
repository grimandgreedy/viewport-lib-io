use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::IoError;
use crate::types::IoPointCloud;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    LittleEndian,
    BigEndian,
}

#[derive(Clone, Copy)]
enum ScalarType {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
}

impl ScalarType {
    fn from_str(value: &str) -> Option<Self> {
        match value {
            "char" | "int8" => Some(Self::Int8),
            "uchar" | "uint8" => Some(Self::Uint8),
            "short" | "int16" => Some(Self::Int16),
            "ushort" | "uint16" => Some(Self::Uint16),
            "int" | "int32" => Some(Self::Int32),
            "uint" | "uint32" => Some(Self::Uint32),
            "float" | "float32" => Some(Self::Float32),
            "double" | "float64" => Some(Self::Float64),
            _ => None,
        }
    }

    fn byte_size(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 => 1,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Float64 => 8,
        }
    }

    fn is_uchar(self) -> bool {
        matches!(self, Self::Uint8 | Self::Int8)
    }
}

struct PropertySpec {
    name: String,
    ty: ScalarType,
}

/// Decode a PLY file into a point cloud.
///
/// This entrypoint is point-cloud-first: files that contain faces are rejected
/// so the family boundary stays explicit.
pub fn point_cloud_from_path(path: &Path) -> Result<IoPointCloud, IoError> {
    #[cfg(feature = "ply")]
    {
        let bytes = std::fs::read(path)?;
        point_cloud_from_bytes(&bytes)
    }

    #[cfg(not(feature = "ply"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "ply",
            context: "PLY point-cloud decoding",
        })
    }
}

fn point_cloud_from_bytes(bytes: &[u8]) -> Result<IoPointCloud, IoError> {
    let end_marker_lf = b"end_header\n";
    let end_marker_crlf = b"end_header\r\n";

    let (header_bytes, data_offset) =
        if let Some(position) = find_subsequence(bytes, end_marker_crlf) {
            (&bytes[..position], position + end_marker_crlf.len())
        } else if let Some(position) = find_subsequence(bytes, end_marker_lf) {
            (&bytes[..position], position + end_marker_lf.len())
        } else {
            return Err(IoError::Parse("ply: missing end_header".into()));
        };

    let header = std::str::from_utf8(header_bytes)
        .map_err(|_| IoError::Parse("ply: header is not valid UTF-8".into()))?;

    let mut lines = header.lines();
    if lines.next().unwrap_or("").trim() != "ply" {
        return Err(IoError::Parse("ply: not a PLY file".into()));
    }

    let format_line = lines.next().unwrap_or("").trim().to_string();
    let format = if format_line.starts_with("format ascii") {
        Format::Ascii
    } else if format_line.starts_with("format binary_little_endian") {
        Format::LittleEndian
    } else if format_line.starts_with("format binary_big_endian") {
        Format::BigEndian
    } else {
        return Err(IoError::Parse(format!(
            "ply: unknown file format `{format_line}`"
        )));
    };

    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut vertex_props = Vec::new();
    let mut in_vertex = false;

    for line in lines {
        let line = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["element", "vertex", count] => {
                vertex_count = count.parse().unwrap_or(0);
                in_vertex = true;
            }
            ["element", "face", count] => {
                face_count = count.parse().unwrap_or(0);
                in_vertex = false;
            }
            ["element", _, _] => {
                in_vertex = false;
            }
            ["property", scalar, name] if in_vertex => {
                if let Some(ty) = ScalarType::from_str(scalar) {
                    vertex_props.push(PropertySpec {
                        name: (*name).to_string(),
                        ty,
                    });
                }
            }
            _ => {}
        }
    }

    if face_count != 0 {
        return Err(IoError::Parse(
            "ply: file contains faces; use a mesh or scene importer instead".into(),
        ));
    }

    let find_property = |name: &str| vertex_props.iter().position(|prop| prop.name == name);
    let x_index = find_property("x").unwrap_or(0);
    let y_index = find_property("y").unwrap_or(1);
    let z_index = find_property("z").unwrap_or(2);
    let r_index = find_property("red");
    let g_index = find_property("green");
    let b_index = find_property("blue");
    let a_index = find_property("alpha");

    let has_color = r_index.is_some() && g_index.is_some() && b_index.is_some();
    let mut standard = HashSet::new();
    standard.insert(x_index);
    standard.insert(y_index);
    standard.insert(z_index);
    if let Some(index) = r_index {
        standard.insert(index);
    }
    if let Some(index) = g_index {
        standard.insert(index);
    }
    if let Some(index) = b_index {
        standard.insert(index);
    }
    if let Some(index) = a_index {
        standard.insert(index);
    }

    let extra_props: Vec<(usize, &PropertySpec)> = vertex_props
        .iter()
        .enumerate()
        .filter(|(index, _)| !standard.contains(index))
        .collect();

    let mut positions = Vec::with_capacity(vertex_count);
    let mut colors = if has_color {
        Vec::with_capacity(vertex_count)
    } else {
        Vec::new()
    };
    let mut scalar_bufs: Vec<Vec<f32>> = extra_props
        .iter()
        .map(|_| Vec::with_capacity(vertex_count))
        .collect();

    match format {
        Format::Ascii => parse_vertices_ascii(
            bytes,
            data_offset,
            vertex_count,
            &vertex_props,
            x_index,
            y_index,
            z_index,
            r_index,
            g_index,
            b_index,
            a_index,
            has_color,
            &extra_props,
            &mut positions,
            &mut colors,
            &mut scalar_bufs,
        )?,
        binary_format => parse_vertices_binary(
            bytes,
            data_offset,
            vertex_count,
            &vertex_props,
            binary_format,
            x_index,
            y_index,
            z_index,
            r_index,
            g_index,
            b_index,
            a_index,
            has_color,
            &extra_props,
            &mut positions,
            &mut colors,
            &mut scalar_bufs,
        )?,
    }

    let mut point_cloud = IoPointCloud {
        name: "PLY Point Cloud".to_string(),
        positions,
        colors,
        scalars: Vec::new(),
        scalar_attributes: HashMap::new(),
    };
    if scalar_bufs.len() == 1 {
        point_cloud.scalars = scalar_bufs.remove(0);
        if let Some((_, prop)) = extra_props.first() {
            point_cloud
                .scalar_attributes
                .insert(prop.name.clone(), point_cloud.scalars.clone());
        }
    } else {
        for (values, (_, prop)) in scalar_bufs.into_iter().zip(extra_props.iter()) {
            point_cloud
                .scalar_attributes
                .insert(prop.name.clone(), values);
        }
    }

    Ok(point_cloud)
}

#[allow(clippy::too_many_arguments)]
fn parse_vertices_ascii(
    bytes: &[u8],
    data_offset: usize,
    vertex_count: usize,
    vertex_props: &[PropertySpec],
    x_index: usize,
    y_index: usize,
    z_index: usize,
    r_index: Option<usize>,
    g_index: Option<usize>,
    b_index: Option<usize>,
    a_index: Option<usize>,
    has_color: bool,
    extra_props: &[(usize, &PropertySpec)],
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    scalar_bufs: &mut Vec<Vec<f32>>,
) -> Result<(), IoError> {
    let text = std::str::from_utf8(&bytes[data_offset..])
        .map_err(|_| IoError::Parse("ply: data is not valid UTF-8".into()))?;
    let mut lines = text.lines();

    for _ in 0..vertex_count {
        let line = lines
            .next()
            .ok_or_else(|| IoError::Parse("ply: unexpected EOF in vertex data".into()))?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        positions.push([
            parse_ascii_f32(&parts, x_index),
            parse_ascii_f32(&parts, y_index),
            parse_ascii_f32(&parts, z_index),
        ]);

        if has_color {
            let red_index = r_index.unwrap_or_default();
            let green_index = g_index.unwrap_or_default();
            let blue_index = b_index.unwrap_or_default();
            colors.push([
                parse_color_ascii(&parts, red_index, vertex_props[red_index].ty.is_uchar()),
                parse_color_ascii(&parts, green_index, vertex_props[green_index].ty.is_uchar()),
                parse_color_ascii(&parts, blue_index, vertex_props[blue_index].ty.is_uchar()),
                a_index
                    .map(|index| {
                        parse_color_ascii(&parts, index, vertex_props[index].ty.is_uchar())
                    })
                    .unwrap_or(1.0),
            ]);
        }

        for (buffer_index, (property_index, _)) in extra_props.iter().enumerate() {
            scalar_bufs[buffer_index].push(parse_ascii_f32(&parts, *property_index));
        }
    }

    Ok(())
}

fn parse_ascii_f32(parts: &[&str], index: usize) -> f32 {
    parts
        .get(index)
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.0)
}

fn parse_color_ascii(parts: &[&str], index: usize, is_uchar: bool) -> f32 {
    let value = parse_ascii_f32(parts, index);
    if is_uchar { value / 255.0 } else { value }
}

#[allow(clippy::too_many_arguments)]
fn parse_vertices_binary(
    bytes: &[u8],
    data_offset: usize,
    vertex_count: usize,
    vertex_props: &[PropertySpec],
    format: Format,
    x_index: usize,
    y_index: usize,
    z_index: usize,
    r_index: Option<usize>,
    g_index: Option<usize>,
    b_index: Option<usize>,
    a_index: Option<usize>,
    has_color: bool,
    extra_props: &[(usize, &PropertySpec)],
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    scalar_bufs: &mut Vec<Vec<f32>>,
) -> Result<(), IoError> {
    let mut prop_offsets = Vec::with_capacity(vertex_props.len());
    let mut stride = 0usize;
    for prop in vertex_props {
        prop_offsets.push(stride);
        stride += prop.ty.byte_size();
    }

    let data = &bytes[data_offset..];
    if data.len() < vertex_count * stride {
        return Err(IoError::Parse("ply: binary vertex data truncated".into()));
    }

    for vertex in 0..vertex_count {
        let base = vertex * stride;
        let record = &data[base..base + stride];
        let read_prop = |index: usize| -> f32 {
            let offset = prop_offsets[index];
            read_f32(&record[offset..], vertex_props[index].ty, format)
        };

        positions.push([read_prop(x_index), read_prop(y_index), read_prop(z_index)]);

        if has_color {
            let red_index = r_index.unwrap_or_default();
            let green_index = g_index.unwrap_or_default();
            let blue_index = b_index.unwrap_or_default();
            let to_color = |index: usize| -> f32 {
                let value = read_prop(index);
                if vertex_props[index].ty.is_uchar() {
                    value / 255.0
                } else {
                    value
                }
            };
            colors.push([
                to_color(red_index),
                to_color(green_index),
                to_color(blue_index),
                a_index.map(to_color).unwrap_or(1.0),
            ]);
        }

        for (buffer_index, (property_index, _)) in extra_props.iter().enumerate() {
            scalar_bufs[buffer_index].push(read_prop(*property_index));
        }
    }

    Ok(())
}

fn read_f32(buffer: &[u8], scalar_type: ScalarType, format: Format) -> f32 {
    match scalar_type {
        ScalarType::Float32 => {
            let bytes: [u8; 4] = buffer[..4].try_into().unwrap_or_default();
            if format == Format::LittleEndian {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            }
        }
        ScalarType::Float64 => {
            let bytes: [u8; 8] = buffer[..8].try_into().unwrap_or_default();
            let value = if format == Format::LittleEndian {
                f64::from_le_bytes(bytes)
            } else {
                f64::from_be_bytes(bytes)
            };
            value as f32
        }
        ScalarType::Uint8 => buffer[0] as f32,
        ScalarType::Int8 => (buffer[0] as i8) as f32,
        ScalarType::Uint16 => {
            let bytes: [u8; 2] = buffer[..2].try_into().unwrap_or_default();
            (if format == Format::LittleEndian {
                u16::from_le_bytes(bytes)
            } else {
                u16::from_be_bytes(bytes)
            }) as f32
        }
        ScalarType::Int16 => {
            let bytes: [u8; 2] = buffer[..2].try_into().unwrap_or_default();
            (if format == Format::LittleEndian {
                i16::from_le_bytes(bytes)
            } else {
                i16::from_be_bytes(bytes)
            }) as f32
        }
        ScalarType::Uint32 => {
            let bytes: [u8; 4] = buffer[..4].try_into().unwrap_or_default();
            (if format == Format::LittleEndian {
                u32::from_le_bytes(bytes)
            } else {
                u32::from_be_bytes(bytes)
            }) as f32
        }
        ScalarType::Int32 => {
            let bytes: [u8; 4] = buffer[..4].try_into().unwrap_or_default();
            (if format == Format::LittleEndian {
                i32::from_le_bytes(bytes)
            } else {
                i32::from_be_bytes(bytes)
            }) as f32
        }
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
