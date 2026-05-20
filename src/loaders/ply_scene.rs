use std::path::Path;

use crate::error::IoError;
use crate::types::{AttributeData, AttributeDomain, IoMesh, IoPointCloud, IoScene, SurfaceMesh};

/// Decode a PLY file into a CPU-side scene.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    let bytes = std::fs::read(path)?;
    load_ply(&bytes)
}



// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Ascii,
    LittleEndian,
    BigEndian,
}

/// PLY scalar type with all its aliases.
#[derive(Clone, Copy)]
enum Ty {
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Float32,
    Float64,
}

impl Ty {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "char" | "int8"     => Some(Self::Int8),
            "uchar" | "uint8"   => Some(Self::Uint8),
            "short" | "int16"   => Some(Self::Int16),
            "ushort" | "uint16" => Some(Self::Uint16),
            "int" | "int32"     => Some(Self::Int32),
            "uint" | "uint32"   => Some(Self::Uint32),
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

struct PropSpec {
    name: String,
    ty: Ty,
}

// ─── Binary read helpers ─────────────────────────────────────────────────────

fn read_f32(buf: &[u8], ty: Ty, fmt: Format) -> f32 {
    match ty {
        Ty::Float32 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or_default();
            if fmt == Format::LittleEndian { f32::from_le_bytes(b) } else { f32::from_be_bytes(b) }
        }
        Ty::Float64 => {
            let b: [u8; 8] = buf[..8].try_into().unwrap_or_default();
            let v = if fmt == Format::LittleEndian { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) };
            v as f32
        }
        Ty::Uint8  => buf[0] as f32,
        Ty::Int8   => (buf[0] as i8) as f32,
        Ty::Uint16 => {
            let b: [u8; 2] = buf[..2].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) }) as f32
        }
        Ty::Int16 => {
            let b: [u8; 2] = buf[..2].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { i16::from_le_bytes(b) } else { i16::from_be_bytes(b) }) as f32
        }
        Ty::Uint32 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }) as f32
        }
        Ty::Int32 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) }) as f32
        }
    }
}

fn read_u32(buf: &[u8], ty: Ty, fmt: Format) -> u32 {
    match ty {
        Ty::Uint8  => buf[0] as u32,
        Ty::Int8   => (buf[0] as i8) as u32,
        Ty::Uint16 => {
            let b: [u8; 2] = buf[..2].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) }) as u32
        }
        Ty::Int16 => {
            let b: [u8; 2] = buf[..2].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { i16::from_le_bytes(b) } else { i16::from_be_bytes(b) }) as u32
        }
        Ty::Uint32 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or_default();
            if fmt == Format::LittleEndian { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }
        }
        Ty::Int32 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or_default();
            (if fmt == Format::LittleEndian { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) }) as u32
        }
        _ => 0,
    }
}

// ─── Parser ──────────────────────────────────────────────────────────────────

fn load_ply(bytes: &[u8]) -> Result<IoScene, IoError> {
    // ── Locate header / data boundary ────────────────────────────────────────

    // The header is always ASCII; find the "end_header\n" marker.
    // Accept both LF and CRLF line endings.
    let end_marker_lf   = b"end_header\n";
    let end_marker_crlf = b"end_header\r\n";

    let (header_bytes, data_offset) = if let Some(pos) = find_subsequence(bytes, end_marker_crlf) {
        (&bytes[..pos], pos + end_marker_crlf.len())
    } else if let Some(pos) = find_subsequence(bytes, end_marker_lf) {
        (&bytes[..pos], pos + end_marker_lf.len())
    } else {
        return Err(IoError::Parse("Missing end_header".into()));
    };

    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| IoError::Parse("PLY header is not valid UTF-8".into()))?;

    // ── Parse header ─────────────────────────────────────────────────────────

    let mut lines = header_str.lines();

    let first = lines.next().unwrap_or("").trim();
    if first != "ply" {
        return Err(IoError::Parse("Not a PLY file".into()));
    }

    let fmt_line = lines.next().unwrap_or("").trim().to_string();
    let format = if fmt_line.starts_with("format ascii") {
        Format::Ascii
    } else if fmt_line.starts_with("format binary_little_endian") {
        Format::LittleEndian
    } else if fmt_line.starts_with("format binary_big_endian") {
        Format::BigEndian
    } else {
        return Err(IoError::Parse(format!(
            "Unknown PLY format: {fmt_line}"
        )));
    };

    let mut vertex_count = 0usize;
    let mut face_count = 0usize;
    let mut vertex_props: Vec<PropSpec> = Vec::new();
    // Face list property types (count_type, index_type).
    let mut face_list: Option<(Ty, Ty)> = None;
    let mut in_vertex = false;
    let mut in_face   = false;

    for line in lines {
        let line = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.as_slice() {
            ["element", "vertex", n] => {
                vertex_count = n.parse().unwrap_or(0);
                in_vertex = true;
                in_face   = false;
            }
            ["element", "face", n] => {
                face_count = n.parse().unwrap_or(0);
                in_vertex = false;
                in_face   = true;
            }
            ["element", _, _] => {
                in_vertex = false;
                in_face   = false;
            }
            ["property", typ, name] if in_vertex => {
                if let Some(ty) = Ty::from_str(typ) {
                    vertex_props.push(PropSpec { name: name.to_string(), ty });
                }
            }
            ["property", "list", count_typ, idx_typ, _name] if in_face => {
                if let (Some(ct), Some(it)) = (Ty::from_str(count_typ), Ty::from_str(idx_typ)) {
                    face_list = Some((ct, it));
                }
            }
            _ => {}
        }
    }

    // ── Property index lookup ─────────────────────────────────────────────────

    let find = |n: &str| vertex_props.iter().position(|p| p.name == n);
    let x_idx   = find("x").unwrap_or(0);
    let y_idx   = find("y").unwrap_or(1);
    let z_idx   = find("z").unwrap_or(2);
    let nx_idx  = find("nx");
    let ny_idx  = find("ny");
    let nz_idx  = find("nz");
    let r_idx   = find("red");
    let g_idx   = find("green");
    let b_idx   = find("blue");
    let a_idx   = find("alpha");

    let has_color   = r_idx.is_some() && g_idx.is_some() && b_idx.is_some();
    let has_normals = nx_idx.is_some() && ny_idx.is_some() && nz_idx.is_some();

    // Collect "extra" scalar property indices (not geometric or color props).
    let standard = {
        let mut s = std::collections::HashSet::new();
        s.insert(x_idx); s.insert(y_idx); s.insert(z_idx);
        if let Some(i) = nx_idx { s.insert(i); }
        if let Some(i) = ny_idx { s.insert(i); }
        if let Some(i) = nz_idx { s.insert(i); }
        if let Some(i) = r_idx  { s.insert(i); }
        if let Some(i) = g_idx  { s.insert(i); }
        if let Some(i) = b_idx  { s.insert(i); }
        if let Some(i) = a_idx  { s.insert(i); }
        s
    };
    let extra_props: Vec<(usize, &PropSpec)> = vertex_props.iter().enumerate()
        .filter(|(i, _)| !standard.contains(i))
        .collect();

    // ── Vertex data ───────────────────────────────────────────────────────────

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vertex_count);
    let mut stored_normals: Vec<[f32; 3]> =
        if has_normals { Vec::with_capacity(vertex_count) } else { Vec::new() };
    let mut colors: Vec<[f32; 4]> =
        if has_color { Vec::with_capacity(vertex_count) } else { Vec::new() };
    // Per-extra-prop scalar buffers, one Vec<f32> per extra property.
    let mut scalar_bufs: Vec<Vec<f32>> = extra_props.iter().map(|_| Vec::with_capacity(vertex_count)).collect();

    match format {
        Format::Ascii => {
            parse_vertices_ascii(
                bytes, data_offset, vertex_count, &vertex_props,
                x_idx, y_idx, z_idx,
                nx_idx, ny_idx, nz_idx, has_normals,
                r_idx, g_idx, b_idx, a_idx, has_color,
                &extra_props,
                &mut positions, &mut stored_normals, &mut colors, &mut scalar_bufs,
            )?;
        }
        le_or_be => {
            parse_vertices_binary(
                bytes, data_offset, vertex_count, &vertex_props, le_or_be,
                x_idx, y_idx, z_idx,
                nx_idx, ny_idx, nz_idx, has_normals,
                r_idx, g_idx, b_idx, a_idx, has_color,
                &extra_props,
                &mut positions, &mut stored_normals, &mut colors, &mut scalar_bufs,
            )?;
        }
    }

    // ── No faces → point cloud ────────────────────────────────────────────────

    if face_count == 0 {
        return Ok(IoScene {
            meshes: Vec::new(),
            materials: Vec::new(),
            point_sets: vec![IoPointCloud {
                name: "Point Cloud".to_string(),
                positions,
                colors,
                scalars: Vec::new(),
                scalar_attributes: std::collections::HashMap::new(),
            }],
        });
    }

    // ── Face data ─────────────────────────────────────────────────────────────

    // Compute where face data starts (after vertex data).
    let face_data_offset = match format {
        Format::Ascii => {
            // Skip vertex_count lines from data_offset.
            skip_ascii_lines(bytes, data_offset, vertex_count)
        }
        _ => {
            let vertex_stride: usize = vertex_props.iter().map(|p| p.ty.byte_size()).sum();
            data_offset + vertex_count * vertex_stride
        }
    };

    let mut indices: Vec<u32> = Vec::with_capacity(face_count * 3);
    match format {
        Format::Ascii => {
            parse_faces_ascii(bytes, face_data_offset, face_count, &mut indices)?;
        }
        le_or_be => {
            let (count_ty, idx_ty) = face_list.unwrap_or((Ty::Uint8, Ty::Int32));
            parse_faces_binary(bytes, face_data_offset, face_count, count_ty, idx_ty, le_or_be, &mut indices)?;
        }
    }

    let normals = if has_normals {
        stored_normals
    } else {
        compute_smooth_normals(&positions, &indices)
    };

    // Store extra scalar attributes in MeshData.
    let mut mesh_data = SurfaceMesh::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    for (buf, (_, prop)) in scalar_bufs.iter().zip(extra_props.iter()) {
        if !buf.is_empty() {
            mesh_data.attributes.insert(
                prop.name.clone(),
                AttributeData::scalars(AttributeDomain::Point, buf.clone()),
            );
        }
    }

    let vertex_attribute_names: Vec<String> = extra_props.iter().map(|(_, p)| p.name.clone()).collect();

    Ok(IoScene {
        meshes: vec![IoMesh {
            name: "PLY Mesh".to_string(),
            mesh: mesh_data,
            material_index: None,
            transform: glam::Mat4::IDENTITY,
            two_sided: false,
            parent_index: None,
            vertex_attribute_names,
            metadata: std::collections::HashMap::new(),
        }],
        materials: Vec::new(),
        point_sets: Vec::new(),
    })
}

// ─── ASCII vertex parsing ────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn parse_vertices_ascii(
    bytes: &[u8],
    data_offset: usize,
    vertex_count: usize,
    vertex_props: &[PropSpec],
    x_idx: usize, y_idx: usize, z_idx: usize,
    nx_idx: Option<usize>, ny_idx: Option<usize>, nz_idx: Option<usize>, has_normals: bool,
    r_idx: Option<usize>, g_idx: Option<usize>, b_idx: Option<usize>, a_idx: Option<usize>, has_color: bool,
    extra_props: &[(usize, &PropSpec)],
    positions: &mut Vec<[f32; 3]>,
    stored_normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    scalar_bufs: &mut Vec<Vec<f32>>,
) -> Result<(), IoError> {
    let text = std::str::from_utf8(&bytes[data_offset..])
        .map_err(|_| IoError::Parse("PLY data is not valid UTF-8".into()))?;
    let mut lines = text.lines();

    for _ in 0..vertex_count {
        let line = lines.next()
            .ok_or_else(|| IoError::Parse("Unexpected EOF in vertex data".into()))?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();

        positions.push([
            parse_ascii_f32(&parts, x_idx),
            parse_ascii_f32(&parts, y_idx),
            parse_ascii_f32(&parts, z_idx),
        ]);

        if has_normals {
            stored_normals.push([
                parse_ascii_f32(&parts, nx_idx.unwrap()),
                parse_ascii_f32(&parts, ny_idx.unwrap()),
                parse_ascii_f32(&parts, nz_idx.unwrap()),
            ]);
        }

        if has_color {
            let ri = r_idx.unwrap();
            let gi = g_idx.unwrap();
            let bi = b_idx.unwrap();
            colors.push([
                parse_color_chan_ascii(&parts, ri, vertex_props[ri].ty.is_uchar()),
                parse_color_chan_ascii(&parts, gi, vertex_props[gi].ty.is_uchar()),
                parse_color_chan_ascii(&parts, bi, vertex_props[bi].ty.is_uchar()),
                a_idx.map(|ai| parse_color_chan_ascii(&parts, ai, vertex_props[ai].ty.is_uchar()))
                    .unwrap_or(1.0),
            ]);
        }

        for (buf_idx, (prop_idx, _)) in extra_props.iter().enumerate() {
            scalar_bufs[buf_idx].push(parse_ascii_f32(&parts, *prop_idx));
        }
    }

    Ok(())
}

fn parse_ascii_f32(parts: &[&str], idx: usize) -> f32 {
    parts.get(idx).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0)
}

fn parse_color_chan_ascii(parts: &[&str], idx: usize, is_uchar: bool) -> f32 {
    let v = parse_ascii_f32(parts, idx);
    if is_uchar { v / 255.0 } else { v }
}

// ─── Binary vertex parsing ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn parse_vertices_binary(
    bytes: &[u8],
    data_offset: usize,
    vertex_count: usize,
    vertex_props: &[PropSpec],
    fmt: Format,
    x_idx: usize, y_idx: usize, z_idx: usize,
    nx_idx: Option<usize>, ny_idx: Option<usize>, nz_idx: Option<usize>, has_normals: bool,
    r_idx: Option<usize>, g_idx: Option<usize>, b_idx: Option<usize>, a_idx: Option<usize>, has_color: bool,
    extra_props: &[(usize, &PropSpec)],
    positions: &mut Vec<[f32; 3]>,
    stored_normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    scalar_bufs: &mut Vec<Vec<f32>>,
) -> Result<(), IoError> {
    // Compute per-property byte offsets within one vertex record.
    let mut prop_offsets: Vec<usize> = Vec::with_capacity(vertex_props.len());
    let mut off = 0usize;
    for prop in vertex_props.iter() {
        prop_offsets.push(off);
        off += prop.ty.byte_size();
    }
    let stride = off;

    let data = &bytes[data_offset..];
    if data.len() < vertex_count * stride {
        return Err(IoError::Parse("PLY binary vertex data truncated".into()));
    }

    for v in 0..vertex_count {
        let base = v * stride;
        let vrec = &data[base..base + stride];

        let read_prop = |idx: usize| -> f32 {
            let o = prop_offsets[idx];
            read_f32(&vrec[o..], vertex_props[idx].ty, fmt)
        };

        positions.push([read_prop(x_idx), read_prop(y_idx), read_prop(z_idx)]);

        if has_normals {
            stored_normals.push([
                read_prop(nx_idx.unwrap()),
                read_prop(ny_idx.unwrap()),
                read_prop(nz_idx.unwrap()),
            ]);
        }

        if has_color {
            let ri = r_idx.unwrap();
            let gi = g_idx.unwrap();
            let bi = b_idx.unwrap();
            let to_linear = |idx: usize| -> f32 {
                let v = read_prop(idx);
                if vertex_props[idx].ty.is_uchar() { v / 255.0 } else { v }
            };
            colors.push([
                to_linear(ri),
                to_linear(gi),
                to_linear(bi),
                a_idx.map(to_linear).unwrap_or(1.0),
            ]);
        }

        for (buf_idx, (prop_idx, _)) in extra_props.iter().enumerate() {
            scalar_bufs[buf_idx].push(read_prop(*prop_idx));
        }
    }

    Ok(())
}

// ─── ASCII face parsing ───────────────────────────────────────────────────────

fn parse_faces_ascii(
    bytes: &[u8],
    face_data_offset: usize,
    face_count: usize,
    indices: &mut Vec<u32>,
) -> Result<(), IoError> {
    let text = std::str::from_utf8(&bytes[face_data_offset..])
        .map_err(|_| IoError::Parse("PLY face data is not valid UTF-8".into()))?;
    let mut lines = text.lines();

    for _ in 0..face_count {
        let line = lines.next()
            .ok_or_else(|| IoError::Parse("Unexpected EOF in face data".into()))?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        let n: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        if n >= 3 {
            let verts: Vec<u32> = parts[1..=n]
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect();
            for i in 1..(n - 1) {
                indices.push(verts[0]);
                indices.push(verts[i]);
                indices.push(verts[i + 1]);
            }
        }
    }

    Ok(())
}

// ─── Binary face parsing ──────────────────────────────────────────────────────

fn parse_faces_binary(
    bytes: &[u8],
    face_data_offset: usize,
    face_count: usize,
    count_ty: Ty,
    idx_ty: Ty,
    fmt: Format,
    indices: &mut Vec<u32>,
) -> Result<(), IoError> {
    let mut cursor = face_data_offset;
    let data = bytes;

    for _ in 0..face_count {
        if cursor + count_ty.byte_size() > data.len() {
            return Err(IoError::Parse("PLY binary face data truncated".into()));
        }
        let n = read_u32(&data[cursor..], count_ty, fmt) as usize;
        cursor += count_ty.byte_size();

        if cursor + n * idx_ty.byte_size() > data.len() {
            return Err(IoError::Parse("PLY binary face index data truncated".into()));
        }
        let mut verts: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            verts.push(read_u32(&data[cursor..], idx_ty, fmt));
            cursor += idx_ty.byte_size();
        }

        if n >= 3 {
            for i in 1..(n - 1) {
                indices.push(verts[0]);
                indices.push(verts[i]);
                indices.push(verts[i + 1]);
            }
        }
    }

    Ok(())
}

// ─── Utilities ────────────────────────────────────────────────────────────────

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Skip `n` newline-terminated lines in `bytes` starting at `offset`.
/// Returns the byte offset just after the n-th line.
fn skip_ascii_lines(bytes: &[u8], mut offset: usize, n: usize) -> usize {
    for _ in 0..n {
        while offset < bytes.len() && bytes[offset] != b'\n' {
            offset += 1;
        }
        if offset < bytes.len() { offset += 1; } // skip the '\n'
    }
    offset
}

/// Accumulate face normals at each vertex and normalise for smooth shading.
fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = glam::Vec3::from(positions[tri[0] as usize]);
        let b = glam::Vec3::from(positions[tri[1] as usize]);
        let c = glam::Vec3::from(positions[tri[2] as usize]);
        let face_normal = (b - a).cross(c - a);
        for &v in tri {
            let n = &mut normals[v as usize];
            n[0] += face_normal.x;
            n[1] += face_normal.y;
            n[2] += face_normal.z;
        }
    }
    normals.iter_mut().for_each(|n| {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > 1e-6 {
            n[0] /= len; n[1] /= len; n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    });
    normals
}
