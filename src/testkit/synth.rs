//! Valid input built in memory, so a test does not need a committed asset.
//!
//! Each builder emits the smallest file its format allows that still exercises
//! the thing under test, and every one of them is a real file: written to a
//! temp path it decodes through its loader and passes the invariant checks.
//! FBX is the gap, and it is not fixable here: the format has no writer in the
//! Rust ecosystem, so FBX keeps committed fixtures and pure-function tests.
//!
//! The geometry throughout is the same two-triangle quad in the XY plane, so a
//! test can compare one format's decode against another's.

use std::path::{Path, PathBuf};

/// A fresh directory under the system temp dir, named for the caller plus the
/// process id and a nanosecond stamp so parallel runs cannot collide.
///
/// The directory is left behind on failure, on purpose: a test that fails has
/// something worth looking at.
pub fn temp_dir(name: &str) -> PathBuf {
    let unique = format!(
        "viewport_lib_io_{name}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A temp file path with the given name, inside a fresh [`temp_dir`].
pub fn temp_path(dir_name: &str, file_name: &str) -> PathBuf {
    temp_dir(dir_name).join(file_name)
}

/// Write `bytes` to `path`, creating the parent directory if needed.
pub fn write(path: &Path, bytes: impl AsRef<[u8]>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, bytes).unwrap();
}

/// The quad every builder emits: four corners of the unit square in the XY
/// plane, two triangles, counter-clockwise seen from +Z.
pub const QUAD_POSITIONS: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];

/// Triangle indices into [`QUAD_POSITIONS`].
pub const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// An OBJ of the quad. `material` names an accompanying `.mtl` when given,
/// which the caller has to write beside it.
pub fn obj_quad(material: Option<&str>) -> String {
    let mut out = String::from("# viewport-lib-io test fixture\n");
    if let Some(mtl) = material {
        out.push_str(&format!("mtllib {mtl}\n"));
    }
    for p in QUAD_POSITIONS {
        out.push_str(&format!("v {} {} {}\n", p[0], p[1], p[2]));
    }
    out.push_str("vn 0 0 1\n");
    out.push_str("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n");
    if material.is_some() {
        out.push_str("usemtl quad\n");
    }
    // OBJ indices are 1-based, and each face names position/uv/normal.
    out.push_str("f 1/1/1 2/2/1 3/3/1\n");
    out.push_str("f 1/1/1 3/3/1 4/4/1\n");
    out
}

/// A minimal MTL to pair with [`obj_quad`].
pub fn mtl_quad() -> String {
    "newmtl quad\nKd 0.8 0.2 0.1\nd 1.0\nNs 0.0\n".to_string()
}

/// An ASCII STL of the quad.
pub fn stl_ascii_quad() -> String {
    let mut out = String::from("solid quad\n");
    for tri in QUAD_INDICES.chunks(3) {
        out.push_str("  facet normal 0 0 1\n    outer loop\n");
        for &i in tri {
            let p = QUAD_POSITIONS[i as usize];
            out.push_str(&format!("      vertex {} {} {}\n", p[0], p[1], p[2]));
        }
        out.push_str("    endloop\n  endfacet\n");
    }
    out.push_str("endsolid quad\n");
    out
}

/// A binary STL of the quad: 80-byte header, triangle count, then 50 bytes per
/// triangle.
pub fn stl_binary_quad() -> Vec<u8> {
    let mut out = vec![0u8; 80];
    let triangles = (QUAD_INDICES.len() / 3) as u32;
    out.extend_from_slice(&triangles.to_le_bytes());
    for tri in QUAD_INDICES.chunks(3) {
        for v in [0.0f32, 0.0, 1.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &i in tri {
            for v in QUAD_POSITIONS[i as usize] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count
    }
    out
}

/// An ASCII PLY of the quad's four vertices and two faces, with per-vertex
/// colours when `colours` is set.
pub fn ply_ascii_quad(colours: bool) -> String {
    let mut out = String::from("ply\nformat ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", QUAD_POSITIONS.len()));
    out.push_str("property float x\nproperty float y\nproperty float z\n");
    if colours {
        out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    out.push_str("element face 2\nproperty list uchar int vertex_indices\nend_header\n");
    for (i, p) in QUAD_POSITIONS.iter().enumerate() {
        out.push_str(&format!("{} {} {}", p[0], p[1], p[2]));
        if colours {
            let c = (i * 60) as u8;
            out.push_str(&format!(" {c} {} {}", 255 - c, 128));
        }
        out.push('\n');
    }
    out.push_str("3 0 1 2\n3 0 2 3\n");
    out
}

/// An ASCII PLY of the quad's four vertices and nothing else, with per-vertex
/// colours when `colours` is set.
///
/// Face-less on purpose: the point-set loader rejects a PLY carrying faces and
/// points the caller at the mesh loader instead, so a point-cloud fixture has
/// to be vertices only.
pub fn ply_ascii_points(colours: bool) -> String {
    let mut out = String::from("ply\nformat ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", QUAD_POSITIONS.len()));
    out.push_str("property float x\nproperty float y\nproperty float z\n");
    if colours {
        out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    out.push_str("end_header\n");
    for (i, p) in QUAD_POSITIONS.iter().enumerate() {
        out.push_str(&format!("{} {} {}", p[0], p[1], p[2]));
        if colours {
            let c = (i * 60) as u8;
            out.push_str(&format!(" {c} {} {}", 255 - c, 128));
        }
        out.push('\n');
    }
    out
}

/// A binary little-endian PLY of the quad.
pub fn ply_binary_quad() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"ply\nformat binary_little_endian 1.0\n");
    out.extend_from_slice(format!("element vertex {}\n", QUAD_POSITIONS.len()).as_bytes());
    out.extend_from_slice(b"property float x\nproperty float y\nproperty float z\n");
    out.extend_from_slice(b"element face 2\nproperty list uchar int vertex_indices\nend_header\n");
    for p in QUAD_POSITIONS {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for tri in QUAD_INDICES.chunks(3) {
        out.push(3);
        for &i in tri {
            out.extend_from_slice(&(i as i32).to_le_bytes());
        }
    }
    out
}

/// A CSV point cloud: `x,y,z` plus a named scalar column.
pub fn csv_points(count: usize, scalar: &str) -> String {
    let mut out = format!("x,y,z,{scalar}\n");
    for i in 0..count {
        let f = i as f32;
        out.push_str(&format!("{f},{},{},{}\n", f * 2.0, f * 3.0, f / 10.0));
    }
    out
}

/// A `.npy` array of `f32`, C-ordered, in the shape given.
///
/// Version 1.0 of the format: magic, version, a two-byte little-endian header
/// length, then a padded Python dict describing dtype, order, and shape.
pub fn npy_f32(shape: &[usize], values: &[f32]) -> Vec<u8> {
    assert_eq!(
        shape.iter().product::<usize>(),
        values.len(),
        "shape does not describe the value count"
    );
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(usize::to_string).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    // The header, including its trailing newline, is padded to a multiple of 64.
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - (unpadded % 64)) % 64));
    header.push('\n');

    let mut out = Vec::new();
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // major
    out.push(0); // minor
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// A legacy ASCII VTK unstructured grid holding the quad as two triangles,
/// with one point scalar field.
pub fn vtk_legacy_quad(field: &str) -> String {
    let mut out = String::from("# vtk DataFile Version 3.0\nviewport-lib-io test fixture\nASCII\n");
    out.push_str("DATASET UNSTRUCTURED_GRID\n");
    out.push_str(&format!("POINTS {} float\n", QUAD_POSITIONS.len()));
    for p in QUAD_POSITIONS {
        out.push_str(&format!("{} {} {}\n", p[0], p[1], p[2]));
    }
    out.push_str("CELLS 2 8\n3 0 1 2\n3 0 2 3\n");
    out.push_str("CELL_TYPES 2\n5\n5\n");
    out.push_str(&format!("POINT_DATA {}\n", QUAD_POSITIONS.len()));
    out.push_str(&format!("SCALARS {field} float 1\nLOOKUP_TABLE default\n"));
    for i in 0..QUAD_POSITIONS.len() {
        out.push_str(&format!("{}\n", i as f32 / 10.0));
    }
    out
}

/// Pack glTF JSON and a binary buffer into a self-contained GLB blob, for a
/// fixture with no external `.bin` on disk.
///
/// GLB layout: a 12-byte header, then each chunk as an 8-byte header plus a
/// body padded to four bytes.
pub fn glb(json: &[u8], bin: &[u8]) -> Vec<u8> {
    fn pad4(len: usize) -> usize {
        (4 - (len & 3)) & 3
    }
    let json_pad = pad4(json.len());
    let bin_pad = pad4(bin.len());
    let json_len = json.len() + json_pad;
    let bin_len = bin.len() + bin_pad;
    let total = 12 + 8 + json_len + 8 + bin_len;

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());

    out.extend_from_slice(&(json_len as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json);
    out.extend(std::iter::repeat_n(b' ', json_pad));

    out.extend_from_slice(&(bin_len as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(bin);
    out.extend(std::iter::repeat_n(0u8, bin_pad));
    out
}

/// An SVG holding one filled path, for the vector loader.
pub fn svg_filled_path(d: &str, fill: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">\
         <path d=\"{d}\" fill=\"{fill}\"/></svg>"
    )
}
