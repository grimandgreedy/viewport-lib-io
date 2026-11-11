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
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh directory under the system temp dir, named for the caller plus the
/// process id, a nanosecond stamp, and a per-process counter.
///
/// The counter is what makes two calls in one process distinct: the clock alone
/// is not enough, since on macOS it ticks in microseconds and two tests starting
/// together can read the same stamp.
///
/// The directory is left behind on failure, on purpose: a test that fails has
/// something worth looking at.
pub fn temp_dir(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let unique = format!(
        "viewport_lib_io_{name}_{}_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
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

/// A Gmsh 2.2 ASCII `.msh` of the quad, as two triangles. Node tags start at 1
/// and are deliberately not dense indices, which is what the format allows and
/// the loader has to remap.
pub fn msh_v2_quad() -> String {
    let mut out = String::from("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");
    out.push_str(&format!("$Nodes\n{}\n", QUAD_POSITIONS.len()));
    for (i, p) in QUAD_POSITIONS.iter().enumerate() {
        out.push_str(&format!("{} {} {} {}\n", i + 1, p[0], p[1], p[2]));
    }
    out.push_str("$EndNodes\n");
    // Element line: tag, type (2 = triangle), tag count, the tags, then nodes.
    out.push_str("$Elements\n2\n");
    out.push_str("1 2 2 0 1 1 2 3\n");
    out.push_str("2 2 2 0 1 1 3 4\n");
    out.push_str("$EndElements\n");
    out
}

/// A Gmsh 4.1 ASCII `.msh` of the same quad. The two versions describe the same
/// geometry through completely different section layouts, so decoding both and
/// comparing is a real check rather than a transcript.
pub fn msh_v4_quad() -> String {
    let mut out = String::from("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    // One entity block: 4 nodes, tags 1..4, listed as tags then coordinates.
    out.push_str(&format!(
        "$Nodes\n1 {n} 1 {n}\n2 1 0 {n}\n",
        n = QUAD_POSITIONS.len()
    ));
    for i in 0..QUAD_POSITIONS.len() {
        out.push_str(&format!("{}\n", i + 1));
    }
    for p in QUAD_POSITIONS {
        out.push_str(&format!("{} {} {}\n", p[0], p[1], p[2]));
    }
    out.push_str("$EndNodes\n");
    // One block of two triangles; each element line is its tag then its nodes.
    out.push_str("$Elements\n1 2 1 2\n2 1 2 2\n");
    out.push_str("1 1 2 3\n");
    out.push_str("2 1 3 4\n");
    out.push_str("$EndElements\n");
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

/// A Basis Universal UASTC texture in a KTX2 container, with a full mip chain
/// down to 1x1 when `mips` is set and only the base level otherwise.
///
/// Every 4x4 block is a UASTC solid-colour block, coloured by `colour(level, bx,
/// by)` with the block's level and block coordinates, so a test can tell
/// levels and blocks apart after a transcode. `alpha` marks the data
/// descriptor's channel as RGBA rather than RGB, which is how the transcoder
/// decides the image carries alpha; `srgb` sets the transfer function. No
/// supercompression.
pub fn ktx2_uastc(
    width: u32,
    height: u32,
    mips: bool,
    alpha: bool,
    srgb: bool,
    colour: impl Fn(u32, u32, u32) -> [u8; 4],
) -> Vec<u8> {
    let levels = if mips {
        32 - width.max(height).leading_zeros()
    } else {
        1
    };

    let mut level_data = Vec::new();
    for level in 0..levels {
        let w = (width >> level).max(1);
        let h = (height >> level).max(1);
        let mut data = Vec::new();
        for by in 0..h.div_ceil(4) {
            for bx in 0..w.div_ceil(4) {
                data.extend_from_slice(&uastc_solid_block(colour(level, bx, by)));
            }
        }
        level_data.push(data);
    }

    // Header (80 bytes), level index (24 per level), then the data format
    // descriptor: a total size word plus one basic block with one sample.
    let index_end = 80 + 24 * levels as usize;
    let dfd_offset = index_end;
    let dfd_len = 44usize;
    let mut data_offset = (dfd_offset + dfd_len).next_multiple_of(16);

    let mut out = Vec::new();
    out.extend_from_slice(&[
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
    ]);
    for word in [0u32, 1, width, height, 0, 0, 1, levels, 0] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    for word in [dfd_offset as u32, dfd_len as u32, 0, 0] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());

    let mut offsets = Vec::new();
    for data in &level_data {
        offsets.push(data_offset);
        data_offset = (data_offset + data.len()).next_multiple_of(16);
    }
    for (data, offset) in level_data.iter().zip(&offsets) {
        out.extend_from_slice(&(*offset as u64).to_le_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    }

    // Colour model 166 (UASTC), primaries BT.709, transfer 2 (sRGB) or 1
    // (linear); 4x4 texel blocks of 16 bytes; one 128-bit sample whose channel
    // id is 3 (RGBA) or 0 (RGB).
    let transfer: u32 = if srgb { 2 } else { 1 };
    let channel: u32 = if alpha { 3 } else { 0 };
    for word in [
        dfd_len as u32,
        0,
        2 | (40 << 16),
        166 | (1 << 8) | (transfer << 16),
        3 | (3 << 8),
        16,
        0,
        127 << 16 | channel << 24,
        0,
        0,
        u32::MAX,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }

    for (data, offset) in level_data.iter().zip(&offsets) {
        out.resize(*offset, 0);
        out.extend_from_slice(data);
    }
    out
}

/// A UASTC solid-colour block: mode 8, the colour, and the ETC1 hint fields
/// a transcoder to ETC reads.
fn uastc_solid_block(colour: [u8; 4]) -> [u8; 16] {
    let mut block = [0u8; 16];
    let mut bit = 0usize;
    let mut put = |value: u32, bits: usize| {
        for i in 0..bits {
            if value >> i & 1 != 0 {
                block[bit / 8] |= 1 << (bit % 8);
            }
            bit += 1;
        }
    };
    // Mode 8's prefix code, five bits.
    put(0x17, 5);
    for channel in colour {
        put(u32::from(channel), 8);
    }
    // ETC1 hints: differential, intensity table 0, selector 0, and the colour
    // at five bits per channel.
    put(1, 1);
    put(0, 3);
    put(0, 2);
    for channel in &colour[..3] {
        put(u32::from(*channel) >> 3, 5);
    }
    block
}

/// An SVG holding one filled path, for the vector loader.
pub fn svg_filled_path(d: &str, fill: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\">\
         <path d=\"{d}\" fill=\"{fill}\"/></svg>"
    )
}

/// The global tile ids of the ground layer in [`tiled_tmx`] and [`tiled_tmj`],
/// row by row: plain, flipped each way, empty, and one from the second tileset.
pub const TILED_GROUND: [u32; 6] = [1, 2 | 0x8000_0000, 3 | 0x4000_0000 | 0x2000_0000, 0, 4, 5];

/// Standard base64 with padding, for encoded tile layer data.
pub fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, b)| acc | (*b as u32) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn ground_base64() -> String {
    let bytes: Vec<u8> = TILED_GROUND.iter().flat_map(|g| g.to_le_bytes()).collect();
    base64(&bytes)
}

/// A 3x2 Tiled map as TMX, its ground layer stored as `encoding`: `"xml"`
/// (`<tile>` elements), `"csv"` or `"base64"`.
///
/// It has an embedded atlas tileset with an animated tile and a tile with a
/// collision shape; an external image-collection tileset, [`tiled_tsx`], at
/// `props.tsx`; an object layer with one object of every shape plus a
/// template instance of [`tiled_template`] at `crate.tx`; an image layer; a
/// group; and a property of every type. [`tiled_tmj`] is the same map as JSON,
/// and decodes identically.
pub fn tiled_tmx(encoding: &str) -> String {
    let data = match encoding {
        "xml" => format!(
            "<data>{}</data>",
            TILED_GROUND
                .iter()
                .map(|g| format!("<tile gid=\"{g}\"/>"))
                .collect::<String>()
        ),
        "csv" => format!(
            "<data encoding=\"csv\">\n{}\n</data>",
            TILED_GROUND.map(|g| g.to_string()).join(",")
        ),
        "base64" => format!(
            "<data encoding=\"base64\">\n   {}\n  </data>",
            ground_base64()
        ),
        other => panic!("no TMX encoding `{other}`"),
    };
    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<map version="1.10" orientation="orthogonal" renderorder="right-down" width="3" height="2" tilewidth="16" tileheight="16" infinite="0" backgroundcolor="#336699" class="Level">
 <properties>
  <property name="title" value="Cave"/>
  <property name="level" type="int" value="3"/>
  <property name="gravity" type="float" value="9.5"/>
  <property name="dark" type="bool" value="true"/>
  <property name="fog" type="color" value="#80ff0000"/>
  <property name="music" type="file" value="audio/cave.ogg"/>
  <property name="door" type="object" value="2"/>
  <property name="spawn" type="class" propertytype="Spawn">
   <properties>
    <property name="count" type="int" value="4"/>
   </properties>
  </property>
 </properties>
 <tileset firstgid="1" name="terrain" tilewidth="16" tileheight="16" tilecount="4" columns="2" spacing="0" margin="0">
  <image source="terrain.png" width="32" height="32"/>
  <tile id="0">
   <properties>
    <property name="solid" type="bool" value="true"/>
   </properties>
   <objectgroup draworder="index">
    <object id="1" x="0" y="8" width="16" height="8"/>
   </objectgroup>
  </tile>
  <tile id="1">
   <animation>
    <frame tileid="1" duration="100"/>
    <frame tileid="2" duration="150"/>
   </animation>
  </tile>
 </tileset>
 <tileset firstgid="5" source="props.tsx"/>
 <layer id="1" name="ground" width="3" height="2" opacity="0.5" offsetx="4" offsety="-2" parallaxx="0.5" tintcolor="#ff00ff00">
  {data}
 </layer>
 <objectgroup id="2" name="things" color="#a0a0a4" draworder="index">
  <object id="1" name="box" type="Trigger" x="1" y="2" width="16" height="8" rotation="45">
   <properties>
    <property name="target" type="object" value="3"/>
   </properties>
  </object>
  <object id="2" name="pond" x="20" y="4" width="10" height="6">
   <ellipse/>
  </object>
  <object id="3" name="spawn" x="8" y="8">
   <point/>
  </object>
  <object id="4" name="wall" x="0" y="0">
   <polygon points="0,0 16,0 8,12"/>
  </object>
  <object id="5" name="path" x="2" y="30" visible="0">
   <polyline points="0,0 10,5.5"/>
  </object>
  <object id="6" name="sign" x="0" y="20" width="40" height="12">
   <text fontfamily="Mono" pixelsize="10" wrap="1" color="#ffffff00" bold="1" halign="center" valign="bottom">Hello, cave</text>
  </object>
  <object id="7" name="barrel" gid="2147483653" x="16" y="32" width="16" height="24"/>
  <object id="8" template="crate.tx" x="40" y="32"/>
 </objectgroup>
 <imagelayer id="3" name="sky" repeatx="1">
  <image source="sky.png" width="64" height="32"/>
 </imagelayer>
 <group id="4" name="decor" visible="0">
  <layer id="5" name="detail" width="3" height="2">
   <data encoding="csv">0,0,0,0,0,1</data>
  </layer>
 </group>
</map>
"##
    )
}

/// The map of [`tiled_tmx`] as Tiled JSON, its ground layer stored as
/// `encoding`: `"array"` or `"base64"`.
pub fn tiled_tmj(encoding: &str) -> String {
    let data = match encoding {
        "array" => format!(
            "\"data\": [{}]",
            TILED_GROUND.map(|g| g.to_string()).join(",")
        ),
        "base64" => format!(
            "\"encoding\": \"base64\", \"data\": \"{}\"",
            ground_base64()
        ),
        other => panic!("no TMJ encoding `{other}`"),
    };
    format!(
        r##"{{
 "type": "map", "version": "1.10", "orientation": "orthogonal", "renderorder": "right-down",
 "width": 3, "height": 2, "tilewidth": 16, "tileheight": 16, "infinite": false,
 "backgroundcolor": "#336699", "class": "Level",
 "properties": [
  {{"name": "title", "type": "string", "value": "Cave"}},
  {{"name": "level", "type": "int", "value": 3}},
  {{"name": "gravity", "type": "float", "value": 9.5}},
  {{"name": "dark", "type": "bool", "value": true}},
  {{"name": "fog", "type": "color", "value": "#80ff0000"}},
  {{"name": "music", "type": "file", "value": "audio/cave.ogg"}},
  {{"name": "door", "type": "object", "value": 2}},
  {{"name": "spawn", "type": "class", "propertytype": "Spawn", "value": {{"count": 4}}}}
 ],
 "tilesets": [
  {{"firstgid": 1, "name": "terrain", "tilewidth": 16, "tileheight": 16, "tilecount": 4,
    "columns": 2, "spacing": 0, "margin": 0,
    "image": "terrain.png", "imagewidth": 32, "imageheight": 32,
    "tiles": [
     {{"id": 0,
       "properties": [{{"name": "solid", "type": "bool", "value": true}}],
       "objectgroup": {{"type": "objectgroup", "draworder": "index",
         "objects": [{{"id": 1, "x": 0, "y": 8, "width": 16, "height": 8}}]}}}},
     {{"id": 1, "animation": [{{"tileid": 1, "duration": 100}}, {{"tileid": 2, "duration": 150}}]}}
    ]}},
  {{"firstgid": 5, "source": "props.tsx"}}
 ],
 "layers": [
  {{"type": "tilelayer", "id": 1, "name": "ground", "width": 3, "height": 2, "opacity": 0.5,
    "offsetx": 4, "offsety": -2, "parallaxx": 0.5, "tintcolor": "#ff00ff00", "visible": true,
    {data}}},
  {{"type": "objectgroup", "id": 2, "name": "things", "color": "#a0a0a4", "draworder": "index",
    "objects": [
     {{"id": 1, "name": "box", "type": "Trigger", "x": 1, "y": 2, "width": 16, "height": 8,
       "rotation": 45, "properties": [{{"name": "target", "type": "object", "value": 3}}]}},
     {{"id": 2, "name": "pond", "x": 20, "y": 4, "width": 10, "height": 6, "ellipse": true}},
     {{"id": 3, "name": "spawn", "x": 8, "y": 8, "point": true}},
     {{"id": 4, "name": "wall", "x": 0, "y": 0,
       "polygon": [{{"x": 0, "y": 0}}, {{"x": 16, "y": 0}}, {{"x": 8, "y": 12}}]}},
     {{"id": 5, "name": "path", "x": 2, "y": 30, "visible": false,
       "polyline": [{{"x": 0, "y": 0}}, {{"x": 10, "y": 5.5}}]}},
     {{"id": 6, "name": "sign", "x": 0, "y": 20, "width": 40, "height": 12,
       "text": {{"text": "Hello, cave", "fontfamily": "Mono", "pixelsize": 10, "wrap": true,
                "color": "#ffffff00", "bold": true, "halign": "center", "valign": "bottom"}}}},
     {{"id": 7, "name": "barrel", "gid": 2147483653, "x": 16, "y": 32, "width": 16, "height": 24}},
     {{"id": 8, "template": "crate.tx", "x": 40, "y": 32}}
    ]}},
  {{"type": "imagelayer", "id": 3, "name": "sky", "repeatx": true,
    "image": "sky.png", "imagewidth": 64, "imageheight": 32}},
  {{"type": "group", "id": 4, "name": "decor", "visible": false,
    "layers": [{{"type": "tilelayer", "id": 5, "name": "detail", "width": 3, "height": 2,
                 "data": [0,0,0,0,0,1]}}]}}
 ]
}}
"##
    )
}

/// The external tileset [`tiled_tmx`] references at `props.tsx`: an image
/// collection of a barrel and a crate, one image file per tile.
pub fn tiled_tsx() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<tileset version="1.10" name="props" tilewidth="16" tileheight="24" tilecount="2" columns="0">
 <tile id="0" type="Barrel">
  <image source="art/barrel.png" width="16" height="24"/>
 </tile>
 <tile id="1">
  <image source="art/crate.png" width="16" height="16"/>
 </tile>
</tileset>
"#
    .to_string()
}

/// The object template [`tiled_tmx`] instances at `crate.tx`: a crate tile
/// from `props.tsx`, with a property the instance does not override.
pub fn tiled_template() -> String {
    r#"<?xml version="1.0" encoding="UTF-8"?>
<template>
 <tileset firstgid="1" source="props.tsx"/>
 <object name="crate" type="Prop" gid="2" width="16" height="16">
  <properties>
   <property name="breakable" type="bool" value="true"/>
  </properties>
 </object>
</template>
"#
    .to_string()
}

fn ldtk_level_0() -> String {
    r##"{"identifier": "Level_0", "iid": "level-0", "uid": 0, "worldX": 0, "worldY": 0, "worldDepth": 0,
 "pxWid": 32, "pxHei": 16, "__bgColor": "#112233", "bgRelPath": "bg/sky.png",
 "__bgPos": {"topLeftPx": [-4, 2], "scale": [1, 1], "cropRect": [0, 0, 32, 16]},
 "fieldInstances": [{"__identifier": "music", "__type": "String", "__value": "cave", "defUid": 1}],
 "layerInstances": [
  {"__identifier": "Entities", "__type": "Entities", "__cWid": 2, "__cHei": 1, "__gridSize": 16,
   "__opacity": 1, "__pxTotalOffsetX": 0, "__pxTotalOffsetY": 0, "layerDefUid": 10, "visible": true,
   "entityInstances": [
    {"__identifier": "Player", "iid": "p-1", "px": [8, 16], "__pivot": [0.5, 1], "width": 16, "height": 16,
     "__tile": {"tilesetUid": 7, "x": 32, "y": 16, "w": 16, "h": 16},
     "fieldInstances": [
      {"__identifier": "hp", "__type": "Int", "__value": 3},
      {"__identifier": "target", "__type": "EntityRef", "__value": {"entityIid": "c-1", "layerIid": "l", "levelIid": "level-0", "worldIid": "w"}}
     ]},
    {"__identifier": "Chest", "iid": "c-1", "px": [16, 0], "__pivot": [0, 0], "width": 16, "height": 16,
     "__tile": null, "fieldInstances": []}
   ]},
  {"__identifier": "Walls", "__type": "IntGrid", "__cWid": 4, "__cHei": 2, "__gridSize": 8,
   "__opacity": 0.5, "__pxTotalOffsetX": 0, "__pxTotalOffsetY": 0, "__tilesetDefUid": 7,
   "layerDefUid": 11, "visible": true, "intGridCsv": [1, 1, 0, 0, 0, 2, 2, 0],
   "autoLayerTiles": [{"px": [0, 0], "src": [16, 16], "f": 0, "t": 5, "d": [0], "a": 1}]},
  {"__identifier": "Ground", "__type": "Tiles", "__cWid": 2, "__cHei": 1, "__gridSize": 16,
   "__opacity": 1, "__pxTotalOffsetX": 4, "__pxTotalOffsetY": -2, "__tilesetDefUid": 7,
   "layerDefUid": 12, "visible": true,
   "gridTiles": [
    {"px": [0, 0], "src": [0, 0], "f": 0, "t": 0, "d": [0], "a": 1},
    {"px": [16, 0], "src": [16, 0], "f": 1, "t": 1, "d": [1], "a": 1},
    {"px": [16, 0], "src": [48, 0], "f": 2, "t": 3, "d": [1], "a": 1}
   ]},
  {"__identifier": "Shadows", "__type": "AutoLayer", "__cWid": 2, "__cHei": 1, "__gridSize": 16,
   "__opacity": 1, "__pxTotalOffsetX": 0, "__pxTotalOffsetY": 0, "__tilesetDefUid": 7,
   "layerDefUid": 13, "visible": false,
   "autoLayerTiles": [{"px": [0, 0], "src": [48, 16], "f": 3, "t": 7, "d": [0], "a": 1}]}
 ]}"##
        .to_string()
}

fn ldtk_level_1() -> String {
    r##"{"identifier": "Level_1", "iid": "level-1", "uid": 1, "worldX": 256, "worldY": 0, "worldDepth": 1,
 "pxWid": 16, "pxHei": 16, "__bgColor": "#000000", "bgRelPath": null, "fieldInstances": [],
 "layerInstances": [
  {"__identifier": "Ground", "__type": "Tiles", "__cWid": 1, "__cHei": 1, "__gridSize": 16,
   "__opacity": 1, "__pxTotalOffsetX": 0, "__pxTotalOffsetY": 0, "__tilesetDefUid": 7,
   "layerDefUid": 12, "visible": true,
   "gridTiles": [{"px": [0, 0], "src": [32, 0], "f": 0, "t": 2, "d": [0], "a": 1}]}
 ]}"##
        .to_string()
}

/// An LDtk project of two levels in a GridVania world, sharing one tileset.
///
/// The first level has a layer of every kind: entities with fields and a
/// tile, an IntGrid that also draws auto-layer tiles, a tile layer with two
/// tiles stacked in one cell and a parallax factor, and a hidden auto layer,
/// plus a background colour, image and a level field. The second is one tile.
///
/// With `external` the levels are saved separately, as LDtk does with "save
/// levels to separate files": the project lists them by `externalRelPath`
/// with no layers, and the returned files (path relative to the project, and
/// contents) hold them. Either way the project decodes the same.
pub fn ldtk_project(external: bool) -> (String, Vec<(String, String)>) {
    let levels = [("Level_0", ldtk_level_0()), ("Level_1", ldtk_level_1())];
    let mut files = Vec::new();
    let listed: Vec<String> = levels
        .iter()
        .enumerate()
        .map(|(i, (name, level))| {
            if !external {
                return level.clone();
            }
            let rel = format!("game/{name}.ldtkl");
            files.push((rel.clone(), level.clone()));
            format!(
                r#"{{"identifier": "{name}", "iid": "level-{i}", "uid": {i}, "worldX": {}, "worldY": 0,
 "pxWid": 16, "pxHei": 16, "externalRelPath": "{rel}", "layerInstances": null, "fieldInstances": []}}"#,
                i * 256
            )
        })
        .collect();

    let project = format!(
        r##"{{"__header__": {{"fileType": "LDtk Project JSON", "app": "LDtk"}},
 "jsonVersion": "1.5.3", "iid": "project", "defaultGridSize": 16, "bgColor": "#40465B",
 "worldLayout": "GridVania", "worldGridWidth": 256, "worldGridHeight": 256,
 "externalLevels": {external},
 "defs": {{
  "layers": [
   {{"uid": 10, "identifier": "Entities", "type": "Entities", "gridSize": 16}},
   {{"uid": 11, "identifier": "Walls", "type": "IntGrid", "gridSize": 8,
     "intGridValues": [{{"value": 1, "identifier": "wall", "color": "#FF0000"}},
                       {{"value": 2, "identifier": "water", "color": "#0000FF"}}]}},
   {{"uid": 12, "identifier": "Ground", "type": "Tiles", "gridSize": 16,
     "parallaxFactorX": 0.5, "parallaxFactorY": 0}},
   {{"uid": 13, "identifier": "Shadows", "type": "AutoLayer", "gridSize": 16}}
  ],
  "entities": [],
  "tilesets": [
   {{"uid": 7, "identifier": "Terrain", "relPath": "atlas/terrain.png", "pxWid": 64, "pxHei": 32,
     "tileGridSize": 16, "spacing": 0, "padding": 0, "__cWid": 4, "__cHei": 2,
     "customData": [{{"tileId": 1, "data": "grass"}}],
     "enumTags": [{{"enumValueId": "Solid", "tileIds": [0, 2]}}]}}
  ],
  "enums": [], "externalEnums": [], "levelFields": []
 }},
 "levels": [{}],
 "worlds": []
}}
"##,
        listed.join(",\n")
    );
    (project, files)
}

/// An Aseprite sprite sheet export, in the JSON hash layout or the array one:
/// four 16x16 frames of a 64x16 atlas, two of them trimmed, with per-frame
/// durations, a forward `walk` tag over frames 0 to 2 and a ping-pong `blink`
/// that repeats twice, and a `hitbox` slice with a nine-slice centre and a
/// pivot keyed on two frames. Both layouts decode identically.
pub fn aseprite_sheet(hash: bool) -> String {
    let frames = [
        ("hero 0.aseprite", 0, false, (0, 0, 16, 16), 100),
        ("hero 1.aseprite", 16, true, (2, 1, 12, 14), 120),
        ("hero 2.aseprite", 32, false, (0, 0, 16, 16), 100),
        ("hero 10.aseprite", 48, true, (3, 0, 10, 16), 200),
    ];
    let body: Vec<String> = frames
        .iter()
        .map(|(name, x, trimmed, (sx, sy, w, h), duration)| {
            let fields = format!(
                r#""frame": {{"x": {x}, "y": 0, "w": {w}, "h": {h}}}, "rotated": false, "trimmed": {trimmed},
   "spriteSourceSize": {{"x": {sx}, "y": {sy}, "w": {w}, "h": {h}}}, "sourceSize": {{"w": 16, "h": 16}},
   "duration": {duration}"#
            );
            if hash {
                format!("  \"{name}\": {{{fields}}}")
            } else {
                format!("  {{\"filename\": \"{name}\", {fields}}}")
            }
        })
        .collect();
    let (open, close) = if hash { ('{', '}') } else { ('[', ']') };
    format!(
        r##"{{ "frames": {open}
{}
 {close},
 "meta": {{
  "app": "https://www.aseprite.org/", "version": "1.3.7", "image": "hero.png",
  "format": "RGBA8888", "size": {{"w": 64, "h": 16}}, "scale": "1",
  "frameTags": [
   {{"name": "walk", "from": 0, "to": 2, "direction": "forward", "color": "#000000ff"}},
   {{"name": "blink", "from": 2, "to": 3, "direction": "pingpong", "repeat": "2", "color": "#ff0000ff", "data": "eyes"}}
  ],
  "layers": [{{"name": "Layer 1", "opacity": 255, "blendMode": "normal"}}],
  "slices": [
   {{"name": "hitbox", "color": "#0000ffff", "data": "solid", "keys": [
     {{"frame": 0, "bounds": {{"x": 2, "y": 2, "w": 12, "h": 12}}, "center": {{"x": 3, "y": 3, "w": 6, "h": 6}}, "pivot": {{"x": 6, "y": 12}}}},
     {{"frame": 2, "bounds": {{"x": 4, "y": 2, "w": 8, "h": 12}}}}
   ]}}
  ]
 }}
}}
"##,
        body.join(",\n")
    )
}

/// A TexturePacker export in the JSON hash layout or the array one: two
/// sprites in a 64x32 atlas, the second rotated to pack and trimmed, each
/// with a pivot.
pub fn texturepacker_sheet(hash: bool) -> String {
    let frames = [
        (
            "coin.png",
            r#""frame": {"x": 0, "y": 0, "w": 32, "h": 32}, "rotated": false, "trimmed": false,
   "spriteSourceSize": {"x": 0, "y": 0, "w": 32, "h": 32}, "sourceSize": {"w": 32, "h": 32},
   "pivot": {"x": 0.5, "y": 0.5}"#,
        ),
        (
            "sword.png",
            r#""frame": {"x": 32, "y": 0, "w": 30, "h": 14}, "rotated": true, "trimmed": true,
   "spriteSourceSize": {"x": 1, "y": 9, "w": 30, "h": 14}, "sourceSize": {"w": 32, "h": 32},
   "pivot": {"x": 0.1, "y": 0.9}"#,
        ),
    ];
    let body: Vec<String> = frames
        .iter()
        .map(|(name, fields)| {
            if hash {
                format!("  \"{name}\": {{{fields}}}")
            } else {
                format!("  {{\"filename\": \"{name}\", {fields}}}")
            }
        })
        .collect();
    let (open, close) = if hash { ('{', '}') } else { ('[', ']') };
    format!(
        r#"{{"frames": {open}
{}
 {close},
 "meta": {{"app": "https://www.codeandweb.com/texturepacker", "version": "1.0",
  "image": "items.png", "format": "RGBA8888", "size": {{"w": 64, "h": 32}}, "scale": "0.5"}}
}}
"#,
        body.join(",\n")
    )
}

/// A BMFont descriptor as text (`xml` false) or XML: a packed font over two
/// 64x64 pages, with `A` in the red channel of page 0, `B` in green, `C` in
/// blue, a space in every channel of page 1, and an `A` `B` kerning pair. Both
/// forms decode identically.
pub fn bmfont(xml: bool) -> String {
    let glyphs = [
        (32, 0, 0, 0, 0, 0, 1, 15),
        (65, 0, 0, 10, 12, 1, 0, 4),
        (66, 12, 0, 9, 12, 1, 0, 2),
        (67, 24, 0, 11, 13, 0, 0, 1),
    ];
    let char_attrs = |(id, x, y, w, h, xo, page, chnl): (
        u32,
        u32,
        u32,
        u32,
        u32,
        i32,
        u32,
        u32,
    )| {
        format!(
            r#"id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" xoffset="{xo}" yoffset="-2" xadvance="{}" page="{page}" chnl="{chnl}""#,
            w + 1
        )
    };
    if xml {
        let chars: String = glyphs
            .iter()
            .map(|g| format!("  <char {} />\n", char_attrs(*g)))
            .collect();
        format!(
            r#"<?xml version="1.0"?>
<font>
 <info face="Pixel Sans" size="-16" bold="1" italic="0" charset="" unicode="1" stretchH="100" smooth="0" aa="1" padding="1,2,3,4" spacing="1,1" outline="0"/>
 <common lineHeight="18" base="14" scaleW="64" scaleH="64" pages="2" packed="1" alphaChnl="0" redChnl="0" greenChnl="0" blueChnl="0"/>
 <pages>
  <page id="1" file="glyphs_1.png" />
  <page id="0" file="glyphs_0.png" />
 </pages>
 <chars count="{}">
{chars} </chars>
 <kernings count="1">
  <kerning first="65" second="66" amount="-1" />
 </kernings>
</font>
"#,
            glyphs.len()
        )
    } else {
        let chars: String = glyphs
            .iter()
            .map(|g| format!("char {}\n", char_attrs(*g).replace('"', "")))
            .collect();
        format!(
            r#"info face="Pixel Sans" size=-16 bold=1 italic=0 charset="" unicode=1 stretchH=100 smooth=0 aa=1 padding=1,2,3,4 spacing=1,1 outline=0
common lineHeight=18 base=14 scaleW=64 scaleH=64 pages=2 packed=1 alphaChnl=0 redChnl=0 greenChnl=0 blueChnl=0
page id=1 file="glyphs_1.png"
page id=0 file="glyphs_0.png"
chars count={}
{chars}kernings count=1
kerning first=65  second=66  amount=-1
"#,
            glyphs.len()
        )
    }
}

/// An Aseprite document to write with [`AseDocument::to_bytes`]: the header
/// fields, layers, frames of cels, palette, tags and slices, in the file's own
/// numbering (blend mode ids, layer flag bits, colour depth in bits).
#[derive(Debug, Clone, Default)]
pub struct AseDocument {
    /// Canvas width and height in pixels.
    pub size: [u16; 2],
    /// 32 for RGBA, 16 for greyscale, 8 for indexed.
    pub depth: u16,
    /// Header flags: 1 layer opacity is valid, 2 groups composite on their own.
    pub flags: u32,
    /// The palette index drawn as transparent, in an indexed document.
    pub transparent_index: u8,
    /// Width and height of one pixel; `[0, 0]` and `[1, 1]` are square.
    pub pixel_ratio: [u8; 2],
    /// RGBA palette entries, written as a new palette chunk when not empty.
    pub palette: Vec<[u8; 4]>,
    /// Layers, bottom first, as the file lists them.
    pub layers: Vec<AseLayer>,
    /// Frames, each with its duration and cels.
    pub frames: Vec<AseFrame>,
    /// Tags, written in the first frame.
    pub tags: Vec<AseTag>,
    /// Slices, written in the first frame.
    pub slices: Vec<AseSlice>,
    /// Further chunks for the first frame, as (type, data), written before the
    /// palette: old palettes, colour profiles, unknown chunks.
    pub extra_chunks: Vec<(u16, Vec<u8>)>,
}

/// A layer of an [`AseDocument`].
#[derive(Debug, Clone, Default)]
pub struct AseLayer {
    /// The layer's name.
    pub name: String,
    /// Flag bits: 1 visible, 8 background, 64 reference.
    pub flags: u16,
    /// 0 image, 1 group, 2 tilemap.
    pub kind: u16,
    /// Nesting depth: a layer's parent is the nearest earlier layer one level up.
    pub child_level: u16,
    /// Blend mode id, 0 normal to 18 divide.
    pub blend: u16,
    /// Opacity, 0 to 255.
    pub opacity: u8,
    /// User data text, written as a user data chunk after the layer.
    pub user_text: Option<String>,
}

impl AseLayer {
    /// A visible, opaque image layer with normal blending.
    pub fn image(name: &str) -> Self {
        Self {
            name: name.into(),
            flags: 1,
            opacity: 255,
            ..Self::default()
        }
    }
}

/// One frame of an [`AseDocument`].
#[derive(Debug, Clone, Default)]
pub struct AseFrame {
    /// Duration in milliseconds.
    pub duration_ms: u16,
    /// The frame's cels.
    pub cels: Vec<AseCel>,
}

/// A cel of an [`AseFrame`].
#[derive(Debug, Clone)]
pub struct AseCel {
    /// Index of the layer, counting every layer including groups.
    pub layer: u16,
    /// Top-left position on the canvas.
    pub position: [i16; 2],
    /// Opacity, 0 to 255.
    pub opacity: u8,
    /// Order offset against the other cels in the frame.
    pub z_index: i16,
    /// The cel's image.
    pub image: AseImage,
}

impl AseCel {
    /// An opaque cel at `position` holding `image`.
    pub fn new(layer: u16, position: [i16; 2], image: AseImage) -> Self {
        Self {
            layer,
            position,
            opacity: 255,
            z_index: 0,
            image,
        }
    }
}

/// The image of an [`AseCel`]. Pixels are in the document's depth: four bytes
/// RGBA, two bytes value and alpha, or one palette index.
#[derive(Debug, Clone)]
pub enum AseImage {
    /// Stored as is.
    Raw {
        /// Width and height.
        size: [u16; 2],
        /// The pixels.
        pixels: Vec<u8>,
    },
    /// Stored as a zlib stream, written with uncompressed deflate blocks.
    Compressed {
        /// Width and height.
        size: [u16; 2],
        /// The pixels, before compression.
        pixels: Vec<u8>,
    },
    /// The same layer's cel in an earlier frame.
    Linked(u16),
}

/// A tag of an [`AseDocument`].
#[derive(Debug, Clone, Default)]
pub struct AseTag {
    /// First frame.
    pub from: u16,
    /// Last frame, inclusive.
    pub to: u16,
    /// 0 forward, 1 reverse, 2 ping-pong, 3 ping-pong reverse.
    pub direction: u8,
    /// Times to play, 0 for ever.
    pub repeat: u16,
    /// The tag's name.
    pub name: String,
    /// The tag colour in the chunk itself, which newer files leave to user data.
    pub rgb: [u8; 3],
    /// User data text and colour, written after the tags chunk.
    pub user: Option<(String, [u8; 4])>,
}

/// A slice of an [`AseDocument`].
#[derive(Debug, Clone, Default)]
pub struct AseSlice {
    /// The slice's name.
    pub name: String,
    /// Its keys.
    pub keys: Vec<AseSliceKey>,
    /// User data text and colour, written after the slice chunk.
    pub user: Option<(String, [u8; 4])>,
}

/// A key of an [`AseSlice`]. A nine-slice centre or a pivot on any key sets
/// the flag for every key, as the format requires.
#[derive(Debug, Clone, Default)]
pub struct AseSliceKey {
    /// The frame the key starts on.
    pub frame: u32,
    /// x, y, width, height.
    pub bounds: [i32; 4],
    /// Nine-slice centre relative to the bounds.
    pub centre: Option<[i32; 4]>,
    /// Pivot relative to the bounds.
    pub pivot: Option<[i32; 2]>,
}

fn ase_string(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u16).to_le_bytes());
    out.extend(s.as_bytes());
}

fn ase_chunk(out: &mut Vec<Vec<u8>>, kind: u16, data: Vec<u8>) {
    let mut chunk = Vec::with_capacity(data.len() + 6);
    chunk.extend((data.len() as u32 + 6).to_le_bytes());
    chunk.extend(kind.to_le_bytes());
    chunk.extend(data);
    out.push(chunk);
}

fn ase_user_data(out: &mut Vec<Vec<u8>>, text: Option<&str>, colour: Option<[u8; 4]>) {
    let mut d = Vec::new();
    let flags = text.is_some() as u32 | (colour.is_some() as u32) << 1;
    d.extend(flags.to_le_bytes());
    if let Some(text) = text {
        ase_string(&mut d, text);
    }
    if let Some(c) = colour {
        d.extend(c);
    }
    ase_chunk(out, 0x2020, d);
}

/// `bytes` as a zlib stream of uncompressed deflate blocks: valid zlib that
/// any inflater reads, with no compressor needed to write it.
pub fn zlib_stored(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = if bytes.is_empty() {
        vec![&[]]
    } else {
        bytes.chunks(65535).collect()
    };
    for (i, block) in blocks.iter().enumerate() {
        out.push((i == blocks.len() - 1) as u8);
        let len = block.len() as u16;
        out.extend(len.to_le_bytes());
        out.extend((!len).to_le_bytes());
        out.extend(*block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for byte in bytes {
        a = (a + *byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend(((b << 16) | a).to_be_bytes());
    out
}

impl AseDocument {
    /// A document of `size` pixels at `depth` bits per pixel, with valid layer
    /// opacity and nothing in it.
    pub fn new(size: [u16; 2], depth: u16) -> Self {
        Self {
            size,
            depth,
            flags: 1,
            ..Self::default()
        }
    }

    /// The document as the bytes of an `.aseprite` file.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut frames = Vec::new();
        for (index, frame) in self.frames.iter().enumerate() {
            let mut chunks = Vec::new();
            if index == 0 {
                for (kind, data) in &self.extra_chunks {
                    ase_chunk(&mut chunks, *kind, data.clone());
                }
                if !self.palette.is_empty() {
                    let mut d = Vec::new();
                    d.extend((self.palette.len() as u32).to_le_bytes());
                    d.extend(0u32.to_le_bytes());
                    d.extend((self.palette.len() as u32 - 1).to_le_bytes());
                    d.extend([0; 8]);
                    for c in &self.palette {
                        d.extend(0u16.to_le_bytes());
                        d.extend(c);
                    }
                    ase_chunk(&mut chunks, 0x2019, d);
                }
                for layer in &self.layers {
                    let mut d = Vec::new();
                    for v in [
                        layer.flags,
                        layer.kind,
                        layer.child_level,
                        0,
                        0,
                        layer.blend,
                    ] {
                        d.extend(v.to_le_bytes());
                    }
                    d.push(layer.opacity);
                    d.extend([0; 3]);
                    ase_string(&mut d, &layer.name);
                    if layer.kind == 2 {
                        d.extend(0u32.to_le_bytes());
                    }
                    ase_chunk(&mut chunks, 0x2004, d);
                    if let Some(text) = &layer.user_text {
                        ase_user_data(&mut chunks, Some(text), None);
                    }
                }
                if !self.tags.is_empty() {
                    let mut d = Vec::new();
                    d.extend((self.tags.len() as u16).to_le_bytes());
                    d.extend([0; 8]);
                    for tag in &self.tags {
                        d.extend(tag.from.to_le_bytes());
                        d.extend(tag.to.to_le_bytes());
                        d.push(tag.direction);
                        d.extend(tag.repeat.to_le_bytes());
                        d.extend([0; 6]);
                        d.extend(tag.rgb);
                        d.push(0);
                        ase_string(&mut d, &tag.name);
                    }
                    ase_chunk(&mut chunks, 0x2018, d);
                    for tag in &self.tags {
                        match &tag.user {
                            Some((text, colour)) => {
                                ase_user_data(&mut chunks, Some(text), Some(*colour))
                            }
                            None => ase_user_data(&mut chunks, None, None),
                        }
                    }
                }
            }
            for cel in &frame.cels {
                let mut d = Vec::new();
                d.extend(cel.layer.to_le_bytes());
                d.extend(cel.position[0].to_le_bytes());
                d.extend(cel.position[1].to_le_bytes());
                d.push(cel.opacity);
                let kind: u16 = match cel.image {
                    AseImage::Raw { .. } => 0,
                    AseImage::Linked(_) => 1,
                    AseImage::Compressed { .. } => 2,
                };
                d.extend(kind.to_le_bytes());
                d.extend(cel.z_index.to_le_bytes());
                d.extend([0; 5]);
                match &cel.image {
                    AseImage::Raw { size, pixels } => {
                        d.extend(size[0].to_le_bytes());
                        d.extend(size[1].to_le_bytes());
                        d.extend(pixels);
                    }
                    AseImage::Compressed { size, pixels } => {
                        d.extend(size[0].to_le_bytes());
                        d.extend(size[1].to_le_bytes());
                        d.extend(zlib_stored(pixels));
                    }
                    AseImage::Linked(frame) => d.extend(frame.to_le_bytes()),
                }
                ase_chunk(&mut chunks, 0x2005, d);
            }
            if index == 0 {
                for slice in &self.slices {
                    let nine = slice.keys.iter().any(|k| k.centre.is_some()) as u32;
                    let pivot = slice.keys.iter().any(|k| k.pivot.is_some()) as u32;
                    let mut d = Vec::new();
                    d.extend((slice.keys.len() as u32).to_le_bytes());
                    d.extend((nine | pivot << 1).to_le_bytes());
                    d.extend(0u32.to_le_bytes());
                    ase_string(&mut d, &slice.name);
                    for key in &slice.keys {
                        d.extend(key.frame.to_le_bytes());
                        d.extend(key.bounds[0].to_le_bytes());
                        d.extend(key.bounds[1].to_le_bytes());
                        d.extend((key.bounds[2] as u32).to_le_bytes());
                        d.extend((key.bounds[3] as u32).to_le_bytes());
                        if nine == 1 {
                            let c = key.centre.unwrap_or_default();
                            d.extend(c[0].to_le_bytes());
                            d.extend(c[1].to_le_bytes());
                            d.extend((c[2] as u32).to_le_bytes());
                            d.extend((c[3] as u32).to_le_bytes());
                        }
                        if pivot == 1 {
                            let p = key.pivot.unwrap_or_default();
                            d.extend(p[0].to_le_bytes());
                            d.extend(p[1].to_le_bytes());
                        }
                    }
                    ase_chunk(&mut chunks, 0x2022, d);
                    if let Some((text, colour)) = &slice.user {
                        ase_user_data(&mut chunks, Some(text), Some(*colour));
                    }
                }
            }

            let body: Vec<u8> = chunks.concat();
            let count = chunks.len() as u32;
            let mut f = Vec::new();
            f.extend((body.len() as u32 + 16).to_le_bytes());
            f.extend(0xF1FAu16.to_le_bytes());
            f.extend((count.min(0xFFFF) as u16).to_le_bytes());
            f.extend(frame.duration_ms.to_le_bytes());
            f.extend([0; 2]);
            f.extend(count.to_le_bytes());
            f.extend(body);
            frames.push(f);
        }

        let body: Vec<u8> = frames.concat();
        let mut out = Vec::with_capacity(128 + body.len());
        out.extend((128 + body.len() as u32).to_le_bytes());
        out.extend(0xA5E0u16.to_le_bytes());
        out.extend((self.frames.len() as u16).to_le_bytes());
        out.extend(self.size[0].to_le_bytes());
        out.extend(self.size[1].to_le_bytes());
        out.extend(self.depth.to_le_bytes());
        out.extend(self.flags.to_le_bytes());
        out.extend(100u16.to_le_bytes());
        out.extend([0; 8]);
        out.push(self.transparent_index);
        out.extend([0; 3]);
        out.extend((self.palette.len() as u16).to_le_bytes());
        out.extend(self.pixel_ratio);
        out.extend([0; 8]);
        out.extend([0; 84]);
        debug_assert_eq!(out.len(), 128);
        out.extend(body);
        out
    }
}
