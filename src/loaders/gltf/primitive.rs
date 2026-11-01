//! One glTF primitive to one `SceneMesh`.

use viewport_lib_types::data::mesh::MeshData;

use crate::error::IoError;
use crate::types::{IoMesh, MorphTarget, SkinWeights, smooth_normals};

use super::attributes::read_vectors;
use super::skin::normalise_skin_weights;

/// The per-vertex streams of one primitive, before they become a mesh.
#[derive(Default)]
pub(super) struct VertexStreams {
    pub(super) positions: Vec<[f32; 3]>,
    /// Triangle-list indices. `None` for a non-indexed primitive.
    pub(super) indices: Option<Vec<u32>>,
    pub(super) normals: Option<Vec<[f32; 3]>>,
    pub(super) tangents: Option<Vec<[f32; 4]>>,
    pub(super) uvs: Option<Vec<[f32; 2]>>,
    /// TEXCOORD_1: the second set a material slot can select through its UV
    /// transform's `uv_set`. Sets beyond it are not carried.
    pub(super) uvs1: Option<Vec<[f32; 2]>>,
    /// COLOR_0, expanded to RGBA.
    pub(super) colours: Option<Vec<[f32; 4]>>,
    pub(super) joints: Option<Vec<[u16; 4]>>,
    pub(super) weights: Option<Vec<[f32; 4]>>,
}

impl VertexStreams {
    /// Read the streams from the primitive's accessors.
    fn from_accessors(primitive: &gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Self {
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
        let attribute = |semantic: gltf::Semantic| primitive.get(&semantic);
        Self {
            positions: attribute(gltf::Semantic::Positions)
                .and_then(|accessor| read_vectors(accessor, buffers))
                .unwrap_or_default(),
            indices: reader.read_indices().map(|iter| iter.into_u32().collect()),
            normals: attribute(gltf::Semantic::Normals)
                .and_then(|accessor| read_vectors(accessor, buffers)),
            tangents: attribute(gltf::Semantic::Tangents)
                .and_then(|accessor| read_vectors(accessor, buffers)),
            uvs: attribute(gltf::Semantic::TexCoords(0))
                .and_then(|accessor| read_vectors(accessor, buffers)),
            uvs1: attribute(gltf::Semantic::TexCoords(1))
                .and_then(|accessor| read_vectors(accessor, buffers)),
            // glTF stores colour as vec3 or vec4 in u8, u16, or f32;
            // `into_rgba_f32` normalises all of them, expanding vec3 with
            // alpha 1.0.
            colours: reader
                .read_colors(0)
                .map(|iter| iter.into_rgba_f32().collect()),
            joints: reader.read_joints(0).map(|iter| iter.into_u16().collect()),
            weights: reader.read_weights(0).map(|iter| iter.into_f32().collect()),
        }
    }
}

/// Convert one glTF primitive into a mesh: positions, indices, normals,
/// tangents, UVs, vertex colours, skin weights, and morph targets.
///
/// A primitive carrying `KHR_draco_mesh_compression` is read from its Draco
/// stream, which the extension makes authoritative over any uncompressed
/// accessors it also has. Returns `Ok(None)` for a primitive with no positions,
/// and an error for a Draco stream that does not decode. Coordinates stay in
/// glTF's Y-up frame here; the caller reorients the finished mesh.
pub(super) fn convert_primitive(
    document: &gltf::Document,
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mesh: &gltf::Mesh,
    primitive_index: usize,
) -> Result<Option<IoMesh>, IoError> {
    #[cfg(feature = "gltf-draco")]
    let streams = match primitive.extension_value(draco::EXTENSION) {
        Some(extension) => draco::read_streams(document, primitive, extension, buffers)?,
        None => VertexStreams::from_accessors(primitive, buffers),
    };
    #[cfg(not(feature = "gltf-draco"))]
    let streams = {
        let _ = document;
        VertexStreams::from_accessors(primitive, buffers)
    };
    Ok(build_mesh(
        streams,
        primitive,
        buffers,
        mesh,
        primitive_index,
    ))
}

/// Assemble the mesh from its streams, with the primitive's morph targets,
/// material and name.
fn build_mesh(
    streams: VertexStreams,
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mesh: &gltf::Mesh,
    primitive_index: usize,
) -> Option<IoMesh> {
    let VertexStreams {
        positions,
        indices,
        normals,
        tangents,
        uvs,
        uvs1,
        colours,
        joints,
        weights,
    } = streams;
    if positions.is_empty() {
        return None;
    }

    let indices = indices.unwrap_or_else(|| (0..positions.len() as u32).collect());
    let normals = normals.unwrap_or_else(|| smooth_normals(&positions, &indices));

    // Skin attributes. glTF stores joint indices as either u8 or u16; we
    // normalise to u8 because the runtime substrate uses [u8; 4] today. Joint
    // indices above 255 are clamped with a warning.
    let skin_weights = match (joints, weights) {
        (Some(ji), Some(jw)) => Some(SkinWeights {
            joint_indices: ji
                .into_iter()
                .map(|q| [q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8])
                .collect(),
            joint_weights: jw.into_iter().map(normalise_skin_weights).collect(),
        }),
        _ => None,
    };

    // Morph targets (blend shapes): each is a set of per-vertex position (and
    // optionally normal / tangent) displacements from the base geometry, in
    // the same vertex order as `positions`. glTF does not store target names on
    // the accessor; the conventional `mesh.extras().targetNames` needs the
    // `extras` feature and raw-JSON parsing, so targets are named by index here
    // and real names are a follow-up. Displacements are reoriented into Z-up
    // alongside the base attributes in `reorient_mesh_z_up`.
    let base_vertex_count = positions.len();
    let morph_targets: Vec<MorphTarget> = primitive
        .morph_targets()
        .enumerate()
        .map(|(i, target)| {
            // Keep every target, even one with no POSITION displacement (some
            // face rigs carry a normals-only target): a dropped target would
            // shift the indices a weight animation drives by, and change the
            // count so the whole clip no longer matches. Zero-fill instead.
            let position_deltas: Vec<[f32; 3]> = target
                .positions()
                .and_then(|accessor| read_vectors(accessor, buffers))
                .unwrap_or_else(|| vec![[0.0, 0.0, 0.0]; base_vertex_count]);
            MorphTarget {
                name: format!("target_{i}"),
                position_deltas,
                normal_deltas: target
                    .normals()
                    .and_then(|accessor| read_vectors(accessor, buffers)),
                tangent_deltas: target
                    .tangents()
                    .and_then(|accessor| read_vectors(accessor, buffers)),
            }
        })
        .collect();

    let material_index = primitive.material().index();

    let base_name = mesh
        .name()
        .map(std::borrow::ToOwned::to_owned)
        .unwrap_or_else(|| format!("mesh_{}", mesh.index()));
    let name = if mesh.primitives().len() > 1 {
        format!("{base_name}.{primitive_index}")
    } else {
        base_name
    };

    let mut mesh_data = MeshData::new(positions, normals, indices);
    mesh_data.uvs = uvs;
    mesh_data.uvs1 = uvs1;
    mesh_data.tangents = tangents;
    mesh_data.vertex_colours = colours;

    Some(IoMesh {
        name,
        mesh: mesh_data,
        skin_weights,
        morph_targets,
        material_index,
        transform: glam::Mat4::IDENTITY,
        two_sided: primitive.material().double_sided(),
        parent_index: None,
        ..IoMesh::default()
    })
}

#[cfg(feature = "gltf-draco")]
mod draco {
    use draco_core::{DataType, DecoderBuffer, Mesh, MeshDecoder, PointAttribute};
    use serde_json::Value;

    use super::VertexStreams;
    use crate::error::IoError;

    pub(super) const EXTENSION: &str = "KHR_draco_mesh_compression";

    /// Decode the primitive's Draco stream into its vertex streams.
    ///
    /// The extension's `attributes` object maps each glTF semantic to a Draco
    /// attribute id; the semantic's accessor still says how many components it
    /// has and whether integer values are normalised. Semantics the loader
    /// does not carry are skipped. Indices come from the decoded faces.
    pub(super) fn read_streams(
        document: &gltf::Document,
        primitive: &gltf::Primitive,
        extension: &Value,
        buffers: &[gltf::buffer::Data],
    ) -> Result<VertexStreams, IoError> {
        let invalid = |reason: &str| IoError::Parse(format!("glTF {EXTENSION}: {reason}"));
        let view_index = extension
            .get("bufferView")
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid("no bufferView"))? as usize;
        let attributes = extension
            .get("attributes")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("no attributes"))?;
        let view = document
            .views()
            .nth(view_index)
            .ok_or_else(|| invalid("bufferView out of range"))?;
        let bytes = buffers
            .get(view.buffer().index())
            .and_then(|data| data.0.get(view.offset()..view.offset() + view.length()))
            .ok_or_else(|| invalid("bufferView range is outside its buffer"))?;

        let mut decoded = Mesh::new();
        MeshDecoder::new()
            .decode(&mut DecoderBuffer::new(bytes), &mut decoded)
            .map_err(|error| invalid(&error.to_string()))?;
        let point_count = decoded.num_points();

        let mut streams = VertexStreams {
            indices: Some(
                decoded
                    .faces()
                    .iter()
                    .flat_map(|face| face.map(|point| point.0))
                    .collect(),
            ),
            ..VertexStreams::default()
        };
        for (semantic, id) in attributes {
            let Some(id) = id.as_u64() else {
                return Err(invalid(&format!("{semantic} has a non-integer id")));
            };
            let attribute = decoded
                .attribute_by_unique_id(id as u32)
                .ok_or_else(|| invalid(&format!("{semantic} names missing attribute {id}")))?;
            let accessor = primitive
                .attributes()
                .find(|(s, _)| s.to_string() == *semantic)
                .map(|(_, accessor)| accessor);
            let normalized =
                accessor.as_ref().is_some_and(gltf::Accessor::normalized) || attribute.normalized();
            let components = accessor
                .as_ref()
                .map_or(attribute.num_components() as usize, |a| {
                    a.dimensions().multiplicity()
                });
            let read = |n: usize| read_values(attribute, point_count, n, normalized);
            match semantic.as_str() {
                "POSITION" => streams.positions = vectors(read(3)),
                "NORMAL" => streams.normals = Some(vectors(read(3))),
                "TANGENT" => streams.tangents = Some(vectors(read(4))),
                "TEXCOORD_0" => streams.uvs = Some(vectors(read(2))),
                "TEXCOORD_1" => streams.uvs1 = Some(vectors(read(2))),
                "COLOR_0" => {
                    let values = read(components.clamp(3, 4));
                    streams.colours = Some(if components == 3 {
                        values
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2], 1.0])
                            .collect()
                    } else {
                        vectors(values)
                    });
                }
                "JOINTS_0" => {
                    streams.joints = Some(
                        read(4)
                            .chunks_exact(4)
                            .map(|j| [j[0] as u16, j[1] as u16, j[2] as u16, j[3] as u16])
                            .collect(),
                    )
                }
                "WEIGHTS_0" => streams.weights = Some(vectors(read(4))),
                _ => {}
            }
        }
        Ok(streams)
    }

    /// `components` values per point as `f32`, normalised from the stored
    /// integer type when `normalized` is set.
    fn read_values(
        attribute: &PointAttribute,
        point_count: usize,
        components: usize,
        normalized: bool,
    ) -> Vec<f32> {
        let mut values = attribute.read_f32s(point_count, components);
        if normalized {
            let scale = match attribute.data_type() {
                DataType::Uint8 => Some((255.0, 0.0)),
                DataType::Int8 => Some((127.0, -1.0)),
                DataType::Uint16 => Some((65535.0, 0.0)),
                DataType::Int16 => Some((32767.0, -1.0)),
                _ => None,
            };
            if let Some((max, min)) = scale {
                for value in &mut values {
                    *value = (*value / max).max(min);
                }
            }
        }
        values
    }

    fn vectors<const N: usize>(values: Vec<f32>) -> Vec<[f32; N]> {
        values
            .chunks_exact(N)
            .map(|chunk| std::array::from_fn(|i| chunk[i]))
            .collect()
    }
    #[cfg(test)]
    mod tests {
        use draco_core::{
            DataType, EncoderBuffer, EncoderOptions, FaceIndex, GeometryAttributeType, Mesh,
            MeshEncoder, PointAttribute,
        };
        use viewport_lib_io_testkit::synth::glb;

        use super::super::super::scene_from_slice;

        fn attribute(
            kind: GeometryAttributeType,
            components: u8,
            values: &[f32],
        ) -> PointAttribute {
            let mut attribute = PointAttribute::new();
            let count = values.len() / components as usize;
            attribute.init(kind, components, DataType::Float32, false, count);
            for (i, value) in values.iter().enumerate() {
                attribute.buffer_mut().write(i * 4, &value.to_le_bytes());
            }
            attribute
        }

        /// A quad whose Draco stream holds UVs as attribute 0 and positions as
        /// attribute 1, beside uncompressed accessors holding other values:
        /// the stream wins, and the ids map through the extension.
        #[test]
        fn reads_the_draco_stream_through_the_attribute_id_table() {
            let positions = [
                0.0f32, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0,
            ];
            let uvs = [0.0f32, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
            let mut mesh = Mesh::new();
            let uv_id = mesh.add_attribute(attribute(GeometryAttributeType::TexCoord, 2, &uvs));
            let position_id =
                mesh.add_attribute(attribute(GeometryAttributeType::Position, 3, &positions));
            mesh.set_num_faces(2);
            mesh.set_face(FaceIndex(0), [0u32.into(), 1u32.into(), 2u32.into()]);
            mesh.set_face(FaceIndex(1), [0u32.into(), 2u32.into(), 3u32.into()]);
            let mut encoder = MeshEncoder::new();
            encoder.set_mesh(mesh);
            let mut encoded = EncoderBuffer::new();
            encoder
                .encode(&EncoderOptions::new(), &mut encoded)
                .unwrap();

            // A decoy uncompressed triangle the loader must not read.
            let mut bin = Vec::new();
            for value in [9.0f32, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0] {
                bin.extend_from_slice(&value.to_le_bytes());
            }
            let draco_offset = bin.len();
            bin.extend_from_slice(encoded.data());
            let draco_len = encoded.data().len();
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            let json = format!(
                r#"{{"asset":{{"version":"2.0"}},
                "extensionsUsed":["KHR_draco_mesh_compression"],
                "extensionsRequired":["KHR_draco_mesh_compression"],
                "scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
                "meshes":[{{"primitives":[{{
                  "attributes":{{"POSITION":0,"TEXCOORD_0":1}},"indices":2,
                  "extensions":{{"KHR_draco_mesh_compression":{{"bufferView":1,
                    "attributes":{{"POSITION":{position_id},"TEXCOORD_0":{uv_id}}}}}}}}}]}}],
                "accessors":[
                  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[9,9,9],"max":[9,9,9]}},
                  {{"componentType":5126,"count":4,"type":"VEC2"}},
                  {{"componentType":5125,"count":6,"type":"SCALAR"}}],
                "bufferViews":[
                  {{"buffer":0,"byteOffset":0,"byteLength":36}},
                  {{"buffer":0,"byteOffset":{draco_offset},"byteLength":{draco_len}}}],
                "buffers":[{{"byteLength":{bin_len}}}]}}"#,
                bin_len = bin.len(),
            );

            let scene = scene_from_slice(&glb(json.as_bytes(), &bin), None).unwrap();
            let mesh = &scene.meshes[0].mesh;
            assert_eq!(mesh.positions.len(), 4);
            assert_eq!(mesh.indices.len(), 6);
            // Every decoded corner is a quad corner (glTF (x, y, 0) is (x, 0, y)
            // in Z-up), with the UV that went in beside it.
            let uv_of = |p: [f32; 3]| [p[0] / 2.0, p[2] / 2.0];
            let decoded_uvs = mesh.uvs.as_ref().expect("uvs");
            for (position, uv) in mesh.positions.iter().zip(decoded_uvs) {
                assert!(position[1].abs() < 1e-6 && position.iter().all(|v| *v != 9.0));
                assert_eq!(*uv, uv_of(*position));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use viewport_lib_io_testkit::synth::temp_dir;

    #[test]
    fn loads_vertex_colours_from_color_0() {
        // Minimal triangle carrying a COLOR_0 (VEC4 f32) attribute.
        let dir = temp_dir("gltf_colour");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        // Binary layout (little-endian):
        //   positions: 3 vec3  (36 bytes) offset 0
        //   indices:   3 u32   (12 bytes) offset 36
        //   colours:   3 vec4  (48 bytes) offset 48
        // Total: 96 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let colours = [
            [1.0f32, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.5],
        ];
        for c in colours {
            for v in c {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        assert_eq!(bin.len(), 96);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "coloured_tri" }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0, "COLOR_0": 2 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 96 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 48 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC4" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let mesh = scene.meshes.first().expect("mesh missing");
        let got = mesh.mesh.vertex_colours.as_ref().expect("colours missing");
        assert_eq!(got.len(), 3);
        // Colours are direction-independent, so the Z-up reorientation leaves
        // them exactly as authored.
        for (g, e) in got.iter().zip(colours.iter()) {
            for k in 0..4 {
                assert!(
                    (g[k] - e[k]).abs() < 1e-5,
                    "colour mismatch: {g:?} vs {e:?}"
                );
            }
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn loads_second_uv_set_from_texcoord_1() {
        // Minimal triangle carrying both TEXCOORD_0 and TEXCOORD_1.
        let dir = temp_dir("gltf_texcoord1");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        // Binary layout (little-endian):
        //   positions: 3 vec3 (36 bytes) offset 0
        //   indices:   3 u32  (12 bytes) offset 36
        //   uv0:       3 vec2 (24 bytes) offset 48
        //   uv1:       3 vec2 (24 bytes) offset 72
        // Total: 96 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let uv0 = [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let uv1 = [[0.25f32, 0.5], [0.75, 0.5], [0.25, 0.9]];
        for uv in uv0.iter().chain(uv1.iter()) {
            for v in uv {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        assert_eq!(bin.len(), 96);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "two_uv_sets" }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0, "TEXCOORD_0": 2, "TEXCOORD_1": 3 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 96 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 24 },
    { "buffer": 0, "byteOffset": 72, "byteLength": 24 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
    { "bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let mesh = scene.meshes.first().expect("mesh missing");
        let first = mesh.mesh.uvs.as_ref().expect("uvs missing");
        let second = mesh.mesh.uvs1.as_ref().expect("uvs1 missing");
        assert_eq!(first.as_slice(), uv0.as_slice());
        assert_eq!(second.as_slice(), uv1.as_slice());

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    /// One UV set stays one UV set: a mesh without TEXCOORD_1 carries no second
    /// channel rather than a copy of the first.
    #[test]
    fn single_uv_set_leaves_uvs1_absent() {
        let dir = temp_dir("gltf_texcoord0_only");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for v in uv {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0 }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0, "TEXCOORD_0": 2 },
      "indices": 1
    }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 72 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 24 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let mesh = scene.meshes.first().expect("mesh missing");
        assert!(mesh.mesh.uvs.is_some());
        assert!(mesh.mesh.uvs1.is_none());

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn loads_morph_target_deltas_reoriented_to_z_up() {
        // Minimal triangle with one morph target carrying POSITION deltas.
        let dir = temp_dir("gltf_morph");
        let gltf_path = dir.join("tri.gltf");
        let bin_path = dir.join("tri.bin");

        // Binary layout (little-endian):
        //   positions:    3 vec3  (36 bytes) offset 0
        //   indices:      3 u32   (12 bytes) offset 36
        //   morph deltas: 3 vec3  (36 bytes) offset 48
        // Total: 84 bytes.
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0u32, 1, 2] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        // Per-vertex deltas in Y-up authoring space, chosen so the Z-up
        // reorientation `(x, y, z) -> (x, -z, y)` moves each onto a different
        // axis: Y->Z, Z->-Y, X unchanged.
        let deltas_y_up = [[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        for d in deltas_y_up {
            for v in d {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        assert_eq!(bin.len(), 84);
        std::fs::write(&bin_path, &bin).unwrap();

        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "mesh": 0, "name": "morph_tri" }],
  "meshes": [{
    "primitives": [{
      "attributes": { "POSITION": 0 },
      "indices": 1,
      "targets": [{ "POSITION": 2 }]
    }]
  }],
  "buffers": [{ "uri": "tri.bin", "byteLength": 84 }],
  "bufferViews": [
    { "buffer": 0, "byteOffset": 0,  "byteLength": 36 },
    { "buffer": 0, "byteOffset": 36, "byteLength": 12, "target": 34963 },
    { "buffer": 0, "byteOffset": 48, "byteLength": 36 }
  ],
  "accessors": [
    { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] },
    { "bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR" },
    { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,1] }
  ]
}"#;
        std::fs::write(&gltf_path, json).unwrap();

        let scene = scene_from_path(&gltf_path).unwrap();
        let mesh = scene.meshes.first().expect("mesh missing");
        assert_eq!(mesh.morph_targets.len(), 1, "one target expected");
        let target = &mesh.morph_targets[0];
        assert_eq!(target.name, "target_0");
        assert!(target.normal_deltas.is_none());
        assert!(target.tangent_deltas.is_none());

        let expected_z_up = [[0.0f32, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(target.position_deltas.len(), 3);
        for (g, e) in target.position_deltas.iter().zip(expected_z_up.iter()) {
            for k in 0..3 {
                assert!((g[k] - e[k]).abs() < 1e-5, "delta mismatch: {g:?} vs {e:?}");
            }
        }

        let _ = std::fs::remove_file(gltf_path);
        let _ = std::fs::remove_file(bin_path);
        let _ = std::fs::remove_dir(dir);
    }
}
