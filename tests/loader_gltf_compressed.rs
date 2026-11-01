//! Compressed glTF: Draco and meshopt twins of the committed Fox asset decode to
//! the same geometry as the original, and KTX2 textures transcode to the format,
//! size and mip count each target asks for.
//!
//! The twins are built here from `fox.glb` with the Draco and meshopt encoders,
//! so no compressed copy of the asset is committed. The KTX2 images are a
//! synthesised UASTC texture and one real encoder output,
//! `color_grid_uastc_zstd.ktx2` (Zstandard-supercompressed UASTC).

#![cfg(any(
    feature = "gltf-draco",
    feature = "gltf-meshopt",
    feature = "gltf-ktx2"
))]
// The fixture helpers serve whichever of the three features is on.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use viewport_lib_io::SceneData;
use viewport_lib_io::loaders::gltf;
use viewport_lib_io_testkit::synth;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// A GLB split into its JSON and binary chunks.
struct Glb {
    json: Value,
    bin: Vec<u8>,
}

impl Glb {
    fn read(path: &Path) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let chunk = |offset: usize| {
            let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            &bytes[offset + 8..offset + 8 + len]
        };
        let json = chunk(12);
        let bin = chunk(12 + 8 + json.len()).to_vec();
        Self {
            json: serde_json::from_slice(json).unwrap(),
            bin,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bin = self.bin.clone();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let mut json = self.json.clone();
        json["buffers"][0]["byteLength"] = json!(bin.len());
        synth::glb(serde_json::to_string(&json).unwrap().as_bytes(), &bin)
    }

    /// Append `bytes` to the binary chunk, four-byte aligned, returning their
    /// offset.
    fn append(&mut self, bytes: &[u8]) -> usize {
        while self.bin.len() % 4 != 0 {
            self.bin.push(0);
        }
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        offset
    }

    /// Add a buffer view over buffer 0, returning its index.
    fn add_view(&mut self, view: Value) -> usize {
        let views = self.json["bufferViews"].as_array_mut().unwrap();
        views.push(view);
        views.len() - 1
    }

    fn require(&mut self, extension: &str) {
        for key in ["extensionsUsed", "extensionsRequired"] {
            let list = self.json[key].as_array().cloned().unwrap_or_default();
            let mut list: Vec<Value> = list;
            list.push(json!(extension));
            self.json[key] = Value::Array(list);
        }
    }

    /// An accessor's elements, tightly packed, with the element size.
    fn accessor_bytes(&self, index: usize) -> (Vec<u8>, usize) {
        let accessor = &self.json["accessors"][index];
        let view = &self.json["bufferViews"][accessor["bufferView"].as_u64().unwrap() as usize];
        let components = match accessor["type"].as_str().unwrap() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            other => panic!("unexpected accessor type {other}"),
        };
        let width = match accessor["componentType"].as_u64().unwrap() {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4,
        };
        let size = components * width;
        let count = accessor["count"].as_u64().unwrap() as usize;
        let stride = view["byteStride"].as_u64().map_or(size, |s| s as usize);
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize
            + accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let mut out = Vec::with_capacity(count * size);
        for i in 0..count {
            out.extend_from_slice(&self.bin[start + i * stride..start + i * stride + size]);
        }
        (out, size)
    }

    fn primitive(&mut self) -> &mut Value {
        &mut self.json["meshes"][0]["primitives"][0]
    }

    fn attribute_accessors(&self) -> Vec<(String, usize)> {
        self.json["meshes"][0]["primitives"][0]["attributes"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(semantic, index)| (semantic.clone(), index.as_u64().unwrap() as usize))
            .collect()
    }
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// World-space bounds over every mesh in the scene.
fn bounds(scene: &SceneData) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for mesh in &scene.meshes {
        for p in &mesh.mesh.positions {
            let world = mesh.transform.transform_point3(glam::Vec3::from(*p));
            for k in 0..3 {
                min[k] = min[k].min(world[k]);
                max[k] = max[k].max(world[k]);
            }
        }
    }
    (min, max)
}

/// The twin decodes to the original's vertex and index counts and bounds,
/// within `tolerance` of the asset's size per axis.
fn assert_same_geometry(original: &SceneData, twin: &SceneData, tolerance: f32) {
    assert_eq!(original.meshes.len(), twin.meshes.len());
    for (a, b) in original.meshes.iter().zip(&twin.meshes) {
        assert_eq!(
            a.mesh.positions.len(),
            b.mesh.positions.len(),
            "vertex count"
        );
        assert_eq!(a.mesh.indices.len(), b.mesh.indices.len(), "index count");
        assert_eq!(
            a.skin_weights.is_some(),
            b.skin_weights.is_some(),
            "skinning"
        );
        assert_eq!(a.mesh.uvs.is_some(), b.mesh.uvs.is_some(), "uvs");
    }
    let (min_a, max_a) = bounds(original);
    let (min_b, max_b) = bounds(twin);
    for k in 0..3 {
        let extent = max_a[k] - min_a[k];
        assert!(
            (min_a[k] - min_b[k]).abs() <= extent * tolerance
                && (max_a[k] - max_b[k]).abs() <= extent * tolerance,
            "axis {k}: original {:?}..{:?}, twin {:?}..{:?}",
            min_a,
            max_a,
            min_b,
            max_b,
        );
    }
    assert_eq!(original.skeletons.len(), twin.skeletons.len());
    assert_eq!(original.animations.len(), twin.animations.len());
}

#[cfg(feature = "gltf-draco")]
mod draco {
    use draco_core::{
        DataType, EncoderBuffer, EncoderOptions, FaceIndex, GeometryAttributeType, Mesh,
        MeshEncoder, PointAttribute,
    };
    use viewport_lib_io_testkit::conformance;

    use super::*;

    /// Fox with its one primitive replaced by a Draco stream: the attribute
    /// accessors lose their buffer views, as the extension allows.
    fn draco_fox(sequential: bool) -> Vec<u8> {
        let mut glb = Glb::read(&fixture("fox.glb"));
        let mut mesh = Mesh::new();
        let mut ids = serde_json::Map::new();
        let mut vertex_count = 0;
        for (semantic, accessor) in glb.attribute_accessors() {
            let (bytes, size) = glb.accessor_bytes(accessor);
            let count = bytes.len() / size;
            vertex_count = count;
            let (kind, components, data_type) = match semantic.as_str() {
                "POSITION" => (GeometryAttributeType::Position, 3, DataType::Float32),
                "NORMAL" => (GeometryAttributeType::Normal, 3, DataType::Float32),
                "TEXCOORD_0" => (GeometryAttributeType::TexCoord, 2, DataType::Float32),
                "JOINTS_0" => (GeometryAttributeType::Generic, 4, DataType::Uint16),
                "WEIGHTS_0" => (GeometryAttributeType::Generic, 4, DataType::Float32),
                other => panic!("fox has no {other}"),
            };
            let mut attribute = PointAttribute::new();
            attribute.init(kind, components, data_type, false, count);
            attribute.buffer_mut().write(0, &bytes);
            ids.insert(semantic, json!(mesh.add_attribute(attribute)));
            glb.json["accessors"][accessor]
                .as_object_mut()
                .unwrap()
                .remove("bufferView");
        }
        // Fox is not indexed; the stream gets the identity triangle list.
        mesh.set_num_faces(vertex_count / 3);
        for face in 0..vertex_count / 3 {
            let i = face as u32 * 3;
            mesh.set_face(
                FaceIndex(face as u32),
                [i.into(), (i + 1).into(), (i + 2).into()],
            );
        }

        let mut options = EncoderOptions::new();
        if sequential {
            // Sequential connectivity keeps the point order, so the vertex
            // count matches the source exactly.
            options.set_encoding_method(0);
        }
        let mut encoder = MeshEncoder::new();
        encoder.set_mesh(mesh);
        let mut encoded = EncoderBuffer::new();
        encoder.encode(&options, &mut encoded).unwrap();

        let offset = glb.append(encoded.data());
        let view = glb.add_view(json!({
            "buffer": 0, "byteOffset": offset, "byteLength": encoded.data().len()
        }));
        let accessors = glb.json["accessors"].as_array_mut().unwrap();
        accessors.push(json!({"componentType": 5125, "count": vertex_count, "type": "SCALAR"}));
        let indices = accessors.len() - 1;
        let primitive = glb.primitive();
        primitive["indices"] = json!(indices);
        primitive["extensions"] = json!({
            "KHR_draco_mesh_compression": {"bufferView": view, "attributes": ids}
        });
        glb.require("KHR_draco_mesh_compression");
        glb.to_bytes()
    }

    #[test]
    fn a_draco_fox_decodes_to_the_same_geometry() {
        let original = gltf::scene_from_path(&fixture("fox.glb")).unwrap();
        let blob = draco_fox(true);
        let twin = conformance::scene("fox_draco.glb", || gltf::scene_from_slice(&blob, None));
        assert_same_geometry(&original, &twin, 1e-3);
    }

    #[test]
    fn an_edgebreaker_draco_fox_keeps_its_triangles_and_bounds() {
        let original = gltf::scene_from_path(&fixture("fox.glb")).unwrap();
        let blob = draco_fox(false);
        let twin = gltf::scene_from_slice(&blob, None).unwrap();
        // Edgebreaker may renumber and merge points, so only the triangle
        // count and the bounds have to match.
        assert_eq!(
            original.meshes[0].mesh.indices.len(),
            twin.meshes[0].mesh.indices.len()
        );
        let (min_a, max_a) = bounds(&original);
        let (min_b, max_b) = bounds(&twin);
        for k in 0..3 {
            let extent = max_a[k] - min_a[k];
            assert!((min_a[k] - min_b[k]).abs() <= extent * 1e-2);
            assert!((max_a[k] - max_b[k]).abs() <= extent * 1e-2);
        }
    }
}

#[cfg(feature = "gltf-meshopt")]
mod meshopt {
    use viewport_lib_io_testkit::conformance;

    use super::*;

    fn encode_vertices(bytes: &[u8], size: usize) -> Vec<u8> {
        let count = bytes.len() / size;
        let bound = meshoptimizer::encode_vertex_buffer_bound(count, size).unwrap();
        let mut out = vec![0; bound];
        let len =
            meshoptimizer::encode_vertex_buffer_level(&mut out, bound, bytes, count, size, 2, 0)
                .unwrap();
        out.truncate(len);
        out
    }

    /// Fox with every attribute in a meshopt view over a fallback buffer that
    /// has no data, positions through the exponential filter, plus an
    /// identity index list in a TRIANGLES view.
    fn meshopt_fox() -> Vec<u8> {
        let mut glb = Glb::read(&fixture("fox.glb"));
        let mut fallback_len = 0usize;
        let mut vertex_count = 0usize;
        let mut add = |glb: &mut Glb, raw: &[u8], stride: usize, mode: &str, filter: &str| {
            let count = raw.len() / stride;
            let encoded = match mode {
                "ATTRIBUTES" => encode_vertices(raw, stride),
                _ => {
                    let indices: Vec<u32> = raw
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let bound =
                        meshoptimizer::encode_index_buffer_bound(indices.len(), indices.len())
                            .unwrap();
                    let mut out = vec![0; bound];
                    let len = meshoptimizer::encode_index_buffer(
                        &mut out,
                        bound,
                        &indices,
                        indices.len(),
                    )
                    .unwrap();
                    out.truncate(len);
                    out
                }
            };
            let offset = glb.append(&encoded);
            let view = glb.add_view(json!({
                "buffer": 1, "byteOffset": fallback_len, "byteLength": raw.len(),
                "byteStride": if mode == "ATTRIBUTES" { Some(stride) } else { None },
                "extensions": {"EXT_meshopt_compression": {
                    "buffer": 0, "byteOffset": offset, "byteLength": encoded.len(),
                    "byteStride": stride, "count": count, "mode": mode, "filter": filter
                }}
            }));
            if mode != "ATTRIBUTES" {
                glb.json["bufferViews"][view]
                    .as_object_mut()
                    .unwrap()
                    .remove("byteStride");
            }
            fallback_len += raw.len().next_multiple_of(4);
            view
        };

        for (semantic, accessor) in glb.attribute_accessors() {
            let (bytes, size) = glb.accessor_bytes(accessor);
            vertex_count = bytes.len() / size;
            let view = if semantic == "POSITION" {
                // Positions as 16-bit-mantissa exponential floats.
                let values = floats(&bytes);
                let mut encoded = vec![0u8; values.len() * 4];
                meshoptimizer::encode_filter_exp(
                    &mut encoded,
                    vertex_count,
                    12,
                    16,
                    &values,
                    meshoptimizer::EncodeExpMode::Separate,
                )
                .unwrap();
                add(&mut glb, &encoded, 12, "ATTRIBUTES", "EXPONENTIAL")
            } else {
                add(&mut glb, &bytes, size, "ATTRIBUTES", "NONE")
            };
            let accessor = glb.json["accessors"][accessor].as_object_mut().unwrap();
            accessor.insert("bufferView".into(), json!(view));
            accessor.insert("byteOffset".into(), json!(0));
        }
        let identity: Vec<u8> = (0..vertex_count as u32)
            .flat_map(u32::to_le_bytes)
            .collect();
        let view = add(&mut glb, &identity, 4, "TRIANGLES", "NONE");
        let accessors = glb.json["accessors"].as_array_mut().unwrap();
        accessors.push(json!({
            "bufferView": view, "componentType": 5125, "count": vertex_count, "type": "SCALAR"
        }));
        let indices = accessors.len() - 1;
        glb.primitive()["indices"] = json!(indices);
        glb.json["buffers"]
            .as_array_mut()
            .unwrap()
            .push(json!({"byteLength": fallback_len,
                "extensions": {"EXT_meshopt_compression": {"fallback": true}}}));
        glb.require("EXT_meshopt_compression");
        glb.to_bytes()
    }

    #[test]
    fn a_meshopt_fox_decodes_to_the_same_geometry() {
        let original = gltf::scene_from_path(&fixture("fox.glb")).unwrap();
        let blob = meshopt_fox();
        let twin = conformance::scene("fox_meshopt.glb", || gltf::scene_from_slice(&blob, None));
        assert_same_geometry(&original, &twin, 1e-3);

        // The vertex streams other than positions are lossless.
        let (a, b) = (&original.meshes[0], &twin.meshes[0]);
        assert_eq!(a.mesh.uvs, b.mesh.uvs);
        let (wa, wb) = (a.skin_weights.as_ref(), b.skin_weights.as_ref());
        assert_eq!(wa.map(|w| &w.joint_indices), wb.map(|w| &w.joint_indices));
    }
}

#[cfg(feature = "gltf-ktx2")]
mod ktx2 {
    use viewport_lib_io::TextureSource;
    use viewport_lib_io::loaders::gltf::{GltfLoadOptions, GltfTextureTarget};
    use viewport_lib_io::viewport_lib_types::colour::ColourSpace;
    use viewport_lib_io::viewport_lib_types::data::texture::{
        AstcBlock, CompressedFormat, TextureData, TexturePayload,
    };

    use super::*;

    /// A triangle whose material takes base colour, normal and occlusion from
    /// `KHR_texture_basisu` images embedded in the GLB, with no fallback image.
    fn basisu_triangle(colour: &[u8], normal: &[u8], occlusion: &[u8]) -> Vec<u8> {
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let mut views = vec![json!({"buffer": 0, "byteOffset": 0, "byteLength": 36})];
        for image in [colour, normal, occlusion] {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            views.push(json!({"buffer": 0, "byteOffset": bin.len(), "byteLength": image.len()}));
            bin.extend_from_slice(image);
        }
        let texture =
            |image: usize| json!({"extensions": {"KHR_texture_basisu": {"source": image}}});
        let json = json!({
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_texture_basisu"],
            "extensionsRequired": ["KHR_texture_basisu"],
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
            "materials": [{
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1},
                "occlusionTexture": {"index": 2}
            }],
            "textures": [texture(0), texture(1), texture(2)],
            "images": [
                {"bufferView": 1, "mimeType": "image/ktx2"},
                {"bufferView": 2, "mimeType": "image/ktx2"},
                {"bufferView": 3, "mimeType": "image/ktx2"}
            ],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0, 0, 0], "max": [1, 1, 0]}],
            "bufferViews": views,
            "buffers": [{"byteLength": bin.len()}]
        });
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        synth::glb(serde_json::to_string(&json).unwrap().as_bytes(), &bin)
    }

    fn decoded(source: &Option<TextureSource>) -> &TextureData {
        match source {
            Some(TextureSource::Decoded(data)) => data,
            other => panic!("expected a decoded texture, got {other:?}"),
        }
    }

    /// Format and mip count, `None` for an uncompressed payload.
    fn format(data: &TextureData) -> Option<(CompressedFormat, usize)> {
        match data.payload() {
            TexturePayload::Compressed { format, mip_levels } => Some((*format, mip_levels.len())),
            _ => None,
        }
    }

    fn load(blob: &[u8], target: GltfTextureTarget) -> SceneData {
        let options = GltfLoadOptions::default().with_texture_target(target);
        gltf::scene_from_slice_with_options(blob, None, options).unwrap()
    }

    fn synthetic() -> Vec<u8> {
        let colour = synth::ktx2_uastc(32, 16, true, true, true, |level, bx, by| {
            [(level * 30) as u8, (bx * 30) as u8, (by * 60) as u8, 128]
        });
        // A two-channel normal map, X in grey RGB and Y in alpha.
        let normal = synth::ktx2_uastc(16, 16, true, true, false, |_, bx, _| {
            [100 + bx as u8 * 20; 4]
        });
        let occlusion = synth::ktx2_uastc(8, 8, false, false, false, |_, _, _| [200, 0, 0, 255]);
        basisu_triangle(&colour, &normal, &occlusion)
    }

    #[test]
    fn each_target_gives_its_formats_sizes_and_mip_counts() {
        let blob = synthetic();
        let astc = CompressedFormat::Astc {
            block: AstcBlock::B4x4,
            hdr: false,
        };
        // (target, base colour, normal, occlusion); `None` is RGBA8.
        let cases = [
            (GltfTextureTarget::Rgba8, None, None, None),
            (
                GltfTextureTarget::Bc,
                Some((CompressedFormat::Bc7Rgba, 6)),
                Some((CompressedFormat::Bc5Rg, 5)),
                Some((CompressedFormat::Bc4R, 1)),
            ),
            (
                GltfTextureTarget::Etc2Eac,
                Some((CompressedFormat::Etc2Rgba8, 6)),
                Some((CompressedFormat::EacRg11, 5)),
                Some((CompressedFormat::EacR11, 1)),
            ),
            (
                GltfTextureTarget::Astc4x4,
                Some((astc, 6)),
                // X and Y in red and alpha cannot stay ASTC.
                None,
                Some((astc, 1)),
            ),
        ];
        for (target, colour, normal, occlusion) in cases {
            let scene = load(&blob, target);
            let material = &scene.materials[0];
            let base = decoded(&material.base_color_texture);
            let normal_map = decoded(&material.normal_map_texture);
            let ao = decoded(&material.ao_texture);
            for data in [base, normal_map, ao] {
                data.validate()
                    .unwrap_or_else(|e| panic!("{target:?}: {e}"));
            }
            assert_eq!((base.width(), base.height()), (32, 16), "{target:?}");
            assert_eq!((normal_map.width(), normal_map.height()), (16, 16));
            assert_eq!((ao.width(), ao.height()), (8, 8));
            assert_eq!(format(base), colour, "{target:?} base colour");
            assert_eq!(format(normal_map), normal, "{target:?} normal");
            assert_eq!(format(ao), occlusion, "{target:?} occlusion");
            assert_eq!(base.colour_space(), ColourSpace::Srgb);
            assert_eq!(ao.colour_space(), ColourSpace::Linear);
        }
    }

    #[test]
    fn a_zstd_uastc_file_from_a_real_encoder_transcodes_to_each_target() {
        let image = std::fs::read(fixture("color_grid_uastc_zstd.ktx2")).unwrap();
        let blob = basisu_triangle(&image, &image, &image);
        let cases = [
            (GltfTextureTarget::Rgba8, None),
            (GltfTextureTarget::Bc, Some((CompressedFormat::Bc7Rgba, 1))),
            // RGB only, so ETC2 without alpha.
            (
                GltfTextureTarget::Etc2Eac,
                Some((CompressedFormat::Etc2Rgb8, 1)),
            ),
            (
                GltfTextureTarget::Astc4x4,
                Some((
                    CompressedFormat::Astc {
                        block: AstcBlock::B4x4,
                        hdr: false,
                    },
                    1,
                )),
            ),
        ];
        for (target, expected) in cases {
            let scene = load(&blob, target);
            let base = decoded(&scene.materials[0].base_color_texture);
            base.validate().unwrap();
            assert_eq!((base.width(), base.height()), (1024, 1024));
            assert_eq!(format(base), expected, "{target:?}");
        }
    }

    /// A 1x1 RGBA PNG.
    const PNG_1X1: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F,
        0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0xF8,
        0xCF, 0xC0, 0xF0, 0x1F, 0x00, 0x05, 0x00, 0x01, 0xFF, 0x89, 0x99, 0x3D, 0x1D, 0x00, 0x00,
        0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];

    /// The `KHR_texture_basisu` image wins over the texture's plain `source`,
    /// and the plain `source` is the fallback when the KTX2 image is unreadable.
    #[test]
    fn the_basisu_image_wins_and_the_plain_source_is_the_fallback() {
        let ktx2 = synth::ktx2_uastc(8, 8, false, false, true, |_, _, _| [1, 2, 3, 255]);
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let mut views = vec![json!({"buffer": 0, "byteOffset": 0, "byteLength": 36})];
        for image in [PNG_1X1, &ktx2, b"KTX2 that is not"] {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
            views.push(json!({"buffer": 0, "byteOffset": bin.len(), "byteLength": image.len()}));
            bin.extend_from_slice(image);
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let json = json!({
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_texture_basisu"],
            "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": [{"mesh": 0}, {"mesh": 1}],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]},
                {"primitives": [{"attributes": {"POSITION": 0}, "material": 1}]}
            ],
            "materials": [
                {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}},
                {"pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}}
            ],
            "textures": [
                {"source": 0, "extensions": {"KHR_texture_basisu": {"source": 1}}},
                {"source": 0, "extensions": {"KHR_texture_basisu": {"source": 2}}}
            ],
            "images": [
                {"bufferView": 1, "mimeType": "image/png"},
                {"bufferView": 2, "mimeType": "image/ktx2"},
                {"bufferView": 3, "mimeType": "image/ktx2"}
            ],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [0, 0, 0], "max": [1, 1, 0]}],
            "bufferViews": views,
            "buffers": [{"byteLength": bin.len()}]
        });
        let blob = synth::glb(serde_json::to_string(&json).unwrap().as_bytes(), &bin);
        let scene = load(&blob, GltfTextureTarget::Bc);

        let ktx2_image = decoded(&scene.materials[0].base_color_texture);
        assert_eq!((ktx2_image.width(), ktx2_image.height()), (8, 8));
        assert_eq!(format(ktx2_image), Some((CompressedFormat::Bc7Rgba, 1)));

        let fallback = decoded(&scene.materials[1].base_color_texture);
        assert_eq!((fallback.width(), fallback.height()), (1, 1));
        assert_eq!(format(fallback), None);
    }
}
