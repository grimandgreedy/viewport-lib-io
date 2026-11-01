//! Buffer loading: the document's buffers read into memory, with every
//! `EXT_meshopt_compression` buffer view decoded into place before any
//! accessor reads it.
//!
//! A meshopt view keeps its uncompressed layout in the `buffer`, `byteOffset`
//! and `byteLength` it declares, and carries the compressed bytes in a second
//! buffer named by the extension. The decoder writes each view's decoded bytes
//! back into the declared range, so the accessors and the gltf crate's readers
//! see an ordinary buffer. The declared buffer is often a fallback buffer with
//! no data of its own (`"fallback": true` and no URI); it is allocated zeroed
//! and filled by the decoded views.

use std::path::Path;

use crate::error::IoError;

const MESHOPT: &str = "EXT_meshopt_compression";

/// Read every buffer of `document`, then decode its meshopt views.
pub(super) fn load_buffers(
    document: &gltf::Document,
    base: Option<&Path>,
    mut blob: Option<Vec<u8>>,
) -> Result<Vec<gltf::buffer::Data>, IoError> {
    let mut buffers = Vec::with_capacity(document.buffers().len());
    for buffer in document.buffers() {
        let fallback = is_meshopt_fallback(&buffer);
        let data = if fallback && cfg!(feature = "gltf-meshopt") {
            // Only decoded views are ever read from it, so any URI it names
            // is skipped.
            gltf::buffer::Data(vec![0; buffer.length().next_multiple_of(4)])
        } else if fallback && matches!(buffer.source(), gltf::buffer::Source::Bin) {
            // Without the decoder a fallback buffer with no URI has nothing
            // to read.
            return Err(IoError::MissingFeature {
                feature: "gltf-meshopt",
                context: "glTF buffers compressed with EXT_meshopt_compression",
            });
        } else {
            let data =
                gltf::buffer::Data::from_source_and_blob(buffer.source(), base, &mut blob)
                    .map_err(|error| IoError::Parse(format!("glTF buffers failed: {error:?}")))?;
            if data.0.len() < buffer.length() {
                return Err(IoError::Parse(format!(
                    "glTF buffer {} holds {} bytes, declares {}",
                    buffer.index(),
                    data.0.len(),
                    buffer.length(),
                )));
            }
            data
        };
        buffers.push(data);
    }

    #[cfg(feature = "gltf-meshopt")]
    meshopt::decode_views(document, &mut buffers)?;

    Ok(buffers)
}

/// Whether `buffer` is a meshopt fallback buffer: one whose bytes are only the
/// decoded output of meshopt views.
fn is_meshopt_fallback(buffer: &gltf::Buffer) -> bool {
    buffer
        .extension_value(MESHOPT)
        .and_then(|extension| extension.get("fallback"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

#[cfg(feature = "gltf-meshopt")]
mod meshopt {
    use serde_json::Value;

    use super::MESHOPT;
    use crate::error::IoError;

    /// One `EXT_meshopt_compression` view description.
    struct CompressedView {
        buffer: usize,
        offset: usize,
        length: usize,
        stride: usize,
        count: usize,
        mode: Mode,
        filter: Filter,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mode {
        Attributes,
        Triangles,
        Indices,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Filter {
        None,
        Octahedral,
        Quaternion,
        Exponential,
    }

    /// Decode every meshopt view of `document` into the range of the buffer it
    /// declares.
    pub(super) fn decode_views(
        document: &gltf::Document,
        buffers: &mut [gltf::buffer::Data],
    ) -> Result<(), IoError> {
        for view in document.views() {
            let Some(extension) = view.extension_value(MESHOPT) else {
                continue;
            };
            let compressed = parse(extension)
                .ok_or_else(|| invalid(view.index(), "malformed extension object"))?;
            let source = buffers
                .get(compressed.buffer)
                .and_then(|data| {
                    data.0
                        .get(compressed.offset..compressed.offset + compressed.length)
                })
                .ok_or_else(|| invalid(view.index(), "compressed range is outside its buffer"))?;
            let decoded =
                decode(&compressed, source).map_err(|reason| invalid(view.index(), &reason))?;

            let target = buffers
                .get_mut(view.buffer().index())
                .and_then(|data| data.0.get_mut(view.offset()..view.offset() + view.length()))
                .ok_or_else(|| invalid(view.index(), "view range is outside its buffer"))?;
            let len = decoded.len().min(target.len());
            target[..len].copy_from_slice(&decoded[..len]);
        }
        Ok(())
    }

    fn invalid(view: usize, reason: &str) -> IoError {
        IoError::Parse(format!("glTF buffer view {view}: {MESHOPT}: {reason}"))
    }

    fn parse(extension: &Value) -> Option<CompressedView> {
        let int = |key: &str| {
            extension
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        };
        let mode = match extension.get("mode")?.as_str()? {
            "ATTRIBUTES" => Mode::Attributes,
            "TRIANGLES" => Mode::Triangles,
            "INDICES" => Mode::Indices,
            _ => return None,
        };
        let filter = match extension
            .get("filter")
            .and_then(Value::as_str)
            .unwrap_or("NONE")
        {
            "NONE" => Filter::None,
            "OCTAHEDRAL" => Filter::Octahedral,
            "QUATERNION" => Filter::Quaternion,
            "EXPONENTIAL" => Filter::Exponential,
            _ => return None,
        };
        Some(CompressedView {
            buffer: int("buffer")?,
            offset: int("byteOffset").unwrap_or(0),
            length: int("byteLength")?,
            stride: int("byteStride")?,
            count: int("count")?,
            mode,
            filter,
        })
    }

    /// Decode one view to `count * stride` bytes.
    fn decode(view: &CompressedView, source: &[u8]) -> Result<Vec<u8>, String> {
        let size = view
            .count
            .checked_mul(view.stride)
            .ok_or_else(|| "decoded size overflows".to_owned())?;
        let error = |error: meshoptimizer::Error| error.to_string();
        let mut out = vec![0u8; size];
        match view.mode {
            Mode::Attributes => {
                meshoptimizer::decode_vertex_buffer(
                    &mut out,
                    view.count,
                    view.stride,
                    source,
                    source.len(),
                )
                .map_err(error)?;
                match view.filter {
                    Filter::None => {}
                    Filter::Octahedral => {
                        meshoptimizer::decode_filter_oct(&mut out, view.count, view.stride)
                            .map_err(error)?
                    }
                    Filter::Quaternion => {
                        meshoptimizer::decode_filter_quat(&mut out, view.count, view.stride)
                            .map_err(error)?
                    }
                    Filter::Exponential => {
                        meshoptimizer::decode_filter_exp(&mut out, view.count, view.stride)
                            .map_err(error)?
                    }
                }
            }
            Mode::Triangles | Mode::Indices => {
                if view.filter != Filter::None {
                    return Err("index data takes no filter".to_owned());
                }
                let sequence = view.mode == Mode::Indices;
                match view.stride {
                    2 => {
                        let mut indices = vec![0u16; view.count];
                        decode_indices(&mut indices, sequence, source).map_err(error)?;
                        for (bytes, index) in out.chunks_exact_mut(2).zip(indices) {
                            bytes.copy_from_slice(&index.to_le_bytes());
                        }
                    }
                    4 => {
                        let mut indices = vec![0u32; view.count];
                        decode_indices(&mut indices, sequence, source).map_err(error)?;
                        for (bytes, index) in out.chunks_exact_mut(4).zip(indices) {
                            bytes.copy_from_slice(&index.to_le_bytes());
                        }
                    }
                    other => return Err(format!("index stride {other} is not 2 or 4")),
                }
            }
        }
        Ok(out)
    }

    fn decode_indices<T: meshoptimizer::IndexType>(
        indices: &mut [T],
        sequence: bool,
        source: &[u8],
    ) -> meshoptimizer::Result<()> {
        let count = indices.len();
        if sequence {
            meshoptimizer::decode_index_sequence(indices, count, source, source.len())
        } else {
            meshoptimizer::decode_index_buffer(indices, count, source, source.len())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::super::scene_from_slice;
        use viewport_lib_io_testkit::synth::glb;

        fn encode_vertices(bytes: &[u8], count: usize, stride: usize) -> Vec<u8> {
            let bound = meshoptimizer::encode_vertex_buffer_bound(count, stride).unwrap();
            let mut out = vec![0; bound];
            // Version 0: the only vertex encoding EXT_meshopt_compression allows.
            let len = meshoptimizer::encode_vertex_buffer_level(
                &mut out, bound, bytes, count, stride, 2, 0,
            )
            .unwrap();
            out.truncate(len);
            out
        }

        fn pad(bin: &mut Vec<u8>) {
            while bin.len() % 4 != 0 {
                bin.push(0);
            }
        }

        /// A triangle whose positions, octahedral-filtered normals and indices
        /// all live in meshopt views over a fallback buffer with no data.
        #[test]
        fn decodes_attribute_filter_and_index_views_into_a_fallback_buffer() {
            let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
            let position_bytes: Vec<u8> = positions
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let mut normal_bytes = vec![0u8; 12];
            meshoptimizer::encode_filter_oct(
                &mut normal_bytes,
                3,
                4,
                8,
                &[0.0, 0.0, 1.0, 0.0].repeat(3),
            )
            .unwrap();
            let indices = [0u16, 1, 2];
            let bound = meshoptimizer::encode_index_buffer_bound(3, 3).unwrap();
            let mut index_bytes = vec![0; bound];
            let len =
                meshoptimizer::encode_index_buffer(&mut index_bytes, bound, &indices, 3).unwrap();
            index_bytes.truncate(len);

            let mut bin = Vec::new();
            let mut ranges = Vec::new();
            for encoded in [
                encode_vertices(&position_bytes, 3, 12),
                encode_vertices(&normal_bytes, 3, 4),
                index_bytes,
            ] {
                ranges.push((bin.len(), encoded.len()));
                bin.extend_from_slice(&encoded);
                pad(&mut bin);
            }
            let ext = |(offset, length): (usize, usize),
                       stride: usize,
                       mode: &str,
                       filter: &str| {
                format!(
                    r#"{{"EXT_meshopt_compression":{{"buffer":0,"byteOffset":{offset},"byteLength":{length},"byteStride":{stride},"count":3,"mode":"{mode}","filter":"{filter}"}}}}"#
                )
            };
            let json = format!(
                r#"{{"asset":{{"version":"2.0"}},
                "extensionsUsed":["EXT_meshopt_compression"],
                "extensionsRequired":["EXT_meshopt_compression"],
                "scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
                "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}}],
                "accessors":[
                  {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
                  {{"bufferView":1,"componentType":5120,"normalized":true,"count":3,"type":"VEC3"}},
                  {{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}],
                "bufferViews":[
                  {{"buffer":1,"byteOffset":0,"byteLength":36,"byteStride":12,"extensions":{p}}},
                  {{"buffer":1,"byteOffset":36,"byteLength":12,"byteStride":4,"extensions":{n}}},
                  {{"buffer":1,"byteOffset":48,"byteLength":6,"extensions":{i}}}],
                "buffers":[
                  {{"byteLength":{bin_len}}},
                  {{"byteLength":56,"extensions":{{"EXT_meshopt_compression":{{"fallback":true}}}}}}]}}"#,
                p = ext(ranges[0], 12, "ATTRIBUTES", "NONE"),
                n = ext(ranges[1], 4, "ATTRIBUTES", "OCTAHEDRAL"),
                i = ext(ranges[2], 2, "TRIANGLES", "NONE"),
                bin_len = bin.len(),
            );

            let scene = scene_from_slice(&glb(json.as_bytes(), &bin), None).unwrap();
            let mesh = &scene.meshes[0].mesh;
            // glTF (x, y, z) is (x, -z, y) in Z-up.
            assert_eq!(
                mesh.positions,
                vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            );
            assert_eq!(mesh.indices, vec![0, 1, 2]);
            for normal in &mesh.normals {
                assert!(
                    (normal[1] + 1.0).abs() < 0.02,
                    "glTF +Z is Z-up -Y: {normal:?}"
                );
            }
        }
    }
}
