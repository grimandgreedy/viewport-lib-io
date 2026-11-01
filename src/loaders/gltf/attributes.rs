//! Vertex attribute reading for any component type an accessor may store.
//!
//! The gltf crate reads positions, normals and tangents as `f32` whatever the
//! accessor says, so the integer encodings `KHR_mesh_quantization` allows
//! (and compressors such as gltfpack write) would come back as garbage. These
//! readers widen every component type to `f32`, normalising where the accessor
//! is marked normalised.

use gltf::accessor::{DataType, Item, Iter};

/// Read an accessor of `N`-component vectors as `f32`, widening integer
/// components and normalising them when the accessor is normalised. `None`
/// when the accessor has no readable data or holds a different component
/// count.
pub(super) fn read_vectors<const N: usize>(
    accessor: gltf::Accessor,
    buffers: &[gltf::buffer::Data],
) -> Option<Vec<[f32; N]>>
where
    [i8; N]: Item,
    [u8; N]: Item,
    [i16; N]: Item,
    [u16; N]: Item,
    [u32; N]: Item,
    [f32; N]: Item,
{
    if accessor.dimensions().multiplicity() != N {
        return None;
    }
    let normalized = accessor.normalized();
    let get = |buffer: gltf::Buffer| buffers.get(buffer.index()).map(|data| data.0.as_slice());
    match accessor.data_type() {
        DataType::F32 => Some(Iter::<[f32; N]>::new(accessor, get)?.collect()),
        DataType::I8 => Some(widen(Iter::<[i8; N]>::new(accessor, get)?, |v| {
            if normalized {
                (f32::from(v) / 127.0).max(-1.0)
            } else {
                f32::from(v)
            }
        })),
        DataType::U8 => Some(widen(Iter::<[u8; N]>::new(accessor, get)?, |v| {
            if normalized {
                f32::from(v) / 255.0
            } else {
                f32::from(v)
            }
        })),
        DataType::I16 => Some(widen(Iter::<[i16; N]>::new(accessor, get)?, |v| {
            if normalized {
                (f32::from(v) / 32767.0).max(-1.0)
            } else {
                f32::from(v)
            }
        })),
        DataType::U16 => Some(widen(Iter::<[u16; N]>::new(accessor, get)?, |v| {
            if normalized {
                f32::from(v) / 65535.0
            } else {
                f32::from(v)
            }
        })),
        DataType::U32 => Some(widen(Iter::<[u32; N]>::new(accessor, get)?, |v| v as f32)),
    }
}

fn widen<T: Copy, const N: usize>(
    values: impl Iterator<Item = [T; N]>,
    convert: impl Fn(T) -> f32,
) -> Vec<[f32; N]> {
    values.map(|value| value.map(&convert)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::scene_from_slice;
    use viewport_lib_io_testkit::synth::glb;

    /// A triangle with KHR_mesh_quantization attributes: positions as unsigned
    /// shorts, normals as normalised bytes, UVs as normalised unsigned shorts.
    #[test]
    fn reads_quantised_positions_normals_and_uvs() {
        let mut bin = Vec::new();
        for p in [[0u16, 0, 0], [2, 0, 0], [0, 2, 0]] {
            for v in p {
                bin.extend_from_slice(&v.to_le_bytes());
            }
            bin.extend_from_slice(&[0, 0]); // pad each vertex to 8 bytes
        }
        let normals_offset = bin.len();
        for _ in 0..3 {
            bin.extend_from_slice(&[0, 0, 127, 0]); // +Z, padded to 4 bytes
        }
        let uv_offset = bin.len();
        for uv in [[0u16, 0], [65535, 0], [0, 65535]] {
            for v in uv {
                bin.extend_from_slice(&v.to_le_bytes());
            }
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},
            "extensionsUsed":["KHR_mesh_quantization"],
            "extensionsRequired":["KHR_mesh_quantization"],
            "scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}}}}]}}],
            "accessors":[
              {{"bufferView":0,"componentType":5123,"count":3,"type":"VEC3","min":[0,0,0],"max":[2,2,0]}},
              {{"bufferView":1,"componentType":5120,"normalized":true,"count":3,"type":"VEC3"}},
              {{"bufferView":2,"componentType":5123,"normalized":true,"count":3,"type":"VEC2"}}],
            "bufferViews":[
              {{"buffer":0,"byteOffset":0,"byteLength":24,"byteStride":8}},
              {{"buffer":0,"byteOffset":{normals_offset},"byteLength":12,"byteStride":4}},
              {{"buffer":0,"byteOffset":{uv_offset},"byteLength":12}}],
            "buffers":[{{"byteLength":{len}}}]}}"#,
            len = bin.len(),
        );
        let scene = scene_from_slice(&glb(json.as_bytes(), &bin), None).unwrap();
        let mesh = &scene.meshes[0].mesh;

        // glTF (x, y, z) is (x, -z, y) in Z-up.
        assert_eq!(
            mesh.positions,
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
        );
        for normal in &mesh.normals {
            assert!(
                (normal[1] + 1.0).abs() < 1e-6,
                "glTF +Z is Z-up -Y: {normal:?}"
            );
        }
        assert_eq!(
            mesh.uvs.as_deref(),
            Some([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]].as_slice())
        );
    }
}
