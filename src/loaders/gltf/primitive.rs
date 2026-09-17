//! One glTF primitive to one `SurfaceMesh`.

use crate::types::{IoMesh, MorphTarget, SkinWeights, SurfaceMesh};

use super::skin::normalise_skin_weights;

/// Convert one glTF primitive into a mesh: positions, indices, normals,
/// tangents, UVs, vertex colours, skin weights, and morph targets.
///
/// Returns `None` for a primitive with no positions, or one whose mode is not
/// triangles. Coordinates stay in glTF's Y-up frame here; the caller reorients
/// the finished mesh.
pub(super) fn convert_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mesh: &gltf::Mesh,
    primitive_index: usize,
) -> Option<IoMesh> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    if positions.is_empty() {
        return None;
    }

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let normals = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| compute_vertex_normals(&positions, &indices));

    let uvs = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect());

    let tangents = reader.read_tangents().map(|iter| iter.collect());

    // COLOR_0: per-vertex colour. glTF stores it as vec3 or vec4 in u8, u16, or
    // f32; `into_rgba_f32` normalises all of them to `[f32; 4]` (expanding vec3
    // with alpha 1.0).
    let colours: Option<Vec<[f32; 4]>> = reader
        .read_colors(0)
        .map(|iter| iter.into_rgba_f32().collect());

    // Skin attributes. glTF stores joint indices as either u8 or u16; we
    // normalise to u8 because the runtime substrate uses [u8; 4] today. Joint
    // indices above 255 are clamped with a warning.
    let joint_indices_u16: Option<Vec<[u16; 4]>> =
        reader.read_joints(0).map(|iter| iter.into_u16().collect());
    let joint_weights: Option<Vec<[f32; 4]>> =
        reader.read_weights(0).map(|iter| iter.into_f32().collect());
    let skin_weights = match (joint_indices_u16, joint_weights) {
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
    let morph_targets: Vec<MorphTarget> = reader
        .read_morph_targets()
        .enumerate()
        .map(|(i, (positions, normals, tangents))| {
            // Keep every target, even one with no POSITION displacement (some
            // face rigs carry a normals-only target): a dropped target would
            // shift the indices a weight animation drives by, and change the
            // count so the whole clip no longer matches. Zero-fill instead.
            let position_deltas: Vec<[f32; 3]> = match positions {
                Some(iter) => iter.collect(),
                None => vec![[0.0, 0.0, 0.0]; base_vertex_count],
            };
            MorphTarget {
                name: format!("target_{i}"),
                position_deltas,
                normal_deltas: normals.map(|iter| iter.collect()),
                tangent_deltas: tangents.map(|iter| iter.collect()),
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

    let mut mesh_data = SurfaceMesh::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    mesh_data.uvs = uvs;
    mesh_data.tangents = tangents;
    mesh_data.colours = colours;
    mesh_data.skin_weights = skin_weights;
    mesh_data.morph_targets = morph_targets;

    Some(IoMesh {
        name,
        mesh: mesh_data,
        material_index,
        transform: glam::Mat4::IDENTITY,
        two_sided: primitive.material().double_sided(),
        parent_index: None,
        ..IoMesh::default()
    })
}

/// Area-weighted vertex normals, for a primitive that ships without them.
///
/// Each face contributes its unnormalised cross product, so larger triangles
/// pull harder, then each vertex normal is normalised at the end.
pub(super) fn compute_vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let (i0, i1, i2) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let v0 = glam::Vec3::from_array(positions[i0]);
        let v1 = glam::Vec3::from_array(positions[i1]);
        let v2 = glam::Vec3::from_array(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0);
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }

    normals
        .into_iter()
        .map(|n| {
            let n = n.normalize_or_zero();
            [n.x, n.y, n.z]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::testkit::synth::temp_dir;
    use super::super::*;

    #[cfg(feature = "gltf")]
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
        let got = mesh.mesh.colours.as_ref().expect("colours missing");
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

    #[cfg(feature = "gltf")]
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
        assert_eq!(mesh.mesh.morph_targets.len(), 1, "one target expected");
        let target = &mesh.mesh.morph_targets[0];
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
