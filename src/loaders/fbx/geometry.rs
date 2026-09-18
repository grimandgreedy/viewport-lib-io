//! Per-mesh geometry that needs no document access: UV layer choice, triangulation,
//! and flat normals.

use anyhow::Error as AnyhowError;
use fbxcel_dom::v7400::data::mesh::{PolygonVertexIndex, PolygonVertices};

/// Choose which of a mesh's several UV layers the base map samples, given the
/// per-vertex UVs of each (in authored order, so index 0 is UV0).
///
/// The rule is Unity's: the base map samples the **first authored layer** (UV0);
/// UV value ranges are never a selection signal. The one case that departs from
/// "just take UV0" is a layer that is not a texture coordinate at all: a packed
/// per-vertex scalar (a wind phase in V with U held constant), whose value on
/// one axis does not vary across the mesh. Such a layer is stepped over in
/// favour of the next. An atlas layer that bakes an integer per-card offset into
/// V (foliage wind) still varies on both axes: its raw range is large but real,
/// and repeat-wrap sampling resolves the offset, so it reads as the texture
/// coordinate it is and, being UV0, is chosen. Magnitude is deliberately not
/// consulted, only whether each axis varies at all.
///
/// Returns the chosen index; `0` when every layer is degenerate (better to keep
/// UV0's coordinates than to drop UVs). `candidates` must be non-empty.
#[cfg(feature = "fbx")]
pub(super) fn pick_uv_channel(candidates: &[Vec<[f32; 2]>]) -> usize {
    candidates
        .iter()
        .position(|uvs| is_texture_coordinate(uvs))
        .unwrap_or(0)
}

/// Whether a layer is a texture coordinate at all, as opposed to a packed
/// per-vertex scalar.
///
/// Raw span of the layer's coordinates on each axis. A genuine 2D texture
/// coordinate varies on both; a packed scalar (a wind phase in V) holds one axis
/// constant. Wrapping is intentionally NOT applied: a value at exactly 1.0 would
/// fold onto 0.0 and make a clean [0,1] square look constant, and magnitude is
/// not a signal anyway.
#[cfg(feature = "fbx")]
pub(super) fn is_texture_coordinate(uvs: &[[f32; 2]]) -> bool {
    // An axis that varies less than this across the whole mesh is a constant,
    // not a texture coordinate.
    const CONSTANT: f32 = 1e-4;
    let (mut umin, mut umax, mut vmin, mut vmax) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    for uv in uvs {
        umin = umin.min(uv[0]);
        umax = umax.max(uv[0]);
        vmin = vmin.min(uv[1]);
        vmax = vmax.max(uv[1]);
    }
    (umax - umin).max(0.0) > CONSTANT && (vmax - vmin).max(0.0) > CONSTANT
}

/// The layer that becomes the second UV set, given the one chosen as UV0.
///
/// The first genuine texture-coordinate layer authored after it, by the same
/// test [`pick_uv_channel`] uses: a layer holding one axis constant is a packed
/// per-vertex scalar rather than a UV set, and is stepped over here as it is
/// there. `None` when there is no later layer that qualifies.
#[cfg(feature = "fbx")]
pub(super) fn pick_second_uv_channel(candidates: &[Vec<[f32; 2]>], first: usize) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .skip(first + 1)
        .find(|(_, uvs)| is_texture_coordinate(uvs))
        .map(|(index, _)| index)
}

pub(super) fn fan_triangulator(
    polygon_vertices: &PolygonVertices<'_>,
    polygon_vertex_indices: &[PolygonVertexIndex],
    results: &mut Vec<[PolygonVertexIndex; 3]>,
) -> Result<(), AnyhowError> {
    match polygon_vertex_indices.len() {
        0..=2 => {}
        3 => {
            results.push([
                polygon_vertex_indices[0],
                polygon_vertex_indices[1],
                polygon_vertex_indices[2],
            ]);
        }
        4 => {
            let point = |i: usize| -> Option<glam::Vec3> {
                let point = polygon_vertices.control_point(polygon_vertex_indices[i])?;
                Some(glam::Vec3::new(
                    point.x as f32,
                    point.y as f32,
                    point.z as f32,
                ))
            };
            if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                (point(0), point(1), point(2), point(3))
            {
                let n1 = (p0 - p1).cross(p2 - p1);
                let n3 = (p2 - p3).cross(p0 - p3);
                if n1.dot(n3) >= 0.0 {
                    results.push([
                        polygon_vertex_indices[0],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[2],
                    ]);
                    results.push([
                        polygon_vertex_indices[2],
                        polygon_vertex_indices[3],
                        polygon_vertex_indices[0],
                    ]);
                } else {
                    results.push([
                        polygon_vertex_indices[0],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[3],
                    ]);
                    results.push([
                        polygon_vertex_indices[3],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[2],
                    ]);
                }
            } else {
                results.push([
                    polygon_vertex_indices[0],
                    polygon_vertex_indices[1],
                    polygon_vertex_indices[2],
                ]);
                results.push([
                    polygon_vertex_indices[2],
                    polygon_vertex_indices[3],
                    polygon_vertex_indices[0],
                ]);
            }
        }
        n => {
            for i in 1..n - 1 {
                results.push([
                    polygon_vertex_indices[0],
                    polygon_vertex_indices[i],
                    polygon_vertex_indices[i + 1],
                ]);
            }
        }
    }
    Ok(())
}

pub(super) fn compute_flat_normals(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32, 0.0, 1.0]; positions.len()];
    for (i, triangle) in positions.chunks_exact(3).enumerate() {
        let v0 = glam::Vec3::from(triangle[0]);
        let v1 = glam::Vec3::from(triangle[1]);
        let v2 = glam::Vec3::from(triangle[2]);
        let n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
        let base = i * 3;
        normals[base] = [n.x, n.y, n.z];
        normals[base + 1] = [n.x, n.y, n.z];
        normals[base + 2] = [n.x, n.y, n.z];
    }
    normals
}

// ---------------------------------------------------------------------------
// Skin extraction
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A single square patch of texture coordinates on both layers: the base map
    /// samples UV0, the first authored layer, the way Unity binds it.
    #[test]
    fn uv_pick_prefers_first_layer() {
        let square = || vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_uv_channel(&[square(), square()]), 0);
    }

    /// A foliage atlas that bakes an integer per-card offset into V (V running to
    /// tens) is still a texture coordinate once wrapped, so UV0 is chosen, not
    /// rejected as a data channel. This is the RomanStreet tree's dark-branches
    /// case: the offset channel is the correct albedo.
    #[test]
    fn uv_pick_keeps_offset_atlas_on_uv0() {
        // UV0: real texcoords plus a large integer card offset in V.
        let offset_atlas = vec![
            [0.1, 12.0],
            [0.9, 12.0],
            [0.9, 12.9],
            [0.1, 12.9],
            [0.2, 47.1],
            [0.8, 47.8],
        ];
        // UV1: an ordinary lightmap-style unwrap in [0, 1].
        let lightmap = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.3, 0.7],
        ];
        assert_eq!(pick_uv_channel(&[offset_atlas, lightmap]), 0);
    }

    /// A packed per-vertex scalar in UV0 (a wind phase in V, U held constant) is
    /// not a texture coordinate (it collapses one axis even after wrapping), so
    /// the pick steps over it to the real texture layer, UV1.
    #[test]
    fn uv_pick_steps_over_packed_data_channel() {
        // UV0: U constant at 0, V an arbitrary per-vertex scalar (wind phase).
        let packed = vec![[0.0, 3.2], [0.0, 41.7], [0.0, 8.9], [0.0, 70.0]];
        // UV1: a proper texture-coordinate square.
        let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_uv_channel(&[packed, texcoords]), 1);
    }

    /// The layer after UV0 becomes the second UV set: the lightmap-style unwrap
    /// an FBX carries alongside its atlas channel is kept rather than dropped.
    #[test]
    fn second_uv_set_is_the_next_authored_layer() {
        let atlas = vec![[0.1, 12.0], [0.9, 12.0], [0.9, 12.9], [0.1, 12.9]];
        let lightmap = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_second_uv_channel(&[atlas, lightmap], 0), Some(1));
    }

    /// A packed per-vertex scalar is not a UV set, so it is stepped over for the
    /// second channel exactly as it is for the first.
    #[test]
    fn second_uv_set_steps_over_packed_data_channel() {
        let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let packed = vec![[0.0, 3.2], [0.0, 41.7], [0.0, 8.9], [0.0, 70.0]];
        let lightmap = vec![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
        assert_eq!(
            pick_second_uv_channel(&[texcoords.clone(), packed.clone()], 0),
            None
        );
        assert_eq!(
            pick_second_uv_channel(&[texcoords, packed, lightmap], 0),
            Some(2)
        );
    }

    /// One UV layer means one UV set, not a second one pointing at the same data.
    #[test]
    fn second_uv_set_absent_with_a_single_layer() {
        let square = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_second_uv_channel(&[square], 0), None);
    }

    /// Every layer degenerate: keep UV0's coordinates rather than drop UVs.
    #[test]
    fn uv_pick_falls_back_to_uv0_when_all_degenerate() {
        let flat_u = vec![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
        let flat_v = vec![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        assert_eq!(pick_uv_channel(&[flat_u, flat_v]), 0);
    }
}
