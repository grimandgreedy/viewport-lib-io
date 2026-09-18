//! Blend shape geometry, in both the modern deformer form and the legacy form nested
//! inside the geometry node.

use fbxcel_dom::fbxcel;

use super::raw::{read_f64_array, read_i32_array};

/// One blend shape harvested from a geometry's BlendShape deformers: a name and
/// a dense per-control-point position delta (zero where the shape leaves a
/// control point in place).
#[cfg(feature = "fbx")]
pub(super) struct ExtractedBlendShape {
    pub(super) name: String,
    pub(super) cp_deltas: Vec<[f32; 3]>,
}

/// Read a geometry's blend shapes as dense per-control-point deltas.
///
/// FBX stores each blend shape as a `Deformer`(BlendShape) -> `SubDeformer`
/// (BlendShapeChannel) -> `Geometry`(Shape) chain. A Shape carries a **sparse**
/// `Indexes` (control-point indices) + `Vertices` (the position delta for each,
/// flat x/y/z) pair, which this expands into a dense array over all control
/// points. Deltas sit in the same local control-point space as the base
/// positions (the axis / unit conversion rides the mesh transform), so no
/// reorientation applies here. A channel with several in-between shapes keeps
/// the last shape's displacement per control point (the full-weight target);
/// progressive in-betweens are not blended.
///
/// Older FBX (version < 7.5, e.g. 7.1 / 7.2 as many game-character exports
/// still are) instead nests the `Shape` sub-nodes **directly inside the
/// `Geometry` node**, with no separate `BlendShape` deformer / `BlendShapeChannel`
/// objects in the connection graph, so the modern `blendshapes()` walk finds
/// nothing. When the modern chain yields no shapes, [`extract_legacy_shapes`]
/// falls back to reading those nested `Shape` sub-nodes off the raw geometry
/// tree, in the same sparse `Indexes` / `Vertices` layout.
#[cfg(feature = "fbx")]
pub(super) fn extract_blend_shapes(
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    control_point_count: usize,
) -> Vec<ExtractedBlendShape> {
    let mut out: Vec<ExtractedBlendShape> = Vec::new();
    for blendshape in geometry.blendshapes() {
        for channel in blendshape.blendshape_channels() {
            let name = channel.name().unwrap_or("").to_string();
            let mut cp_deltas = vec![[0.0f32; 3]; control_point_count];
            let mut any = false;
            for shape in channel.shapes() {
                let node = shape.node();
                if accumulate_shape_deltas(&node, &mut cp_deltas) {
                    any = true;
                }
            }
            if any {
                out.push(ExtractedBlendShape { name, cp_deltas });
            }
        }
    }
    // Legacy fallback: pre-7.5 exports carry the shapes as nested `Shape` sub-nodes
    // of the geometry rather than as connected `BlendShape` deformer objects.
    if out.is_empty() {
        out = extract_legacy_shapes(geometry, control_point_count);
    }
    out
}

/// Expand one `Shape` node's sparse `Indexes` + `Vertices` position deltas into
/// `cp_deltas` (dense over all control points), returning whether it touched any.
/// Shared by the modern deformer walk and the legacy nested-shape fallback.
#[cfg(feature = "fbx")]
pub(super) fn accumulate_shape_deltas(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    cp_deltas: &mut [[f32; 3]],
) -> bool {
    let indexes = read_i32_array(node, "Indexes").unwrap_or_default();
    let verts = read_f64_array(node, "Vertices").unwrap_or_default();
    expand_sparse_shape_deltas(&indexes, &verts, cp_deltas)
}

/// Scatter a Shape's sparse `Indexes` + flat `Vertices` (x/y/z per index) into
/// the dense per-control-point `cp_deltas`, returning whether it wrote anything.
/// The pure core shared by the modern deformer walk and the legacy nested-shape
/// fallback: out-of-range indices are skipped, and a truncated `Vertices` buffer
/// (fewer than `indexes.len() * 3`) is rejected wholesale rather than read past.
#[cfg(feature = "fbx")]
pub(super) fn expand_sparse_shape_deltas(
    indexes: &[i32],
    verts: &[f64],
    cp_deltas: &mut [[f32; 3]],
) -> bool {
    if indexes.is_empty() || verts.len() < indexes.len() * 3 {
        return false;
    }
    let mut any = false;
    for (k, &cp_i32) in indexes.iter().enumerate() {
        let cp = cp_i32 as usize;
        if cp >= cp_deltas.len() {
            continue;
        }
        let base = k * 3;
        cp_deltas[cp] = [
            verts[base] as f32,
            verts[base + 1] as f32,
            verts[base + 2] as f32,
        ];
        any = true;
    }
    any
}

/// Read legacy (pre-7.5) blend shapes nested as `Shape` sub-nodes inside the
/// geometry node. Each `Shape` node names the target in its first string
/// attribute and carries the same sparse `Indexes` / `Vertices` delta layout as
/// a modern Shape geometry. The ARKit facial rigs many game characters ship
/// (jawOpen / eyeBlinkLeft / ...) are stored this way.
#[cfg(feature = "fbx")]
pub(super) fn extract_legacy_shapes(
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    control_point_count: usize,
) -> Vec<ExtractedBlendShape> {
    let mut out: Vec<ExtractedBlendShape> = Vec::new();
    for shape in geometry.node().children_by_name("Shape") {
        let name = shape
            .attributes()
            .iter()
            .find_map(|a| match a {
                fbxcel::low::v7400::AttributeValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let mut cp_deltas = vec![[0.0f32; 3]; control_point_count];
        if accumulate_shape_deltas(&shape, &mut cp_deltas) {
            out.push(ExtractedBlendShape { name, cp_deltas });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blend shape's sparse `Indexes` + flat `Vertices` scatter into the dense
    /// per-control-point delta array, leaving untouched control points at zero.
    /// The shared core of the modern deformer walk and the legacy nested-shape
    /// fallback (the ARKit rigs pre-7.5 FBX stores inside the geometry node).
    #[test]
    fn sparse_shape_deltas_scatter_into_dense_control_points() {
        // Two of four control points move; index order need not be ascending.
        let indexes = [2, 0];
        let verts = [1.0, 2.0, 3.0, -4.0, 0.0, 0.5];
        let mut dense = vec![[0.0f32; 3]; 4];
        assert!(expand_sparse_shape_deltas(&indexes, &verts, &mut dense));
        assert_eq!(dense[2], [1.0, 2.0, 3.0]);
        assert_eq!(dense[0], [-4.0, 0.0, 0.5]);
        // Unreferenced control points stay at rest.
        assert_eq!(dense[1], [0.0, 0.0, 0.0]);
        assert_eq!(dense[3], [0.0, 0.0, 0.0]);
    }

    /// An out-of-range control-point index is skipped, and a `Vertices` buffer
    /// too short for the index count is rejected wholesale rather than read past.
    #[test]
    fn sparse_shape_deltas_reject_truncated_and_skip_out_of_range() {
        let mut dense = vec![[0.0f32; 3]; 2];
        // 9 (>= len 2) is skipped; index 1 still applies, so it wrote something.
        assert!(expand_sparse_shape_deltas(&[9, 1], &[7.0; 6], &mut dense));
        assert_eq!(dense[1], [7.0, 7.0, 7.0]);
        assert_eq!(dense[0], [0.0, 0.0, 0.0]);

        // Truncated Vertices (needs 6 floats for 2 indices, has 3) writes nothing.
        let mut dense2 = vec![[0.0f32; 3]; 2];
        assert!(!expand_sparse_shape_deltas(
            &[0, 1],
            &[1.0, 2.0, 3.0],
            &mut dense2
        ));
        assert_eq!(dense2, vec![[0.0f32; 3]; 2]);

        // Empty is a no-op.
        assert!(!expand_sparse_shape_deltas(&[], &[], &mut dense2));
    }
}
