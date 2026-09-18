//! Connectivity and structured-grid layout: cell faces, surface extraction, the
//! triangulators, and the position grids an image or rectilinear extent implies.

use std::collections::HashMap;

use vtkio::model::{CellType, Cells, VertexNumbers};

use super::buffers::collect_cells;

pub(super) fn extract_surface_unstructured(cells: &Cells) -> Vec<u32> {
    let cell_verts = collect_cells(&cells.cell_verts);
    let mut face_counts: HashMap<[u32; 3], usize> = HashMap::new();

    for (cell_type, verts) in cells.types.iter().zip(cell_verts.iter()) {
        for tri in cell_faces(*cell_type, verts) {
            *face_counts.entry(sorted_tri(tri)).or_insert(0) += 1;
        }
    }

    // Walk the cells a second time rather than the map. A `HashMap` yields its
    // keys in an order seeded per process, so collecting the surface straight
    // out of it gave the same file a different triangle order on every run.
    // Going back through the cells emits them in the order the file wrote them,
    // which is both stable and what every other mesh loader here produces.
    let mut indices = Vec::new();
    for (cell_type, verts) in cells.types.iter().zip(cell_verts.iter()) {
        for tri in cell_faces(*cell_type, verts) {
            if face_counts.get(&sorted_tri(tri)) == Some(&1) {
                indices.extend_from_slice(&tri);
            }
        }
    }
    indices
}

pub(super) fn cell_faces(cell_type: CellType, verts: &[u32]) -> Vec<[u32; 3]> {
    match cell_type {
        CellType::Triangle | CellType::QuadraticTriangle | CellType::BiquadraticTriangle => {
            if verts.len() >= 3 {
                vec![[verts[0], verts[1], verts[2]]]
            } else {
                vec![]
            }
        }
        CellType::Quad
        | CellType::QuadraticQuad
        | CellType::BiquadraticQuad
        | CellType::QuadraticLinearQuad => {
            if verts.len() >= 4 {
                vec![
                    [verts[0], verts[1], verts[2]],
                    [verts[0], verts[2], verts[3]],
                ]
            } else {
                vec![]
            }
        }
        CellType::Tetra | CellType::QuadraticTetra => {
            if verts.len() >= 4 {
                let (v0, v1, v2, v3) = (verts[0], verts[1], verts[2], verts[3]);
                vec![[v0, v1, v2], [v0, v1, v3], [v1, v2, v3], [v0, v2, v3]]
            } else {
                vec![]
            }
        }
        CellType::Hexahedron
        | CellType::QuadraticHexahedron
        | CellType::TriquadraticHexahedron
        | CellType::BiquadraticQuadraticHexahedron => {
            if verts.len() >= 8 {
                hex_faces(
                    verts[0], verts[1], verts[2], verts[3], verts[4], verts[5], verts[6], verts[7],
                )
            } else {
                vec![]
            }
        }
        CellType::Wedge
        | CellType::QuadraticWedge
        | CellType::QuadraticLinearWedge
        | CellType::BiquadraticQuadraticWedge => {
            if verts.len() >= 6 {
                wedge_faces(verts[0], verts[1], verts[2], verts[3], verts[4], verts[5])
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}

pub(super) fn hex_faces(
    v0: u32,
    v1: u32,
    v2: u32,
    v3: u32,
    v4: u32,
    v5: u32,
    v6: u32,
    v7: u32,
) -> Vec<[u32; 3]> {
    vec![
        [v0, v1, v2],
        [v0, v2, v3],
        [v4, v6, v5],
        [v4, v7, v6],
        [v0, v1, v5],
        [v0, v5, v4],
        [v3, v7, v6],
        [v3, v6, v2],
        [v0, v4, v7],
        [v0, v7, v3],
        [v1, v2, v6],
        [v1, v6, v5],
    ]
}

pub(super) fn wedge_faces(v0: u32, v1: u32, v2: u32, v3: u32, v4: u32, v5: u32) -> Vec<[u32; 3]> {
    vec![
        [v0, v1, v2],
        [v3, v5, v4],
        [v0, v1, v4],
        [v0, v4, v3],
        [v1, v2, v5],
        [v1, v5, v4],
        [v2, v0, v3],
        [v2, v3, v5],
    ]
}

pub(super) fn sorted_tri(mut tri: [u32; 3]) -> [u32; 3] {
    tri.sort_unstable();
    tri
}

pub(super) fn vtk_extent_to_dims(extent: vtkio::model::Extent) -> [usize; 3] {
    let [ni, nj, nk] = extent.into_dims();
    [ni as usize, nj as usize, nk as usize]
}

pub(super) fn structured_surface_indices(ni: usize, nj: usize, nk: usize) -> Vec<u32> {
    let idx = |i: usize, j: usize, k: usize| -> u32 { (i + j * ni + k * ni * nj) as u32 };
    let mut out = Vec::new();
    let k_faces: Vec<usize> = if nk <= 1 { vec![0] } else { vec![0, nk - 1] };
    let j_faces: Vec<usize> = if nj <= 1 { vec![0] } else { vec![0, nj - 1] };
    let i_faces: Vec<usize> = if ni <= 1 { vec![0] } else { vec![0, ni - 1] };

    for &k in &k_faces {
        for j in 0..nj.saturating_sub(1) {
            for i in 0..ni.saturating_sub(1) {
                let (a, b, c, d) = (
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                );
                out.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
    for &j in &j_faces {
        for k in 0..nk.saturating_sub(1) {
            for i in 0..ni.saturating_sub(1) {
                let (a, b, c, d) = (
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j, k + 1),
                    idx(i, j, k + 1),
                );
                out.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
    for &i in &i_faces {
        for k in 0..nk.saturating_sub(1) {
            for j in 0..nj.saturating_sub(1) {
                let (a, b, c, d) = (
                    idx(i, j, k),
                    idx(i, j + 1, k),
                    idx(i, j + 1, k + 1),
                    idx(i, j, k + 1),
                );
                out.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
    out
}

pub(super) fn image_data_positions(
    ni: usize,
    nj: usize,
    nk: usize,
    origin: [f32; 3],
    spacing: [f32; 3],
) -> Vec<[f32; 3]> {
    let flat_z = nk <= 1;
    let mut positions = Vec::with_capacity(ni * nj * nk.max(1));
    for k in 0..nk.max(1) {
        for j in 0..nj {
            for i in 0..ni {
                let x = origin[0] + i as f32 * spacing[0];
                let (y, z) = if flat_z {
                    (
                        origin[2] + k as f32 * spacing[2],
                        origin[1] + j as f32 * spacing[1],
                    )
                } else {
                    (
                        origin[1] + j as f32 * spacing[1],
                        origin[2] + k as f32 * spacing[2],
                    )
                };
                positions.push([x, y, z]);
            }
        }
    }
    positions
}

pub(super) fn rectilinear_positions(xs: &[f32], ys: &[f32], zs: &[f32]) -> Vec<[f32; 3]> {
    let flat_z = zs.len() <= 1;
    let mut positions = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &fz in zs {
        for &fy in ys {
            for &x in xs {
                let (y, z) = if flat_z { (fz, fy) } else { (fy, fz) };
                positions.push([x, y, z]);
            }
        }
    }
    positions
}

pub(super) fn triangulate_vertex_numbers(vn: &VertexNumbers, out: &mut Vec<u32>) {
    for cell in collect_cells(vn) {
        fan_triangulate(&cell, out);
    }
}

pub(super) fn triangulate_strips(vn: &VertexNumbers, out: &mut Vec<u32>) {
    for strip in collect_cells(vn) {
        for index in 0..strip.len().saturating_sub(2) {
            if index % 2 == 0 {
                out.extend_from_slice(&[strip[index], strip[index + 1], strip[index + 2]]);
            } else {
                out.extend_from_slice(&[strip[index + 1], strip[index], strip[index + 2]]);
            }
        }
    }
}

pub(super) fn fan_triangulate(verts: &[u32], out: &mut Vec<u32>) {
    if verts.len() < 3 {
        return;
    }
    for index in 1..verts.len() - 1 {
        out.extend_from_slice(&[verts[0], verts[index], verts[index + 1]]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tetrahedron has four faces, and they are the four distinct triples of
    /// its vertices.
    #[test]
    fn a_tetra_yields_its_four_faces() {
        let faces = cell_faces(CellType::Tetra, &[0, 1, 2, 3]);
        assert_eq!(faces.len(), 4);
        let mut keys: Vec<[u32; 3]> = faces.into_iter().map(sorted_tri).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]);
    }

    /// A quad splits into two triangles that between them use all four corners
    /// and share the 0-2 diagonal.
    #[test]
    fn a_quad_splits_on_its_diagonal() {
        assert_eq!(
            cell_faces(CellType::Quad, &[10, 11, 12, 13]),
            vec![[10, 11, 12], [10, 12, 13]]
        );
    }

    /// A cell whose vertex list is shorter than its type requires is skipped
    /// rather than indexed out of bounds. Truncated cell arrays are the common
    /// shape of a damaged file.
    #[test]
    fn a_short_cell_yields_nothing() {
        assert!(cell_faces(CellType::Tetra, &[0, 1, 2]).is_empty());
        assert!(cell_faces(CellType::Hexahedron, &[0, 1, 2, 3, 4]).is_empty());
        assert!(cell_faces(CellType::Triangle, &[0, 1]).is_empty());
    }

    /// The face key ignores winding, which is what lets the surface extractor
    /// recognise the same face reached from either side of it.
    #[test]
    fn the_face_key_ignores_winding() {
        assert_eq!(sorted_tri([2, 0, 1]), sorted_tri([0, 1, 2]));
        assert_eq!(sorted_tri([1, 2, 0]), [0, 1, 2]);
    }

    /// The surface comes back in the order the file wrote its cells. This is
    /// the guard on the actual bug: collecting the faces out of the counting
    /// map instead put them in a per-process random order, so the same file
    /// decoded to a different triangle list on every run.
    #[test]
    fn the_surface_keeps_the_order_the_cells_were_written_in() {
        let cells = Cells {
            cell_verts: VertexNumbers::Legacy {
                num_cells: 3,
                vertices: vec![3, 0, 1, 2, 3, 3, 4, 5, 3, 6, 7, 8],
            },
            types: vec![CellType::Triangle; 3],
        };

        assert_eq!(
            extract_surface_unstructured(&cells),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            "three unshared faces, in file order"
        );
    }

    /// Surface extraction keeps the faces belonging to one cell and drops the
    /// one two cells share. Two tetrahedra glued on 0-1-2 have eight faces
    /// between them, of which six are surface.
    #[test]
    fn a_shared_face_is_not_surface() {
        let cells = Cells {
            cell_verts: VertexNumbers::Legacy {
                num_cells: 2,
                vertices: vec![4, 0, 1, 2, 3, 4, 0, 1, 2, 4],
            },
            types: vec![CellType::Tetra, CellType::Tetra],
        };

        let indices = extract_surface_unstructured(&cells);
        assert_eq!(
            indices.len(),
            6 * 3,
            "six surface faces, three indices each"
        );

        let keys: Vec<[u32; 3]> = indices
            .chunks_exact(3)
            .map(|t| sorted_tri([t[0], t[1], t[2]]))
            .collect();
        assert!(
            !keys.contains(&[0, 1, 2]),
            "the glued face is interior, not surface"
        );
    }

    /// A polygon fans from its first vertex, so an n-gon becomes n - 2
    /// triangles, and anything thinner than a triangle becomes none.
    #[test]
    fn a_polygon_fans_from_its_first_vertex() {
        let mut out = Vec::new();
        fan_triangulate(&[5, 6, 7, 8], &mut out);
        assert_eq!(out, vec![5, 6, 7, 5, 7, 8]);

        out.clear();
        fan_triangulate(&[5, 6], &mut out);
        assert!(out.is_empty());
    }
}
