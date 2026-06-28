use std::collections::HashMap;
use std::path::Path;

use crate::error::IoError;
use crate::types::{
    AttributeData, AttributeDomain, CELL_SENTINEL, IoDataSet, IoPointCloud, IoSparseVolume,
    IoVolume, IoVolumeGeometry, IoVolumeMesh, SurfaceMesh,
};

/// Decode a VTK-family file into one scientific dataset per piece.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    #[cfg(feature = "vtk")]
    {
        imp::datasets_from_path(path)
    }

    #[cfg(not(feature = "vtk"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "vtk",
            context: "VTK dataset decoding",
        })
    }
}

#[cfg(feature = "vtk")]
mod imp {
    use super::*;
    use vtkio::Vtk;
    use vtkio::model::{
        Attribute, Attributes, CellType, Cells, DataSet, PolyDataPiece, UnstructuredGridPiece,
        VertexNumbers,
    };

    pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
        let vtk = Vtk::import(path).map_err(|error| IoError::Parse(format!("vtk: {error}")))?;
        datasets_from_vtk(vtk, path)
    }

    fn datasets_from_vtk(vtk: Vtk, source: &Path) -> Result<Vec<IoDataSet>, IoError> {
        match vtk.data {
            DataSet::PolyData { pieces, .. } => pieces
                .into_iter()
                .enumerate()
                .map(|(index, piece)| {
                    let piece = piece
                        .load_piece_data(Some(source))
                        .map_err(|error| IoError::Parse(format!("vtk polydata piece: {error}")))?;
                    from_polydata(piece, format!("polydata_piece_{index}"))
                })
                .collect(),
            DataSet::UnstructuredGrid { pieces, .. } => pieces
                .into_iter()
                .enumerate()
                .map(|(index, piece)| {
                    let piece = piece.load_piece_data(Some(source)).map_err(|error| {
                        IoError::Parse(format!("vtk unstructured piece: {error}"))
                    })?;
                    from_unstructured(piece, format!("unstructured_piece_{index}"))
                })
                .collect(),
            DataSet::StructuredGrid { pieces, .. } => pieces
                .into_iter()
                .enumerate()
                .map(|(index, piece)| {
                    let piece = piece.load_piece_data(Some(source)).map_err(|error| {
                        IoError::Parse(format!("vtk structured piece: {error}"))
                    })?;
                    let positions = iobuf_to_positions(piece.points)?;
                    if positions.is_empty() {
                        return Err(IoError::Parse("vtk: structured grid has no points".into()));
                    }
                    let [ni, nj, nk] = vtk_extent_to_dims(piece.extent);
                    let indices = structured_surface_indices(ni, nj, nk);
                    let normals = compute_normals(&positions, &indices);
                    let (point_data, cell_data, edge_data) =
                        extract_attributes(&piece.data, positions.len(), indices.len() / 3, true);
                    Ok(IoDataSet {
                        name: format!("structured_grid_piece_{index}"),
                        surface_mesh: Some(build_mesh_data(
                            positions, indices, normals, point_data, cell_data, edge_data,
                        )),
                        ..IoDataSet::default()
                    })
                })
                .collect(),
            DataSet::RectilinearGrid { pieces, .. } => pieces
                .into_iter()
                .enumerate()
                .map(|(index, piece)| {
                    let piece = piece.load_piece_data(Some(source)).map_err(|error| {
                        IoError::Parse(format!("vtk rectilinear piece: {error}"))
                    })?;
                    let [ni, nj, nk] = vtk_extent_to_dims(piece.extent);
                    let xs = iobuf_to_f32_vec(piece.coords.x)?;
                    let ys = iobuf_to_f32_vec(piece.coords.y)?;
                    let zs = iobuf_to_f32_vec(piece.coords.z)?;
                    let positions = rectilinear_positions(&xs, &ys, &zs);
                    let indices = structured_surface_indices(ni, nj, nk);
                    let normals = compute_normals(&positions, &indices);
                    let (point_data, cell_data, edge_data) =
                        extract_attributes(&piece.data, positions.len(), indices.len() / 3, false);
                    Ok(IoDataSet {
                        name: format!("rectilinear_grid_piece_{index}"),
                        surface_mesh: Some(build_mesh_data(
                            positions,
                            indices,
                            normals,
                            point_data.clone(),
                            cell_data.clone(),
                            edge_data,
                        )),
                        volume: Some(IoVolume {
                            name: format!("rectilinear_grid_piece_{index}"),
                            dims: [ni as u32, nj as u32, nk as u32],
                            geometry: IoVolumeGeometry::Rectilinear { xs, ys, zs },
                            point_fields: point_data,
                            cell_fields: cell_data,
                        }),
                        ..IoDataSet::default()
                    })
                })
                .collect(),
            DataSet::ImageData {
                origin,
                spacing,
                pieces,
                ..
            } => pieces
                .into_iter()
                .enumerate()
                .map(|(index, piece)| {
                    let piece = piece
                        .load_piece_data(Some(source))
                        .map_err(|error| IoError::Parse(format!("vtk image piece: {error}")))?;
                    let [ni, nj, nk] = vtk_extent_to_dims(piece.extent);
                    let num_points = ni
                        .checked_mul(nj)
                        .and_then(|value| value.checked_mul(nk))
                        .ok_or_else(|| {
                            IoError::Parse("vtk: ImageData point count overflow".into())
                        })?;
                    let num_cells = ni
                        .saturating_sub(1)
                        .checked_mul(nj.saturating_sub(1))
                        .and_then(|value| value.checked_mul(nk.saturating_sub(1)))
                        .unwrap_or(0);
                    let (point_data, cell_data, edge_data) =
                        extract_attributes(&piece.data, num_points, num_cells, false);
                    let is_2d = ni <= 1 || nj <= 1 || nk <= 1;
                    if is_2d {
                        let positions = image_data_positions(ni, nj, nk, origin, spacing);
                        let indices = structured_surface_indices(ni, nj, nk);
                        let normals = compute_normals(&positions, &indices);
                        return Ok(IoDataSet {
                            name: format!("image_data_piece_{index}"),
                            surface_mesh: Some(build_mesh_data(
                                positions, indices, normals, point_data, cell_data, edge_data,
                            )),
                            ..IoDataSet::default()
                        });
                    }

                    let volume_mesh = build_image_data_volume_mesh(
                        ni,
                        nj,
                        nk,
                        origin,
                        spacing,
                        &point_data,
                        &cell_data,
                    );
                    Ok(IoDataSet {
                        name: format!("image_data_piece_{index}"),
                        volume: Some(IoVolume {
                            name: format!("image_data_piece_{index}"),
                            dims: [ni as u32, nj as u32, nk as u32],
                            geometry: IoVolumeGeometry::Uniform { origin, spacing },
                            point_fields: point_data.clone(),
                            cell_fields: cell_data.clone(),
                        }),
                        volume_mesh: volume_mesh.map(Box::new),
                        ..IoDataSet::default()
                    })
                })
                .collect(),
            other => Err(IoError::UnsupportedFormat(format!(
                "vtk: unsupported dataset type {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    fn from_polydata(piece: PolyDataPiece, name: String) -> Result<IoDataSet, IoError> {
        let positions = iobuf_to_positions(piece.points)?;
        if positions.is_empty() {
            return Err(IoError::Parse("vtk: polydata has no points".into()));
        }

        let mut indices = Vec::new();
        if let Some(polys) = &piece.polys {
            triangulate_vertex_numbers(polys, &mut indices);
        }
        if let Some(strips) = &piece.strips {
            triangulate_strips(strips, &mut indices);
        }

        let (point_data, _, _) =
            extract_attributes(&piece.data, positions.len(), indices.len() / 3, false);
        if indices.is_empty() {
            return Ok(IoDataSet {
                name,
                point_set: Some(build_point_cloud(
                    "VTK Point Cloud".into(),
                    positions,
                    point_data,
                )),
                ..IoDataSet::default()
            });
        }

        let num_tris = indices.len() / 3;
        let normals = compute_normals(&positions, &indices);
        let (point_data, cell_data, edge_data) =
            extract_attributes(&piece.data, positions.len(), num_tris, true);
        Ok(IoDataSet {
            name,
            surface_mesh: Some(build_mesh_data(
                positions, indices, normals, point_data, cell_data, edge_data,
            )),
            ..IoDataSet::default()
        })
    }

    fn from_unstructured(piece: UnstructuredGridPiece, name: String) -> Result<IoDataSet, IoError> {
        let positions = iobuf_to_positions(piece.points)?;
        if positions.is_empty() {
            return Err(IoError::Parse(
                "vtk: unstructured grid has no points".into(),
            ));
        }

        let all_voxel = !piece.cells.types.is_empty()
            && piece.cells.types.iter().all(|&t| t == CellType::Voxel);
        if all_voxel {
            let num_vox = piece.cells.types.len();
            let (point_data, cell_data, _) =
                extract_attributes(&piece.data, positions.len(), num_vox, false);
            if let Some(sparse) = try_sparse_voxel_grid(&positions, &piece.cells, &cell_data) {
                return Ok(IoDataSet {
                    name,
                    point_set: Some(build_point_cloud(
                        "VTK Voxel Points".into(),
                        positions,
                        point_data,
                    )),
                    sparse_grid: Some(Box::new(sparse)),
                    ..IoDataSet::default()
                });
            }
        }

        let indices = extract_surface_unstructured(&piece.cells);
        if indices.is_empty() {
            let point_data = extract_attributes(&piece.data, positions.len(), 0, false).0;
            return Ok(IoDataSet {
                name,
                point_set: Some(build_point_cloud(
                    "VTK Points".into(),
                    positions,
                    point_data,
                )),
                ..IoDataSet::default()
            });
        }

        let num_cells = piece.cells.types.len();
        let num_tris = indices.len() / 3;
        let normals = compute_normals(&positions, &indices);
        let (point_data, cell_data, edge_data) =
            extract_attributes(&piece.data, positions.len(), num_tris, true);
        let (raw_point_data, raw_cell_data, _) =
            extract_attributes(&piece.data, positions.len(), num_cells, false);
        let volume_mesh =
            build_volume_mesh_data(&positions, &piece.cells, &raw_cell_data, &raw_point_data);

        Ok(IoDataSet {
            name,
            surface_mesh: Some(build_mesh_data(
                positions, indices, normals, point_data, cell_data, edge_data,
            )),
            volume_mesh: volume_mesh.map(Box::new),
            ..IoDataSet::default()
        })
    }

    fn build_mesh_data(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        normals: Vec<[f32; 3]>,
        point_data: HashMap<String, Vec<f32>>,
        cell_data: HashMap<String, Vec<f32>>,
        edge_data: HashMap<String, Vec<f32>>,
    ) -> SurfaceMesh {
        let mut mesh = SurfaceMesh::default();
        mesh.positions = positions;
        mesh.normals = normals;
        mesh.indices = indices;
        for (name, values) in point_data {
            mesh.attributes
                .insert(name, AttributeData::scalars(AttributeDomain::Point, values));
        }
        for (name, values) in cell_data {
            mesh.attributes
                .insert(name, AttributeData::scalars(AttributeDomain::Cell, values));
        }
        for (name, values) in edge_data {
            mesh.attributes.insert(
                name,
                AttributeData::scalars(AttributeDomain::Halfedge, values),
            );
        }
        mesh
    }

    fn build_point_cloud(
        name: String,
        positions: Vec<[f32; 3]>,
        scalar_attributes: HashMap<String, Vec<f32>>,
    ) -> IoPointCloud {
        let scalars = if scalar_attributes.len() == 1 {
            scalar_attributes
                .values()
                .next()
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        IoPointCloud {
            name,
            positions,
            colors: Vec::new(),
            scalars,
            scalar_attributes,
        }
    }

    fn build_image_data_volume_mesh(
        ni: usize,
        nj: usize,
        nk: usize,
        origin: [f32; 3],
        spacing: [f32; 3],
        point_data: &HashMap<String, Vec<f32>>,
        cell_data: &HashMap<String, Vec<f32>>,
    ) -> Option<IoVolumeMesh> {
        let nci = ni.saturating_sub(1);
        let ncj = nj.saturating_sub(1);
        let nck = nk.saturating_sub(1);
        let num_cells = nci * ncj * nck;
        if num_cells == 0 {
            return None;
        }

        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(ni * nj * nk);
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    positions.push([
                        origin[0] + i as f32 * spacing[0],
                        origin[1] + j as f32 * spacing[1],
                        origin[2] + k as f32 * spacing[2],
                    ]);
                }
            }
        }

        let vid = |i: usize, j: usize, k: usize| -> usize { i + j * ni + k * ni * nj };
        let mut cells: Vec<[u32; 8]> = Vec::with_capacity(num_cells);
        let mut cell_vids: Vec<[usize; 8]> = Vec::with_capacity(num_cells);
        for ck in 0..nck {
            for cj in 0..ncj {
                for ci in 0..nci {
                    let v = [
                        vid(ci, cj, ck),
                        vid(ci + 1, cj, ck),
                        vid(ci + 1, cj, ck + 1),
                        vid(ci, cj, ck + 1),
                        vid(ci, cj + 1, ck),
                        vid(ci + 1, cj + 1, ck),
                        vid(ci + 1, cj + 1, ck + 1),
                        vid(ci, cj + 1, ck + 1),
                    ];
                    cells.push(v.map(|index| index as u32));
                    cell_vids.push(v);
                }
            }
        }

        let mut volume_mesh = IoVolumeMesh::default();
        volume_mesh.positions = positions;
        volume_mesh.cells = cells;
        for (name, values) in cell_data {
            volume_mesh.cell_fields.insert(name.clone(), values.clone());
        }
        for (name, values) in point_data {
            if volume_mesh.cell_fields.contains_key(name) {
                continue;
            }
            let averaged = cell_vids
                .iter()
                .map(|verts| {
                    verts
                        .iter()
                        .map(|&vi| values.get(vi).copied().unwrap_or(0.0))
                        .sum::<f32>()
                        / 8.0
                })
                .collect();
            volume_mesh.cell_fields.insert(name.clone(), averaged);
        }
        Some(volume_mesh)
    }

    fn build_volume_mesh_data(
        positions: &[[f32; 3]],
        cells: &Cells,
        cell_data: &HashMap<String, Vec<f32>>,
        point_data: &HashMap<String, Vec<f32>>,
    ) -> Option<IoVolumeMesh> {
        let cell_verts = collect_cells(&cells.cell_verts);
        let mut vm_cells = Vec::new();
        let mut original_indices = Vec::new();

        for (cell_index, (cell_type, verts)) in
            cells.types.iter().zip(cell_verts.iter()).enumerate()
        {
            let entry = match cell_type {
                CellType::Hexahedron
                | CellType::QuadraticHexahedron
                | CellType::TriquadraticHexahedron
                | CellType::BiquadraticQuadraticHexahedron
                    if verts.len() >= 8 =>
                {
                    Some([
                        verts[0], verts[1], verts[2], verts[3], verts[4], verts[5], verts[6],
                        verts[7],
                    ])
                }
                CellType::Tetra | CellType::QuadraticTetra if verts.len() >= 4 => Some([
                    verts[0],
                    verts[1],
                    verts[2],
                    verts[3],
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                ]),
                CellType::Wedge
                | CellType::QuadraticWedge
                | CellType::QuadraticLinearWedge
                | CellType::BiquadraticQuadraticWedge
                    if verts.len() >= 6 =>
                {
                    Some([
                        verts[0],
                        verts[1],
                        verts[2],
                        verts[3],
                        verts[4],
                        verts[5],
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                    ])
                }
                CellType::Pyramid if verts.len() >= 5 => Some([
                    verts[0],
                    verts[1],
                    verts[2],
                    verts[3],
                    verts[4],
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                ]),
                _ => None,
            };
            if let Some(cell) = entry {
                vm_cells.push(cell);
                original_indices.push(cell_index);
            }
        }

        if vm_cells.is_empty() {
            return None;
        }

        let mut volume_mesh = IoVolumeMesh::default();
        volume_mesh.positions = positions.to_vec();
        volume_mesh.cells = vm_cells;
        for (name, values) in cell_data {
            let mapped = original_indices
                .iter()
                .map(|&index| values.get(index).copied().unwrap_or(0.0))
                .collect();
            volume_mesh.cell_fields.insert(name.clone(), mapped);
        }
        for (name, values) in point_data {
            if volume_mesh.cell_fields.contains_key(name) {
                continue;
            }
            let averaged = volume_mesh
                .cells
                .iter()
                .map(|cell| {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for &vi in cell {
                        if vi != CELL_SENTINEL {
                            if let Some(&value) = values.get(vi as usize) {
                                sum += value;
                                count += 1;
                            }
                        }
                    }
                    if count > 0 { sum / count as f32 } else { 0.0 }
                })
                .collect();
            volume_mesh.cell_fields.insert(name.clone(), averaged);
        }
        Some(volume_mesh)
    }

    fn try_sparse_voxel_grid(
        positions: &[[f32; 3]],
        cells: &Cells,
        cell_data: &HashMap<String, Vec<f32>>,
    ) -> Option<IoSparseVolume> {
        if cells.types.is_empty() || cells.types.iter().any(|&t| t != CellType::Voxel) {
            return None;
        }

        let cell_verts = collect_cells(&cells.cell_verts);
        let num_cells = cell_verts.len();
        if num_cells < 2 {
            return None;
        }

        let mut min_corners = Vec::with_capacity(num_cells);
        let mut cell_sizes = Vec::with_capacity(num_cells);
        for verts in &cell_verts {
            if verts.len() != 8 {
                return None;
            }
            let v0 = *positions.get(verts[0] as usize)?;
            let v7 = *positions.get(verts[7] as usize)?;
            let dx = v7[0] - v0[0];
            let dy = v7[1] - v0[1];
            let dz = v7[2] - v0[2];
            if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
                return None;
            }
            let avg = (dx + dy + dz) / 3.0;
            if (dx - avg).abs() > 0.05 * avg
                || (dy - avg).abs() > 0.05 * avg
                || (dz - avg).abs() > 0.05 * avg
            {
                return None;
            }
            min_corners.push(v0);
            cell_sizes.push(avg);
        }

        let mut sorted_sizes = cell_sizes.clone();
        sorted_sizes.sort_unstable_by(|a, b| a.total_cmp(b));
        let cell_size = sorted_sizes[num_cells / 2];
        let tolerance = cell_size * 0.05;
        for &size in &cell_sizes {
            if (size - cell_size).abs() > tolerance {
                return None;
            }
        }

        let origin = min_corners.iter().fold([f32::INFINITY; 3], |acc, &corner| {
            [
                acc[0].min(corner[0]),
                acc[1].min(corner[1]),
                acc[2].min(corner[2]),
            ]
        });

        let mut active_cells = Vec::with_capacity(num_cells);
        for &corner in &min_corners {
            let ci = ((corner[0] - origin[0]) / cell_size).round() as u32;
            let cj = ((corner[1] - origin[1]) / cell_size).round() as u32;
            let ck = ((corner[2] - origin[2]) / cell_size).round() as u32;
            let ex = origin[0] + ci as f32 * cell_size;
            let ey = origin[1] + cj as f32 * cell_size;
            let ez = origin[2] + ck as f32 * cell_size;
            if (corner[0] - ex).abs() > tolerance
                || (corner[1] - ey).abs() > tolerance
                || (corner[2] - ez).abs() > tolerance
            {
                return None;
            }
            active_cells.push([ci, cj, ck]);
        }

        let mut seen = std::collections::HashSet::new();
        for &cell in &active_cells {
            if !seen.insert(cell) {
                return None;
            }
        }

        let max_idx = active_cells.iter().fold([0u32; 3], |acc, &cell| {
            [
                acc[0].max(cell[0]),
                acc[1].max(cell[1]),
                acc[2].max(cell[2]),
            ]
        });
        let grid_volume =
            (max_idx[0] + 1) as u64 * (max_idx[1] + 1) as u64 * (max_idx[2] + 1) as u64;
        if num_cells as f64 / grid_volume as f64 >= 0.30 {
            return None;
        }

        let mut sparse = IoSparseVolume::default();
        sparse.origin = origin;
        sparse.cell_size = cell_size;
        sparse.active_cells = active_cells;
        for (name, values) in cell_data {
            if values.len() == num_cells {
                sparse.cell_fields.insert(name.clone(), values.clone());
            }
        }
        Some(sparse)
    }

    fn extract_surface_unstructured(cells: &Cells) -> Vec<u32> {
        let cell_verts = collect_cells(&cells.cell_verts);
        let mut face_counts: HashMap<[u32; 3], usize> = HashMap::new();
        let mut face_verts: HashMap<[u32; 3], [u32; 3]> = HashMap::new();

        for (cell_type, verts) in cells.types.iter().zip(cell_verts.iter()) {
            for tri in cell_faces(*cell_type, verts) {
                let key = sorted_tri(tri);
                *face_counts.entry(key).or_insert(0) += 1;
                face_verts.entry(key).or_insert(tri);
            }
        }

        let mut indices = Vec::new();
        for (key, count) in face_counts {
            if count == 1 {
                let tri = face_verts[&key];
                indices.extend_from_slice(&tri);
            }
        }
        indices
    }

    fn cell_faces(cell_type: CellType, verts: &[u32]) -> Vec<[u32; 3]> {
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
                        verts[0], verts[1], verts[2], verts[3], verts[4], verts[5], verts[6],
                        verts[7],
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

    fn hex_faces(
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

    fn wedge_faces(v0: u32, v1: u32, v2: u32, v3: u32, v4: u32, v5: u32) -> Vec<[u32; 3]> {
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

    fn sorted_tri(mut tri: [u32; 3]) -> [u32; 3] {
        tri.sort_unstable();
        tri
    }

    fn vtk_extent_to_dims(extent: vtkio::model::Extent) -> [usize; 3] {
        let [ni, nj, nk] = extent.into_dims();
        [ni as usize, nj as usize, nk as usize]
    }

    fn structured_surface_indices(ni: usize, nj: usize, nk: usize) -> Vec<u32> {
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

    fn image_data_positions(
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

    fn rectilinear_positions(xs: &[f32], ys: &[f32], zs: &[f32]) -> Vec<[f32; 3]> {
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

    fn extract_attributes(
        attrs: &Attributes,
        num_points: usize,
        num_cells: usize,
        detect_halfedges: bool,
    ) -> (
        HashMap<String, Vec<f32>>,
        HashMap<String, Vec<f32>>,
        HashMap<String, Vec<f32>>,
    ) {
        let mut point_data = HashMap::new();
        let mut cell_data = HashMap::new();
        let mut edge_data = HashMap::new();
        let halfedge_len = if detect_halfedges { 3 * num_cells } else { 0 };

        for attr in &attrs.point {
            extract_attribute_into(
                attr,
                num_points,
                halfedge_len,
                &mut point_data,
                &mut edge_data,
            );
        }
        for attr in &attrs.cell {
            extract_attribute_into(
                attr,
                num_cells,
                halfedge_len,
                &mut cell_data,
                &mut edge_data,
            );
        }

        (point_data, cell_data, edge_data)
    }

    fn extract_attribute_into(
        attr: &Attribute,
        expected_len: usize,
        halfedge_len: usize,
        out: &mut HashMap<String, Vec<f32>>,
        edge_out: &mut HashMap<String, Vec<f32>>,
    ) {
        match attr {
            Attribute::DataArray(arr) => {
                let num_comp = arr.elem.num_comp() as usize;
                let Some(floats) = arr.data.clone().cast_into::<f32>() else {
                    return;
                };
                let n_tuples = if num_comp > 0 {
                    floats.len() / num_comp
                } else {
                    0
                };
                if n_tuples == expected_len {
                    emit_scalar_arrays(&arr.name, num_comp, floats, expected_len, out);
                } else if halfedge_len > 0 && num_comp == 1 && floats.len() == halfedge_len {
                    edge_out.insert(arr.name.clone(), floats);
                }
            }
            Attribute::Field { data_array, .. } => {
                for field in data_array {
                    let num_comp = field.elem as usize;
                    let Some(floats) = field.data.clone().cast_into::<f32>() else {
                        continue;
                    };
                    let n_tuples = if num_comp > 0 {
                        floats.len() / num_comp
                    } else {
                        0
                    };
                    if n_tuples == expected_len {
                        emit_scalar_arrays(&field.name, num_comp, floats, expected_len, out);
                    } else if halfedge_len > 0 && num_comp == 1 && floats.len() == halfedge_len {
                        edge_out.insert(field.name.clone(), floats);
                    }
                }
            }
        }
    }

    fn emit_scalar_arrays(
        name: &str,
        num_comp: usize,
        floats: Vec<f32>,
        expected_len: usize,
        out: &mut HashMap<String, Vec<f32>>,
    ) {
        if num_comp == 0 || floats.len() != num_comp * expected_len {
            return;
        }
        if num_comp == 1 {
            out.insert(name.to_string(), floats);
            return;
        }
        if num_comp == 3 {
            let mut xs = Vec::with_capacity(expected_len);
            let mut ys = Vec::with_capacity(expected_len);
            let mut zs = Vec::with_capacity(expected_len);
            let mut mags = Vec::with_capacity(expected_len);
            for chunk in floats.chunks_exact(3) {
                let (x, y, z) = (chunk[0], chunk[1], chunk[2]);
                xs.push(x);
                ys.push(y);
                zs.push(z);
                mags.push((x * x + y * y + z * z).sqrt());
            }
            out.insert(format!("{name}_x"), xs);
            out.insert(format!("{name}_y"), ys);
            out.insert(format!("{name}_z"), zs);
            out.insert(format!("{name}_mag"), mags);
            return;
        }
        let mut mags = Vec::with_capacity(expected_len);
        for chunk in floats.chunks_exact(num_comp) {
            mags.push(chunk.iter().map(|value| value * value).sum::<f32>().sqrt());
        }
        out.insert(format!("{name}_mag"), mags);
    }

    fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
        let mut normals = vec![[0.0f32; 3]; positions.len()];
        for tri in indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let p0 = glam::Vec3::from(positions[i0]);
            let p1 = glam::Vec3::from(positions[i1]);
            let p2 = glam::Vec3::from(positions[i2]);
            let normal = (p1 - p0).cross(p2 - p0);
            for index in [i0, i1, i2] {
                normals[index][0] += normal.x;
                normals[index][1] += normal.y;
                normals[index][2] += normal.z;
            }
        }
        for normal in &mut normals {
            *normal = glam::Vec3::from(*normal).normalize_or_zero().into();
        }
        normals
    }

    fn iobuf_to_positions(buf: vtkio::model::IOBuffer) -> Result<Vec<[f32; 3]>, IoError> {
        let floats = buf.cast_into::<f32>().ok_or_else(|| {
            IoError::UnsupportedFormat("vtk: could not cast point coordinates to f32".into())
        })?;
        Ok(floats
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect())
    }

    fn iobuf_to_f32_vec(buf: vtkio::model::IOBuffer) -> Result<Vec<f32>, IoError> {
        buf.cast_into::<f32>().ok_or_else(|| {
            IoError::UnsupportedFormat("vtk: could not cast coordinate array to f32".into())
        })
    }

    fn collect_cells(vn: &VertexNumbers) -> Vec<Vec<u32>> {
        match vn {
            VertexNumbers::XML {
                connectivity,
                offsets,
            } => {
                let mut cells = Vec::with_capacity(offsets.len());
                let mut prev = 0usize;
                for &end in offsets {
                    let end = end as usize;
                    cells.push(
                        connectivity[prev..end]
                            .iter()
                            .map(|&value| value as u32)
                            .collect(),
                    );
                    prev = end;
                }
                cells
            }
            VertexNumbers::Legacy { vertices, .. } => {
                let mut cells = Vec::new();
                let mut index = 0;
                while index < vertices.len() {
                    let len = vertices[index] as usize;
                    if index + len + 1 > vertices.len() {
                        break;
                    }
                    cells.push(vertices[index + 1..index + 1 + len].to_vec());
                    index += 1 + len;
                }
                cells
            }
        }
    }

    fn triangulate_vertex_numbers(vn: &VertexNumbers, out: &mut Vec<u32>) {
        for cell in collect_cells(vn) {
            fan_triangulate(&cell, out);
        }
    }

    fn triangulate_strips(vn: &VertexNumbers, out: &mut Vec<u32>) {
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

    fn fan_triangulate(verts: &[u32], out: &mut Vec<u32>) {
        if verts.len() < 3 {
            return;
        }
        for index in 1..verts.len() - 1 {
            out.extend_from_slice(&[verts[0], verts[index], verts[index + 1]]);
        }
    }
}
