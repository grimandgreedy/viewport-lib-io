use std::collections::HashMap;
use std::path::Path;

use hdf5::File as Hdf5File;

use super::common::{Dataset, VolumeGeometry, VolumeGrid};
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};
use crate::{error::IoError, types::IoDataSet};

/// Decode a CGNS file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

/// Read a CGNS file and return one [`Dataset`] per zone.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    let file = open_cgns(path)?;
    let mut datasets = Vec::new();
    for base in find_children_by_label(&file, "CGNSBase_t")? {
        let cell_dim = read_base_dimensions(&base)?.0;
        for zone in find_children_by_label(&base, "Zone_t")? {
            if let Ok(ds) = read_zone(&zone, cell_dim) {
                datasets.push(ds);
            }
        }
    }
    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }
    Ok(datasets)
}

/// Read time-series metadata from a CGNS file.
///
/// Looks for `BaseIterativeData_t` with `TimeValues` to build timestep entries.
/// Each timestep selector encodes `base_idx/time_idx` for later retrieval.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let file = open_cgns(path)?;
    let bases = find_children_by_label(&file, "CGNSBase_t")?;

    for (base_idx, base) in bases.iter().enumerate() {
        if let Ok(iter_data) = find_children_by_label(base, "BaseIterativeData_t") {
            for bid in &iter_data {
                if let Some(times) = read_time_values(bid) {
                    let timesteps = times
                        .iter()
                        .enumerate()
                        .map(|(i, &t)| TimestepEntry {
                            time: t as f64,
                            file: path.to_path_buf(),
                            selector: Some(format!("{base_idx}/{i}")),
                        })
                        .collect();
                    return Ok(PvdSeries { timesteps });
                }
            }
        }
    }

    // Non-temporal: return a single entry.
    Ok(PvdSeries {
        timesteps: vec![TimestepEntry {
            time: 0.0,
            file: path.to_path_buf(),
            selector: Some("0".to_string()),
        }],
    })
}

/// Read zones for a specific selector from a CGNS file.
///
/// Selector format: `"base_idx"` reads all zones from that base,
/// `"base_idx/time_idx"` reads zones and selects the matching flow solution
/// for the given time index.
pub fn read_selected(path: &Path, selector: &str) -> Result<Vec<Dataset>, ReadError> {
    let file = open_cgns(path)?;
    let parts: Vec<&str> = selector.split('/').collect();
    let base_idx: usize = parts
        .first()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| ReadError::Cgns(format!("Invalid CGNS selector: {selector}")))?;
    let time_idx: Option<usize> = parts.get(1).and_then(|s| s.parse().ok());

    let bases = find_children_by_label(&file, "CGNSBase_t")?;
    let base = bases
        .get(base_idx)
        .ok_or_else(|| ReadError::Cgns(format!("Base index {base_idx} out of range")))?;
    let cell_dim = read_base_dimensions(base)?.0;

    // Find the flow solution name for this time index if iterative data exists.
    let solution_name = time_idx.and_then(|ti| {
        find_children_by_label(base, "BaseIterativeData_t")
            .ok()
            .and_then(|bids| {
                for bid in &bids {
                    if let Some(names) = read_flow_solution_pointers(bid) {
                        return names.get(ti).cloned();
                    }
                }
                None
            })
    });

    let mut datasets = Vec::new();
    for zone in find_children_by_label(base, "Zone_t")? {
        if let Ok(ds) = read_zone_with_solution(&zone, cell_dim, solution_name.as_deref()) {
            datasets.push(ds);
        }
    }
    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }
    Ok(datasets)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn open_cgns(path: &Path) -> Result<Hdf5File, ReadError> {
    Hdf5File::open(path)
        .map_err(|err| ReadError::Cgns(format!("Failed to open {}: {err}", path.display())))
}

/// Return (cell_dim, phys_dim) from a CGNSBase_t node's " data" dataset.
fn read_base_dimensions(base: &hdf5::Group) -> Result<(u32, u32), ReadError> {
    let data = base
        .dataset(" data")
        .map_err(|e| ReadError::Cgns(format!("Base missing data: {e}")))?;
    let vals = read_dataset_i32(&data)?;
    let cell_dim = *vals.first().unwrap_or(&3) as u32;
    let phys_dim = *vals.get(1).unwrap_or(&3) as u32;
    Ok((cell_dim, phys_dim))
}

fn read_zone(zone: &hdf5::Group, cell_dim: u32) -> Result<Dataset, ReadError> {
    read_zone_with_solution(zone, cell_dim, None)
}

fn read_zone_with_solution(
    zone: &hdf5::Group,
    cell_dim: u32,
    solution_name: Option<&str>,
) -> Result<Dataset, ReadError> {
    let zone_type = read_zone_type(zone);
    let is_structured = zone_type.eq_ignore_ascii_case("Structured");

    // Read zone dimensions from " data".
    let zone_dims = read_zone_dims(zone)?;

    // Read coordinates.
    let (coords_x, coords_y, coords_z) = read_coordinates(zone)?;
    let n_vertices = coords_x.len();

    if is_structured {
        read_structured_zone(
            zone,
            &zone_dims,
            coords_x,
            coords_y,
            coords_z,
            cell_dim,
            solution_name,
        )
    } else {
        read_unstructured_zone(
            zone,
            coords_x,
            coords_y,
            coords_z,
            n_vertices,
            solution_name,
        )
    }
}

fn read_structured_zone(
    zone: &hdf5::Group,
    zone_dims: &[i32],
    coords_x: Vec<f32>,
    coords_y: Vec<f32>,
    coords_z: Vec<f32>,
    cell_dim: u32,
    solution_name: Option<&str>,
) -> Result<Dataset, ReadError> {
    // Structured zone dims: [vertex_i, vertex_j, vertex_k, cell_i, cell_j, cell_k, ...]
    // For 3D: 3 vertex dims then 3 cell dims.
    let ni = *zone_dims.first().unwrap_or(&1) as u32;
    let nj = *zone_dims.get(1).unwrap_or(&1) as u32;
    let nk = if cell_dim >= 3 {
        *zone_dims.get(2).unwrap_or(&1) as u32
    } else {
        1
    };

    let positions: Vec<[f32; 3]> = coords_x
        .iter()
        .zip(coords_y.iter())
        .zip(coords_z.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    let (point_data, cell_data) = read_flow_solutions(zone, solution_name)?;

    let is_3d = ni > 1 && nj > 1 && nk > 1;
    if is_3d {
        // Build a VolumeGrid from the structured coordinates.
        // Check if uniform or rectilinear.
        let dims = [ni, nj, nk];
        let volume = build_volume_grid(
            dims,
            &coords_x,
            &coords_y,
            &coords_z,
            &point_data,
            &cell_data,
        );
        let indices = structured_surface_indices(ni as usize, nj as usize, nk as usize);
        let normals = compute_normals(&positions, &indices);
        Ok(Dataset {
            positions,
            indices,
            normals,
            point_data,
            cell_data,
            edge_data: HashMap::new(),
            sparse_volume: None,
            volume: Some(volume),
            volume_mesh: None,
        })
    } else {
        // 2D structured surface.
        let indices = structured_surface_indices(ni as usize, nj as usize, nk as usize);
        if indices.is_empty() || positions.is_empty() {
            return Err(ReadError::Empty);
        }
        let normals = compute_normals(&positions, &indices);
        Ok(Dataset {
            positions,
            indices,
            normals,
            point_data,
            cell_data,
            edge_data: HashMap::new(),
            sparse_volume: None,
            volume: None,
            volume_mesh: None,
        })
    }
}

fn read_unstructured_zone(
    zone: &hdf5::Group,
    coords_x: Vec<f32>,
    coords_y: Vec<f32>,
    coords_z: Vec<f32>,
    n_vertices: usize,
    solution_name: Option<&str>,
) -> Result<Dataset, ReadError> {
    let positions: Vec<[f32; 3]> = coords_x
        .iter()
        .zip(coords_y.iter())
        .zip(coords_z.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    // Read element connectivity and triangulate.
    let indices = read_elements_triangulated(zone, n_vertices)?;
    if indices.is_empty() || positions.is_empty() {
        return Err(ReadError::Empty);
    }

    let normals = compute_normals(&positions, &indices);
    let (point_data, cell_data) = read_flow_solutions(zone, solution_name)?;

    Ok(Dataset {
        positions,
        indices,
        normals,
        point_data,
        cell_data,
        edge_data: HashMap::new(),
        sparse_volume: None,
        volume: None,
        volume_mesh: None,
    })
}

// ---------------------------------------------------------------------------
// CGNS node traversal
// ---------------------------------------------------------------------------

/// Find child groups with a given CGNS `label` attribute.
fn find_children_by_label(
    parent: &hdf5::Group,
    label: &str,
) -> Result<Vec<hdf5::Group>, ReadError> {
    let mut result = Vec::new();
    let members = parent
        .member_names()
        .map_err(|e| ReadError::Cgns(format!("Failed to list members: {e}")))?;
    for name in &members {
        if let Ok(group) = parent.group(name) {
            if node_label(&group).as_deref() == Some(label) {
                result.push(group);
            }
        }
    }
    Ok(result)
}

/// Read the "label" attribute of a CGNS node.
fn node_label(group: &hdf5::Group) -> Option<String> {
    group
        .attr("label")
        .ok()
        .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            // Some writers store label as fixed-length ASCII.
            group
                .attr("label")
                .ok()
                .and_then(|attr| attr.read_scalar::<hdf5::types::FixedAscii<33>>().ok())
                .map(|s| s.as_str().trim_end_matches('\0').to_string())
        })
}

/// Read the ZoneType_t child to determine "Structured" or "Unstructured".
fn read_zone_type(zone: &hdf5::Group) -> String {
    find_children_by_label(zone, "ZoneType_t")
        .ok()
        .and_then(|groups| groups.into_iter().next())
        .and_then(|zt| read_data_string(&zt))
        .unwrap_or_else(|| "Unstructured".to_string())
}

/// Read the " data" dataset of a node as a string.
fn read_data_string(group: &hdf5::Group) -> Option<String> {
    let ds = group.dataset(" data").ok()?;
    // Try reading as a 1D char array (common CGNS pattern).
    if let Ok(chars) = ds.read_raw::<u8>() {
        let s = String::from_utf8_lossy(&chars)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        if !s.is_empty() {
            return Some(s);
        }
    }
    None
}

/// Read zone dimension data from " data" dataset.
fn read_zone_dims(zone: &hdf5::Group) -> Result<Vec<i32>, ReadError> {
    let ds = zone
        .dataset(" data")
        .map_err(|e| ReadError::Cgns(format!("Zone missing data: {e}")))?;
    read_dataset_i32(&ds)
}

// ---------------------------------------------------------------------------
// Coordinate reading
// ---------------------------------------------------------------------------

fn read_coordinates(zone: &hdf5::Group) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), ReadError> {
    let grid_coords = find_children_by_label(zone, "GridCoordinates_t")?;
    let gc = grid_coords
        .first()
        .ok_or_else(|| ReadError::Cgns("Zone missing GridCoordinates".to_string()))?;

    let coord_arrays = find_children_by_label(gc, "DataArray_t")?;
    let mut x = None;
    let mut y = None;
    let mut z = None;

    for arr in &coord_arrays {
        let name = arr.name();
        let short_name = name.rsplit('/').next().unwrap_or(&name);
        let values = read_data_array_f32(arr)?;
        match short_name {
            "CoordinateX" => x = Some(values),
            "CoordinateY" => y = Some(values),
            "CoordinateZ" => z = Some(values),
            _ => {}
        }
    }

    let x = x.ok_or_else(|| ReadError::Cgns("Missing CoordinateX".to_string()))?;
    let n = x.len();
    let y = y.unwrap_or_else(|| vec![0.0; n]);
    let z = z.unwrap_or_else(|| vec![0.0; n]);

    Ok((x, y, z))
}

/// Read a DataArray_t node's " data" dataset as f32 values.
fn read_data_array_f32(group: &hdf5::Group) -> Result<Vec<f32>, ReadError> {
    let ds = group
        .dataset(" data")
        .map_err(|e| ReadError::Cgns(format!("DataArray missing data: {e}")))?;
    read_dataset_f32(&ds)
}

// ---------------------------------------------------------------------------
// Element connectivity
// ---------------------------------------------------------------------------

/// CGNS element type IDs (from SIDS).
const TRI_3: i32 = 5;
const QUAD_4: i32 = 7;
const TETRA_4: i32 = 10;
const PYRA_5: i32 = 12;
const PENTA_6: i32 = 14;
const HEXA_8: i32 = 17;
const MIXED: i32 = 20;

fn read_elements_triangulated(
    zone: &hdf5::Group,
    _n_vertices: usize,
) -> Result<Vec<u32>, ReadError> {
    let elem_sections = find_children_by_label(zone, "Elements_t")?;
    let mut all_indices = Vec::new();

    for section in &elem_sections {
        let elem_type = read_element_type(section)?;
        let connectivity = read_element_connectivity(section)?;

        if elem_type == MIXED {
            triangulate_mixed(&connectivity, &mut all_indices);
        } else {
            triangulate_typed(elem_type, &connectivity, &mut all_indices);
        }
    }

    Ok(all_indices)
}

fn read_element_type(section: &hdf5::Group) -> Result<i32, ReadError> {
    let ds = section
        .dataset(" data")
        .map_err(|e| ReadError::Cgns(format!("Elements missing data: {e}")))?;
    let vals = read_dataset_i32(&ds)?;
    vals.first()
        .copied()
        .ok_or_else(|| ReadError::Cgns("Elements data empty".to_string()))
}

fn read_element_connectivity(section: &hdf5::Group) -> Result<Vec<i32>, ReadError> {
    let conn_nodes = find_children_by_label(section, "DataArray_t")?;
    for node in &conn_nodes {
        let name = node.name();
        let short = name.rsplit('/').next().unwrap_or(&name);
        if short == "ElementConnectivity" {
            let ds = node
                .dataset(" data")
                .map_err(|e| ReadError::Cgns(format!("ElementConnectivity missing data: {e}")))?;
            return read_dataset_i32(&ds);
        }
    }
    Err(ReadError::Cgns("Missing ElementConnectivity".to_string()))
}

/// Triangulate elements of a single known type.
fn triangulate_typed(elem_type: i32, connectivity: &[i32], out: &mut Vec<u32>) {
    let stride = match elem_type {
        TRI_3 => 3,
        QUAD_4 => 4,
        TETRA_4 => 4,
        PYRA_5 => 5,
        PENTA_6 => 6,
        HEXA_8 => 8,
        _ => return,
    };

    for elem in connectivity.chunks_exact(stride) {
        triangulate_element(elem_type, elem, out);
    }
}

/// Triangulate a MIXED element section where each element is prefixed by its type.
fn triangulate_mixed(connectivity: &[i32], out: &mut Vec<u32>) {
    let mut i = 0;
    while i < connectivity.len() {
        let etype = connectivity[i];
        i += 1;
        let n_nodes = element_node_count(etype);
        if n_nodes == 0 || i + n_nodes > connectivity.len() {
            break;
        }
        let nodes = &connectivity[i..i + n_nodes];
        triangulate_element(etype, nodes, out);
        i += n_nodes;
    }
}

fn element_node_count(elem_type: i32) -> usize {
    match elem_type {
        TRI_3 => 3,
        QUAD_4 => 4,
        TETRA_4 => 4,
        PYRA_5 => 5,
        PENTA_6 => 6,
        HEXA_8 => 8,
        _ => 0,
    }
}

/// Convert a single element to triangles. CGNS uses 1-based indices.
fn triangulate_element(elem_type: i32, nodes: &[i32], out: &mut Vec<u32>) {
    // Convert 1-based to 0-based.
    let v = |i: usize| -> u32 { (nodes[i] - 1).max(0) as u32 };

    match elem_type {
        TRI_3 => {
            out.extend_from_slice(&[v(0), v(1), v(2)]);
        }
        QUAD_4 => {
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
        }
        TETRA_4 => {
            // 4 triangular faces.
            out.extend_from_slice(&[v(0), v(2), v(1)]);
            out.extend_from_slice(&[v(0), v(1), v(3)]);
            out.extend_from_slice(&[v(1), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(3), v(2)]);
        }
        PYRA_5 => {
            // Quad base + 4 triangular sides.
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(1), v(4)]);
            out.extend_from_slice(&[v(1), v(2), v(4)]);
            out.extend_from_slice(&[v(2), v(3), v(4)]);
            out.extend_from_slice(&[v(3), v(0), v(4)]);
        }
        PENTA_6 => {
            // 2 triangular caps + 3 quad sides.
            out.extend_from_slice(&[v(0), v(1), v(2)]);
            out.extend_from_slice(&[v(3), v(5), v(4)]);
            out.extend_from_slice(&[v(0), v(3), v(4), v(0), v(4), v(1)]);
            out.extend_from_slice(&[v(1), v(4), v(5), v(1), v(5), v(2)]);
            out.extend_from_slice(&[v(2), v(5), v(3), v(2), v(3), v(0)]);
        }
        HEXA_8 => {
            // 6 quad faces → 12 triangles.
            out.extend_from_slice(&[v(0), v(3), v(2), v(0), v(2), v(1)]); // bottom
            out.extend_from_slice(&[v(4), v(5), v(6), v(4), v(6), v(7)]); // top
            out.extend_from_slice(&[v(0), v(1), v(5), v(0), v(5), v(4)]); // front
            out.extend_from_slice(&[v(2), v(3), v(7), v(2), v(7), v(6)]); // back
            out.extend_from_slice(&[v(0), v(4), v(7), v(0), v(7), v(3)]); // left
            out.extend_from_slice(&[v(1), v(2), v(6), v(1), v(6), v(5)]); // right
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Flow solution reading
// ---------------------------------------------------------------------------

fn read_flow_solutions(
    zone: &hdf5::Group,
    preferred_name: Option<&str>,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<f32>>), ReadError> {
    let solutions = find_children_by_label(zone, "FlowSolution_t")?;
    if solutions.is_empty() {
        return Ok((HashMap::new(), HashMap::new()));
    }

    // Pick the preferred solution by name, or the first one.
    let sol = if let Some(pref) = preferred_name {
        solutions
            .iter()
            .find(|s| {
                let name = s.name();
                let short = name.rsplit('/').next().unwrap_or(&name);
                short == pref
            })
            .unwrap_or(&solutions[0])
    } else {
        &solutions[0]
    };

    let grid_location = read_grid_location(sol);
    let is_cell_center = grid_location.eq_ignore_ascii_case("CellCenter");

    let mut point_data = HashMap::new();
    let mut cell_data = HashMap::new();

    let arrays = find_children_by_label(sol, "DataArray_t")?;
    for arr in &arrays {
        let full_name = arr.name();
        let name = full_name
            .rsplit('/')
            .next()
            .unwrap_or(&full_name)
            .to_string();
        if let Ok(values) = read_data_array_f32(arr) {
            if is_cell_center {
                cell_data.insert(name, values);
            } else {
                point_data.insert(name, values);
            }
        }
    }

    Ok((point_data, cell_data))
}

fn read_grid_location(sol: &hdf5::Group) -> String {
    find_children_by_label(sol, "GridLocation_t")
        .ok()
        .and_then(|groups| groups.into_iter().next())
        .and_then(|gl| read_data_string(&gl))
        .unwrap_or_else(|| "Vertex".to_string())
}

// ---------------------------------------------------------------------------
// Time-series helpers
// ---------------------------------------------------------------------------

fn read_time_values(bid: &hdf5::Group) -> Option<Vec<f32>> {
    let arrays = find_children_by_label(bid, "DataArray_t").ok()?;
    for arr in &arrays {
        let name = arr.name();
        let short = name.rsplit('/').next().unwrap_or(&name);
        if short == "TimeValues" {
            return read_data_array_f32(arr).ok();
        }
    }
    None
}

fn read_flow_solution_pointers(bid: &hdf5::Group) -> Option<Vec<String>> {
    let arrays = find_children_by_label(bid, "DataArray_t").ok()?;
    for arr in &arrays {
        let name = arr.name();
        let short = name.rsplit('/').next().unwrap_or(&name);
        if short == "FlowSolutionPointers" {
            let ds = arr.dataset(" data").ok()?;
            if let Ok(chars) = ds.read_raw::<u8>() {
                // FlowSolutionPointers is stored as a 2D char array [nsteps][32].
                let shape = ds.shape();
                if shape.len() == 2 {
                    let step_len = shape[1];
                    let names: Vec<String> = chars
                        .chunks(step_len)
                        .map(|chunk| {
                            String::from_utf8_lossy(chunk)
                                .trim_end_matches('\0')
                                .trim()
                                .to_string()
                        })
                        .collect();
                    return Some(names);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Volume grid construction for structured zones
// ---------------------------------------------------------------------------

fn build_volume_grid(
    dims: [u32; 3],
    coords_x: &[f32],
    coords_y: &[f32],
    coords_z: &[f32],
    point_data: &HashMap<String, Vec<f32>>,
    cell_data: &HashMap<String, Vec<f32>>,
) -> VolumeGrid {
    let [ni, nj, nk] = dims;
    let n = (ni as usize) * (nj as usize) * (nk as usize);

    // Check if the grid is uniform by sampling coordinate differences.
    if let Some((origin, spacing)) = detect_uniform_grid(dims, coords_x, coords_y, coords_z) {
        return VolumeGrid {
            dims,
            geometry: VolumeGeometry::Uniform { origin, spacing },
            point_data: point_data.clone(),
            cell_data: cell_data.clone(),
        };
    }

    // Otherwise build rectilinear coordinate arrays.
    // Extract unique sorted coordinates along each axis.
    let xs = extract_axis_coords(coords_x, n, ni as usize, |idx| idx % ni as usize);
    let ys = extract_axis_coords(coords_y, n, nj as usize, |idx| {
        (idx / ni as usize) % nj as usize
    });
    let zs = extract_axis_coords(coords_z, n, nk as usize, |idx| {
        idx / (ni as usize * nj as usize)
    });

    VolumeGrid {
        dims,
        geometry: VolumeGeometry::Rectilinear { xs, ys, zs },
        point_data: point_data.clone(),
        cell_data: cell_data.clone(),
    }
}

fn detect_uniform_grid(
    dims: [u32; 3],
    coords_x: &[f32],
    coords_y: &[f32],
    coords_z: &[f32],
) -> Option<([f32; 3], [f32; 3])> {
    let [ni, nj, nk] = dims;
    if coords_x.is_empty() || coords_y.is_empty() || coords_z.is_empty() {
        return None;
    }

    let origin = [coords_x[0], coords_y[0], coords_z[0]];

    // Sample spacing from first axis steps.
    let dx = if ni > 1 {
        coords_x[1] - coords_x[0]
    } else {
        1.0
    };
    let dy = if nj > 1 {
        coords_y[ni as usize] - coords_y[0]
    } else {
        1.0
    };
    let dz = if nk > 1 {
        coords_z[(ni as usize) * (nj as usize)] - coords_z[0]
    } else {
        1.0
    };

    if dx.abs() < f32::EPSILON || dy.abs() < f32::EPSILON || dz.abs() < f32::EPSILON {
        return None;
    }

    // Verify uniformity by sampling a few points.
    let eps = (dx.abs().max(dy.abs()).max(dz.abs())) * 1e-4;
    let check_count = coords_x.len().min(100);
    for i in 0..check_count {
        let ix = i % ni as usize;
        let iy = (i / ni as usize) % nj as usize;
        let iz = i / (ni as usize * nj as usize);
        let expected_x = origin[0] + dx * ix as f32;
        let expected_y = origin[1] + dy * iy as f32;
        let expected_z = origin[2] + dz * iz as f32;
        if (coords_x[i] - expected_x).abs() > eps
            || (coords_y[i] - expected_y).abs() > eps
            || (coords_z[i] - expected_z).abs() > eps
        {
            return None;
        }
    }

    Some((origin, [dx, dy, dz]))
}

fn extract_axis_coords(
    all_coords: &[f32],
    _total: usize,
    axis_len: usize,
    axis_index: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut values = vec![0.0f32; axis_len];
    let mut counts = vec![0u32; axis_len];
    for (i, &val) in all_coords.iter().enumerate() {
        let ai = axis_index(i);
        if ai < axis_len {
            values[ai] += val;
            counts[ai] += 1;
        }
    }
    for (v, c) in values.iter_mut().zip(counts.iter()) {
        if *c > 0 {
            *v /= *c as f32;
        }
    }
    values
}

// ---------------------------------------------------------------------------
// HDF5 dataset reading helpers
// ---------------------------------------------------------------------------

fn read_dataset_f32(ds: &hdf5::Dataset) -> Result<Vec<f32>, ReadError> {
    if let Ok(values) = ds.read_raw::<f32>() {
        return Ok(values);
    }
    if let Ok(values) = ds.read_raw::<f64>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = ds.read_raw::<i32>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = ds.read_raw::<i64>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    Err(ReadError::Cgns(
        "Unsupported HDF5 dataset type for float conversion".to_string(),
    ))
}

fn read_dataset_i32(ds: &hdf5::Dataset) -> Result<Vec<i32>, ReadError> {
    if let Ok(values) = ds.read_raw::<i32>() {
        return Ok(values);
    }
    if let Ok(values) = ds.read_raw::<i64>() {
        return Ok(values.into_iter().map(|v| v as i32).collect());
    }
    if let Ok(values) = ds.read_raw::<u32>() {
        return Ok(values.into_iter().map(|v| v as i32).collect());
    }
    if let Ok(values) = ds.read_raw::<f64>() {
        return Ok(values.into_iter().map(|v| v as i32).collect());
    }
    Err(ReadError::Cgns(
        "Unsupported HDF5 dataset type for int conversion".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Geometry helpers (shared with xdmf.rs patterns)
// ---------------------------------------------------------------------------

fn structured_surface_indices(ni: usize, nj: usize, nk: usize) -> Vec<u32> {
    if ni >= 2 && nj >= 2 && nk <= 1 {
        return quad_grid_indices(ni, nj);
    }
    if ni >= 2 && nk >= 2 && nj <= 1 {
        return quad_grid_indices(ni, nk);
    }
    if nj >= 2 && nk >= 2 && ni <= 1 {
        return quad_grid_indices(nj, nk);
    }
    if ni < 2 || nj < 2 || nk < 2 {
        return Vec::new();
    }

    let idx = |i: usize, j: usize, k: usize| -> u32 { (i + j * ni + k * ni * nj) as u32 };
    let mut indices = Vec::new();

    // -Z and +Z faces.
    for j in 0..(nj - 1) {
        for i in 0..(ni - 1) {
            let v00 = idx(i, j, 0);
            let v10 = idx(i + 1, j, 0);
            let v01 = idx(i, j + 1, 0);
            let v11 = idx(i + 1, j + 1, 0);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);

            let v00 = idx(i, j, nk - 1);
            let v10 = idx(i + 1, j, nk - 1);
            let v01 = idx(i, j + 1, nk - 1);
            let v11 = idx(i + 1, j + 1, nk - 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }
    // -Y and +Y faces.
    for k in 0..(nk - 1) {
        for i in 0..(ni - 1) {
            let v00 = idx(i, 0, k);
            let v10 = idx(i + 1, 0, k);
            let v01 = idx(i, 0, k + 1);
            let v11 = idx(i + 1, 0, k + 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);

            let v00 = idx(i, nj - 1, k);
            let v10 = idx(i + 1, nj - 1, k);
            let v01 = idx(i, nj - 1, k + 1);
            let v11 = idx(i + 1, nj - 1, k + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
        }
    }
    // -X and +X faces.
    for k in 0..(nk - 1) {
        for j in 0..(nj - 1) {
            let v00 = idx(0, j, k);
            let v10 = idx(0, j + 1, k);
            let v01 = idx(0, j, k + 1);
            let v11 = idx(0, j + 1, k + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);

            let v00 = idx(ni - 1, j, k);
            let v10 = idx(ni - 1, j + 1, k);
            let v01 = idx(ni - 1, j, k + 1);
            let v11 = idx(ni - 1, j + 1, k + 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }

    indices
}

fn quad_grid_indices(nx: usize, ny: usize) -> Vec<u32> {
    let idx = |i: usize, j: usize| -> u32 { (i + j * nx) as u32 };
    let mut indices = Vec::with_capacity((nx - 1) * (ny - 1) * 6);
    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let v00 = idx(i, j);
            let v10 = idx(i + 1, j);
            let v01 = idx(i, j + 1);
            let v11 = idx(i + 1, j + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
        }
    }
    indices
}

fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = glam::Vec3::from_array(positions[tri[0] as usize]);
        let b = glam::Vec3::from_array(positions[tri[1] as usize]);
        let c = glam::Vec3::from_array(positions[tri[2] as usize]);
        let n = (b - a).cross(c - a);
        for &idx in tri {
            let acc = glam::Vec3::from_array(normals[idx as usize]) + n;
            normals[idx as usize] = acc.to_array();
        }
    }
    for n in &mut normals {
        *n = glam::Vec3::from_array(*n).normalize_or_zero().to_array();
    }
    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Create a minimal CGNS-like HDF5 file with a single unstructured zone
    /// containing a triangle, and verify the reader produces correct output.
    #[test]
    fn reads_unstructured_triangle_zone() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("brimcast-cgns-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        let cgns_path = dir.join("test.cgns");

        // Build a minimal CGNS HDF5 structure.
        let file = Hdf5File::create(&cgns_path).unwrap();

        // CGNSLibraryVersion
        let ver = file
            .new_dataset::<f32>()
            .shape([1])
            .create(" data")
            .unwrap();
        ver.write(&[3.4f32]).unwrap();
        write_label(&file, "CGNSLibraryVersion_t");

        // Base
        let base = file.create_group("Base").unwrap();
        write_label(&base, "CGNSBase_t");
        let base_data = base
            .new_dataset::<i32>()
            .shape([2])
            .create(" data")
            .unwrap();
        base_data.write(&[3i32, 3]).unwrap(); // cell_dim=3, phys_dim=3

        // Zone
        let zone = base.create_group("Zone1").unwrap();
        write_label(&zone, "Zone_t");
        let zone_data = zone
            .new_dataset::<i32>()
            .shape([3])
            .create(" data")
            .unwrap();
        zone_data.write(&[3i32, 1, 0]).unwrap(); // 3 vertices, 1 cell, 0 boundary

        // ZoneType
        let zt = zone.create_group("ZoneType").unwrap();
        write_label(&zt, "ZoneType_t");
        let zt_data = zt.new_dataset::<u8>().shape([12]).create(" data").unwrap();
        zt_data.write(b"Unstructured").unwrap();

        // GridCoordinates
        let gc = zone.create_group("GridCoordinates").unwrap();
        write_label(&gc, "GridCoordinates_t");

        let cx = gc.create_group("CoordinateX").unwrap();
        write_label(&cx, "DataArray_t");
        let cx_data = cx.new_dataset::<f64>().shape([3]).create(" data").unwrap();
        cx_data.write(&[0.0f64, 1.0, 0.5]).unwrap();

        let cy = gc.create_group("CoordinateY").unwrap();
        write_label(&cy, "DataArray_t");
        let cy_data = cy.new_dataset::<f64>().shape([3]).create(" data").unwrap();
        cy_data.write(&[0.0f64, 0.0, 1.0]).unwrap();

        let cz = gc.create_group("CoordinateZ").unwrap();
        write_label(&cz, "DataArray_t");
        let cz_data = cz.new_dataset::<f64>().shape([3]).create(" data").unwrap();
        cz_data.write(&[0.0f64, 0.0, 0.0]).unwrap();

        // Elements (TRI_3)
        let elem = zone.create_group("Elements").unwrap();
        write_label(&elem, "Elements_t");
        let elem_data = elem
            .new_dataset::<i32>()
            .shape([2])
            .create(" data")
            .unwrap();
        elem_data.write(&[5i32, 0]).unwrap(); // TRI_3, boundary flag

        let conn = elem.create_group("ElementConnectivity").unwrap();
        write_label(&conn, "DataArray_t");
        let conn_data = conn
            .new_dataset::<i32>()
            .shape([3])
            .create(" data")
            .unwrap();
        conn_data.write(&[1i32, 2, 3]).unwrap(); // 1-based

        // FlowSolution
        let sol = zone.create_group("FlowSolution").unwrap();
        write_label(&sol, "FlowSolution_t");

        let gl = sol.create_group("GridLocation").unwrap();
        write_label(&gl, "GridLocation_t");
        let gl_data = gl.new_dataset::<u8>().shape([6]).create(" data").unwrap();
        gl_data.write(b"Vertex").unwrap();

        let pressure = sol.create_group("Pressure").unwrap();
        write_label(&pressure, "DataArray_t");
        let p_data = pressure
            .new_dataset::<f64>()
            .shape([3])
            .create(" data")
            .unwrap();
        p_data.write(&[100.0f64, 200.0, 150.0]).unwrap();

        drop(file);

        // Read it back.
        let datasets = read(&cgns_path).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 3);
        assert_eq!(ds.indices.len(), 3);
        assert_eq!(ds.indices, vec![0, 1, 2]);
        assert!(ds.point_data.contains_key("Pressure"));
        let p = ds.point_data.get("Pressure").unwrap();
        assert_eq!(p.len(), 3);
        assert!((p[0] - 100.0).abs() < 0.01);
        assert!((p[1] - 200.0).abs() < 0.01);
        assert!((p[2] - 150.0).abs() < 0.01);

        let _ = std::fs::remove_file(&cgns_path);
        let _ = std::fs::remove_dir(&dir);
    }

    fn write_label(group: &hdf5::Group, label: &str) {
        let attr = group
            .new_attr::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("label")
            .unwrap();
        let val: hdf5::types::VarLenUnicode = label.parse().unwrap();
        attr.write_scalar(&val).unwrap();
    }
}
