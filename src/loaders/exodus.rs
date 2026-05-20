use std::collections::HashMap;
use std::path::Path;

use hdf5::File as Hdf5File;

use crate::{error::IoError, types::IoDataSet};
use super::common::Dataset;
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};

/// Decode an Exodus II file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

/// Read an Exodus II file and return one [`Dataset`] per element block.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    read_at_timestep(path, None)
}

/// Read time-series metadata from an Exodus II file.
///
/// Returns timestep entries where each selector is `"time_idx"`.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let file = open_exodus(path)?;
    let times = read_time_values(&file)?;

    if times.is_empty() {
        return Ok(PvdSeries {
            timesteps: vec![TimestepEntry {
                time: 0.0,
                file: path.to_path_buf(),
                selector: Some("0".to_string()),
            }],
        });
    }

    let timesteps = times
        .iter()
        .enumerate()
        .map(|(i, &t)| TimestepEntry {
            time: t as f64,
            file: path.to_path_buf(),
            selector: Some(i.to_string()),
        })
        .collect();

    Ok(PvdSeries { timesteps })
}

/// Read datasets at a specific timestep index.
pub fn read_selected(path: &Path, selector: &str) -> Result<Vec<Dataset>, ReadError> {
    let time_idx: usize = selector
        .parse()
        .map_err(|_| ReadError::Exodus(format!("Invalid Exodus selector: {selector}")))?;
    read_at_timestep(path, Some(time_idx))
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn open_exodus(path: &Path) -> Result<Hdf5File, ReadError> {
    Hdf5File::open(path)
        .map_err(|e| ReadError::Exodus(format!("Failed to open {}: {e}", path.display())))
}

fn read_at_timestep(path: &Path, time_idx: Option<usize>) -> Result<Vec<Dataset>, ReadError> {
    let file = open_exodus(path)?;

    let num_dim = read_dimension(&file, "num_dim").unwrap_or(3) as usize;

    // Read coordinates.
    let coord_x = read_dataset_f32_from(&file, "coordx")?;
    let coord_y = read_dataset_f32_from(&file, "coordy")?;
    let coord_z = if num_dim >= 3 {
        read_dataset_f32_from(&file, "coordz").unwrap_or_default()
    } else {
        vec![0.0; coord_x.len()]
    };

    let n_nodes = coord_x.len();
    let positions: Vec<[f32; 3]> = coord_x
        .iter()
        .zip(coord_y.iter())
        .zip(coord_z.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    // Read nodal variables for the requested timestep.
    let nodal_vars = read_nodal_variables(&file, n_nodes, time_idx)?;

    // Read element blocks.
    let num_el_blk = read_dimension(&file, "num_el_blk").unwrap_or(0) as usize;
    let mut datasets = Vec::with_capacity(num_el_blk.max(1));

    for blk_idx in 1..=num_el_blk {
        if let Ok(ds) = read_element_block(&file, blk_idx, &positions, &nodal_vars, time_idx) {
            datasets.push(ds);
        }
    }

    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }
    Ok(datasets)
}

/// Read a single element block and produce a Dataset.
fn read_element_block(
    file: &Hdf5File,
    blk_idx: usize,
    all_positions: &[[f32; 3]],
    nodal_vars: &HashMap<String, Vec<f32>>,
    time_idx: Option<usize>,
) -> Result<Dataset, ReadError> {
    let connect_name = format!("connect{blk_idx}");
    let ds = file
        .dataset(&connect_name)
        .map_err(|e| ReadError::Exodus(format!("Missing {connect_name}: {e}")))?;

    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(ReadError::Exodus(format!(
            "{connect_name} has unexpected shape: {shape:?}"
        )));
    }
    let num_elem = shape[0];
    let nodes_per_elem = shape[1];

    // Read connectivity (1-based indices).
    let connectivity = read_dataset_i32(&ds)?;

    // Determine element type from the `elem_type` attribute on the connectivity dataset.
    let elem_type = ds
        .attr("elem_type")
        .ok()
        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|s| s.to_string().to_uppercase())
        .or_else(|| {
            ds.attr("elem_type")
                .ok()
                .and_then(|a| a.read_scalar::<hdf5::types::FixedAscii<33>>().ok())
                .map(|s| s.as_str().trim_end_matches('\0').to_uppercase())
        })
        .unwrap_or_default();

    // Collect unique node indices used by this block and build a local index map.
    let mut global_indices: Vec<i32> = connectivity.clone();
    global_indices.sort_unstable();
    global_indices.dedup();

    let mut global_to_local: HashMap<i32, u32> = HashMap::with_capacity(global_indices.len());
    let mut positions = Vec::with_capacity(global_indices.len());
    for (local_idx, &global_idx) in global_indices.iter().enumerate() {
        global_to_local.insert(global_idx, local_idx as u32);
        let gi = (global_idx - 1).max(0) as usize;
        if gi < all_positions.len() {
            positions.push(all_positions[gi]);
        } else {
            positions.push([0.0; 3]);
        }
    }

    // Triangulate elements.
    let mut indices = Vec::new();
    for elem in connectivity.chunks_exact(nodes_per_elem) {
        let local: Vec<u32> = elem
            .iter()
            .map(|&g| *global_to_local.get(&g).unwrap_or(&0))
            .collect();
        triangulate_element(&elem_type, nodes_per_elem, &local, &mut indices);
    }

    if indices.is_empty() || positions.is_empty() {
        return Err(ReadError::Empty);
    }

    let normals = compute_normals(&positions, &indices);

    // Map nodal variables to local node ordering.
    let mut point_data = HashMap::new();
    for (name, all_values) in nodal_vars {
        let mut local_values = Vec::with_capacity(global_indices.len());
        for &global_idx in &global_indices {
            let gi = (global_idx - 1).max(0) as usize;
            local_values.push(if gi < all_values.len() {
                all_values[gi]
            } else {
                0.0
            });
        }
        point_data.insert(name.clone(), local_values);
    }

    // Read element variables for this block.
    let cell_data = read_element_variables(file, blk_idx, num_elem, time_idx)?;

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
// Nodal variables
// ---------------------------------------------------------------------------

fn read_nodal_variables(
    file: &Hdf5File,
    _n_nodes: usize,
    time_idx: Option<usize>,
) -> Result<HashMap<String, Vec<f32>>, ReadError> {
    let mut result = HashMap::new();

    let num_nod_var = read_dimension(file, "num_nod_var").unwrap_or(0) as usize;
    if num_nod_var == 0 {
        return Ok(result);
    }

    let var_names = read_variable_names(file, "name_nod_var", num_nod_var);
    let step = time_idx.unwrap_or(0);

    for (i, name) in var_names.iter().enumerate() {
        let ds_name = format!("vals_nod_var{}", i + 1);
        if let Ok(ds) = file.dataset(&ds_name) {
            let shape = ds.shape();
            if shape.len() == 2 && step < shape[0] {
                // Read the full dataset and extract the requested timestep.
                if let Ok(all_values) = read_dataset_f32(&ds) {
                    let start = step * shape[1];
                    let end = start + shape[1];
                    if end <= all_values.len() {
                        result.insert(name.clone(), all_values[start..end].to_vec());
                    }
                }
            } else if shape.len() == 1 {
                // Single timestep.
                if let Ok(values) = read_dataset_f32(&ds) {
                    result.insert(name.clone(), values);
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Element variables
// ---------------------------------------------------------------------------

fn read_element_variables(
    file: &Hdf5File,
    blk_idx: usize,
    num_elem: usize,
    time_idx: Option<usize>,
) -> Result<HashMap<String, Vec<f32>>, ReadError> {
    let mut result = HashMap::new();

    let num_elem_var = read_dimension(file, "num_elem_var").unwrap_or(0) as usize;
    if num_elem_var == 0 {
        return Ok(result);
    }

    let var_names = read_variable_names(file, "name_elem_var", num_elem_var);
    let step = time_idx.unwrap_or(0);

    for (i, name) in var_names.iter().enumerate() {
        let ds_name = format!("vals_elem_var{}eb{}", i + 1, blk_idx);
        if let Ok(ds) = file.dataset(&ds_name) {
            let shape = ds.shape();
            if shape.len() == 2 && step < shape[0] {
                if let Ok(all_values) = read_dataset_f32(&ds) {
                    let start = step * shape[1];
                    let end = start + shape[1];
                    if end <= all_values.len() {
                        result.insert(name.clone(), all_values[start..end].to_vec());
                    }
                }
            } else if shape.len() == 1 {
                if let Ok(values) = read_dataset_f32(&ds) {
                    result.insert(name.clone(), values);
                }
            }
        }
    }

    let _ = num_elem;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Variable name reading
// ---------------------------------------------------------------------------

fn read_variable_names(file: &Hdf5File, ds_name: &str, count: usize) -> Vec<String> {
    let ds = match file.dataset(ds_name) {
        Ok(ds) => ds,
        Err(_) => {
            return (1..=count).map(|i| format!("var_{i}")).collect();
        }
    };

    let shape = ds.shape();
    if shape.len() == 2 {
        // [num_vars, name_length] char array.
        let name_len = shape[1];
        if let Ok(chars) = ds.read_raw::<u8>() {
            return chars
                .chunks(name_len)
                .take(count)
                .map(|chunk| {
                    String::from_utf8_lossy(chunk)
                        .trim_end_matches('\0')
                        .trim()
                        .to_string()
                })
                .collect();
        }
    }

    (1..=count).map(|i| format!("var_{i}")).collect()
}

// ---------------------------------------------------------------------------
// Time values
// ---------------------------------------------------------------------------

fn read_time_values(file: &Hdf5File) -> Result<Vec<f32>, ReadError> {
    match file.dataset("time_whole") {
        Ok(ds) => read_dataset_f32(&ds),
        Err(_) => Ok(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// Dimension reading
// ---------------------------------------------------------------------------

/// Read a NetCDF dimension value.
///
/// In NetCDF-4/HDF5, dimensions are stored as HDF5 dimension scales.
/// We look for the dataset with the dimension name and read its size.
fn read_dimension(file: &Hdf5File, name: &str) -> Option<u64> {
    // Try reading as a root attribute first (some writers store dimensions there).
    if let Ok(attr) = file.attr(name) {
        if let Ok(val) = attr.read_scalar::<i32>() {
            return Some(val as u64);
        }
        if let Ok(val) = attr.read_scalar::<i64>() {
            return Some(val as u64);
        }
    }

    // NetCDF-4 stores dimensions as datasets in a `_netcdf_dim_info` convention,
    // but the actual size is derivable from the shape of datasets that use them.
    // For Exodus II, we can infer dimensions from dataset shapes.
    match name {
        "num_dim" => {
            // Check if coordz exists.
            if file.dataset("coordz").is_ok() {
                Some(3)
            } else {
                Some(2)
            }
        }
        "num_nodes" => file.dataset("coordx").ok().map(|ds| ds.shape()[0] as u64),
        "num_el_blk" => {
            // Count connect{N} datasets.
            let mut count = 0u64;
            for i in 1..=1000 {
                if file.dataset(&format!("connect{i}")).is_ok() {
                    count = i as u64;
                } else {
                    break;
                }
            }
            Some(count)
        }
        "num_nod_var" => {
            // Count vals_nod_var{N} datasets.
            let mut count = 0u64;
            for i in 1..=1000 {
                if file.dataset(&format!("vals_nod_var{i}")).is_ok() {
                    count = i as u64;
                } else {
                    break;
                }
            }
            Some(count)
        }
        "num_elem_var" => {
            // Count vals_elem_var{N}eb1 datasets.
            let mut count = 0u64;
            for i in 1..=1000 {
                if file.dataset(&format!("vals_elem_var{i}eb1")).is_ok() {
                    count = i as u64;
                } else {
                    break;
                }
            }
            Some(count)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Element triangulation
// ---------------------------------------------------------------------------

fn triangulate_element(elem_type: &str, nodes_per_elem: usize, local: &[u32], out: &mut Vec<u32>) {
    if local.len() < nodes_per_elem {
        return;
    }

    // Determine triangulation from Exodus element type string or node count.
    let kind = if !elem_type.is_empty() {
        classify_element_type(elem_type)
    } else {
        classify_by_node_count(nodes_per_elem)
    };

    let v = |i: usize| local[i];

    match kind {
        ElemKind::Tri => {
            out.extend_from_slice(&[v(0), v(1), v(2)]);
        }
        ElemKind::Quad => {
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
        }
        ElemKind::Tet => {
            out.extend_from_slice(&[v(0), v(2), v(1)]);
            out.extend_from_slice(&[v(0), v(1), v(3)]);
            out.extend_from_slice(&[v(1), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(3), v(2)]);
        }
        ElemKind::Pyramid => {
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(1), v(4)]);
            out.extend_from_slice(&[v(1), v(2), v(4)]);
            out.extend_from_slice(&[v(2), v(3), v(4)]);
            out.extend_from_slice(&[v(3), v(0), v(4)]);
        }
        ElemKind::Wedge => {
            out.extend_from_slice(&[v(0), v(1), v(2)]);
            out.extend_from_slice(&[v(3), v(5), v(4)]);
            out.extend_from_slice(&[v(0), v(3), v(4), v(0), v(4), v(1)]);
            out.extend_from_slice(&[v(1), v(4), v(5), v(1), v(5), v(2)]);
            out.extend_from_slice(&[v(2), v(5), v(3), v(2), v(3), v(0)]);
        }
        ElemKind::Hex => {
            out.extend_from_slice(&[v(0), v(3), v(2), v(0), v(2), v(1)]); // bottom
            out.extend_from_slice(&[v(4), v(5), v(6), v(4), v(6), v(7)]); // top
            out.extend_from_slice(&[v(0), v(1), v(5), v(0), v(5), v(4)]); // front
            out.extend_from_slice(&[v(2), v(3), v(7), v(2), v(7), v(6)]); // back
            out.extend_from_slice(&[v(0), v(4), v(7), v(0), v(7), v(3)]); // left
            out.extend_from_slice(&[v(1), v(2), v(6), v(1), v(6), v(5)]); // right
        }
        ElemKind::Unknown => {}
    }
}

#[derive(Clone, Copy)]
enum ElemKind {
    Tri,
    Quad,
    Tet,
    Pyramid,
    Wedge,
    Hex,
    Unknown,
}

fn classify_element_type(elem_type: &str) -> ElemKind {
    let upper = elem_type.to_uppercase();
    if upper.starts_with("TRI") {
        ElemKind::Tri
    } else if upper.starts_with("QUAD") || upper.starts_with("SHELL") {
        ElemKind::Quad
    } else if upper.starts_with("TET") || upper.starts_with("TETRA") {
        ElemKind::Tet
    } else if upper.starts_with("PYR") || upper.starts_with("PYRA") {
        ElemKind::Pyramid
    } else if upper.starts_with("WEDGE") || upper.starts_with("PENTA") {
        ElemKind::Wedge
    } else if upper.starts_with("HEX") {
        ElemKind::Hex
    } else {
        ElemKind::Unknown
    }
}

fn classify_by_node_count(n: usize) -> ElemKind {
    match n {
        3 | 7 => ElemKind::Tri,
        4 | 10 | 11 => ElemKind::Tet, // 4-node defaults to tet in 3D context
        5 | 13 | 14 => ElemKind::Pyramid,
        6 | 15 | 16 | 21 => ElemKind::Wedge,
        8 | 20 | 27 => ElemKind::Hex,
        _ => ElemKind::Unknown,
    }
}

// ---------------------------------------------------------------------------
// HDF5 helpers
// ---------------------------------------------------------------------------

fn read_dataset_f32_from(file: &Hdf5File, name: &str) -> Result<Vec<f32>, ReadError> {
    let ds = file
        .dataset(name)
        .map_err(|e| ReadError::Exodus(format!("Missing dataset {name}: {e}")))?;
    read_dataset_f32(&ds)
}

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
    Err(ReadError::Exodus(
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
    Err(ReadError::Exodus(
        "Unsupported HDF5 dataset type for int conversion".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

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

    /// Create a minimal Exodus-II-like HDF5 file with one element block
    /// containing a single triangle, and verify the reader produces correct output.
    #[test]
    fn reads_single_triangle_block() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("brimcast-exodus-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        let exo_path = dir.join("test.exo");

        let file = Hdf5File::create(&exo_path).unwrap();

        // Coordinates: 3 nodes forming a triangle.
        let cx = file
            .new_dataset::<f64>()
            .shape([3])
            .create("coordx")
            .unwrap();
        cx.write(&[0.0f64, 1.0, 0.5]).unwrap();
        let cy = file
            .new_dataset::<f64>()
            .shape([3])
            .create("coordy")
            .unwrap();
        cy.write(&[0.0f64, 0.0, 1.0]).unwrap();
        let cz = file
            .new_dataset::<f64>()
            .shape([3])
            .create("coordz")
            .unwrap();
        cz.write(&[0.0f64, 0.0, 0.0]).unwrap();

        // Element block 1: one TRI3 element.
        let conn = file
            .new_dataset::<i32>()
            .shape([1, 3])
            .create("connect1")
            .unwrap();
        conn.write_raw(&[1i32, 2, 3]).unwrap(); // 1-based
        let attr = conn
            .new_attr::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("elem_type")
            .unwrap();
        let val: hdf5::types::VarLenUnicode = "TRI3".parse().unwrap();
        attr.write_scalar(&val).unwrap();

        // One nodal variable: "temperature".
        let name_ds = file
            .new_dataset::<u8>()
            .shape([1, 33])
            .create("name_nod_var")
            .unwrap();
        let mut name_buf = [0u8; 33];
        name_buf[..11].copy_from_slice(b"temperature");
        name_ds.write_raw(&name_buf).unwrap();

        // Nodal variable values: 1 timestep, 3 nodes.
        let vals = file
            .new_dataset::<f64>()
            .shape([1, 3])
            .create("vals_nod_var1")
            .unwrap();
        vals.write_raw(&[300.0f64, 350.0, 325.0]).unwrap();

        // Time values.
        let time = file
            .new_dataset::<f64>()
            .shape([1])
            .create("time_whole")
            .unwrap();
        time.write(&[0.0f64]).unwrap();

        drop(file);

        // Read it back.
        let datasets = read(&exo_path).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 3);
        assert_eq!(ds.indices.len(), 3);
        assert_eq!(ds.indices, vec![0, 1, 2]);

        assert!(ds.point_data.contains_key("temperature"));
        let temps = ds.point_data.get("temperature").unwrap();
        assert_eq!(temps.len(), 3);
        assert!((temps[0] - 300.0).abs() < 0.01);
        assert!((temps[1] - 350.0).abs() < 0.01);
        assert!((temps[2] - 325.0).abs() < 0.01);

        // Test series reading.
        let series = read_series(&exo_path).unwrap();
        assert_eq!(series.timesteps.len(), 1);
        assert_eq!(series.timesteps[0].time, 0.0);

        let _ = std::fs::remove_file(&exo_path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn reads_hex_block_with_two_timesteps() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("brimcast-exodus-hex-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        let exo_path = dir.join("hex.exo");

        let file = Hdf5File::create(&exo_path).unwrap();

        // 8 nodes for a unit cube.
        let coords: [[f64; 8]; 3] = [
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], // x
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], // y
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], // z
        ];
        let cx = file
            .new_dataset::<f64>()
            .shape([8])
            .create("coordx")
            .unwrap();
        cx.write(&coords[0]).unwrap();
        let cy = file
            .new_dataset::<f64>()
            .shape([8])
            .create("coordy")
            .unwrap();
        cy.write(&coords[1]).unwrap();
        let cz = file
            .new_dataset::<f64>()
            .shape([8])
            .create("coordz")
            .unwrap();
        cz.write(&coords[2]).unwrap();

        // One HEX8 element.
        let conn = file
            .new_dataset::<i32>()
            .shape([1, 8])
            .create("connect1")
            .unwrap();
        conn.write_raw(&[1i32, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let attr = conn
            .new_attr::<hdf5::types::VarLenUnicode>()
            .shape(())
            .create("elem_type")
            .unwrap();
        let val: hdf5::types::VarLenUnicode = "HEX8".parse().unwrap();
        attr.write_scalar(&val).unwrap();

        // Nodal variable: "pressure", 2 timesteps.
        let name_ds = file
            .new_dataset::<u8>()
            .shape([1, 33])
            .create("name_nod_var")
            .unwrap();
        let mut name_buf = [0u8; 33];
        name_buf[..8].copy_from_slice(b"pressure");
        name_ds.write_raw(&name_buf).unwrap();

        let vals = file
            .new_dataset::<f64>()
            .shape([2, 8])
            .create("vals_nod_var1")
            .unwrap();
        vals.write_raw(&[
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // step 0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, // step 1
        ])
        .unwrap();

        let time = file
            .new_dataset::<f64>()
            .shape([2])
            .create("time_whole")
            .unwrap();
        time.write(&[0.0f64, 1.0]).unwrap();

        drop(file);

        // Verify series.
        let series = read_series(&exo_path).unwrap();
        assert_eq!(series.timesteps.len(), 2);
        assert_eq!(series.timesteps[1].time, 1.0);

        // Read step 0.
        let ds0 = read(&exo_path).unwrap();
        let p0 = ds0[0].point_data.get("pressure").unwrap();
        assert!((p0[0] - 1.0).abs() < 0.01);

        // Read step 1 via selector.
        let ds1 = read_selected(&exo_path, "1").unwrap();
        let p1 = ds1[0].point_data.get("pressure").unwrap();
        assert!((p1[0] - 10.0).abs() < 0.01);

        // HEX8 produces 12 triangles = 36 indices.
        assert_eq!(ds0[0].indices.len(), 36);

        let _ = std::fs::remove_file(&exo_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
