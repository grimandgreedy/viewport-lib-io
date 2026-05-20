use std::collections::HashMap;
use std::path::Path;

use netcdf_reader::{NcFile, NcType, NcVariable};

use crate::{error::IoError, types::IoDataSet};
use super::common::{Dataset, VolumeGeometry, VolumeGrid};
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};

/// Decode a NetCDF file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

/// Read a NetCDF file and return one [`Dataset`] per data variable.
///
/// Coordinate variables (dimensions sharing the variable name) are consumed
/// as axis positions rather than emitted as separate datasets. Structured 3-D
/// grids are exposed via [`VolumeGrid`]; 2-D grids are triangulated as surfaces.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    let file = open(path)?;
    let ctx = FileContext::new(&file)?;

    // Read first timestep (index 0) or the whole file if no time axis.
    read_at_time(&file, &ctx, 0)
}

/// Read time-series metadata from a NetCDF file.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let file = open(path)?;
    let ctx = FileContext::new(&file)?;

    let timesteps = match &ctx.time {
        Some(t) => t
            .values
            .iter()
            .enumerate()
            .map(|(i, &tv)| TimestepEntry {
                time: tv,
                file: path.to_path_buf(),
                selector: Some(i.to_string()),
            })
            .collect(),
        None => vec![TimestepEntry {
            time: 0.0,
            file: path.to_path_buf(),
            selector: None,
        }],
    };

    Ok(PvdSeries { timesteps })
}

/// Read datasets at a specific timestep index.
pub fn read_selected(path: &Path, selector: &str) -> Result<Vec<Dataset>, ReadError> {
    let step: usize = selector
        .parse()
        .map_err(|_| ReadError::NetCdf(format!("Invalid NetCDF selector: {selector}")))?;

    let file = open(path)?;
    let ctx = FileContext::new(&file)?;
    read_at_time(&file, &ctx, step)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn open(path: &Path) -> Result<NcFile, ReadError> {
    NcFile::open(path).map_err(|e| ReadError::NetCdf(format!("{e}")))
}

/// Cached information about the file's coordinate structure.
struct FileContext {
    /// Time coordinate values (if present).
    time: Option<TimeCoord>,
    /// Spatial coordinate axis names and values.
    spatial: SpatialAxes,
    /// Names of data variables (not coordinates, not bounds).
    data_vars: Vec<String>,
}

struct TimeCoord {
    name: String,
    values: Vec<f64>,
}

struct SpatialAxes {
    /// X-axis coordinate values (lon, x, etc.).
    x: Option<AxisValues>,
    /// Y-axis coordinate values (lat, y, etc.).
    y: Option<AxisValues>,
    /// Z-axis coordinate values (depth, height, z, etc.).
    z: Option<AxisValues>,
}

struct AxisValues {
    name: String,
    values: Vec<f64>,
}

impl FileContext {
    fn new(file: &NcFile) -> Result<Self, ReadError> {
        let vars = file
            .variables()
            .map_err(|e| ReadError::NetCdf(format!("{e}")))?;
        let dims = file
            .dimensions()
            .map_err(|e| ReadError::NetCdf(format!("{e}")))?;

        // Find coordinate variables (1-D vars whose name matches their dimension).
        let coord_names: Vec<String> = vars
            .iter()
            .filter(|v| v.is_coordinate_variable())
            .map(|v| v.name.clone())
            .collect();

        // Detect time coordinate.
        let time = detect_time_coord(file, vars)?;

        // Detect spatial axes.
        let spatial = detect_spatial_axes(file, vars, dims)?;

        // Data variables: everything that isn't a coordinate, bounds, or auxiliary.
        let data_vars: Vec<String> = vars
            .iter()
            .filter(|v| {
                !coord_names.contains(&v.name)
                    && !v.name.contains("_bnds")
                    && !v.name.contains("_bounds")
                    && v.ndim() >= 1
                    && is_numeric_type(&v.dtype)
            })
            .map(|v| v.name.clone())
            .collect();

        Ok(FileContext {
            time,
            spatial,
            data_vars,
        })
    }
}

fn detect_time_coord(file: &NcFile, vars: &[NcVariable]) -> Result<Option<TimeCoord>, ReadError> {
    // Look for a coordinate variable named "time", or one with axis="T" / units containing "since".
    let time_var = vars.iter().find(|v| {
        if !v.is_coordinate_variable() {
            return false;
        }
        let name_lower = v.name.to_lowercase();
        if name_lower == "time" || name_lower == "t" {
            return true;
        }
        // Check CF axis attribute.
        if let Some(attr) = v.attribute("axis") {
            if attr.value.as_string().map_or(false, |s| s == "T") {
                return true;
            }
        }
        // Check units for "since" (CF time convention).
        if let Some(attr) = v.attribute("units") {
            if let Some(units) = attr.value.as_string() {
                return units.contains("since");
            }
        }
        false
    });

    match time_var {
        Some(tv) => {
            let values = read_var_as_f64(file, &tv.name)?;
            Ok(Some(TimeCoord {
                name: tv.name.clone(),
                values,
            }))
        }
        None => Ok(None),
    }
}

fn detect_spatial_axes(
    file: &NcFile,
    vars: &[NcVariable],
    _dims: &[netcdf_reader::NcDimension],
) -> Result<SpatialAxes, ReadError> {
    let mut x = None;
    let mut y = None;
    let mut z = None;

    for v in vars.iter().filter(|v| v.is_coordinate_variable()) {
        let name_lower = v.name.to_lowercase();
        let axis = v
            .attribute("axis")
            .and_then(|a| a.value.as_string())
            .unwrap_or_default()
            .to_uppercase();

        let is_x = axis == "X"
            || name_lower == "x"
            || name_lower == "lon"
            || name_lower == "longitude"
            || name_lower == "xc";
        let is_y = axis == "Y"
            || name_lower == "y"
            || name_lower == "lat"
            || name_lower == "latitude"
            || name_lower == "yc";
        let is_z = axis == "Z"
            || name_lower == "z"
            || name_lower == "depth"
            || name_lower == "height"
            || name_lower == "level"
            || name_lower == "lev"
            || name_lower == "zc";

        if is_x && x.is_none() {
            x = Some(AxisValues {
                name: v.name.clone(),
                values: read_var_as_f64(file, &v.name)?,
            });
        } else if is_y && y.is_none() {
            y = Some(AxisValues {
                name: v.name.clone(),
                values: read_var_as_f64(file, &v.name)?,
            });
        } else if is_z && z.is_none() {
            z = Some(AxisValues {
                name: v.name.clone(),
                values: read_var_as_f64(file, &v.name)?,
            });
        }
    }

    Ok(SpatialAxes { x, y, z })
}

/// Read datasets for a given timestep index.
fn read_at_time(
    file: &NcFile,
    ctx: &FileContext,
    time_step: usize,
) -> Result<Vec<Dataset>, ReadError> {
    let mut datasets = Vec::new();

    for var_name in &ctx.data_vars {
        let var = file
            .variable(var_name)
            .map_err(|e| ReadError::NetCdf(format!("{e}")))?;

        // Determine which dimensions are spatial vs time.
        let dim_names: Vec<String> = var.dimensions().iter().map(|d| d.name.clone()).collect();
        let has_time = ctx
            .time
            .as_ref()
            .map_or(false, |t| dim_names.contains(&t.name));

        // Read the variable data.
        let raw = if has_time {
            read_var_slice_at_time(file, var_name, &dim_names, ctx, time_step)?
        } else {
            read_var_as_f32(file, var_name)?
        };

        // Determine spatial shape (excluding time dimension).
        let spatial_dims: Vec<(String, u64)> = var
            .dimensions()
            .iter()
            .filter(|d| ctx.time.as_ref().map_or(true, |t| d.name != t.name))
            .map(|d| (d.name.clone(), d.size))
            .collect();

        let ds = if spatial_dims.len() >= 3 {
            build_volume_dataset(ctx, &spatial_dims, var_name, &raw)?
        } else if spatial_dims.len() == 2 {
            build_surface_dataset(ctx, &spatial_dims, var_name, &raw)?
        } else {
            // 1-D or scalar — skip for visualization purposes.
            continue;
        };

        if !ds.positions.is_empty() || ds.volume.is_some() {
            datasets.push(ds);
        }
    }

    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }

    Ok(datasets)
}

/// Build a 3-D volume dataset.
fn build_volume_dataset(
    ctx: &FileContext,
    spatial_dims: &[(String, u64)],
    var_name: &str,
    data: &[f32],
) -> Result<Dataset, ReadError> {
    // Map dimension names to coordinate values.
    let (x_vals, y_vals, z_vals) = resolve_spatial_coords(ctx, spatial_dims)?;

    let nx = x_vals.len() as u32;
    let ny = y_vals.len() as u32;
    let nz = z_vals.len() as u32;

    let geometry = if is_uniform(&x_vals) && is_uniform(&y_vals) && is_uniform(&z_vals) {
        let origin = [
            *x_vals.first().unwrap_or(&0.0) as f32,
            *y_vals.first().unwrap_or(&0.0) as f32,
            *z_vals.first().unwrap_or(&0.0) as f32,
        ];
        let spacing = [
            if nx > 1 {
                (x_vals.last().unwrap() - x_vals.first().unwrap()) as f32 / (nx - 1) as f32
            } else {
                1.0
            },
            if ny > 1 {
                (y_vals.last().unwrap() - y_vals.first().unwrap()) as f32 / (ny - 1) as f32
            } else {
                1.0
            },
            if nz > 1 {
                (z_vals.last().unwrap() - z_vals.first().unwrap()) as f32 / (nz - 1) as f32
            } else {
                1.0
            },
        ];
        VolumeGeometry::Uniform { origin, spacing }
    } else {
        VolumeGeometry::Rectilinear {
            xs: x_vals.iter().map(|&v| v as f32).collect(),
            ys: y_vals.iter().map(|&v| v as f32).collect(),
            zs: z_vals.iter().map(|&v| v as f32).collect(),
        }
    };

    // NetCDF data order is typically (z, y, x). The VolumeGrid expects (x, y, z) indexing
    // with x varying fastest. Reorder if necessary.
    let reordered = reorder_zyx_to_xyz(data, nx as usize, ny as usize, nz as usize);

    let mut point_data = HashMap::new();
    point_data.insert(var_name.to_string(), reordered);

    let volume = VolumeGrid {
        dims: [nx, ny, nz],
        geometry,
        point_data,
        cell_data: HashMap::new(),
    };

    Ok(Dataset {
        positions: Vec::new(),
        indices: Vec::new(),
        normals: Vec::new(),
        point_data: HashMap::new(),
        cell_data: HashMap::new(),
        edge_data: HashMap::new(),
        sparse_volume: None,
        volume: Some(volume),
        volume_mesh: None,
    })
}

/// Build a 2-D surface dataset by triangulating a structured grid.
fn build_surface_dataset(
    ctx: &FileContext,
    spatial_dims: &[(String, u64)],
    var_name: &str,
    data: &[f32],
) -> Result<Dataset, ReadError> {
    let (x_vals, y_vals, _z_vals) = resolve_spatial_coords(ctx, spatial_dims)?;

    let nx = x_vals.len();
    let ny = y_vals.len();

    if nx == 0 || ny == 0 {
        return Err(ReadError::Empty);
    }

    // Build positions on the XZ plane (Y = 0), matching the app convention
    // that 2D grids lie in the x-z plane.
    let mut positions = Vec::with_capacity(nx * ny);
    for iy in 0..ny {
        for ix in 0..nx {
            positions.push([x_vals[ix] as f32, 0.0, y_vals[iy] as f32]);
        }
    }

    // Triangulate.
    let mut indices = Vec::new();
    if nx >= 2 && ny >= 2 {
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                let v00 = (iy * nx + ix) as u32;
                let v10 = (iy * nx + (ix + 1)) as u32;
                let v01 = ((iy + 1) * nx + ix) as u32;
                let v11 = ((iy + 1) * nx + (ix + 1)) as u32;
                indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }
    }

    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];

    // Reorder data from (y, x) to row-major matching positions.
    let reordered = if data.len() == nx * ny {
        data.to_vec()
    } else {
        data.to_vec()
    };

    let mut point_data = HashMap::new();
    point_data.insert(var_name.to_string(), reordered);

    Ok(Dataset {
        positions,
        indices,
        normals,
        point_data,
        cell_data: HashMap::new(),
        edge_data: HashMap::new(),
        sparse_volume: None,
        volume: None,
        volume_mesh: None,
    })
}

/// Resolve coordinate values for spatial dimensions, in (x, y, z) order.
fn resolve_spatial_coords(
    ctx: &FileContext,
    spatial_dims: &[(String, u64)],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), ReadError> {
    // Try to match each spatial dimension to known axes.
    let mut x_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();

    // Reverse order: NetCDF typically stores (z, y, x), so last dim is often x.
    let mut unmatched = Vec::new();
    for (name, size) in spatial_dims.iter().rev() {
        let matched = try_match_axis(name, ctx);
        match matched {
            Some(('x', vals)) => x_vals = vals,
            Some(('y', vals)) => y_vals = vals,
            Some(('z', vals)) => z_vals = vals,
            _ => unmatched.push((name.clone(), *size)),
        }
    }

    // For any unmatched dims, assign in reverse order to z, y, x (whichever is still empty).
    for (name, size) in &unmatched {
        let synthetic: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let name_lower = name.to_lowercase();

        if x_vals.is_empty()
            && (name_lower.contains('x')
                || name_lower.contains("lon")
                || name_lower.contains("col"))
        {
            x_vals = synthetic;
        } else if y_vals.is_empty()
            && (name_lower.contains('y')
                || name_lower.contains("lat")
                || name_lower.contains("row"))
        {
            y_vals = synthetic;
        } else if z_vals.is_empty() {
            z_vals = synthetic;
        } else if y_vals.is_empty() {
            y_vals = synthetic;
        } else if x_vals.is_empty() {
            x_vals = synthetic;
        }
    }

    // Final fallback: if still empty, generate synthetic indices.
    for (_, size) in spatial_dims.iter().rev() {
        if x_vals.is_empty() {
            x_vals = (0..*size).map(|i| i as f64).collect();
        } else if y_vals.is_empty() {
            y_vals = (0..*size).map(|i| i as f64).collect();
        } else if z_vals.is_empty() {
            z_vals = (0..*size).map(|i| i as f64).collect();
        }
    }

    Ok((x_vals, y_vals, z_vals))
}

fn try_match_axis(dim_name: &str, ctx: &FileContext) -> Option<(char, Vec<f64>)> {
    if let Some(ref ax) = ctx.spatial.x {
        if ax.name == dim_name {
            return Some(('x', ax.values.clone()));
        }
    }
    if let Some(ref ax) = ctx.spatial.y {
        if ax.name == dim_name {
            return Some(('y', ax.values.clone()));
        }
    }
    if let Some(ref ax) = ctx.spatial.z {
        if ax.name == dim_name {
            return Some(('z', ax.values.clone()));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Data reading helpers
// ---------------------------------------------------------------------------

fn read_var_as_f64(file: &NcFile, name: &str) -> Result<Vec<f64>, ReadError> {
    let arr = file
        .read_variable_as_f64(name)
        .map_err(|e| ReadError::NetCdf(format!("Failed to read '{name}': {e}")))?;
    Ok(arr.into_raw_vec_and_offset().0)
}

fn read_var_as_f32(file: &NcFile, name: &str) -> Result<Vec<f32>, ReadError> {
    let arr = file
        .read_variable_as_f64(name)
        .map_err(|e| ReadError::NetCdf(format!("Failed to read '{name}': {e}")))?;
    Ok(arr
        .into_raw_vec_and_offset()
        .0
        .into_iter()
        .map(|v| v as f32)
        .collect())
}

fn read_var_slice_at_time(
    file: &NcFile,
    var_name: &str,
    dim_names: &[String],
    ctx: &FileContext,
    time_step: usize,
) -> Result<Vec<f32>, ReadError> {
    let time_name = ctx.time.as_ref().map(|t| &t.name);

    let var = file
        .variable(var_name)
        .map_err(|e| ReadError::NetCdf(format!("{e}")))?;

    let mut selections = Vec::with_capacity(var.ndim());
    for dim in var.dimensions() {
        if time_name.map_or(false, |tn| dim.name == *tn) {
            selections.push(netcdf_reader::NcSliceInfoElem::Index(time_step as u64));
        } else {
            selections.push(netcdf_reader::NcSliceInfoElem::Slice {
                start: 0,
                end: u64::MAX,
                step: 1,
            });
        }
    }

    let slice_info = netcdf_reader::NcSliceInfo { selections };
    let arr = file
        .read_variable_slice_as_f64(var_name, &slice_info)
        .map_err(|e| ReadError::NetCdf(format!("Failed to slice '{var_name}': {e}")))?;

    let _ = dim_names;
    Ok(arr
        .into_raw_vec_and_offset()
        .0
        .into_iter()
        .map(|v| v as f32)
        .collect())
}

fn is_numeric_type(dtype: &NcType) -> bool {
    matches!(
        dtype,
        NcType::Byte
            | NcType::Short
            | NcType::Int
            | NcType::Float
            | NcType::Double
            | NcType::UByte
            | NcType::UShort
            | NcType::UInt
            | NcType::Int64
            | NcType::UInt64
    )
}

fn is_uniform(vals: &[f64]) -> bool {
    if vals.len() < 2 {
        return true;
    }
    let step = vals[1] - vals[0];
    if step.abs() < 1e-30 {
        return false;
    }
    vals.windows(2)
        .all(|w| ((w[1] - w[0]) - step).abs() < step.abs() * 1e-6)
}

/// Reorder data from (z, y, x) to (x, y, z) order for VolumeGrid.
///
/// VolumeGrid expects x to vary fastest: `index = ix + iy * nx + iz * nx * ny`.
/// NetCDF typically stores in C order with last dim varying fastest,
/// i.e. `(z, y, x)` means x varies fastest — which is already the right order.
fn reorder_zyx_to_xyz(data: &[f32], _nx: usize, _ny: usize, _nz: usize) -> Vec<f32> {
    // NetCDF C-order with dims (z, y, x): x varies fastest, which matches VolumeGrid.
    data.to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("brimcast-{prefix}-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a minimal valid CDF-1 file with given dimensions, variables, and data.
    fn build_cdf1_bytes(
        dims: &[(&str, u32)],
        global_attrs: &[(&str, &str)],
        variables: &[CdfVar<'_>],
    ) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic + numrecs.
        data.extend_from_slice(b"CDF\x01");
        data.extend_from_slice(&0u32.to_be_bytes());

        // dim_list.
        if dims.is_empty() {
            data.extend_from_slice(&0u32.to_be_bytes());
            data.extend_from_slice(&0u32.to_be_bytes());
        } else {
            data.extend_from_slice(&0x0000_000Au32.to_be_bytes()); // NC_DIMENSION
            data.extend_from_slice(&(dims.len() as u32).to_be_bytes());
            for (name, size) in dims {
                write_nc_string(&mut data, name);
                data.extend_from_slice(&size.to_be_bytes());
            }
        }

        // att_list.
        if global_attrs.is_empty() {
            data.extend_from_slice(&0u32.to_be_bytes());
            data.extend_from_slice(&0u32.to_be_bytes());
        } else {
            data.extend_from_slice(&0x0000_000Cu32.to_be_bytes()); // NC_ATTRIBUTE
            data.extend_from_slice(&(global_attrs.len() as u32).to_be_bytes());
            for (name, value) in global_attrs {
                write_nc_string(&mut data, name);
                data.extend_from_slice(&2u32.to_be_bytes()); // NC_CHAR
                write_nc_string_value(&mut data, value);
            }
        }

        // var_list — first pass: compute header size, then fill offsets.
        if variables.is_empty() {
            data.extend_from_slice(&0u32.to_be_bytes());
            data.extend_from_slice(&0u32.to_be_bytes());
            return data;
        }

        data.extend_from_slice(&0x0000_000Bu32.to_be_bytes()); // NC_VARIABLE
        data.extend_from_slice(&(variables.len() as u32).to_be_bytes());

        // We'll write the var headers with placeholder offsets, then fix them.
        let mut var_header_starts = Vec::new();
        for v in variables {
            var_header_starts.push(data.len());
            write_nc_string(&mut data, v.name);
            data.extend_from_slice(&(v.dim_ids.len() as u32).to_be_bytes());
            for &did in v.dim_ids {
                data.extend_from_slice(&(did as u32).to_be_bytes());
            }
            // No per-variable attributes.
            data.extend_from_slice(&0u32.to_be_bytes());
            data.extend_from_slice(&0u32.to_be_bytes());
            // nc_type.
            data.extend_from_slice(&v.nc_type.to_be_bytes());
            // vsize (bytes).
            let vsize = v.values_f32.len() * 4;
            data.extend_from_slice(&(vsize as u32).to_be_bytes());
            // begin offset — placeholder.
            data.extend_from_slice(&0u32.to_be_bytes());
        }

        // Now fix offsets and write data.
        let mut data_offset = data.len();
        for (i, v) in variables.iter().enumerate() {
            // The offset field is at the end of the var header: -4 bytes from after the header.
            // We need to find the exact position. Each var header:
            //   nc_string(name) + ndims(4) + dimids(4*ndims) + att_absent(8) + nc_type(4) + vsize(4) + begin(4)
            let name_padded = ((v.name.len() + 3) / 4) * 4;
            let header_len = 4 + name_padded + 4 + v.dim_ids.len() * 4 + 8 + 4 + 4 + 4;
            let offset_pos = var_header_starts[i] + header_len - 4;
            let offset_bytes = (data_offset as u32).to_be_bytes();
            data[offset_pos..offset_pos + 4].copy_from_slice(&offset_bytes);

            // Write data.
            for &val in v.values_f32 {
                data.extend_from_slice(&val.to_be_bytes());
            }
            // Pad to 4-byte boundary.
            while data.len() % 4 != 0 {
                data.push(0);
            }
            data_offset = data.len();
        }

        data
    }

    struct CdfVar<'a> {
        name: &'a str,
        dim_ids: &'a [usize],
        nc_type: u32, // 5 = NC_FLOAT
        values_f32: &'a [f32],
    }

    fn write_nc_string(data: &mut Vec<u8>, s: &str) {
        data.extend_from_slice(&(s.len() as u32).to_be_bytes());
        data.extend_from_slice(s.as_bytes());
        // Pad to 4-byte boundary.
        let pad = (4 - (s.len() % 4)) % 4;
        for _ in 0..pad {
            data.push(0);
        }
    }

    fn write_nc_string_value(data: &mut Vec<u8>, s: &str) {
        data.extend_from_slice(&(s.len() as u32).to_be_bytes());
        data.extend_from_slice(s.as_bytes());
        let pad = (4 - (s.len() % 4)) % 4;
        for _ in 0..pad {
            data.push(0);
        }
    }

    #[test]
    fn reads_2d_surface_from_cdf1() {
        let dir = temp_dir("netcdf-2d");

        // Build a 3x2 grid: dims x(3), y(2), one data var "temperature".
        let x_vals: Vec<f32> = vec![0.0, 1.0, 2.0];
        let y_vals: Vec<f32> = vec![0.0, 1.0];
        // temperature(y, x): 6 values.
        let temp_vals: Vec<f32> = vec![300.0, 310.0, 320.0, 305.0, 315.0, 325.0];

        let bytes = build_cdf1_bytes(
            &[("x", 3), ("y", 2)],
            &[],
            &[
                CdfVar {
                    name: "x",
                    dim_ids: &[0],
                    nc_type: 5,
                    values_f32: &x_vals,
                },
                CdfVar {
                    name: "y",
                    dim_ids: &[1],
                    nc_type: 5,
                    values_f32: &y_vals,
                },
                CdfVar {
                    name: "temperature",
                    dim_ids: &[1, 0], // (y, x)
                    nc_type: 5,
                    values_f32: &temp_vals,
                },
            ],
        );

        let path = dir.join("test.nc");
        std::fs::write(&path, &bytes).unwrap();

        let datasets = read(&path).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 6);
        // 3x2 grid -> 2*1=2 quads -> 4 triangles -> 12 indices.
        assert_eq!(ds.indices.len(), 12);
        assert!(ds.point_data.contains_key("temperature"));
        let t = ds.point_data.get("temperature").unwrap();
        assert_eq!(t.len(), 6);
        assert!((t[0] - 300.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_series_metadata() {
        let dir = temp_dir("netcdf-series");

        let time_vals: Vec<f32> = vec![0.0, 3600.0, 7200.0];
        let x_vals: Vec<f32> = vec![0.0, 1.0];
        // data(time, x) — 3 timesteps * 2 x values = 6 total.
        let data_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let bytes = build_cdf1_bytes(
            &[("time", 3), ("x", 2)],
            &[],
            &[
                CdfVar {
                    name: "time",
                    dim_ids: &[0],
                    nc_type: 5,
                    values_f32: &time_vals,
                },
                CdfVar {
                    name: "x",
                    dim_ids: &[1],
                    nc_type: 5,
                    values_f32: &x_vals,
                },
                CdfVar {
                    name: "velocity",
                    dim_ids: &[0, 1], // (time, x)
                    nc_type: 5,
                    values_f32: &data_vals,
                },
            ],
        );

        let path = dir.join("series.nc");
        std::fs::write(&path, &bytes).unwrap();

        let series = read_series(&path).unwrap();
        assert_eq!(series.timesteps.len(), 3);
        assert_eq!(series.timesteps[0].time, 0.0);
        assert!((series.timesteps[1].time - 3600.0).abs() < 0.01);
        assert!((series.timesteps[2].time - 7200.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_uniform_spacing() {
        assert!(is_uniform(&[0.0, 1.0, 2.0, 3.0]));
        assert!(is_uniform(&[0.0, 0.5, 1.0, 1.5]));
        assert!(!is_uniform(&[0.0, 1.0, 3.0, 6.0]));
        assert!(is_uniform(&[5.0]));
        assert!(is_uniform(&[]));
    }
}
