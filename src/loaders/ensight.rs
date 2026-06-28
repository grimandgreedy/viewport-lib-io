use std::collections::HashMap;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};

use super::common::Dataset;
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};
use crate::{error::IoError, types::IoDataSet};

/// Decode an EnSight Gold case file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

/// Read an EnSight Gold case file and return one [`Dataset`] per part.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    let case = parse_case_file(path)?;
    let base_dir = path.parent().unwrap_or(Path::new("."));

    let geo_path = resolve_wildcard(
        &case.geometry_file,
        base_dir,
        case.time_set_for_geo(),
        &case,
    )?;
    let parts = read_geometry(&geo_path)?;

    if parts.is_empty() {
        return Err(ReadError::Empty);
    }

    let mut datasets: Vec<Dataset> = parts.into_iter().map(|p| p.into_dataset()).collect();

    // Read variable files for the first timestep.
    read_variables_into(&case, base_dir, 0, &mut datasets)?;

    Ok(datasets)
}

/// Read time-series metadata from an EnSight case file.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let case = parse_case_file(path)?;

    if case.time_sets.is_empty() {
        return Ok(PvdSeries {
            timesteps: vec![TimestepEntry {
                time: 0.0,
                file: path.to_path_buf(),
                selector: Some("0".to_string()),
            }],
        });
    }

    // Use the first time set with values.
    let ts = &case.time_sets[0];
    let timesteps = ts
        .values
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
    let step: usize = selector
        .parse()
        .map_err(|_| ReadError::EnSight(format!("Invalid EnSight selector: {selector}")))?;

    let case = parse_case_file(path)?;
    let base_dir = path.parent().unwrap_or(Path::new("."));

    let geo_path = resolve_wildcard(
        &case.geometry_file,
        base_dir,
        case.time_set_for_geo(),
        &case,
    )?;
    let parts = read_geometry(&geo_path)?;

    if parts.is_empty() {
        return Err(ReadError::Empty);
    }

    let mut datasets: Vec<Dataset> = parts.into_iter().map(|p| p.into_dataset()).collect();

    read_variables_into(&case, base_dir, step, &mut datasets)?;

    Ok(datasets)
}

// ---------------------------------------------------------------------------
// Case file parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
#[allow(dead_code)]
struct CaseFile {
    geometry_file: String,
    geometry_time_set: Option<usize>,
    variables: Vec<CaseVariable>,
    time_sets: Vec<TimeSet>,
}

impl CaseFile {
    fn time_set_for_geo(&self) -> Option<usize> {
        self.geometry_time_set
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct CaseVariable {
    name: String,
    file_pattern: String,
    var_type: VarType,
    time_set: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum VarType {
    ScalarPerNode,
    ScalarPerElement,
    VectorPerNode,
    VectorPerElement,
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct TimeSet {
    id: usize,
    values: Vec<f32>,
}

fn parse_case_file(path: &Path) -> Result<CaseFile, ReadError> {
    let text = std::fs::read_to_string(path)?;
    let mut case = CaseFile::default();
    let mut current_section = String::new();
    let mut current_time_set: Option<TimeSet> = None;
    let mut expecting_time_values = 0usize;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Collect pending time values.
        if expecting_time_values > 0 {
            for token in trimmed.split_whitespace() {
                if let Ok(v) = token.parse::<f32>() {
                    if let Some(ts) = &mut current_time_set {
                        ts.values.push(v);
                        expecting_time_values = expecting_time_values.saturating_sub(1);
                    }
                }
            }
            if expecting_time_values == 0 {
                if let Some(ts) = current_time_set.take() {
                    case.time_sets.push(ts);
                }
            }
            continue;
        }

        // Section headers.
        if trimmed.ends_with(':') || matches!(trimmed, "FORMAT" | "GEOMETRY" | "VARIABLE" | "TIME")
        {
            // Flush any pending time set.
            if let Some(ts) = current_time_set.take() {
                case.time_sets.push(ts);
            }
            current_section = trimmed.trim_end_matches(':').to_uppercase();
            continue;
        }

        match current_section.as_str() {
            "GEOMETRY" => {
                if let Some(rest) = trimmed.strip_prefix("model:") {
                    parse_geometry_line(rest.trim(), &mut case);
                } else if trimmed.starts_with("model") {
                    let rest = trimmed.strip_prefix("model").unwrap_or("").trim();
                    if let Some(rest) = rest.strip_prefix(':') {
                        parse_geometry_line(rest.trim(), &mut case);
                    }
                }
            }
            "VARIABLE" => {
                parse_variable_line(trimmed, &mut case);
            }
            "TIME" => {
                if let Some(rest) = trimmed.strip_prefix("time set:") {
                    let id: usize = rest.trim().parse().unwrap_or(1);
                    current_time_set = Some(TimeSet {
                        id,
                        values: Vec::new(),
                    });
                } else if trimmed.starts_with("number of steps:") {
                    let n: usize = trimmed
                        .strip_prefix("number of steps:")
                        .unwrap_or("0")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                    expecting_time_values = n;
                } else if trimmed.starts_with("time values:") {
                    // Time values may follow on the same line or subsequent lines.
                    let rest = trimmed.strip_prefix("time values:").unwrap_or("").trim();
                    for token in rest.split_whitespace() {
                        if let Ok(v) = token.parse::<f32>() {
                            if let Some(ts) = &mut current_time_set {
                                ts.values.push(v);
                                expecting_time_values = expecting_time_values.saturating_sub(1);
                            }
                        }
                    }
                    if expecting_time_values == 0 {
                        if let Some(ts) = current_time_set.take() {
                            case.time_sets.push(ts);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Flush any remaining time set.
    if let Some(ts) = current_time_set {
        case.time_sets.push(ts);
    }

    if case.geometry_file.is_empty() {
        return Err(ReadError::EnSight(
            "Case file missing geometry model line".to_string(),
        ));
    }

    Ok(case)
}

fn parse_geometry_line(rest: &str, case: &mut CaseFile) {
    // Format: [time_set] [file_set] filename
    // or just: filename
    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.is_empty() {
        return;
    }

    // If first token is numeric, it's a time set reference.
    if tokens.len() >= 2 {
        if let Ok(ts) = tokens[0].parse::<usize>() {
            case.geometry_time_set = Some(ts);
            // Possibly a file set number follows, then the filename.
            if tokens.len() >= 3 {
                case.geometry_file = tokens[2..].join(" ");
            } else {
                case.geometry_file = tokens[1].to_string();
            }
            return;
        }
    }
    case.geometry_file = tokens.join(" ");
}

fn parse_variable_line(line: &str, case: &mut CaseFile) {
    // Format: <type> per <location>: [time_set] [file_set] <description> <filename>
    let (var_type, rest) = if let Some(rest) = line.strip_prefix("scalar per node:") {
        (VarType::ScalarPerNode, rest.trim())
    } else if let Some(rest) = line.strip_prefix("scalar per element:") {
        (VarType::ScalarPerElement, rest.trim())
    } else if let Some(rest) = line.strip_prefix("vector per node:") {
        (VarType::VectorPerNode, rest.trim())
    } else if let Some(rest) = line.strip_prefix("vector per element:") {
        (VarType::VectorPerElement, rest.trim())
    } else {
        return;
    };

    let tokens: Vec<&str> = rest.split_whitespace().collect();
    if tokens.len() < 2 {
        return;
    }

    // Parse optional time_set number, then description and filename.
    let (time_set, name_start) = if let Ok(ts) = tokens[0].parse::<usize>() {
        (Some(ts), 1)
    } else {
        (None, 0)
    };

    // Last token is the filename, everything between is the description/name.
    let file_pattern = tokens.last().unwrap().to_string();
    let name = tokens[name_start..tokens.len() - 1].join(" ");
    let name = if name.is_empty() {
        file_pattern
            .rsplit('/')
            .next()
            .unwrap_or(&file_pattern)
            .to_string()
    } else {
        name
    };

    case.variables.push(CaseVariable {
        name,
        file_pattern,
        var_type,
        time_set,
    });
}

// ---------------------------------------------------------------------------
// Wildcard file resolution
// ---------------------------------------------------------------------------

fn resolve_wildcard(
    pattern: &str,
    base_dir: &Path,
    time_set_id: Option<usize>,
    case: &CaseFile,
) -> Result<PathBuf, ReadError> {
    if !pattern.contains('*') {
        return Ok(base_dir.join(pattern));
    }

    // Replace wildcard with zero-padded step number.
    let star_count = pattern.chars().filter(|&c| c == '*').count();
    let step_str = format!("{:0>width$}", 0, width = star_count);
    let resolved = pattern.replacen(&"*".repeat(star_count), &step_str, 1);
    let path = base_dir.join(&resolved);
    if path.exists() {
        return Ok(path);
    }

    // Try step 1 as fallback.
    let step_str = format!("{:0>width$}", 1, width = star_count);
    let resolved = pattern.replacen(&"*".repeat(star_count), &step_str, 1);
    let path = base_dir.join(&resolved);
    if path.exists() {
        return Ok(path);
    }

    let _ = (time_set_id, case);
    Ok(base_dir.join(pattern.replace('*', "0")))
}

fn resolve_wildcard_at_step(pattern: &str, base_dir: &Path, step: usize) -> PathBuf {
    if !pattern.contains('*') {
        return base_dir.join(pattern);
    }

    let star_count = pattern.chars().filter(|&c| c == '*').count();
    let step_str = format!("{:0>width$}", step, width = star_count);
    let resolved = pattern.replacen(&"*".repeat(star_count), &step_str, 1);
    base_dir.join(&resolved)
}

// ---------------------------------------------------------------------------
// Geometry reading (EnSight Gold ASCII format)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct GeoPart {
    name: String,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl GeoPart {
    fn into_dataset(self) -> Dataset {
        let normals = compute_normals(&self.positions, &self.indices);
        Dataset {
            positions: self.positions,
            indices: self.indices,
            normals,
            point_data: HashMap::new(),
            cell_data: HashMap::new(),
            edge_data: HashMap::new(),
            sparse_volume: None,
            volume: None,
            volume_mesh: None,
        }
    }
}

fn read_geometry(path: &Path) -> Result<Vec<GeoPart>, ReadError> {
    let data = std::fs::read(path)?;

    // Detect binary vs ASCII by checking for the "C Binary" marker.
    if data.len() >= 80 && is_c_binary(&data) {
        read_geometry_binary(&data)
    } else {
        let text = String::from_utf8_lossy(&data);
        read_geometry_ascii(&text)
    }
}

fn is_c_binary(data: &[u8]) -> bool {
    let header = String::from_utf8_lossy(&data[..80.min(data.len())]);
    header.contains("C Binary")
}

fn is_probably_binary(data: &[u8]) -> bool {
    is_c_binary(data) || std::str::from_utf8(data).is_err()
}

fn read_geometry_ascii(text: &str) -> Result<Vec<GeoPart>, ReadError> {
    let mut lines = text.lines().peekable();
    let mut parts = Vec::new();

    // Skip description lines (first two lines).
    lines.next();
    lines.next();

    // Optional "node id" and "element id" lines.
    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.starts_with("node id") || trimmed.starts_with("element id") {
            lines.next();
        } else if trimmed.starts_with("extents") {
            lines.next();
            // Skip 3 lines of extent values.
            for _ in 0..3 {
                lines.next();
            }
        } else {
            break;
        }
    }

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == "part" {
            let part = read_ascii_part(&mut lines)?;
            if !part.positions.is_empty() && !part.indices.is_empty() {
                parts.push(part);
            }
        }
    }

    Ok(parts)
}

fn read_ascii_part(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Result<GeoPart, ReadError> {
    // Part number.
    let _part_num = lines.next().unwrap_or("").trim().to_string();
    // Part description/name.
    let name = lines.next().unwrap_or("").trim().to_string();

    let mut positions = Vec::new();
    let mut indices = Vec::new();

    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed == "coordinates" {
            lines.next();
            positions = read_ascii_coordinates(lines)?;
        } else if trimmed == "block" {
            lines.next();
            let dims = read_ascii_block_dims(lines)?;
            positions = read_ascii_block_coordinates(lines, dims)?;
            indices = structured_surface_indices(dims[0], dims[1], dims[2]);
        } else if trimmed == "part" {
            // Next part — don't consume this line.
            break;
        } else if is_element_type(trimmed) {
            let elem_type = trimmed.to_string();
            lines.next();
            read_ascii_elements(lines, &elem_type, &mut indices)?;
        } else {
            lines.next();
        }
    }

    Ok(GeoPart {
        name,
        positions,
        indices,
    })
}

fn read_ascii_coordinates(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Result<Vec<[f32; 3]>, ReadError> {
    let n: usize = lines
        .next()
        .unwrap_or("0")
        .trim()
        .parse()
        .map_err(|_| ReadError::EnSight("Invalid coordinate count".to_string()))?;

    // EnSight Gold ASCII: all X values, then all Y, then all Z.
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);

    read_n_floats(lines, n, &mut xs)?;
    read_n_floats(lines, n, &mut ys)?;
    read_n_floats(lines, n, &mut zs)?;

    let positions = xs
        .iter()
        .zip(ys.iter())
        .zip(zs.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    Ok(positions)
}

fn read_ascii_block_dims(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Result<[usize; 3], ReadError> {
    let line = lines.next().unwrap_or("");
    let dims: Vec<usize> = line
        .split_whitespace()
        .filter_map(|t| t.parse::<usize>().ok())
        .collect();
    if dims.len() < 3 {
        return Err(ReadError::EnSight(
            "Invalid structured block dimensions".to_string(),
        ));
    }
    Ok([dims[0], dims[1], dims[2]])
}

fn read_ascii_block_coordinates(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    dims: [usize; 3],
) -> Result<Vec<[f32; 3]>, ReadError> {
    let num_points = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ReadError::EnSight("Structured block point count overflow".to_string()))?;
    let mut xs = Vec::with_capacity(num_points);
    let mut ys = Vec::with_capacity(num_points);
    let mut zs = Vec::with_capacity(num_points);
    read_n_floats(lines, num_points, &mut xs)?;
    read_n_floats(lines, num_points, &mut ys)?;
    read_n_floats(lines, num_points, &mut zs)?;
    Ok(zip_structured_positions(xs, ys, zs))
}

fn read_n_floats(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    n: usize,
    out: &mut Vec<f32>,
) -> Result<(), ReadError> {
    let mut count = 0;
    while count < n {
        let line = lines.next().unwrap_or("");
        for token in line.split_whitespace() {
            if let Ok(v) = token.parse::<f32>() {
                out.push(v);
                count += 1;
                if count >= n {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn read_ascii_elements(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    elem_type: &str,
    indices: &mut Vec<u32>,
) -> Result<(), ReadError> {
    let n: usize = lines
        .next()
        .unwrap_or("0")
        .trim()
        .parse()
        .map_err(|_| ReadError::EnSight("Invalid element count".to_string()))?;

    let nodes_per = nodes_per_element(elem_type);
    if nodes_per == 0 {
        // Skip unknown element type lines.
        for _ in 0..n {
            lines.next();
        }
        return Ok(());
    }

    for _ in 0..n {
        let line = lines.next().unwrap_or("");
        let node_ids: Vec<u32> = line
            .split_whitespace()
            .filter_map(|t| t.parse::<u32>().ok())
            .map(|id| id.saturating_sub(1)) // 1-based to 0-based
            .collect();

        if node_ids.len() >= nodes_per {
            triangulate_ensight(elem_type, &node_ids[..nodes_per], indices);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Geometry reading (EnSight Gold C-Binary format)
// ---------------------------------------------------------------------------

fn read_geometry_binary(data: &[u8]) -> Result<Vec<GeoPart>, ReadError> {
    let mut cursor = std::io::Cursor::new(data);
    let mut parts = Vec::new();

    // Skip two 80-char description lines.
    skip_record(&mut cursor)?;
    skip_record(&mut cursor)?;

    // Read "node id" / "element id" lines.
    loop {
        let record = match read_record_string(&mut cursor) {
            Ok(record) => record,
            Err(_) => break,
        };
        let trimmed = record.trim();
        if trimmed.starts_with("node id") || trimmed.starts_with("element id") {
            continue;
        } else if trimmed.starts_with("extents") {
            // Skip 6 floats (3 pairs of min/max).
            let mut buf = [0u8; 24];
            cursor.read_exact(&mut buf).map_err(io_err)?;
            continue;
        } else if trimmed == "part" {
            let part = read_binary_part(&mut cursor)?;
            if !part.positions.is_empty() && !part.indices.is_empty() {
                parts.push(part);
            }
        } else {
            break;
        }
    }

    // Continue reading parts.
    loop {
        match read_record_string(&mut cursor) {
            Ok(record) => {
                if record.trim() == "part" {
                    if let Ok(part) = read_binary_part(&mut cursor) {
                        if !part.positions.is_empty() && !part.indices.is_empty() {
                            parts.push(part);
                        }
                    }
                }
            }
            Err(_) => break,
        }
    }

    Ok(parts)
}

fn read_binary_part(cursor: &mut std::io::Cursor<&[u8]>) -> Result<GeoPart, ReadError> {
    // Part number (int32).
    let _part_num = read_i32(cursor)?;
    // Description (80 chars).
    let name = read_record_string(cursor)?.trim().to_string();

    let mut positions = Vec::new();
    let mut indices = Vec::new();

    loop {
        let pos_before = cursor.position();
        match read_record_string(cursor) {
            Ok(record) => {
                let trimmed = record.trim();
                if trimmed == "coordinates" {
                    positions = read_binary_coordinates(cursor)?;
                } else if trimmed == "block" {
                    let dims = read_binary_block_dims(cursor)?;
                    positions = read_binary_block_coordinates(cursor, dims)?;
                    indices = structured_surface_indices(dims[0], dims[1], dims[2]);
                } else if trimmed == "part" {
                    // Rewind — this is the next part.
                    cursor.set_position(pos_before);
                    break;
                } else if is_element_type(trimmed) {
                    read_binary_elements(cursor, trimmed, &mut indices)?;
                } else {
                    // Unknown record — might be end of geometry.
                    cursor.set_position(pos_before);
                    break;
                }
            }
            Err(_) => break,
        }
    }

    Ok(GeoPart {
        name,
        positions,
        indices,
    })
}

fn read_binary_coordinates(
    cursor: &mut std::io::Cursor<&[u8]>,
) -> Result<Vec<[f32; 3]>, ReadError> {
    let n = read_i32(cursor)? as usize;
    let mut xs = vec![0.0f32; n];
    let mut ys = vec![0.0f32; n];
    let mut zs = vec![0.0f32; n];

    read_f32_array(cursor, &mut xs)?;
    read_f32_array(cursor, &mut ys)?;
    read_f32_array(cursor, &mut zs)?;

    let positions = xs
        .iter()
        .zip(ys.iter())
        .zip(zs.iter())
        .map(|((&x, &y), &z)| [x, y, z])
        .collect();

    Ok(positions)
}

fn read_binary_block_dims(cursor: &mut std::io::Cursor<&[u8]>) -> Result<[usize; 3], ReadError> {
    let ni = read_i32(cursor)?;
    let nj = read_i32(cursor)?;
    let nk = read_i32(cursor)?;
    if ni <= 0 || nj <= 0 || nk <= 0 {
        return Err(ReadError::EnSight(
            "Invalid structured block dimensions".to_string(),
        ));
    }
    Ok([ni as usize, nj as usize, nk as usize])
}

fn read_binary_block_coordinates(
    cursor: &mut std::io::Cursor<&[u8]>,
    dims: [usize; 3],
) -> Result<Vec<[f32; 3]>, ReadError> {
    let num_points = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ReadError::EnSight("Structured block point count overflow".to_string()))?;
    let mut xs = vec![0.0f32; num_points];
    let mut ys = vec![0.0f32; num_points];
    let mut zs = vec![0.0f32; num_points];
    read_f32_array(cursor, &mut xs)?;
    read_f32_array(cursor, &mut ys)?;
    read_f32_array(cursor, &mut zs)?;
    Ok(zip_structured_positions(xs, ys, zs))
}

fn read_binary_elements(
    cursor: &mut std::io::Cursor<&[u8]>,
    elem_type: &str,
    indices: &mut Vec<u32>,
) -> Result<(), ReadError> {
    let n = read_i32(cursor)? as usize;
    let nodes_per = nodes_per_element(elem_type);
    if nodes_per == 0 {
        // Skip connectivity data.
        let skip = n * nodes_per * 4;
        let pos = cursor.position() + skip as u64;
        cursor.set_position(pos);
        return Ok(());
    }

    let total = n * nodes_per;
    let mut connectivity = vec![0i32; total];
    read_i32_array(cursor, &mut connectivity)?;

    for elem in connectivity.chunks_exact(nodes_per) {
        let local: Vec<u32> = elem
            .iter()
            .map(|&id| (id - 1).max(0) as u32) // 1-based to 0-based
            .collect();
        triangulate_ensight(elem_type, &local, indices);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

fn read_record_string(cursor: &mut std::io::Cursor<&[u8]>) -> Result<String, ReadError> {
    let mut buf = [0u8; 80];
    cursor.read_exact(&mut buf).map_err(io_err)?;
    Ok(String::from_utf8_lossy(&buf)
        .trim_end_matches('\0')
        .to_string())
}

fn skip_record(cursor: &mut std::io::Cursor<&[u8]>) -> Result<(), ReadError> {
    let mut buf = [0u8; 80];
    cursor.read_exact(&mut buf).map_err(io_err)?;
    Ok(())
}

fn read_i32(cursor: &mut std::io::Cursor<&[u8]>) -> Result<i32, ReadError> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf).map_err(io_err)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32_array(cursor: &mut std::io::Cursor<&[u8]>, out: &mut [f32]) -> Result<(), ReadError> {
    let mut buf = [0u8; 4];
    for val in out.iter_mut() {
        cursor.read_exact(&mut buf).map_err(io_err)?;
        *val = f32::from_le_bytes(buf);
    }
    Ok(())
}

fn read_i32_array(cursor: &mut std::io::Cursor<&[u8]>, out: &mut [i32]) -> Result<(), ReadError> {
    let mut buf = [0u8; 4];
    for val in out.iter_mut() {
        cursor.read_exact(&mut buf).map_err(io_err)?;
        *val = i32::from_le_bytes(buf);
    }
    Ok(())
}

fn io_err(e: std::io::Error) -> ReadError {
    ReadError::EnSight(format!("Binary read error: {e}"))
}

// ---------------------------------------------------------------------------
// Variable reading
// ---------------------------------------------------------------------------

fn read_variables_into(
    case: &CaseFile,
    base_dir: &Path,
    step: usize,
    datasets: &mut [Dataset],
) -> Result<(), ReadError> {
    for var in &case.variables {
        let var_path = resolve_wildcard_at_step(&var.file_pattern, base_dir, step);
        if !var_path.exists() {
            continue;
        }

        let is_per_node = matches!(
            var.var_type,
            VarType::ScalarPerNode | VarType::VectorPerNode
        );
        let is_vector = matches!(
            var.var_type,
            VarType::VectorPerNode | VarType::VectorPerElement
        );
        let expected_counts: Vec<usize> = if is_per_node {
            datasets
                .iter()
                .map(|dataset| dataset.positions.len())
                .collect()
        } else {
            vec![0; datasets.len()]
        };

        if let Ok(part_values) = read_variable_file(&var_path, &expected_counts, is_vector) {
            for (i, values) in part_values.into_iter().enumerate() {
                if i >= datasets.len() || values.is_empty() {
                    continue;
                }
                if is_vector {
                    // Split vector into magnitude scalar for now.
                    let n = values.len() / 3;
                    let magnitudes: Vec<f32> = (0..n)
                        .map(|j| {
                            let vx = values[j];
                            let vy = values[n + j];
                            let vz = values[2 * n + j];
                            (vx * vx + vy * vy + vz * vz).sqrt()
                        })
                        .collect();

                    if is_per_node {
                        datasets[i]
                            .point_data
                            .insert(format!("{}:magnitude", var.name), magnitudes);
                    } else {
                        datasets[i]
                            .cell_data
                            .insert(format!("{}:magnitude", var.name), magnitudes);
                    }
                } else if is_per_node {
                    datasets[i].point_data.insert(var.name.clone(), values);
                } else {
                    datasets[i].cell_data.insert(var.name.clone(), values);
                }
            }
        }
    }
    Ok(())
}

fn read_variable_file(
    path: &Path,
    expected_part_counts: &[usize],
    is_vector: bool,
) -> Result<Vec<Vec<f32>>, ReadError> {
    let data = std::fs::read(path)?;

    if data.len() >= 80 && is_probably_binary(&data) {
        read_variable_binary(&data, expected_part_counts, is_vector)
    } else {
        let text = String::from_utf8_lossy(&data);
        read_variable_ascii(&text, expected_part_counts, is_vector)
    }
}

fn read_variable_ascii(
    text: &str,
    expected_part_counts: &[usize],
    is_vector: bool,
) -> Result<Vec<Vec<f32>>, ReadError> {
    let mut lines = text.lines().peekable();
    let mut part_values = Vec::new();

    // Skip description line.
    lines.next();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed == "part" {
            let part_idx = part_values.len();
            let _part_num = lines.next(); // part number
            // Next line is block type ("coordinates" for per-node, element type for per-element).
            let block_line = lines.next().unwrap_or("");
            let block_trimmed = block_line.trim();
            // If this line is "coordinates" or an element type, read count from next line.
            // Otherwise this line IS the count.
            let n: usize = if block_trimmed == "block" {
                expected_part_counts.get(part_idx).copied().unwrap_or(0)
            } else if block_trimmed == "coordinates" || is_element_type(block_trimmed) {
                lines.next().unwrap_or("0").trim().parse().unwrap_or(0)
            } else {
                block_trimmed.parse().unwrap_or(0)
            };

            let components = if is_vector { 3 } else { 1 };
            let total = n * components;
            let mut values = Vec::with_capacity(total);
            read_n_floats(&mut lines, total, &mut values)?;
            part_values.push(values);
        }
    }

    Ok(part_values)
}

fn read_variable_binary(
    data: &[u8],
    expected_part_counts: &[usize],
    is_vector: bool,
) -> Result<Vec<Vec<f32>>, ReadError> {
    let mut cursor = std::io::Cursor::new(data);
    let mut part_values = Vec::new();

    // Skip description (80 chars).
    skip_record(&mut cursor)?;

    loop {
        match read_record_string(&mut cursor) {
            Ok(record) => {
                if record.trim() == "part" {
                    let part_idx = part_values.len();
                    let _part_num = read_i32(&mut cursor)?;
                    match read_record_string(&mut cursor) {
                        Ok(record) => {
                            let trimmed = record.trim();
                            if trimmed == "coordinates" || is_element_type(trimmed) {
                                let n = read_i32(&mut cursor)? as usize;
                                let components = if is_vector { 3 } else { 1 };
                                let total = n * components;
                                let mut values = vec![0.0f32; total];
                                read_f32_array(&mut cursor, &mut values)?;
                                part_values.push(values);
                            } else if trimmed == "block" {
                                let n = expected_part_counts.get(part_idx).copied().unwrap_or(0);
                                let components = if is_vector { 3 } else { 1 };
                                let total = n * components;
                                let mut values = vec![0.0f32; total];
                                read_f32_array(&mut cursor, &mut values)?;
                                part_values.push(values);
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
            Err(_) => break,
        }
    }

    Ok(part_values)
}

// ---------------------------------------------------------------------------
// Element type helpers
// ---------------------------------------------------------------------------

fn is_element_type(s: &str) -> bool {
    matches!(
        s,
        "point"
            | "bar2"
            | "bar3"
            | "tria3"
            | "tria6"
            | "quad4"
            | "quad8"
            | "tetra4"
            | "tetra10"
            | "pyramid5"
            | "pyramid13"
            | "penta6"
            | "penta15"
            | "hexa8"
            | "hexa20"
            | "nsided"
            | "nfaced"
    )
}

fn zip_structured_positions(xs: Vec<f32>, ys: Vec<f32>, zs: Vec<f32>) -> Vec<[f32; 3]> {
    xs.into_iter()
        .zip(ys)
        .zip(zs)
        .map(|((x, y), z)| [x, y, z])
        .collect()
}

/// Generate the boundary surface triangles for a structured point grid.
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

fn nodes_per_element(elem_type: &str) -> usize {
    match elem_type {
        "point" => 1,
        "bar2" => 2,
        "bar3" => 3,
        "tria3" => 3,
        "tria6" => 6,
        "quad4" => 4,
        "quad8" => 8,
        "tetra4" => 4,
        "tetra10" => 10,
        "pyramid5" => 5,
        "pyramid13" => 13,
        "penta6" => 6,
        "penta15" => 15,
        "hexa8" => 8,
        "hexa20" => 20,
        _ => 0,
    }
}

fn triangulate_ensight(elem_type: &str, nodes: &[u32], out: &mut Vec<u32>) {
    let v = |i: usize| nodes[i];

    match elem_type {
        "tria3" | "tria6" => {
            out.extend_from_slice(&[v(0), v(1), v(2)]);
        }
        "quad4" | "quad8" => {
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
        }
        "tetra4" | "tetra10" => {
            out.extend_from_slice(&[v(0), v(2), v(1)]);
            out.extend_from_slice(&[v(0), v(1), v(3)]);
            out.extend_from_slice(&[v(1), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(3), v(2)]);
        }
        "pyramid5" | "pyramid13" => {
            out.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
            out.extend_from_slice(&[v(0), v(1), v(4)]);
            out.extend_from_slice(&[v(1), v(2), v(4)]);
            out.extend_from_slice(&[v(2), v(3), v(4)]);
            out.extend_from_slice(&[v(3), v(0), v(4)]);
        }
        "penta6" | "penta15" => {
            out.extend_from_slice(&[v(0), v(1), v(2)]);
            out.extend_from_slice(&[v(3), v(5), v(4)]);
            out.extend_from_slice(&[v(0), v(3), v(4), v(0), v(4), v(1)]);
            out.extend_from_slice(&[v(1), v(4), v(5), v(1), v(5), v(2)]);
            out.extend_from_slice(&[v(2), v(5), v(3), v(2), v(3), v(0)]);
        }
        "hexa8" | "hexa20" => {
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
    use std::io::Write;
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

    #[test]
    fn reads_ascii_case_with_triangle_part() {
        let dir = temp_dir("ensight");

        // Write case file.
        let case_content = "\
FORMAT
type: ensight gold
GEOMETRY
model: test.geo
VARIABLE
scalar per node: 1 pressure test.prs****
TIME
time set: 1
number of steps: 2
time values: 0.0 1.0
";
        std::fs::write(dir.join("test.case"), case_content).unwrap();

        // Write geometry file (ASCII EnSight Gold).
        let geo_content = "\
EnSight Geometry File
Description line 2
node id off
element id off
part
     1
triangle-part
coordinates
     3
 0.00000e+00
 1.00000e+00
 5.00000e-01
 0.00000e+00
 0.00000e+00
 1.00000e+00
 0.00000e+00
 0.00000e+00
 0.00000e+00
tria3
     1
     1     2     3
";
        std::fs::write(dir.join("test.geo"), geo_content).unwrap();

        // Write variable file for step 0.
        let prs_content = "\
Pressure
part
     1
coordinates
     3
 100.0
 200.0
 150.0
";
        std::fs::write(dir.join("test.prs0000"), prs_content).unwrap();

        // Read via case file.
        let datasets = read(&dir.join("test.case")).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 3);
        assert_eq!(ds.indices.len(), 3);
        assert_eq!(ds.indices, vec![0, 1, 2]);

        // Check variable was loaded.
        assert!(ds.point_data.contains_key("pressure"));
        let p = ds.point_data.get("pressure").unwrap();
        assert_eq!(p.len(), 3);
        assert!((p[0] - 100.0).abs() < 0.01);
        assert!((p[1] - 200.0).abs() < 0.01);

        // Test series reading.
        let series = read_series(&dir.join("test.case")).unwrap();
        assert_eq!(series.timesteps.len(), 2);
        assert_eq!(series.timesteps[0].time, 0.0);
        assert_eq!(series.timesteps[1].time, 1.0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_c_binary_record(file: &mut std::fs::File, text: &str) {
        let mut buf = [b' '; 80];
        let bytes = text.as_bytes();
        let len = bytes.len().min(80);
        buf[..len].copy_from_slice(&bytes[..len]);
        file.write_all(&buf).unwrap();
    }

    #[test]
    fn reads_c_binary_block_case() {
        let dir = temp_dir("ensight-block");

        let case_content = "\
FORMAT
type: ensight gold
GEOMETRY
model: block.geo
VARIABLE
scalar per node: pressure pressure.scl
";
        std::fs::write(dir.join("block.case"), case_content).unwrap();

        let mut geo = std::fs::File::create(dir.join("block.geo")).unwrap();
        write_c_binary_record(&mut geo, "C Binary");
        write_c_binary_record(&mut geo, "Structured block");
        write_c_binary_record(&mut geo, "node id off");
        write_c_binary_record(&mut geo, "element id off");
        write_c_binary_record(&mut geo, "part");
        geo.write_all(&1i32.to_le_bytes()).unwrap();
        write_c_binary_record(&mut geo, "curvilinear-surface");
        write_c_binary_record(&mut geo, "block");
        geo.write_all(&2i32.to_le_bytes()).unwrap();
        geo.write_all(&2i32.to_le_bytes()).unwrap();
        geo.write_all(&1i32.to_le_bytes()).unwrap();
        for value in [0.0f32, 1.0, 0.0, 1.0] {
            geo.write_all(&value.to_le_bytes()).unwrap();
        }
        for value in [0.0f32, 0.0, 1.0, 1.0] {
            geo.write_all(&value.to_le_bytes()).unwrap();
        }
        for value in [0.0f32, 0.0, 0.0, 0.0] {
            geo.write_all(&value.to_le_bytes()).unwrap();
        }

        let mut scalar = std::fs::File::create(dir.join("pressure.scl")).unwrap();
        write_c_binary_record(&mut scalar, "pressure");
        write_c_binary_record(&mut scalar, "part");
        scalar.write_all(&1i32.to_le_bytes()).unwrap();
        write_c_binary_record(&mut scalar, "block");
        for value in [10.0f32, 20.0, 30.0, 40.0] {
            scalar.write_all(&value.to_le_bytes()).unwrap();
        }

        let datasets = read(&dir.join("block.case")).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 4);
        assert_eq!(ds.indices, vec![0, 1, 3, 0, 3, 2]);
        assert_eq!(
            ds.point_data.get("pressure").unwrap(),
            &vec![10.0, 20.0, 30.0, 40.0]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_case_file_sections() {
        let dir = temp_dir("ensight-parse");

        let case_content = "\
FORMAT
type: ensight gold
GEOMETRY
model: mesh.geo
VARIABLE
scalar per node: temperature temp.scl
scalar per element: density dens.scl
vector per node: velocity vel.vec
";
        std::fs::write(dir.join("test.case"), case_content).unwrap();

        let case = parse_case_file(&dir.join("test.case")).unwrap();
        assert_eq!(case.geometry_file, "mesh.geo");
        assert_eq!(case.variables.len(), 3);
        assert_eq!(case.variables[0].name, "temperature");
        assert!(matches!(case.variables[0].var_type, VarType::ScalarPerNode));
        assert_eq!(case.variables[1].name, "density");
        assert!(matches!(
            case.variables[1].var_type,
            VarType::ScalarPerElement
        ));
        assert_eq!(case.variables[2].name, "velocity");
        assert!(matches!(case.variables[2].var_type, VarType::VectorPerNode));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
