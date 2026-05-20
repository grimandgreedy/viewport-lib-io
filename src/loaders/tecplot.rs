use std::collections::HashMap;
use std::path::Path;

use crate::{error::IoError, types::IoDataSet};
use super::common::Dataset;
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};

/// Decode a Tecplot file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

/// Read a Tecplot file and return one [`Dataset`] per zone.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    let text = std::fs::read_to_string(path)?;
    let file = parse_tecplot(&text)?;

    if file.zones.is_empty() {
        return Err(ReadError::Empty);
    }

    let datasets = file
        .zones
        .into_iter()
        .filter_map(|z| zone_to_dataset(z, &file.variables).ok())
        .filter(|ds| !ds.positions.is_empty())
        .collect::<Vec<_>>();

    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }

    Ok(datasets)
}

/// Read time-series metadata from a Tecplot file.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let text = std::fs::read_to_string(path)?;
    let file = parse_tecplot(&text)?;

    // Collect zones that have SOLUTIONTIME set.
    let mut timesteps: Vec<TimestepEntry> = Vec::new();
    let mut seen_times = std::collections::HashSet::new();

    for zone in &file.zones {
        if let Some(t) = zone.solution_time {
            let key = t.to_bits();
            if seen_times.insert(key) {
                timesteps.push(TimestepEntry {
                    time: t,
                    file: path.to_path_buf(),
                    selector: Some(format!("{t}")),
                });
            }
        }
    }

    if timesteps.is_empty() {
        timesteps.push(TimestepEntry {
            time: 0.0,
            file: path.to_path_buf(),
            selector: None,
        });
    }

    Ok(PvdSeries { timesteps })
}

/// Read zones matching a specific solution time.
pub fn read_selected(path: &Path, selector: &str) -> Result<Vec<Dataset>, ReadError> {
    let target_time: f64 = selector
        .parse()
        .map_err(|_| ReadError::Tecplot(format!("Invalid Tecplot selector: {selector}")))?;

    let text = std::fs::read_to_string(path)?;
    let file = parse_tecplot(&text)?;

    let datasets: Vec<Dataset> = file
        .zones
        .into_iter()
        .filter(|z| match z.solution_time {
            Some(t) => (t - target_time).abs() < 1e-12,
            None => true,
        })
        .filter_map(|z| zone_to_dataset(z, &file.variables).ok())
        .filter(|ds| !ds.positions.is_empty())
        .collect();

    if datasets.is_empty() {
        return Err(ReadError::Empty);
    }

    Ok(datasets)
}

// ---------------------------------------------------------------------------
// Tecplot file structure
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct TecplotFile {
    variables: Vec<String>,
    zones: Vec<TecplotZone>,
}

#[derive(Debug, Default)]
struct TecplotZone {
    name: String,
    /// Number of nodes (N for FE, or I*J*K for ordered).
    num_nodes: usize,
    /// Number of elements (E for FE zones).
    num_elements: usize,
    /// Structured dimensions.
    i_dim: usize,
    j_dim: usize,
    k_dim: usize,
    /// Data format: POINT (FEPOINT), BLOCK, or default.
    format: DataFormat,
    /// Finite-element type (TRIANGLE, QUADRILATERAL, TETRAHEDRON, BRICK).
    element_type: Option<ElementType>,
    /// SOLUTIONTIME for time-series.
    solution_time: Option<f64>,
    /// All variable data in zone order; data[var_index][node_index].
    data: Vec<Vec<f32>>,
    /// Element connectivity (0-based). Each element is a contiguous slice.
    connectivity: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
enum DataFormat {
    #[default]
    Point,
    Block,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ElementType {
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Brick,
}

impl ElementType {
    fn nodes_per(&self) -> usize {
        match self {
            ElementType::Triangle => 3,
            ElementType::Quadrilateral => 4,
            ElementType::Tetrahedron => 4,
            ElementType::Brick => 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn parse_tecplot(text: &str) -> Result<TecplotFile, ReadError> {
    let mut file = TecplotFile::default();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.peek() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            lines.next();
            continue;
        }

        let upper = trimmed.to_uppercase();

        if upper.starts_with("TITLE") {
            lines.next();
        } else if upper.starts_with("FILETYPE") {
            lines.next();
        } else if upper.starts_with("VARIABLES") {
            file.variables = parse_variables_line(trimmed, &mut lines);
        } else if upper.starts_with("ZONE") {
            let zone = parse_zone(trimmed, &mut lines, file.variables.len())?;
            file.zones.push(zone);
        } else if upper.starts_with("TEXT")
            || upper.starts_with("GEOMETRY")
            || upper.starts_with("CUSTOMLABELS")
        {
            // Skip auxiliary records.
            lines.next();
        } else {
            // Unknown line — skip.
            lines.next();
        }
    }

    Ok(file)
}

fn parse_variables_line(
    first_line: &str,
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
) -> Vec<String> {
    // Consume the first line.
    lines.next();

    // Collect the full variables declaration (may span continuation lines).
    let mut full = strip_key_value(first_line, "VARIABLES");

    // Continuation lines: lines that start with a quote or comma and don't look like a keyword.
    while let Some(next) = lines.peek() {
        let t = next.trim();
        if t.starts_with('"') || t.starts_with(',') {
            full.push(' ');
            full.push_str(t);
            lines.next();
        } else {
            break;
        }
    }

    parse_variable_names(&full)
}

fn strip_key_value(line: &str, key: &str) -> String {
    let upper = line.to_uppercase();
    let rest = if let Some(pos) = upper.find(key) {
        &line[pos + key.len()..]
    } else {
        line
    };
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('=').unwrap_or(rest);
    rest.trim().to_string()
}

fn parse_variable_names(s: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut chars = s.chars().peekable();

    loop {
        // Skip whitespace and commas.
        while let Some(&c) = chars.peek() {
            if c == ' ' || c == ',' || c == '\t' {
                chars.next();
            } else {
                break;
            }
        }

        if chars.peek().is_none() {
            break;
        }

        if chars.peek() == Some(&'"') {
            // Quoted name.
            chars.next(); // consume opening quote
            let mut name = String::new();
            for c in chars.by_ref() {
                if c == '"' {
                    break;
                }
                name.push(c);
            }
            names.push(name);
        } else {
            // Unquoted name — until comma or whitespace.
            let mut name = String::new();
            while let Some(&c) = chars.peek() {
                if c == ',' || c == ' ' || c == '\t' {
                    break;
                }
                name.push(c);
                chars.next();
            }
            if !name.is_empty() {
                names.push(name);
            }
        }
    }

    names
}

fn parse_zone(
    first_line: &str,
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    num_vars: usize,
) -> Result<TecplotZone, ReadError> {
    lines.next(); // consume the ZONE line

    // Collect the full zone header (may span continuation lines with commas).
    let header_rest = strip_key_value(first_line, "ZONE");
    let mut header = header_rest;

    // Continuation: next lines that look like key=value pairs, not numeric data.
    while let Some(next) = lines.peek() {
        let t = next.trim();
        if t.is_empty() {
            lines.next();
            continue;
        }
        // If the line starts with a letter or comma, it's likely a continuation header.
        // If it starts with a digit, sign, or dot, it's data.
        let first_char = t.chars().next().unwrap_or(' ');
        if first_char.is_ascii_alphabetic() || first_char == ',' {
            // But not if it looks like a new section keyword.
            let upper = t.to_uppercase();
            if upper.starts_with("ZONE")
                || upper.starts_with("TITLE")
                || upper.starts_with("VARIABLES")
                || upper.starts_with("TEXT")
                || upper.starts_with("GEOMETRY")
            {
                break;
            }
            header.push_str(", ");
            header.push_str(t);
            lines.next();
        } else {
            break;
        }
    }

    let mut zone = TecplotZone::default();
    parse_zone_header(&header, &mut zone)?;

    // Determine node count for structured zones.
    if zone.element_type.is_none() && zone.num_nodes == 0 {
        let i = if zone.i_dim > 0 { zone.i_dim } else { 1 };
        let j = if zone.j_dim > 0 { zone.j_dim } else { 1 };
        let k = if zone.k_dim > 0 { zone.k_dim } else { 1 };
        zone.num_nodes = i * j * k;
    }

    if zone.num_nodes == 0 || num_vars == 0 {
        return Ok(zone);
    }

    // Read data values.
    match zone.format {
        DataFormat::Point => read_point_data(lines, &mut zone, num_vars)?,
        DataFormat::Block => read_block_data(lines, &mut zone, num_vars)?,
    }

    // Read connectivity for FE zones.
    if let Some(et) = zone.element_type {
        read_connectivity(lines, &mut zone, et)?;
    }

    Ok(zone)
}

fn parse_zone_header(header: &str, zone: &mut TecplotZone) -> Result<(), ReadError> {
    // Tokenize: split on commas first, then on '='.
    for part in header.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((key, value)) = part.split_once('=') {
            let key = key.trim().to_uppercase();
            let value = value.trim().trim_matches('"');

            match key.as_str() {
                "T" => zone.name = value.to_string(),
                "I" => zone.i_dim = value.parse().unwrap_or(0),
                "J" => zone.j_dim = value.parse().unwrap_or(0),
                "K" => zone.k_dim = value.parse().unwrap_or(0),
                "N" | "NODES" => zone.num_nodes = value.parse().unwrap_or(0),
                "E" | "ELEMENTS" => zone.num_elements = value.parse().unwrap_or(0),
                "F" | "DATAPACKING" => {
                    let v = value.to_uppercase();
                    zone.format = if v == "BLOCK" {
                        DataFormat::Block
                    } else {
                        DataFormat::Point
                    };
                }
                "ET" | "ZONETYPE" => {
                    zone.element_type = parse_element_type(value);
                }
                "SOLUTIONTIME" => {
                    zone.solution_time = value.parse().ok();
                }
                "STRANDID" => { /* acknowledged but not used for filtering */ }
                _ => {}
            }
        }
    }

    Ok(())
}

fn parse_element_type(s: &str) -> Option<ElementType> {
    match s.to_uppercase().as_str() {
        "TRIANGLE" | "FETRIANGLE" => Some(ElementType::Triangle),
        "QUADRILATERAL" | "FEQUADRILATERAL" => Some(ElementType::Quadrilateral),
        "TETRAHEDRON" | "FETETRAHEDRON" => Some(ElementType::Tetrahedron),
        "BRICK" | "FEBRICK" => Some(ElementType::Brick),
        _ => None,
    }
}

fn read_point_data(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    zone: &mut TecplotZone,
    num_vars: usize,
) -> Result<(), ReadError> {
    zone.data = vec![Vec::with_capacity(zone.num_nodes); num_vars];

    let mut nodes_read = 0;
    while nodes_read < zone.num_nodes {
        let line = match lines.peek() {
            Some(l) => {
                let t = l.trim();
                if t.is_empty() {
                    lines.next();
                    continue;
                }
                // Stop if we hit a new section.
                let upper = t.to_uppercase();
                if upper.starts_with("ZONE")
                    || upper.starts_with("TITLE")
                    || upper.starts_with("VARIABLES")
                    || upper.starts_with("TEXT")
                {
                    break;
                }
                lines.next();
                t.to_string()
            }
            None => break,
        };

        let values: Vec<f32> = line
            .split_whitespace()
            .filter_map(|tok| tok.parse::<f32>().ok())
            .collect();

        if values.len() >= num_vars {
            for (vi, &val) in values.iter().enumerate().take(num_vars) {
                zone.data[vi].push(val);
            }
            nodes_read += 1;
        } else if !values.is_empty() {
            // Partial line — push what we have and continue collecting.
            for (vi, &val) in values.iter().enumerate() {
                if vi < num_vars {
                    zone.data[vi].push(val);
                }
            }
            nodes_read += 1;
        }
    }

    Ok(())
}

fn read_block_data(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    zone: &mut TecplotZone,
    num_vars: usize,
) -> Result<(), ReadError> {
    zone.data = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        let mut values = Vec::with_capacity(zone.num_nodes);
        while values.len() < zone.num_nodes {
            match lines.peek() {
                Some(l) => {
                    let t = l.trim();
                    if t.is_empty() {
                        lines.next();
                        continue;
                    }
                    let upper = t.to_uppercase();
                    if upper.starts_with("ZONE")
                        || upper.starts_with("TITLE")
                        || upper.starts_with("VARIABLES")
                    {
                        break;
                    }
                    lines.next();
                    for tok in t.split_whitespace() {
                        if let Ok(v) = tok.parse::<f32>() {
                            values.push(v);
                        }
                    }
                }
                None => break,
            }
        }
        zone.data.push(values);
    }

    Ok(())
}

fn read_connectivity(
    lines: &mut std::iter::Peekable<std::str::Lines<'_>>,
    zone: &mut TecplotZone,
    et: ElementType,
) -> Result<(), ReadError> {
    let npe = et.nodes_per();
    let total = zone.num_elements * npe;
    zone.connectivity = Vec::with_capacity(total);

    let mut elems_read = 0;
    while elems_read < zone.num_elements {
        match lines.peek() {
            Some(l) => {
                let t = l.trim();
                if t.is_empty() {
                    lines.next();
                    continue;
                }
                let upper = t.to_uppercase();
                if upper.starts_with("ZONE")
                    || upper.starts_with("TITLE")
                    || upper.starts_with("VARIABLES")
                {
                    break;
                }
                lines.next();
                let ids: Vec<u32> = t
                    .split_whitespace()
                    .filter_map(|tok| tok.parse::<u32>().ok())
                    .map(|id| id.saturating_sub(1)) // 1-based to 0-based
                    .collect();

                if ids.len() >= npe {
                    zone.connectivity.extend_from_slice(&ids[..npe]);
                    elems_read += 1;
                }
            }
            None => break,
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Zone to Dataset conversion
// ---------------------------------------------------------------------------

fn zone_to_dataset(zone: TecplotZone, variables: &[String]) -> Result<Dataset, ReadError> {
    if zone.data.is_empty() || zone.num_nodes == 0 {
        return Err(ReadError::Empty);
    }

    // Find X, Y, Z variable indices.
    let (xi, yi, zi) = find_xyz_indices(variables)?;

    let xs = &zone.data[xi];
    let ys = &zone.data[yi];
    let zs = if let Some(zi) = zi {
        zone.data[zi].as_slice()
    } else {
        &[] // 2D data
    };

    let n = xs.len().min(ys.len());
    let mut positions = Vec::with_capacity(n);
    for i in 0..n {
        let z = if i < zs.len() { zs[i] } else { 0.0 };
        positions.push([xs[i], ys[i], z]);
    }

    // Build triangle indices.
    let indices = if let Some(et) = zone.element_type {
        triangulate_fe(&zone.connectivity, et)
    } else {
        // Structured zone — triangulate the IJ surface.
        triangulate_structured(zone.i_dim, zone.j_dim, zone.k_dim)
    };

    let normals = compute_normals(&positions, &indices);

    // Collect scalar variables (everything except X, Y, Z).
    let mut point_data = HashMap::new();
    for (vi, var_name) in variables.iter().enumerate() {
        if vi == xi || vi == yi || zi.map_or(false, |z| vi == z) {
            continue;
        }
        if vi < zone.data.len() && !zone.data[vi].is_empty() {
            point_data.insert(var_name.clone(), zone.data[vi].clone());
        }
    }

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

fn find_xyz_indices(variables: &[String]) -> Result<(usize, usize, Option<usize>), ReadError> {
    let mut xi = None;
    let mut yi = None;
    let mut zi = None;

    for (i, name) in variables.iter().enumerate() {
        let upper = name.to_uppercase();
        if xi.is_none()
            && (upper == "X"
                || upper == "XC"
                || upper.contains("X-COORD")
                || upper.contains("X COORD"))
        {
            xi = Some(i);
        } else if yi.is_none()
            && (upper == "Y"
                || upper == "YC"
                || upper.contains("Y-COORD")
                || upper.contains("Y COORD"))
        {
            yi = Some(i);
        } else if zi.is_none()
            && (upper == "Z"
                || upper == "ZC"
                || upper.contains("Z-COORD")
                || upper.contains("Z COORD"))
        {
            zi = Some(i);
        }
    }

    // Fallback: first three variables are X, Y, Z.
    let xi = xi.unwrap_or(0);
    let yi = yi.unwrap_or(1.min(variables.len().saturating_sub(1)));
    let zi_val = if variables.len() >= 3 {
        zi.or(Some(2))
    } else {
        zi
    };

    Ok((xi, yi, zi_val))
}

fn triangulate_fe(connectivity: &[u32], et: ElementType) -> Vec<u32> {
    let npe = et.nodes_per();
    let mut indices = Vec::new();

    for elem in connectivity.chunks_exact(npe) {
        let v = |i: usize| elem[i];
        match et {
            ElementType::Triangle => {
                indices.extend_from_slice(&[v(0), v(1), v(2)]);
            }
            ElementType::Quadrilateral => {
                indices.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
            }
            ElementType::Tetrahedron => {
                indices.extend_from_slice(&[v(0), v(2), v(1)]);
                indices.extend_from_slice(&[v(0), v(1), v(3)]);
                indices.extend_from_slice(&[v(1), v(2), v(3)]);
                indices.extend_from_slice(&[v(0), v(3), v(2)]);
            }
            ElementType::Brick => {
                indices.extend_from_slice(&[v(0), v(3), v(2), v(0), v(2), v(1)]); // bottom
                indices.extend_from_slice(&[v(4), v(5), v(6), v(4), v(6), v(7)]); // top
                indices.extend_from_slice(&[v(0), v(1), v(5), v(0), v(5), v(4)]); // front
                indices.extend_from_slice(&[v(2), v(3), v(7), v(2), v(7), v(6)]); // back
                indices.extend_from_slice(&[v(0), v(4), v(7), v(0), v(7), v(3)]); // left
                indices.extend_from_slice(&[v(1), v(2), v(6), v(1), v(6), v(5)]); // right
            }
        }
    }

    indices
}

fn triangulate_structured(i_dim: usize, j_dim: usize, _k_dim: usize) -> Vec<u32> {
    let mut indices = Vec::new();

    if i_dim < 2 || j_dim < 2 {
        return indices;
    }

    // Triangulate the IJ surface (first K-plane if 3D).
    for j in 0..j_dim - 1 {
        for i in 0..i_dim - 1 {
            let v00 = (j * i_dim + i) as u32;
            let v10 = (j * i_dim + (i + 1)) as u32;
            let v01 = ((j + 1) * i_dim + i) as u32;
            let v11 = ((j + 1) * i_dim + (i + 1)) as u32;

            indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        if tri[0] as usize >= positions.len()
            || tri[1] as usize >= positions.len()
            || tri[2] as usize >= positions.len()
        {
            continue;
        }
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

    #[test]
    fn reads_fepoint_triangle_zone() {
        let dir = temp_dir("tecplot-fe");

        let content = r#"TITLE = "Triangle test"
VARIABLES = "X", "Y", "Z", "Pressure"
ZONE T="Zone 1", N=3, E=1, F=FEPOINT, ET=TRIANGLE
0.0 0.0 0.0 100.0
1.0 0.0 0.0 200.0
0.5 1.0 0.0 150.0
1 2 3
"#;
        let path = dir.join("test.dat");
        std::fs::write(&path, content).unwrap();

        let datasets = read(&path).unwrap();
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

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_block_structured_zone() {
        let dir = temp_dir("tecplot-block");

        let content = r#"TITLE = "Structured test"
VARIABLES = "X", "Y", "Z", "Temperature"
ZONE T="Grid", I=3, J=2, K=1, F=BLOCK
0.0 1.0 2.0 0.0 1.0 2.0
0.0 0.0 0.0 1.0 1.0 1.0
0.0 0.0 0.0 0.0 0.0 0.0
300.0 310.0 320.0 305.0 315.0 325.0
"#;
        let path = dir.join("test.dat");
        std::fs::write(&path, content).unwrap();

        let datasets = read(&path).unwrap();
        assert_eq!(datasets.len(), 1);
        let ds = &datasets[0];
        assert_eq!(ds.positions.len(), 6);
        // 3x2 grid => 2*1 = 2 quads => 4 triangles => 12 indices.
        assert_eq!(ds.indices.len(), 12);
        assert!(ds.point_data.contains_key("Temperature"));
        let t = ds.point_data.get("Temperature").unwrap();
        assert_eq!(t.len(), 6);
        assert!((t[0] - 300.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_time_series() {
        let dir = temp_dir("tecplot-time");

        let content = r#"TITLE = "Time series"
VARIABLES = "X", "Y", "Z", "Vel"
ZONE T="Step 0", N=3, E=1, F=FEPOINT, ET=TRIANGLE, SOLUTIONTIME=0.0
0.0 0.0 0.0 1.0
1.0 0.0 0.0 2.0
0.5 1.0 0.0 1.5
1 2 3
ZONE T="Step 1", N=3, E=1, F=FEPOINT, ET=TRIANGLE, SOLUTIONTIME=0.5
0.0 0.0 0.0 3.0
1.0 0.0 0.0 4.0
0.5 1.0 0.0 3.5
1 2 3
"#;
        let path = dir.join("test.dat");
        std::fs::write(&path, content).unwrap();

        let series = read_series(&path).unwrap();
        assert_eq!(series.timesteps.len(), 2);
        assert_eq!(series.timesteps[0].time, 0.0);
        assert_eq!(series.timesteps[1].time, 0.5);

        // Read specific timestep.
        let ds = read_selected(&path, "0.5").unwrap();
        assert_eq!(ds.len(), 1);
        let vel = ds[0].point_data.get("Vel").unwrap();
        assert!((vel[0] - 3.0).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reads_quad_fe_zone() {
        let dir = temp_dir("tecplot-quad");

        let content = r#"TITLE = "Quad test"
VARIABLES = "X", "Y", "Z"
ZONE T="Quad", N=4, E=1, F=FEPOINT, ET=QUADRILATERAL
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
1 2 3 4
"#;
        let path = dir.join("test.dat");
        std::fs::write(&path, content).unwrap();

        let datasets = read(&path).unwrap();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].positions.len(), 4);
        // 1 quad => 2 triangles => 6 indices.
        assert_eq!(datasets[0].indices.len(), 6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_variable_names_quoted_and_unquoted() {
        let names = parse_variable_names(r#""X", "Y", "Z", "Pressure (Pa)""#);
        assert_eq!(names, vec!["X", "Y", "Z", "Pressure (Pa)"]);

        let names2 = parse_variable_names("X Y Z P");
        assert_eq!(names2, vec!["X", "Y", "Z", "P"]);
    }
}
