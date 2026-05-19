use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::IoError;
use crate::types::IoPointCloud;

/// Decode a CSV-like tabular file into a point cloud.
///
/// The first non-comment row is treated as a header. Columns named like `x`,
/// `y`, and `z` become positions; remaining numeric columns become named scalar
/// attributes.
pub fn point_cloud_from_path(path: &Path) -> Result<IoPointCloud, IoError> {
    let text = std::fs::read_to_string(path)?;
    parse_csv(&text)
}

fn parse_csv(text: &str) -> Result<IoPointCloud, IoError> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));

    let header_line = lines
        .next()
        .ok_or_else(|| IoError::Parse("csv: file is empty".into()))?;

    let delimiter = detect_delimiter(header_line);
    let headers: Vec<String> = split_line(header_line, delimiter)
        .into_iter()
        .map(|header| header.trim_matches('"').trim().to_string())
        .collect();
    if headers.is_empty() {
        return Err(IoError::Parse("csv: no columns found in header".into()));
    }

    let x_index = find_coord_col(
        &headers,
        &["x", "xc", "pos_x", "coord_x", "longitude", "lon"],
    );
    let y_index = find_coord_col(
        &headers,
        &["y", "yc", "pos_y", "coord_y", "latitude", "lat"],
    );
    let z_index = find_coord_col(
        &headers,
        &["z", "zc", "pos_z", "coord_z", "depth", "height"],
    );
    let (x_index, y_index, z_index) = match (x_index, y_index, z_index) {
        (Some(x), Some(y), z) => (x, y, z),
        _ => {
            let column_count = headers.len();
            (
                0,
                1.min(column_count.saturating_sub(1)),
                if column_count >= 3 { Some(2) } else { None },
            )
        }
    };

    let mut columns: Vec<Vec<f32>> = vec![Vec::new(); headers.len()];
    for line in lines {
        let fields = split_line(line, delimiter);
        for (column_index, field) in fields.iter().enumerate().take(headers.len()) {
            let value = field
                .trim()
                .trim_matches('"')
                .parse::<f32>()
                .unwrap_or(f32::NAN);
            columns[column_index].push(value);
        }
        for column in columns.iter_mut().skip(fields.len().min(headers.len())) {
            column.push(f32::NAN);
        }
    }

    let point_count = columns.first().map(Vec::len).unwrap_or(0);
    if point_count == 0 {
        return Err(IoError::Parse("csv: file contains no data rows".into()));
    }

    let mut positions = Vec::with_capacity(point_count);
    for row in 0..point_count {
        positions.push([
            columns[x_index][row],
            columns[y_index][row],
            z_index.map(|index| columns[index][row]).unwrap_or(0.0),
        ]);
    }

    let coord_columns: HashSet<usize> = {
        let mut set = HashSet::new();
        set.insert(x_index);
        set.insert(y_index);
        if let Some(index) = z_index {
            set.insert(index);
        }
        set
    };

    let mut scalar_attributes = HashMap::new();
    for (column_index, name) in headers.iter().enumerate() {
        if coord_columns.contains(&column_index) {
            continue;
        }
        if columns[column_index].iter().all(|value| value.is_nan()) {
            continue;
        }
        scalar_attributes.insert(name.clone(), columns[column_index].clone());
    }

    let scalars = if scalar_attributes.len() == 1 {
        scalar_attributes
            .values()
            .next()
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok(IoPointCloud {
        name: "CSV Point Cloud".to_string(),
        positions,
        colors: Vec::new(),
        scalars,
        scalar_attributes,
    })
}

fn detect_delimiter(line: &str) -> u8 {
    let comma = line.chars().filter(|&c| c == ',').count();
    let tab = line.chars().filter(|&c| c == '\t').count();
    let semi = line.chars().filter(|&c| c == ';').count();

    if tab >= comma && tab >= semi && tab > 0 {
        b'\t'
    } else if semi > comma && semi > 0 {
        b';'
    } else if comma > 0 {
        b','
    } else {
        b' '
    }
}

fn split_line(line: &str, delimiter: u8) -> Vec<&str> {
    if delimiter == b' ' {
        line.split_whitespace().collect()
    } else {
        line.split(delimiter as char).collect()
    }
}

fn find_coord_col(headers: &[String], candidates: &[&str]) -> Option<usize> {
    for (index, header) in headers.iter().enumerate() {
        let lower = header.to_ascii_lowercase();
        if candidates.iter().any(|candidate| *candidate == lower) {
            return Some(index);
        }
    }
    None
}
