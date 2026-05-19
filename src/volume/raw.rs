use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::IoError;
use crate::types::{IoVolume, IoVolumeGeometry};

#[derive(Debug, Deserialize)]
struct RawManifest {
    byte_order: Option<String>,
    dims: [usize; 3],
    fields: Vec<RawField>,
    origin: Option<[f32; 3]>,
    spacing: Option<[f32; 3]>,
    raw_file: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawField {
    dtype: String,
    name: String,
    num_components: usize,
    offset: usize,
    size: usize,
}

/// Decode a JSON raw-volume manifest into a dense structured volume.
pub fn volume_from_path(path: &Path) -> Result<IoVolume, IoError> {
    let manifest: RawManifest = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|error| IoError::Parse(format!("raw: failed to parse manifest: {error}")))?;
    let raw_path = resolve_raw_path(path, &manifest);
    let bytes = std::fs::read(raw_path)?;
    volume_from_manifest(manifest, &bytes)
}

fn resolve_raw_path(manifest_path: &Path, manifest: &RawManifest) -> PathBuf {
    if let Some(raw_file) = &manifest.raw_file {
        return manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(raw_file);
    }
    manifest_path.with_extension("raw")
}

fn volume_from_manifest(manifest: RawManifest, bytes: &[u8]) -> Result<IoVolume, IoError> {
    let [nx, ny, nz] = manifest.dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(IoError::Parse("raw: manifest dims must be positive".into()));
    }
    let point_count = nx
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nz))
        .ok_or_else(|| IoError::Parse("raw: point count overflow".into()))?;

    let little_endian = !matches!(
        manifest.byte_order.as_deref(),
        Some("big") | Some("be") | Some("big_endian")
    );

    let mut scalar_fields = HashMap::new();
    for field in manifest.fields {
        let values = decode_field(&field, bytes, point_count, little_endian)?;
        if field.num_components <= 1 {
            scalar_fields.insert(field.name, values);
        } else {
            for component in 0..field.num_components {
                let component_values = (0..point_count)
                    .map(|index| values[index * field.num_components + component])
                    .collect();
                scalar_fields.insert(
                    format!("{}_{}", field.name, component_suffix(component)),
                    component_values,
                );
            }
            let magnitudes = (0..point_count)
                .map(|index| {
                    let base = index * field.num_components;
                    values[base..base + field.num_components]
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        .sqrt()
                })
                .collect();
            scalar_fields.insert(format!("{}:magnitude", field.name), magnitudes);
        }
    }

    Ok(IoVolume {
        name: "Raw Volume".to_string(),
        dims: [nx as u32, ny as u32, nz as u32],
        geometry: IoVolumeGeometry::Uniform {
            origin: manifest.origin.unwrap_or([0.0, 0.0, 0.0]),
            spacing: manifest.spacing.unwrap_or([1.0, 1.0, 1.0]),
        },
        scalar_fields,
    })
}

fn decode_field(
    field: &RawField,
    bytes: &[u8],
    point_count: usize,
    little_endian: bool,
) -> Result<Vec<f32>, IoError> {
    let total_values = point_count
        .checked_mul(field.num_components)
        .ok_or_else(|| IoError::Parse(format!("raw: field {} value count overflow", field.name)))?;
    let bytes_per = match field.dtype.as_str() {
        "f32" => 4,
        "f64" => 8,
        "u8" | "i8" => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" => 4,
        _ => {
            return Err(IoError::Parse(format!(
                "raw: unsupported dtype {} for field {}",
                field.dtype, field.name
            )));
        }
    };
    let expected_size = total_values
        .checked_mul(bytes_per)
        .ok_or_else(|| IoError::Parse(format!("raw: field {} size overflow", field.name)))?;
    if field.size != expected_size {
        return Err(IoError::Parse(format!(
            "raw: field {} size mismatch: manifest says {}, expected {}",
            field.name, field.size, expected_size
        )));
    }
    let end = field
        .offset
        .checked_add(field.size)
        .ok_or_else(|| IoError::Parse(format!("raw: field {} offset overflow", field.name)))?;
    if end > bytes.len() {
        return Err(IoError::Parse(format!(
            "raw: field {} exceeds raw file size",
            field.name
        )));
    }

    let slice = &bytes[field.offset..end];
    let mut out = Vec::with_capacity(total_values);
    match field.dtype.as_str() {
        "f32" => {
            for chunk in slice.chunks_exact(4) {
                out.push(if little_endian {
                    f32::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    f32::from_be_bytes(chunk.try_into().unwrap_or_default())
                });
            }
        }
        "f64" => {
            for chunk in slice.chunks_exact(8) {
                out.push(if little_endian {
                    f64::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    f64::from_be_bytes(chunk.try_into().unwrap_or_default())
                } as f32);
            }
        }
        "u8" => out.extend(slice.iter().map(|byte| *byte as f32)),
        "i8" => out.extend(slice.iter().map(|byte| (*byte as i8) as f32)),
        "u16" => {
            for chunk in slice.chunks_exact(2) {
                out.push(if little_endian {
                    u16::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    u16::from_be_bytes(chunk.try_into().unwrap_or_default())
                } as f32);
            }
        }
        "i16" => {
            for chunk in slice.chunks_exact(2) {
                out.push(if little_endian {
                    i16::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    i16::from_be_bytes(chunk.try_into().unwrap_or_default())
                } as f32);
            }
        }
        "u32" => {
            for chunk in slice.chunks_exact(4) {
                out.push(if little_endian {
                    u32::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    u32::from_be_bytes(chunk.try_into().unwrap_or_default())
                } as f32);
            }
        }
        "i32" => {
            for chunk in slice.chunks_exact(4) {
                out.push(if little_endian {
                    i32::from_le_bytes(chunk.try_into().unwrap_or_default())
                } else {
                    i32::from_be_bytes(chunk.try_into().unwrap_or_default())
                } as f32);
            }
        }
        _ => unreachable!(),
    }
    Ok(out)
}

fn component_suffix(index: usize) -> &'static str {
    match index {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "w",
    }
}
