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
        point_fields: scalar_fields,
        cell_fields: HashMap::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn field(dtype: &str, components: usize, size: usize) -> RawField {
        RawField {
            dtype: dtype.into(),
            name: "v".into(),
            num_components: components,
            offset: 0,
            size,
        }
    }

    fn manifest(dims: [usize; 3], fields: Vec<RawField>, byte_order: Option<&str>) -> RawManifest {
        RawManifest {
            byte_order: byte_order.map(str::to_owned),
            dims,
            fields,
            origin: None,
            spacing: None,
            raw_file: None,
        }
    }

    /// The manifest may name its payload, and does not have to: without
    /// `raw_file` the payload sits beside the manifest under the same stem.
    #[test]
    fn the_payload_path_comes_from_the_manifest_or_its_own_name() {
        let path = Path::new("/data/scan.json");
        let mut m = manifest([1, 1, 1], Vec::new(), None);
        assert_eq!(
            resolve_raw_path(path, &m),
            PathBuf::from("/data/scan.raw"),
            "no raw_file: same stem, .raw extension"
        );

        m.raw_file = Some("volume.bin".into());
        assert_eq!(
            resolve_raw_path(path, &m),
            PathBuf::from("/data/volume.bin"),
            "raw_file resolves against the manifest's directory"
        );
    }

    /// Every dtype the manifest may name reads back the same value, so a width
    /// or sign read wrong shows up here rather than as a skewed volume.
    #[test]
    fn every_dtype_reads_the_same_value() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("f32", 5.0f32.to_le_bytes().to_vec()),
            ("f64", 5.0f64.to_le_bytes().to_vec()),
            ("u8", vec![5]),
            ("i8", vec![5]),
            ("u16", 5u16.to_le_bytes().to_vec()),
            ("i16", 5i16.to_le_bytes().to_vec()),
            ("u32", 5u32.to_le_bytes().to_vec()),
            ("i32", 5i32.to_le_bytes().to_vec()),
        ];
        for (dtype, bytes) in cases {
            let spec = field(dtype, 1, bytes.len());
            let values = decode_field(&spec, &bytes, 1, true)
                .unwrap_or_else(|e| panic!("{dtype} should decode: {e}"));
            assert_eq!(values, vec![5.0], "{dtype}");
        }

        assert_eq!(
            decode_field(&field("i8", 1, 1), &[0xFF], 1, true).unwrap(),
            vec![-1.0]
        );
        assert_eq!(
            decode_field(&field("u8", 1, 1), &[0xFF], 1, true).unwrap(),
            vec![255.0]
        );
    }

    /// `byte_order` decides how multi-byte samples are read, and anything other
    /// than an explicit big-endian spelling means little-endian.
    #[test]
    fn byte_order_defaults_to_little_endian() {
        let value = 258u16; // 0x0102: distinguishable either way round
        let le = value.to_le_bytes().to_vec();
        let be = value.to_be_bytes().to_vec();

        assert_eq!(
            decode_field(&field("u16", 1, 2), &le, 1, true).unwrap(),
            vec![258.0]
        );
        assert_eq!(
            decode_field(&field("u16", 1, 2), &be, 1, false).unwrap(),
            vec![258.0]
        );

        for spelling in [None, Some("little"), Some("le"), Some("anything else")] {
            let m = manifest([1, 1, 1], vec![field("u16", 1, 2)], spelling);
            let volume = volume_from_manifest(m, &le).expect("decode");
            assert_eq!(volume.point_fields["v"], vec![258.0], "{spelling:?}");
        }
        for spelling in ["big", "be", "big_endian"] {
            let m = manifest([1, 1, 1], vec![field("u16", 1, 2)], Some(spelling));
            let volume = volume_from_manifest(m, &be).expect("decode");
            assert_eq!(volume.point_fields["v"], vec![258.0], "{spelling}");
        }
    }

    /// A multi-component field is split into one named scalar per component
    /// plus its magnitude, because the neutral volume carries scalars only.
    #[test]
    fn a_vector_field_becomes_components_and_a_magnitude() {
        // One point, one 3-vector (3, 4, 0): magnitude 5.
        let mut bytes = Vec::new();
        for value in [3.0f32, 4.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let m = manifest([1, 1, 1], vec![field("f32", 3, bytes.len())], None);

        let volume = volume_from_manifest(m, &bytes).expect("decode");
        assert_eq!(volume.point_fields["v_x"], vec![3.0]);
        assert_eq!(volume.point_fields["v_y"], vec![4.0]);
        assert_eq!(volume.point_fields["v_z"], vec![0.0]);
        assert_eq!(volume.point_fields["v:magnitude"], vec![5.0]);
        assert!(
            !volume.point_fields.contains_key("v"),
            "the undecomposed field is not emitted as well"
        );
    }

    /// Components past the third fall back to `w`, so a four-wide field still
    /// names all of them distinctly.
    #[test]
    fn component_names_run_x_y_z_then_w() {
        assert_eq!([0, 1, 2, 3].map(component_suffix), ["x", "y", "z", "w"]);
    }

    /// A manifest that disagrees with its own arithmetic is refused. Reading it
    /// anyway would silently shift every field after the bad one.
    #[test]
    fn a_manifest_that_lies_about_a_field_is_refused() {
        let bytes = vec![0u8; 64];

        let wrong_size = decode_field(&field("f32", 1, 99), &bytes, 1, true);
        assert!(format!("{}", wrong_size.unwrap_err()).contains("size mismatch"));

        let unknown = decode_field(&field("f16", 1, 2), &bytes, 1, true);
        assert!(format!("{}", unknown.unwrap_err()).contains("unsupported dtype"));

        let mut past_end = field("f32", 1, 4);
        past_end.offset = 62;
        assert!(
            format!("{}", decode_field(&past_end, &bytes, 1, true).unwrap_err())
                .contains("exceeds raw file size")
        );
    }

    /// A zero dimension describes no volume at all.
    #[test]
    fn a_zero_dimension_is_refused() {
        let m = manifest([2, 0, 2], Vec::new(), None);
        assert!(volume_from_manifest(m, &[]).is_err());
    }

    /// Origin and spacing are optional, and default to the unit grid at the
    /// origin rather than to zero spacing.
    #[test]
    fn a_manifest_without_a_grid_gets_the_unit_grid() {
        let m = manifest([2, 3, 4], Vec::new(), None);
        let volume = volume_from_manifest(m, &[]).expect("decode");

        assert_eq!(volume.dims, [2, 3, 4]);
        match volume.geometry {
            IoVolumeGeometry::Uniform { origin, spacing } => {
                assert_eq!(origin, [0.0, 0.0, 0.0]);
                assert_eq!(spacing, [1.0, 1.0, 1.0]);
            }
            other => panic!("expected a uniform grid, got {other:?}"),
        }
    }
}
