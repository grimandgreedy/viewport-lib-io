use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use flate2::read::DeflateDecoder;

use crate::error::IoError;
use crate::types::{IoVolume, IoVolumeGeometry};

#[derive(Clone, Debug)]
struct NpyArray {
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// Decode a NumPy `.npy` or `.npz` file into a dense structured volume.
///
/// This entrypoint is volume-first: only rank-3 arrays are accepted.
pub fn volume_from_path(path: &Path) -> Result<IoVolume, IoError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    if extension == "npz" {
        let bytes = std::fs::read(path)?;
        let entries = parse_npz_entries(&bytes)?;
        if entries.is_empty() {
            return Err(IoError::Parse("numpy: NPZ archive is empty".into()));
        }

        let mut dims = None;
        let mut scalar_fields = HashMap::new();
        for (name, payload) in entries {
            let array = parse_npy(&payload)?;
            let array_dims = dims_from_shape(&array.shape)?;
            if let Some(existing_dims) = dims {
                if existing_dims != array_dims {
                    return Err(IoError::Parse(
                        "numpy: NPZ arrays must share the same 3D shape".into(),
                    ));
                }
            } else {
                dims = Some(array_dims);
            }

            let field_name = name.trim_end_matches(".npy").replace('/', ".");
            scalar_fields.insert(field_name, array.values);
        }

        return Ok(IoVolume {
            name: "NumPy Volume".to_string(),
            dims: dims.unwrap_or([0, 0, 0]),
            geometry: IoVolumeGeometry::Uniform {
                origin: [0.0, 0.0, 0.0],
                spacing: [1.0, 1.0, 1.0],
            },
            point_fields: scalar_fields,
            cell_fields: HashMap::new(),
        });
    }

    let bytes = std::fs::read(path)?;
    let array = parse_npy(&bytes)?;
    let dims = dims_from_shape(&array.shape)?;
    let field_name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("value")
        .to_string();
    let mut scalar_fields = HashMap::new();
    scalar_fields.insert(field_name, array.values);

    Ok(IoVolume {
        name: "NumPy Volume".to_string(),
        dims,
        geometry: IoVolumeGeometry::Uniform {
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        },
        point_fields: scalar_fields,
        cell_fields: HashMap::new(),
    })
}

fn dims_from_shape(shape: &[usize]) -> Result<[u32; 3], IoError> {
    match shape {
        [nz, ny, nx] => Ok([*nx as u32, *ny as u32, *nz as u32]),
        _ => Err(IoError::Parse(format!(
            "numpy: expected a rank-3 array, got rank {}",
            shape.len()
        ))),
    }
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray, IoError> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(IoError::Parse("numpy: invalid NPY header".into()));
    }
    let major = bytes[6];
    let header_len = match major {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        2 | 3 => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        _ => {
            return Err(IoError::Parse(format!(
                "numpy: unsupported NPY version {major}"
            )));
        }
    };
    let preamble = if major == 1 { 10 } else { 12 };
    let header_end = preamble + header_len;
    if bytes.len() < header_end {
        return Err(IoError::Parse("numpy: truncated NPY header".into()));
    }

    let header = std::str::from_utf8(&bytes[preamble..header_end])
        .map_err(|_| IoError::Parse("numpy: NPY header is not UTF-8".into()))?;
    let descr = parse_header_string(header, "'descr'")?;
    let fortran_order = parse_header_bool(header, "'fortran_order'")?;
    if fortran_order {
        return Err(IoError::Parse(
            "numpy: Fortran-ordered arrays are not supported".into(),
        ));
    }

    let shape = parse_header_shape(header)?;
    let count = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| IoError::Parse("numpy: shape overflow".into()))?;
    let values = decode_values(&bytes[header_end..], &descr, count)?;
    Ok(NpyArray { shape, values })
}

fn parse_header_string(header: &str, key: &str) -> Result<String, IoError> {
    let key_pos = header
        .find(key)
        .ok_or_else(|| IoError::Parse(format!("numpy: missing header key {key}")))?;
    let rest = &header[key_pos + key.len()..];
    let start = rest
        .find('\'')
        .ok_or_else(|| IoError::Parse(format!("numpy: malformed header key {key}")))?;
    let rest = &rest[start + 1..];
    let end = rest
        .find('\'')
        .ok_or_else(|| IoError::Parse(format!("numpy: malformed header key {key}")))?;
    Ok(rest[..end].to_string())
}

fn parse_header_bool(header: &str, key: &str) -> Result<bool, IoError> {
    let key_pos = header
        .find(key)
        .ok_or_else(|| IoError::Parse(format!("numpy: missing header key {key}")))?;
    let rest = &header[key_pos + key.len()..];
    if rest.contains("True") {
        Ok(true)
    } else if rest.contains("False") {
        Ok(false)
    } else {
        Err(IoError::Parse(format!(
            "numpy: malformed boolean value for {key}"
        )))
    }
}

fn parse_header_shape(header: &str) -> Result<Vec<usize>, IoError> {
    let key_pos = header
        .find("'shape'")
        .ok_or_else(|| IoError::Parse("numpy: missing NPY shape".into()))?;
    let rest = &header[key_pos..];
    let start = rest
        .find('(')
        .ok_or_else(|| IoError::Parse("numpy: malformed NPY shape".into()))?;
    let end = rest[start + 1..]
        .find(')')
        .ok_or_else(|| IoError::Parse("numpy: malformed NPY shape".into()))?
        + start
        + 1;
    let inner = &rest[start + 1..end];
    let dims: Vec<usize> = inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| IoError::Parse(format!("numpy: invalid shape component {part}")))
        })
        .collect::<Result<_, _>>()?;
    if dims.is_empty() {
        return Err(IoError::Parse("numpy: empty NPY shape".into()));
    }
    Ok(dims)
}

fn decode_values(data: &[u8], descr: &str, count: usize) -> Result<Vec<f32>, IoError> {
    let bytes_per = match descr {
        "<f4" | "|f4" => 4,
        "<f8" | "|f8" => 8,
        "<i4" | "|i4" => 4,
        "<i8" | "|i8" => 8,
        "<u4" | "|u4" => 4,
        "<u8" | "|u8" => 8,
        "<i2" | "|i2" => 2,
        "<u2" | "|u2" => 2,
        "|b1" | "|u1" | "|i1" => 1,
        _ => return Err(IoError::Parse(format!("numpy: unsupported dtype {descr}"))),
    };
    let expected = count
        .checked_mul(bytes_per)
        .ok_or_else(|| IoError::Parse("numpy: byte count overflow".into()))?;
    if data.len() < expected {
        return Err(IoError::Parse("numpy: truncated NPY payload".into()));
    }

    let mut values = Vec::with_capacity(count);
    match descr {
        "<f4" | "|f4" => {
            for chunk in data[..expected].chunks_exact(4) {
                values.push(f32::from_le_bytes(chunk.try_into().unwrap_or_default()));
            }
        }
        "<f8" | "|f8" => {
            for chunk in data[..expected].chunks_exact(8) {
                values.push(f64::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<i8" | "|i8" => {
            for chunk in data[..expected].chunks_exact(8) {
                values.push(i64::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<u8" | "|u8" => {
            for chunk in data[..expected].chunks_exact(8) {
                values.push(u64::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<i4" | "|i4" => {
            for chunk in data[..expected].chunks_exact(4) {
                values.push(i32::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<u4" | "|u4" => {
            for chunk in data[..expected].chunks_exact(4) {
                values.push(u32::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<i2" | "|i2" => {
            for chunk in data[..expected].chunks_exact(2) {
                values.push(i16::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "<u2" | "|u2" => {
            for chunk in data[..expected].chunks_exact(2) {
                values.push(u16::from_le_bytes(chunk.try_into().unwrap_or_default()) as f32);
            }
        }
        "|b1" => values.extend(
            data[..expected]
                .iter()
                .map(|byte| if *byte == 0 { 0.0 } else { 1.0 }),
        ),
        "|u1" => values.extend(data[..expected].iter().map(|byte| *byte as f32)),
        "|i1" => values.extend(data[..expected].iter().map(|byte| (*byte as i8) as f32)),
        _ => unreachable!(),
    }
    Ok(values)
}

fn parse_npz_entries(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>, IoError> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let signature =
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap_or_default());
        if signature != 0x0403_4b50 {
            break;
        }
        if offset + 30 > bytes.len() {
            return Err(IoError::Parse("numpy: truncated NPZ local header".into()));
        }

        let compression = u16::from_le_bytes(
            bytes[offset + 8..offset + 10]
                .try_into()
                .unwrap_or_default(),
        );
        let compressed_size = u32::from_le_bytes(
            bytes[offset + 18..offset + 22]
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        let uncompressed_size = u32::from_le_bytes(
            bytes[offset + 22..offset + 26]
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        let name_len = u16::from_le_bytes(
            bytes[offset + 26..offset + 28]
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        let extra_len = u16::from_le_bytes(
            bytes[offset + 28..offset + 30]
                .try_into()
                .unwrap_or_default(),
        ) as usize;
        let name_start = offset + 30;
        let name_end = name_start + name_len;
        let data_start = name_end + extra_len;
        let data_end = data_start + compressed_size;
        if data_end > bytes.len() {
            return Err(IoError::Parse("numpy: truncated NPZ entry".into()));
        }

        let name = std::str::from_utf8(&bytes[name_start..name_end])
            .map_err(|_| IoError::Parse("numpy: NPZ entry name is not UTF-8".into()))?
            .to_string();
        let payload = match compression {
            0 => bytes[data_start..data_end].to_vec(),
            8 => {
                let mut decoder = DeflateDecoder::new(Cursor::new(&bytes[data_start..data_end]));
                let mut buffer = Vec::with_capacity(uncompressed_size);
                decoder.read_to_end(&mut buffer).map_err(|error| {
                    IoError::Parse(format!("numpy: NPZ inflate failed: {error}"))
                })?;
                buffer
            }
            other => {
                return Err(IoError::Parse(format!(
                    "numpy: unsupported NPZ compression method {other}"
                )));
            }
        };
        out.push((name, payload));
        offset = data_end;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_io_testkit::synth;

    /// NumPy writes `(nz, ny, nx)`, the renderer wants `[nx, ny, nz]`. Reading
    /// this the wrong way round transposes the volume without any error.
    #[test]
    fn shape_is_reversed_into_render_order() {
        assert_eq!(dims_from_shape(&[4, 3, 2]).unwrap(), [2, 3, 4]);
        assert!(dims_from_shape(&[3, 2]).is_err(), "rank 2 is not a volume");
        assert!(dims_from_shape(&[2, 2, 2, 2]).is_err());
    }

    #[test]
    fn the_header_dictionary_is_read_by_key() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 4), }";
        assert_eq!(parse_header_string(header, "'descr'").unwrap(), "<f4");
        assert!(!parse_header_bool(header, "'fortran_order'").unwrap());
        assert_eq!(parse_header_shape(header).unwrap(), vec![2, 3, 4]);

        assert!(parse_header_string(header, "'missing'").is_err());
        assert!(parse_header_shape("{'descr': '<f4'}").is_err());
        assert!(parse_header_shape("{'shape': (), }").is_err());
    }

    /// A one-dimensional array's shape is written `(4,)` with a trailing comma,
    /// which is a tuple in Python and an empty component to a naive split.
    #[test]
    fn a_trailing_comma_in_the_shape_is_not_a_dimension() {
        assert_eq!(parse_header_shape("{'shape': (4,), }").unwrap(), vec![4]);
    }

    /// Every dtype the loader claims to read becomes the same value as an f32.
    /// A width read wrong shifts the whole array, not one sample.
    #[test]
    fn every_supported_dtype_decodes_to_the_same_value() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("<f4", 5.0f32.to_le_bytes().to_vec()),
            ("<f8", 5.0f64.to_le_bytes().to_vec()),
            ("<i4", 5i32.to_le_bytes().to_vec()),
            ("<u4", 5u32.to_le_bytes().to_vec()),
            ("<i8", 5i64.to_le_bytes().to_vec()),
            ("<u8", 5u64.to_le_bytes().to_vec()),
            ("<i2", 5i16.to_le_bytes().to_vec()),
            ("<u2", 5u16.to_le_bytes().to_vec()),
            ("|u1", vec![5]),
            ("|i1", vec![5]),
        ];
        for (descr, bytes) in cases {
            let values = decode_values(&bytes, descr, 1)
                .unwrap_or_else(|e| panic!("{descr} should decode: {e}"));
            assert_eq!(values, vec![5.0], "{descr}");
        }

        // A boolean array is not a number: anything non-zero reads as one.
        assert_eq!(decode_values(&[0, 7], "|b1", 2).unwrap(), vec![0.0, 1.0]);
        assert_eq!(decode_values(&[0xFF], "|i1", 1).unwrap(), vec![-1.0]);
        assert_eq!(decode_values(&[0xFF], "|u1", 1).unwrap(), vec![255.0]);
    }

    /// A payload shorter than the shape promises is a truncated file, reported
    /// rather than read past the end.
    #[test]
    fn a_truncated_payload_is_an_error() {
        assert!(decode_values(&[0, 0, 0], "<f4", 1).is_err());
        assert!(decode_values(&[], "not-a-dtype", 0).is_err());
    }

    /// Header versions 1 and 2 differ only in the width of the length field, so
    /// the same array written either way decodes identically.
    #[test]
    fn a_v1_header_decodes() {
        let bytes = synth::npy_f32(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&bytes[..6], b"\x93NUMPY");

        let array = parse_npy(&bytes).expect("decode npy");
        assert_eq!(array.shape, vec![2, 2, 2]);
        assert_eq!(array.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// Anything not opening with the magic, and any version beyond 3, is
    /// refused up front rather than mis-parsed.
    #[test]
    fn a_file_that_is_not_an_npy_is_refused() {
        assert!(parse_npy(b"not numpy at all").is_err());
        assert!(parse_npy(&[]).is_err());

        let mut bad_version = synth::npy_f32(&[1, 1, 1], &[1.0]);
        bad_version[6] = 9;
        assert!(parse_npy(&bad_version).is_err());
    }

    /// Fortran ordering transposes the array, so it is refused rather than
    /// loaded with its axes swapped.
    #[test]
    fn fortran_ordered_arrays_are_refused() {
        let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (1, 1, 1), }";
        assert!(parse_header_bool(header, "'fortran_order'").unwrap());
    }
}
