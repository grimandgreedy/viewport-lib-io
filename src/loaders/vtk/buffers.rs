//! Format plumbing: `IOBuffer` conversions and the cell array a `VertexNumbers`
//! unpacks to.

use vtkio::model::VertexNumbers;

use crate::error::IoError;

pub(super) fn iobuf_to_positions(buf: vtkio::model::IOBuffer) -> Result<Vec<[f32; 3]>, IoError> {
    let floats = buf.cast_into::<f32>().ok_or_else(|| {
        IoError::UnsupportedFormat("vtk: could not cast point coordinates to f32".into())
    })?;
    Ok(floats
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

pub(super) fn iobuf_to_f32_vec(buf: vtkio::model::IOBuffer) -> Result<Vec<f32>, IoError> {
    buf.cast_into::<f32>().ok_or_else(|| {
        IoError::UnsupportedFormat("vtk: could not cast coordinate array to f32".into())
    })
}

pub(super) fn collect_cells(vn: &VertexNumbers) -> Vec<Vec<u32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy and XML cell encodings describe the same cells different ways:
    /// legacy prefixes each cell with its length, XML carries a separate offset
    /// per cell. Both must unpack to the same thing.
    #[test]
    fn both_cell_encodings_unpack_the_same() {
        let legacy = VertexNumbers::Legacy {
            num_cells: 2,
            vertices: vec![3, 0, 1, 2, 4, 2, 3, 4, 5],
        };
        let xml = VertexNumbers::XML {
            connectivity: vec![0, 1, 2, 2, 3, 4, 5],
            offsets: vec![3, 7],
        };

        let expected = vec![vec![0, 1, 2], vec![2, 3, 4, 5]];
        assert_eq!(collect_cells(&legacy), expected);
        assert_eq!(collect_cells(&xml), expected);
    }

    /// A legacy array whose last cell claims more vertices than remain stops at
    /// the cells it can read, rather than panicking on the truncated tail.
    #[test]
    fn a_truncated_legacy_cell_array_stops_early() {
        let truncated = VertexNumbers::Legacy {
            num_cells: 2,
            vertices: vec![3, 0, 1, 2, 4, 3, 4],
        };
        assert_eq!(collect_cells(&truncated), vec![vec![0, 1, 2]]);
    }
}
