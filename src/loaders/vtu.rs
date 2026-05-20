use std::path::Path;

use crate::error::IoError;
use crate::types::IoDataSet;

/// Decode a VTU file into one scientific dataset per piece.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    super::vtk::datasets_from_path(path)
}
