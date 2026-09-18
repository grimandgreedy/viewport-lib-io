//! XML PolyData (`.vtp`) decoding, forwarded to the shared VTK decode.

use std::path::Path;

use crate::error::IoError;
use crate::types::IoDataSet;

/// Decode a VTP file into one scientific dataset per piece.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    super::datasets_from_path(path)
}
