//! VTK-family decoding: the legacy `.vtk` format and the XML `.vtu` / `.vtp`
//! variants, which `vtkio` reads through the same entry point.
//!
//! A VTK file can hold several pieces of different kinds, so the decode returns
//! one dataset per piece and a consumer picks the representations it wants.

use std::path::Path;

use crate::error::IoError;
use crate::types::IoDataSet;

mod attributes;
mod buffers;
mod dataset;
mod topology;

pub mod vtp;
pub mod vtu;

/// Decode a VTK-family file into one scientific dataset per piece.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    #[cfg(feature = "vtk")]
    {
        dataset::datasets_from_path(path)
    }

    #[cfg(not(feature = "vtk"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "vtk",
            context: "VTK dataset decoding",
        })
    }
}
