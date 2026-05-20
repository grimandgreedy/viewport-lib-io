use crate::error::IoError;

use vtkio::Error as VtkError;
use vtkio::model::Error as VtkModelError;

#[derive(thiserror::Error, Debug)]
pub enum ReadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("VTK parse error: {0}")]
    Parse(#[from] VtkError),
    #[error("VTK model error: {0}")]
    Model(#[from] VtkModelError),
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("Empty dataset")]
    Empty,
    #[error("PVD parse error: {0}")]
    Pvd(String),
    #[error("XDMF parse error: {0}")]
    Xdmf(String),
    #[error("HDF5 error: {0}")]
    Hdf5(String),
    #[error("OpenFOAM parse error: {0}")]
    OpenFoam(String),
    #[error("CGNS parse error: {0}")]
    Cgns(String),
    #[error("Exodus II parse error: {0}")]
    Exodus(String),
    #[error("EnSight parse error: {0}")]
    EnSight(String),
    #[error("Tecplot parse error: {0}")]
    Tecplot(String),
    #[error("NetCDF parse error: {0}")]
    NetCdf(String),
    #[error("CSV parse error: {0}")]
    Csv(String),
    #[error("Geometry parse error: {0}")]
    Geometry(String),
    #[error("Raw manifest parse error: {0}")]
    Raw(String),
}

impl From<ReadError> for IoError {
    fn from(value: ReadError) -> Self {
        match value {
            ReadError::Io(error) => IoError::Io(error),
            ReadError::UnsupportedFormat(message) => IoError::UnsupportedFormat(message),
            ReadError::Empty => IoError::Parse("empty dataset".to_string()),
            other => IoError::Parse(other.to_string()),
        }
    }
}
