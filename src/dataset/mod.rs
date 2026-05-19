//! Decoders that may produce multiple scientific viewport data families.

mod common;
mod error;
mod pvd;

/// CGNS dataset decoding.
pub mod cgns;
/// EnSight dataset decoding.
pub mod ensight;
/// Exodus II dataset decoding.
pub mod exodus;
/// NetCDF dataset decoding.
pub mod netcdf;
/// Tecplot dataset decoding.
pub mod tecplot;
/// Legacy VTK dataset decoding.
pub mod vtk;
/// XML PolyData dataset decoding.
pub mod vtp;
/// XML UnstructuredGrid dataset decoding.
pub mod vtu;
/// XDMF dataset decoding.
pub mod xdmf;
