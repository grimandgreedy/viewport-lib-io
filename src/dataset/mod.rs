//! Decoders that may produce multiple scientific viewport data families.

/// Legacy VTK dataset decoding.
pub mod vtk;
/// XML PolyData dataset decoding.
pub mod vtp;
/// XML UnstructuredGrid dataset decoding.
pub mod vtu;
