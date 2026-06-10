//! Source-format loaders.

/// CGNS dataset decoding.
pub mod cgns;
mod common;

/// CSV point-set decoding.
pub mod csv;

/// Shared loader error types.
mod error;

/// EnSight Gold dataset decoding.
pub mod ensight;

/// Exodus II dataset decoding.
pub mod exodus;

/// OpenEXR image decoding.
#[cfg(feature = "exr")]
pub mod exr;

/// FBX scene decoding.
#[cfg(feature = "fbx")]
pub mod fbx;

/// glTF / GLB scene decoding.
#[cfg(feature = "gltf")]
pub mod gltf;

/// Radiance HDR environment decoding.
#[cfg(feature = "hdr")]
pub mod hdr;

/// JPEG image decoding.
#[cfg(feature = "jpeg")]
pub mod jpeg;

/// Gmsh mesh decoding.
pub mod msh;

/// NetCDF dataset decoding.
pub mod netcdf;

/// NumPy volume decoding.
pub mod numpy;

/// OBJ scene decoding.
#[cfg(feature = "obj")]
pub mod obj;

/// Shared PVD/timestep helpers for scientific loaders.
mod pvd;

/// PLY point-set decoding.
pub mod ply_points;

/// PLY scene decoding.
pub mod ply_scene;

/// PNG image decoding.
#[cfg(feature = "png")]
pub mod png;

/// RAW manifest volume decoding.
pub mod raw;

/// STL surface mesh decoding.
#[cfg(feature = "stl")]
pub mod stl;

/// Tecplot dataset decoding.
pub mod tecplot;

/// Legacy VTK-family dataset decoding.
pub mod vtk;

/// XML PolyData dataset decoding.
pub mod vtp;

/// XML UnstructuredGrid dataset decoding.
pub mod vtu;

/// XDMF dataset decoding.
pub mod xdmf;
