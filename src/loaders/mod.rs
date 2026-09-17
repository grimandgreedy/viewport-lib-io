//! Source-format loaders.

/// BMP image decoding.
#[cfg(feature = "bmp")]
pub mod bmp;

/// Biovision Hierarchy (`.bvh`) motion-capture decoding.
#[cfg(feature = "bvh")]
pub mod bvh;

/// CGNS dataset decoding.
#[cfg(feature = "cgns")]
pub mod cgns;
#[cfg(feature = "_scientific-common")]
mod common;

/// CSV point-set decoding.
#[cfg(feature = "csv")]
pub mod csv;

/// Shared loader error types.
#[cfg(feature = "_scientific-common")]
mod error;

/// EnSight Gold dataset decoding.
#[cfg(feature = "ensight")]
pub mod ensight;

/// Exodus II dataset decoding.
#[cfg(feature = "exodus")]
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
#[cfg(feature = "msh")]
pub mod msh;

/// NetCDF dataset decoding.
#[cfg(feature = "netcdf")]
pub mod netcdf;

/// NumPy volume decoding.
#[cfg(feature = "numpy")]
pub mod numpy;

/// OBJ scene decoding.
#[cfg(feature = "obj")]
pub mod obj;

/// Shared PVD/timestep helpers for scientific loaders.
#[cfg(feature = "_scientific-common")]
pub mod pvd;

/// PLY point-set decoding.
#[cfg(feature = "ply")]
pub mod ply_points;

/// PLY scene decoding.
#[cfg(feature = "ply")]
pub mod ply_scene;

/// PNG image decoding.
#[cfg(feature = "png")]
pub mod png;

/// RAW manifest volume decoding.
#[cfg(feature = "raw")]
pub mod raw;

/// STL surface mesh decoding.
#[cfg(feature = "stl")]
pub mod stl;

/// SVG image decoding.
#[cfg(feature = "svg")]
pub mod svg;

/// TGA image decoding.
#[cfg(feature = "tga")]
pub mod tga;

/// Tecplot dataset decoding.
#[cfg(feature = "tecplot")]
pub mod tecplot;

/// Legacy VTK-family dataset decoding.
#[cfg(feature = "vtk")]
pub mod vtk;

/// XML PolyData dataset decoding.
#[cfg(feature = "vtk")]
pub mod vtp;

/// XML UnstructuredGrid dataset decoding.
#[cfg(feature = "vtk")]
pub mod vtu;

/// XDMF dataset decoding.
#[cfg(feature = "xdmf")]
pub mod xdmf;
