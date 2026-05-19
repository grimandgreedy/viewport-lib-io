//! Decoders that primarily produce `viewport_lib::MeshData`.

#[cfg(feature = "msh")]
/// Gmsh `.msh` surface-mesh decoding.
pub mod msh;
#[cfg(feature = "stl")]
/// STL surface-mesh decoding.
pub mod stl;
