#![warn(missing_docs)]
//! Optional IO helpers for `viewport-lib`.
//!
//! This crate exists to keep heavy format-decoding dependencies out of
//! `viewport-lib` itself while still offering a small, reusable bridge for:
//! - file/bytes to `viewport_lib::MeshData`
//! - file/bytes to CPU-side scene structs for multi-object formats
//! - file/bytes to RGBA texture data

/// Error types for `viewport-lib-io`.
pub mod error;
/// CPU-side scene and texture data types returned by decoders.
pub mod scene;

#[cfg(feature = "obj")]
/// OBJ scene decoding.
pub mod obj;
#[cfg(feature = "png")]
/// Image decoding to RGBA8 pixel data.
pub mod png;
#[cfg(feature = "stl")]
/// STL decoding to `viewport_lib::MeshData`.
pub mod stl;

pub use error::IoError;
pub use scene::{
    IoMaterial, IoMesh, IoPointCloud, IoScene, TextureData, TextureSource,
};
