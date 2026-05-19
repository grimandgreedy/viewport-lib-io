#![warn(missing_docs)]
//! Optional IO helpers for `viewport-lib`.
//!
//! This crate keeps format-decoding dependencies out of `viewport-lib` itself.
//! It groups decoders by the `viewport-lib` input data family they primarily
//! produce, while still keeping one Rust source file per source filetype.

/// Error types for `viewport-lib-io`.
pub mod error;
/// Shared decode-side data types.
pub mod types;
/// Surface mesh input family.
pub mod surface_mesh;
/// Scene-container input family.
pub mod scene;
/// Texture input family.
pub mod texture;
/// Dense structured volume input family.
pub mod volume;
/// Unstructured volume-mesh input family.
pub mod volume_mesh;
/// Sparse regular volume-grid input family.
pub mod sparse_volume;
/// Point-cloud input family.
pub mod point_cloud;
/// Gaussian-splat input family.
pub mod gaussian_splat;
/// Path and polyline input family.
pub mod path;
/// Glyph input family.
pub mod glyph;
/// Lighting and environment input family.
pub mod lighting;

pub use error::IoError;
pub use types::{IoMaterial, IoMesh, IoPointCloud, IoScene, TextureData, TextureSource};
