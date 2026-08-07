#![warn(missing_docs)]
//! Source-format loaders and neutral decoded data for applications built on
//! `viewport-lib`.
//!
//! This crate owns file-format decoding and source-agnostic data structures.
//! It does not depend on `viewport-lib` runtime or upload types directly.
//!
//! # Coordinate convention
//!
//! Decoded scene data is right-handed Z-up. Source formats that store data
//! in a different frame (e.g. glTF, which is right-handed Y-up) are
//! reoriented at load. Vertex positions, normals, tangents, mesh
//! transforms, skeleton inverse-bind matrices, and animation samples are
//! all rotated once during decoding; downstream consumers do not need to
//! re-rotate.

/// Error types for `viewport-lib-io`.
pub mod error;
/// Neutral baked-lightmap data.
pub mod lightmap;
/// Source-format loaders.
pub mod loaders;
/// Source-agnostic decoded data types.
pub mod types;

pub use error::IoError;
pub use lightmap::{LightmapData, LightmapEncoding};
pub use types::{
    AlphaMode, AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler,
    AnimationTrack, AnimationTrackValues, AttributeData, AttributeDomain, AttributeValues,
    CELL_SENTINEL, DecodedDataSet, GaussianSplatSet, HdrImageData, Joint, MAX_JOINTS, MaterialData,
    PointSet, RasterImageData, SceneData, SceneMesh, ShDegree, Skeleton, SkinWeights, SparseGrid,
    StructuredVolume, SurfaceMesh, TextureSource, VolumeGridGeometry, VolumeMesh,
};
