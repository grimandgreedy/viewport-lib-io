#![warn(missing_docs)]
//! Source-format loaders and neutral decoded data for applications built on
//! `viewport-lib`.
//!
//! This crate owns file-format decoding and source-agnostic data structures.
//! It does not depend on `viewport-lib` runtime or upload types directly.

/// Error types for `viewport-lib-io`.
pub mod error;
/// Source-agnostic decoded data types.
pub mod types;
/// Source-format loaders.
pub mod loaders;

pub use error::IoError;
pub use types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, AttributeData, AttributeDomain, AttributeValues, CELL_SENTINEL,
    DecodedDataSet, GaussianSplatSet, HdrImageData, Joint, MaterialData, PointSet,
    RasterImageData, SceneData, SceneMesh, ShDegree, Skeleton, SkinWeights, SparseGrid,
    StructuredVolume, SurfaceMesh, TextureSource, VolumeGridGeometry, VolumeMesh,
};
