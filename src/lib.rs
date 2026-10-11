#![warn(missing_docs)]
//! Source-format loaders and neutral decoded data for applications built on
//! `viewport-lib`.
//!
//! This crate owns file-format decoding and source-agnostic data structures.
//! Where `viewport-lib-types` has a payload that says the same thing (an image,
//! a triangle mesh, a vector path), loaders return that type, so the decoded
//! data goes to the renderer unchanged. The crate does not depend on
//! `viewport-lib` itself, so it carries no GPU stack.
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
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, AttributeData, AttributeDomain, AttributeValues, CELL_SENTINEL,
    DecodedDataSet, FillRule, GaussianSplatSet, Joint, MAX_JOINTS, MaterialData, PathSegment,
    PointSet, SceneData, SceneMesh, ShDegree, Skeleton, SkinWeights, SparseGrid, StructuredVolume,
    SubPath, SurfaceMesh, TextureSampler, TextureSource, VectorArt, VectorShape, VectorStroke,
    VolumeGridGeometry, VolumeMesh,
};
/// The shared payload vocabulary, re-exported so a consumer names the types
/// these loaders return without depending on it separately.
pub use viewport_lib_types;
