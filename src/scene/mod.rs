//! Scene-container decoders that primarily produce `IoScene`.

#[cfg(feature = "fbx")]
/// FBX scene decoding.
pub mod fbx;
#[cfg(feature = "gltf")]
/// glTF / GLB scene decoding.
pub mod gltf;
#[cfg(feature = "obj")]
/// OBJ scene decoding.
pub mod obj;
