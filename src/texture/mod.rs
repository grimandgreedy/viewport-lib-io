//! Decoders that primarily produce `TextureData`.

#[cfg(feature = "jpeg")]
/// JPEG texture decoding.
pub mod jpeg;
#[cfg(feature = "png")]
/// PNG texture decoding.
pub mod png;
