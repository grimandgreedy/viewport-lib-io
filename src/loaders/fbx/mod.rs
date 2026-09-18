//! FBX decoding: a binary or ASCII FBX file into a neutral [`SceneData`](crate::types::SceneData).
//!
//! FBX carries its own axis convention in the header, so the decode reorients the
//! scene to Z-up once, here, rather than leaving it to the consumer.

mod animation;
mod blendshape;
mod chain;
mod geometry;
mod material;
mod options;
mod raw;
mod scene;
mod skin;
mod transform;

pub use chain::{FbxChainLink, FbxMeshChain, FbxNodeKind, chain_breakdown};
pub use options::{AxisPolicy, CumulativeOrder, FbxLoadOptions};

use crate::error::IoError;
use crate::types::IoScene;
use std::path::Path;

use scene::build_fbx_scene;

/// Decode an FBX file into a CPU-side scene using default load options.
///
/// Equivalent to [`scene_from_path_with_options`] with
/// `FbxLoadOptions::default()`.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    scene_from_path_with_options(path, FbxLoadOptions::default())
}

/// Decode an FBX file into a CPU-side scene with caller-supplied options.
///
/// See [`FbxLoadOptions`] for what's tunable and why you might want to
/// reach for it.
pub fn scene_from_path_with_options(
    path: &Path,
    options: FbxLoadOptions,
) -> Result<IoScene, IoError> {
    #[cfg(feature = "fbx")]
    {
        let bytes = std::fs::read(path)?;
        build_fbx_scene(&bytes, path.parent(), options)
    }

    #[cfg(not(feature = "fbx"))]
    {
        let _ = (path, options);
        Err(IoError::MissingFeature {
            feature: "fbx",
            context: "FBX scene decoding",
        })
    }
}

/// Decode a binary FBX (7.4/7.5) scene from in-memory bytes. The in-memory sibling of
/// [`scene_from_path`], for a mesh served from a cooked bundle (or fetched over the
/// network for a `wasm32` target) rather than read from a file. Self-contained meshes
/// decode with no base directory; see [`scene_from_bytes_with_options`] to resolve
/// external texture references.
pub fn scene_from_bytes(bytes: &[u8]) -> Result<IoScene, IoError> {
    scene_from_bytes_with_options(bytes, None, FbxLoadOptions::default())
}

/// Decode a binary FBX scene from in-memory bytes with per-call overrides. `base`
/// resolves any external texture paths the FBX references; pass `None` when textures are
/// supplied out of band (as DRAKE does, keying them as separate assets). The in-memory
/// sibling of [`scene_from_path_with_options`].
pub fn scene_from_bytes_with_options(
    bytes: &[u8],
    base: Option<&Path>,
    options: FbxLoadOptions,
) -> Result<IoScene, IoError> {
    #[cfg(feature = "fbx")]
    {
        build_fbx_scene(bytes, base, options)
    }

    #[cfg(not(feature = "fbx"))]
    {
        let _ = (bytes, base, options);
        Err(IoError::MissingFeature {
            feature: "fbx",
            context: "FBX scene decoding",
        })
    }
}
