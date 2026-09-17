//! Shared checks for loader tests: the properties decoded data has to satisfy
//! whatever file it came from, and the helpers for building input to feed a
//! loader.
//!
//! Decoding has no golden-image oracle, so the thing worth defining once is the
//! set of invariants every loader's output obeys: indices in bounds, per-vertex
//! arrays the same length as the positions they belong to, cross-references
//! that resolve, and the same file decoding to the same thing twice. Each is a
//! bug that has shipped in some loader somewhere, and none of them needs to
//! know which format the data came from.
//!
//! Off by default and outside `all-formats`, since a release build has no use
//! for it. Turn it on with `--features testkit`, in this crate's own tests or
//! from a consumer checking its own asset library:
//!
//! ```ignore
//! let scene = viewport_lib_io::loaders::gltf::scene_from_path(path)?;
//! viewport_lib_io::testkit::check_scene("hero.glb", &scene).expect_clean();
//! ```

pub mod determinism;
pub mod invariants;
pub mod synth;

pub use determinism::{decode_twice_identical, decode_twice_matches};
pub use invariants::{
    Report, Violation, check_point_set, check_scene, check_skeleton, check_surface_mesh,
};
