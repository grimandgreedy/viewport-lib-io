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
//! A separate crate rather than a feature on the library, so the library's own
//! tests and an outside consumer reach it the same way, and a release build
//! never compiles it. `viewport-lib-io` dev-depends on this crate, which
//! depends back on the library: cargo allows that cycle between packages.
//!
//! ```ignore
//! let scene = viewport_lib_io::loaders::gltf::scene_from_path(path)?;
//! viewport_lib_io_testkit::check_scene("hero.glb", &scene).expect_clean();
//! ```

pub mod conformance;
pub mod determinism;
pub mod invariants;
pub mod synth;

pub use determinism::{decode_twice_identical, decode_twice_matches};
pub use invariants::{
    Report, Violation, check_point_set, check_scene, check_skeleton, check_surface_mesh,
};
