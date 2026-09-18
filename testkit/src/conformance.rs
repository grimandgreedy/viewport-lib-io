//! The floor every loader has to clear: decode, satisfy the invariants, and do
//! it the same way twice.
//!
//! A loader's own tests then cover what is peculiar to its format. Keeping the
//! floor in one place means a new loader inherits it by calling one function,
//! and a new invariant applies to every loader the day it is written.
//!
//! ```ignore
//! let scene = conformance::scene("quad.obj", || loaders::obj::scene_from_path(&path));
//! assert_eq!(scene.meshes.len(), 1);
//! ```

use viewport_lib_io::error::IoError;
use viewport_lib_io::types::{PointSet, SceneData, SurfaceMesh};

use crate::invariants::{check_point_set, check_scene, check_surface_mesh};

/// One mesh as a consumer addresses it: its name, its size, and what it points
/// at. Enough to catch a reordered or re-indexed decode, small enough to print.
type MeshIdentity = (String, usize, usize, Option<usize>, Option<usize>);

/// What a positional index into a scene addresses. Two decodes that differ here
/// break every index a consumer recorded against the first one.
fn scene_identity(scene: &SceneData) -> Vec<MeshIdentity> {
    scene
        .meshes
        .iter()
        .map(|m| {
            (
                m.name.clone(),
                m.mesh.positions.len(),
                m.mesh.indices.len(),
                m.material_index,
                m.skeleton_index,
            )
        })
        .collect()
}

/// Decode a scene twice, check the invariants, and hand back the first decode
/// for format-specific assertions.
///
/// Panics with the decode error, the invariant violations, or the ordering
/// difference, whichever comes first.
#[track_caller]
pub fn scene<F>(label: &str, decode: F) -> SceneData
where
    F: Fn() -> Result<SceneData, IoError>,
{
    let first = decode().unwrap_or_else(|e| panic!("{label} failed to decode: {e}"));
    check_scene(label, &first).expect_clean();

    let second = decode().unwrap_or_else(|e| panic!("{label} failed to decode a second time: {e}"));
    let (a, b) = (scene_identity(&first), scene_identity(&second));
    assert!(
        a == b,
        "{label} decoded differently the second time\nfirst:  {a:?}\nsecond: {b:?}"
    );

    // Materials and skeletons are addressed by index too, so their order is
    // part of what has to hold still.
    let names = |s: &SceneData| {
        (
            s.materials
                .iter()
                .map(|m| m.name.clone())
                .collect::<Vec<_>>(),
            s.skeletons
                .iter()
                .map(|s| s.name.clone())
                .collect::<Vec<_>>(),
            s.animations
                .iter()
                .map(|a| a.name.clone())
                .collect::<Vec<_>>(),
        )
    };
    assert!(
        names(&first) == names(&second),
        "{label} reordered its materials, skeletons, or animations between decodes"
    );

    first
}

/// Decode a single mesh twice, check the invariants, and hand back the first.
#[track_caller]
pub fn mesh<F>(label: &str, decode: F) -> SurfaceMesh
where
    F: Fn() -> Result<SurfaceMesh, IoError>,
{
    let first = decode().unwrap_or_else(|e| panic!("{label} failed to decode: {e}"));
    check_surface_mesh(label, &first).expect_clean();

    let second = decode().unwrap_or_else(|e| panic!("{label} failed to decode a second time: {e}"));
    assert!(
        first.positions == second.positions && first.indices == second.indices,
        "{label} decoded different geometry the second time"
    );

    first
}

/// Decode a point set twice, check the invariants, and hand back the first.
#[track_caller]
pub fn points<F>(label: &str, decode: F) -> PointSet
where
    F: Fn() -> Result<PointSet, IoError>,
{
    let first = decode().unwrap_or_else(|e| panic!("{label} failed to decode: {e}"));
    check_point_set(label, &first).expect_clean();

    let second = decode().unwrap_or_else(|e| panic!("{label} failed to decode a second time: {e}"));
    assert!(
        first.positions == second.positions,
        "{label} decoded different points the second time"
    );

    first
}
