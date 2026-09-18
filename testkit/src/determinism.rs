//! Decoding the same input twice has to give the same thing.
//!
//! This is not a theoretical property. A loader that walks a parser's hash map
//! emits its output in whatever order that map iterates, which varies per map
//! instance, so two decodes in one process disagree. The damage is downstream
//! and looks nothing like a loader bug: a positional index recorded against one
//! decode (a material assignment, a saved selection, a retarget map) addresses
//! something else in the next.

use std::fmt::Debug;

/// Decode twice and compare, panicking with the difference if the two disagree.
///
/// `label` names the input in the failure message. The comparison is on the
/// decoded value, so a type whose `PartialEq` ignores ordering will not catch
/// an ordering bug: compare the thing a consumer would index into.
///
/// ```ignore
/// decode_twice_identical("fox.glb", || {
///     loaders::gltf::scene_from_path(&path).unwrap()
/// });
/// ```
#[track_caller]
pub fn decode_twice_identical<T, F>(label: &str, mut decode: F)
where
    T: PartialEq + Debug,
    F: FnMut() -> T,
{
    let first = decode();
    let second = decode();
    assert!(
        first == second,
        "{label} decoded differently the second time\nfirst:  {first:?}\nsecond: {second:?}"
    );
}

/// Decode twice and compare a projection of the result, for output that is too
/// large to print or that carries fields a comparison should ignore.
///
/// Reach for this over [`decode_twice_identical`] when the failure message
/// matters: projecting to the identity a consumer depends on (names, counts,
/// the indices they address) gives a diff that can be read, rather than two
/// dumps of a whole scene.
#[track_caller]
pub fn decode_twice_matches<T, P, F, G>(label: &str, mut decode: F, project: G)
where
    P: PartialEq + Debug,
    F: FnMut() -> T,
    G: Fn(&T) -> P,
{
    let first = project(&decode());
    let second = project(&decode());
    assert!(
        first == second,
        "{label} decoded differently the second time\nfirst:  {first:?}\nsecond: {second:?}"
    );
}
