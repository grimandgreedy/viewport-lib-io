//! Public load options: how the axis conversion is chosen, and where the cumulative
//! parent transform sits relative to it.

/// How the loader decides whether to apply the Y-up to Z-up axis transform.
///
/// FBX files don't carry enough metadata to reliably tell whether a given
/// leaf's raw vertices are Y-up or Z-up: the `GlobalSettings.UpAxis` header
/// is often inconsistent with the actual vertex orientation (Unity-exported
/// packs frequently declare `UpAxis = Z` while shipping Y-up geometry), and
/// `Lcl Rotation` bakes can mean either "axis conversion" or "placement".
/// Callers with out-of-band knowledge of their asset pipeline can use this
/// to override the default heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisPolicy {
    /// Apply `+90 degrees` about X by default, with a per-leaf signed-veto check
    /// in `extract_node_transform`: if the cumulative parent chain
    /// already maps +Y to +Z, suppress the axis transform for that leaf.
    /// Best for mixed asset packs where some meshes bake the conversion
    /// and others don't. This is the default.
    HeuristicVeto,
    /// Honour `GlobalSettings.UpAxis` literally. `UpAxis = Y` applies
    /// `+90 degrees` about X; `UpAxis = Z` applies identity; no per-leaf veto.
    /// Use for files you trust to declare their orientation correctly.
    HonourHeader,
    /// Always apply `+90 degrees` about X regardless of header or chain. Use
    /// when you know raw vertices are Y-up.
    ForceYUpRaw,
    /// Never apply any axis transform. Use when you know raw vertices
    /// are already Z-up.
    PassThrough,
}

/// Where the cumulative parent-chain transform sits relative to the
/// axis transform when composing the per-leaf world matrix.
///
/// The right choice depends on what the artist's chain rotations *mean*:
/// a coord-system bake belongs in the source frame ([`PreAxis`]), a
/// display-orientation rotation belongs in the consumer's frame
/// ([`PostAxis`]). FBX has no format-level marker distinguishing the two.
///
/// [`PreAxis`]: CumulativeOrder::PreAxis
/// [`PostAxis`]: CumulativeOrder::PostAxis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CumulativeOrder {
    /// `effective_axis * scale * cumulative * geometric`. The cumulative
    /// chain is composed in the source frame and the axis transform is
    /// the final step. This is the default and matches the loader's
    /// historical behaviour.
    PreAxis,
    /// `scale * cumulative * effective_axis * geometric`. The axis
    /// transform runs first (per-vertex source -> consumer conversion)
    /// and the cumulative chain then places the result in the consumer
    /// frame. Use when the artist authored the prop in a non-display
    /// pose and uses metadata rotations to right it.
    PostAxis,
}

/// Per-call overrides for [`super::scene_from_path_with_options`].
///
/// Defaults preserve the loader's historical behaviour. Callers only
/// need to construct a non-default value when they have specific
/// knowledge about how their asset pipeline emits FBX.
#[derive(Debug, Clone, Copy, Default)]
pub struct FbxLoadOptions {
    /// Strategy for deciding whether to apply the Y-up to Z-up axis
    /// transform per leaf.
    pub axis_policy: AxisPolicy,
    /// Ordering of the cumulative parent-chain transform relative to
    /// the axis transform.
    pub cumulative_order: CumulativeOrder,
}

impl Default for AxisPolicy {
    fn default() -> Self {
        Self::HeuristicVeto
    }
}

impl Default for CumulativeOrder {
    fn default() -> Self {
        Self::PreAxis
    }
}
