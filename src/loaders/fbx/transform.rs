//! Node transforms and the axis conversion, including the heuristic that vetoes the
//! file-level conversion when the parent chain already performed it.

use fbxcel_dom::fbxcel;
use fbxcel_dom::v7400::Document;

use super::options::{AxisPolicy, CumulativeOrder, FbxLoadOptions};
use super::raw::{read_int_property, read_vec3_property};

/// Read the FBX-space local TRS for a model node, plus its geometric
/// offset. Returns `(local, geometric)` matrices without any axis
/// transform or unit scaling applied: those are intended to be applied
/// once at the root of the walk by [`extract_node_transform`].
///
/// Works for any `ModelHandle` variant (Mesh, Null, LimbNode, ...). For
/// nodes without a `Properties70` block (rare; mostly a no-op root),
/// returns identity for both.
pub(super) fn extract_local_components(
    model: &fbxcel_dom::v7400::object::model::ModelHandle<'_>,
) -> (glam::Mat4, glam::Mat4) {
    let Some(props) = model.direct_properties() else {
        return (glam::Mat4::IDENTITY, glam::Mat4::IDENTITY);
    };

    let translation = read_vec3_property(&props, "Lcl Translation").unwrap_or(glam::Vec3::ZERO);
    let rotation_deg = read_vec3_property(&props, "Lcl Rotation").unwrap_or(glam::Vec3::ZERO);
    let scaling = read_vec3_property(&props, "Lcl Scaling").unwrap_or(glam::Vec3::ONE);
    let pre_rotation_deg = read_vec3_property(&props, "PreRotation").unwrap_or(glam::Vec3::ZERO);
    let post_rotation_deg = read_vec3_property(&props, "PostRotation").unwrap_or(glam::Vec3::ZERO);
    let rotation_offset = read_vec3_property(&props, "RotationOffset").unwrap_or(glam::Vec3::ZERO);
    let rotation_pivot = read_vec3_property(&props, "RotationPivot").unwrap_or(glam::Vec3::ZERO);
    let scaling_offset = read_vec3_property(&props, "ScalingOffset").unwrap_or(glam::Vec3::ZERO);
    let scaling_pivot = read_vec3_property(&props, "ScalingPivot").unwrap_or(glam::Vec3::ZERO);
    let geo_translation =
        read_vec3_property(&props, "GeometricTranslation").unwrap_or(glam::Vec3::ZERO);
    let geo_rotation_deg =
        read_vec3_property(&props, "GeometricRotation").unwrap_or(glam::Vec3::ZERO);
    let geo_scaling = read_vec3_property(&props, "GeometricScaling").unwrap_or(glam::Vec3::ONE);
    let rotation_order = read_int_property(&props, "RotationOrder").unwrap_or(0);

    let t = glam::Mat4::from_translation(translation);
    let r_off = glam::Mat4::from_translation(rotation_offset);
    let r_piv = glam::Mat4::from_translation(rotation_pivot);
    let r_piv_inv = glam::Mat4::from_translation(-rotation_pivot);
    let s_off = glam::Mat4::from_translation(scaling_offset);
    let s_piv = glam::Mat4::from_translation(scaling_pivot);
    let s_piv_inv = glam::Mat4::from_translation(-scaling_pivot);
    let pre_r = euler_to_mat4(pre_rotation_deg, 0);
    let r = euler_to_mat4(rotation_deg, rotation_order);
    let post_r_inv = euler_to_mat4(post_rotation_deg, 0).inverse();
    let s = glam::Mat4::from_scale(scaling);

    let geo_t = glam::Mat4::from_translation(geo_translation);
    let geo_r = euler_to_mat4(geo_rotation_deg, 0);
    let geo_s = glam::Mat4::from_scale(geo_scaling);
    let geometric = geo_t * geo_r * geo_s;

    let local =
        t * r_off * r_piv * pre_r * r * post_r_inv * r_piv_inv * s_off * s_piv * s * s_piv_inv;
    (local, geometric)
}

/// Compute the absolute world transform for a Mesh node, walking up the
/// full FBX parent chain through every node type (Mesh, Null, LimbNode,
/// ...). Each ancestor contributes its `local` TRS. Geometric offset is
/// the leaf node's only (it does not propagate to children). Axis
/// transform + unit scale are applied once at the root.
///
/// Unity-exported FBX commonly stacks Mesh -> Null -> Mesh chains where
/// the Null nodes carry the placement transforms (LODGroup containers,
/// "DummyHelper" exporter scaffolding). Earlier versions of this loader
/// only extracted each node's own local transform and tracked Mesh ->
/// Mesh parent links separately; transforms on Null/LimbNode parents
/// silently dropped, producing detached "floating" sub-meshes in the
/// output scene. This walk picks up every ancestor type, so the world
/// position matches what a runtime engine using FBX cumulative
/// transforms would produce.
///
/// The axis transform to apply to one mesh leaf, and whether the heuristic
/// veto suppressed the file-level conversion.
///
/// Only [`AxisPolicy::HeuristicVeto`] varies this per leaf. The others apply
/// `file_axis` as computed for the file as a whole.
///
/// The veto: if the cumulative chain (Lcl Rotation on the mesh or any
/// Null/LimbNode ancestor) already lands +Y on +Z, applying the file-level
/// Y-to-Z fix on top would double it up. The check is signed, deliberately:
///
/// - cumulative +Y lands on +Z: the artist already converted Y-up to Z-up at
///   the vertex level, so suppress the +90 degree X and return identity.
/// - cumulative +Y lands on -Z: the artist baked a rotation targeting a Y-up
///   consumer (a Z-up authored mesh wrapped in a -90 degree X Null for Unity
///   ingestion). Do not veto: the +90 degree X composes with their -90 degree
///   X to identity.
/// - cumulative +Y stays near +Y: ordinary Y-up content, so apply the
///   +90 degree X.
///
/// The `y.abs() < 0.5` half of the test keeps a chain that merely tilts towards
/// +Z from tripping the veto: both components have to agree that the chain is a
/// genuine Y-to-Z conversion.
#[cfg(feature = "fbx")]
pub(super) fn effective_axis_transform(
    policy: AxisPolicy,
    file_axis: &glam::Mat4,
    cumulative: &glam::Mat4,
) -> (glam::Mat4, bool) {
    match policy {
        AxisPolicy::HeuristicVeto => {
            let cumulative_y = cumulative.transform_vector3(glam::Vec3::Y);
            let already_z_up = cumulative_y.z > 0.9 && cumulative_y.y.abs() < 0.5;
            if already_z_up {
                (glam::Mat4::IDENTITY, true)
            } else {
                (*file_axis, false)
            }
        }
        AxisPolicy::HonourHeader | AxisPolicy::ForceYUpRaw | AxisPolicy::PassThrough => {
            (*file_axis, false)
        }
    }
}

pub(super) fn extract_node_transform(
    mesh_model: &fbxcel_dom::v7400::object::model::MeshHandle<'_>,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
    options: &FbxLoadOptions,
) -> glam::Mat4 {
    use fbxcel_dom::v7400::object::model::TypedModelHandle;
    use std::ops::Deref;

    let (mut cumulative, geometric) = extract_local_components(mesh_model.deref());

    // Walk parents up the chain. `parent_model()` returns
    // `Option<TypedModelHandle>`; we dispatch by variant to read its
    // local TRS, then chain to its own parent.
    let mut current: Option<TypedModelHandle<'_>> = mesh_model.parent_model();
    let mut depth_guard = 0;
    while let Some(parent) = current {
        depth_guard += 1;
        if depth_guard > 256 {
            // FBX hierarchies hundreds deep are pathological; bail rather
            // than infinite-loop on a malformed file.
            break;
        }
        let (parent_local, _parent_geometric) = match &parent {
            TypedModelHandle::Mesh(m) => extract_local_components(m.deref()),
            TypedModelHandle::Null(n) => extract_local_components(n.deref()),
            TypedModelHandle::LimbNode(n) => extract_local_components(n.deref()),
            TypedModelHandle::Light(l) => extract_local_components(l.deref()),
            TypedModelHandle::Camera(c) => extract_local_components(c.deref()),
            // Unknown / future variants: stop walking so we don't drop
            // the ones we already accumulated. Mirrors the conservative
            // posture of the rest of this loader.
            _ => break,
        };
        cumulative = parent_local * cumulative;
        current = match parent {
            TypedModelHandle::Mesh(m) => m.parent_model(),
            TypedModelHandle::Null(n) => n.parent_model(),
            TypedModelHandle::LimbNode(n) => n.parent_model(),
            TypedModelHandle::Light(l) => l.parent_model(),
            TypedModelHandle::Camera(c) => c.parent_model(),
            _ => None,
        };
    }

    let (effective_axis, _veto_fired) =
        effective_axis_transform(options.axis_policy, axis_transform, &cumulative);

    let scale = glam::Mat4::from_scale(glam::Vec3::splat(unit_scale));
    match options.cumulative_order {
        // Source-frame interpretation of the chain: bake first, axis
        // last. Matches the loader's historical behaviour.
        CumulativeOrder::PreAxis => effective_axis * scale * cumulative * geometric,
        // Consumer-frame interpretation of the chain: axis converts
        // the leaf to the consumer frame, then the chain places it.
        CumulativeOrder::PostAxis => scale * cumulative * effective_axis * geometric,
    }
}

pub(super) fn euler_to_mat4(degrees: glam::Vec3, order: i32) -> glam::Mat4 {
    let radians = degrees * (std::f32::consts::PI / 180.0);
    let rx = glam::Mat4::from_rotation_x(radians.x);
    let ry = glam::Mat4::from_rotation_y(radians.y);
    let rz = glam::Mat4::from_rotation_z(radians.z);
    match order {
        0 => rz * ry * rx,
        1 => ry * rz * rx,
        2 => rx * rz * ry,
        3 => rz * rx * ry,
        4 => ry * rx * rz,
        5 => rx * ry * rz,
        _ => rz * ry * rx,
    }
}

/// Build the axis-conversion matrix that lands FBX vertices into the
/// loader's canonical output frame: Z-up, right-handed, CCW-front.
///
/// FBX `GlobalSettings.UpAxis` encodes the source up axis as `0 = X`,
/// `1 = Y`, `2 = Z`. In principle a `UpAxis = Z` file is already in our
/// frame, but in practice many Unity-exported asset packs (Leartes Roman
/// Street, etc.) write `UpAxis = Z` while the raw vertex stream is still
/// Y-up: the file's own scene-level convention is contradicted by the
/// geometry. Some sub-meshes in the same pack carry a compensating
/// `Lcl Rotation = (90, 0, 0)` on the leaf; most don't.
///
/// The strategy used here: **always** produce a Y->Z axis transform
/// (`+90 degrees` about X), then in [`extract_node_transform`] suppress it for
/// any leaf whose cumulative parent chain already lands +Y on +Z. That
/// covers the "exporter compensated via `Lcl Rotation`" case (tree
/// foliage) without breaking the "exporter left raw Y-up" case (roofs,
/// pots, arcs, walls).
pub(super) fn get_axis_transform(document: &Document, policy: AxisPolicy) -> (glam::Mat4, f32) {
    let settings = match document.global_settings() {
        Some(settings) => settings,
        None => return (glam::Mat4::IDENTITY, 1.0),
    };

    let props = settings.raw_properties();
    let up_axis = props
        .get_property("UpAxis")
        .and_then(|property| {
            let values = property.value_part();
            values.first().and_then(|value| match value {
                fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i),
                fbxcel::low::v7400::AttributeValue::I16(i) => Some(*i as i32),
                _ => None,
            })
        })
        .unwrap_or(1);

    let unit_scale_factor = props
        .get_property("UnitScaleFactor")
        .and_then(|property| {
            let values = property.value_part();
            values.first().and_then(|value| match value {
                fbxcel::low::v7400::AttributeValue::F64(f) => Some(*f as f32),
                fbxcel::low::v7400::AttributeValue::F32(f) => Some(*f),
                fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i as f32),
                _ => None,
            })
        })
        .unwrap_or(1.0);

    let unit_scale = unit_scale_factor / 100.0;
    // `up_axis`: 0 = X, 1 = Y, 2 = Z. Per the policy:
    //
    // - `HeuristicVeto` / `ForceYUpRaw`: assume raw vertices are Y-up
    //   (the loader's historical assumption: Unity exports lie about
    //   this in the header). X-up sources are vanishingly rare and we
    //   pass them through rather than guess.
    // - `HonourHeader`: trust the header. Y-up gets +90 degrees about X; Z-up gets
    //   identity; X-up passes through.
    // - `PassThrough`: never apply axis correction.
    //
    // `extract_node_transform` may further suppress this per-leaf when
    // the policy is `HeuristicVeto`.
    let axis_transform = match policy {
        AxisPolicy::HeuristicVeto | AxisPolicy::ForceYUpRaw => match up_axis {
            0 => glam::Mat4::IDENTITY,
            _ => glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
        },
        AxisPolicy::HonourHeader => match up_axis {
            1 => glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2),
            // X-up and Z-up both land as identity (Z-up is already in
            // our canonical frame; X-up we pass through).
            _ => glam::Mat4::IDENTITY,
        },
        AxisPolicy::PassThrough => glam::Mat4::IDENTITY,
    };

    (axis_transform, unit_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// +90 degrees about X: the file-level Y-up to Z-up conversion this loader
    /// applies when the header says the content is Y-up.
    fn y_to_z() -> glam::Mat4 {
        glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
    }

    fn maps_y_to(m: &glam::Mat4) -> glam::Vec3 {
        m.transform_vector3(glam::Vec3::Y)
    }

    #[test]
    fn veto_fires_when_the_chain_already_lands_y_on_plus_z() {
        // The artist converted Y-up to Z-up at the vertex level, so applying
        // the file conversion on top would rotate the mesh twice.
        let cumulative = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
        assert!(maps_y_to(&cumulative).z > 0.9, "chain maps +Y to +Z");

        let (axis, vetoed) =
            effective_axis_transform(AxisPolicy::HeuristicVeto, &y_to_z(), &cumulative);

        assert!(vetoed, "the veto should fire");
        assert_eq!(axis, glam::Mat4::IDENTITY);
    }

    #[test]
    fn veto_does_not_fire_when_the_chain_lands_y_on_minus_z() {
        // A Z-up mesh wrapped in a -90 degree X Null for a Y-up consumer. Our
        // +90 degree X composes with their -90 degree X back to identity, so
        // vetoing here double-applies the artist's intent: the Triumph_Arc case.
        let cumulative = glam::Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        assert!(maps_y_to(&cumulative).z < -0.9, "chain maps +Y to -Z");

        let (axis, vetoed) =
            effective_axis_transform(AxisPolicy::HeuristicVeto, &y_to_z(), &cumulative);

        assert!(!vetoed, "the veto must stay signed, not absolute");
        assert_eq!(axis, y_to_z());
    }

    #[test]
    fn veto_does_not_fire_on_ordinary_y_up_content() {
        let (axis, vetoed) =
            effective_axis_transform(AxisPolicy::HeuristicVeto, &y_to_z(), &glam::Mat4::IDENTITY);

        assert!(!vetoed);
        assert_eq!(axis, y_to_z());
    }

    #[test]
    fn veto_does_not_fire_on_a_chain_merely_tilted_towards_z() {
        // 45 degrees puts +Y at z = 0.707, under the 0.9 threshold: a tilted
        // placement is not an axis conversion.
        let cumulative = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_4);
        let y = maps_y_to(&cumulative);
        assert!(y.z > 0.7 && y.z < 0.9, "tilted, not converted: {y:?}");

        let (axis, vetoed) =
            effective_axis_transform(AxisPolicy::HeuristicVeto, &y_to_z(), &cumulative);

        assert!(!vetoed, "a 45 degree tilt is under the threshold");
        assert_eq!(axis, y_to_z());
    }

    #[test]
    fn only_the_heuristic_policy_vetoes() {
        // A chain that would trip the veto, under every other policy.
        let cumulative = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
        for policy in [
            AxisPolicy::HonourHeader,
            AxisPolicy::ForceYUpRaw,
            AxisPolicy::PassThrough,
        ] {
            let (axis, vetoed) = effective_axis_transform(policy, &y_to_z(), &cumulative);
            assert!(!vetoed, "{policy:?} must not veto");
            assert_eq!(axis, y_to_z(), "{policy:?} applies the file axis as given");
        }
    }

    #[test]
    fn a_vetoed_leaf_keeps_its_chain_orientation() {
        // What the veto is for, end to end on the matrices: a chain that
        // already converts, composed with the effective axis, lands +Y on +Z
        // exactly once. Composing the file axis instead would overshoot to -Y.
        let cumulative = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let (effective, _) =
            effective_axis_transform(AxisPolicy::HeuristicVeto, &y_to_z(), &cumulative);

        let vetoed = maps_y_to(&(effective * cumulative));
        assert!(vetoed.z > 0.99, "converted once: {vetoed:?}");

        let doubled = maps_y_to(&(y_to_z() * cumulative));
        assert!(
            doubled.y < -0.99,
            "converting twice overshoots: {doubled:?}"
        );
    }
}
