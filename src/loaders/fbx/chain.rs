//! The per-mesh parent-chain breakdown: the diagnostic that says what the loader
//! composed for a given mesh, and why.

use crate::error::IoError;
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::object::TypedObjectHandle;
use fbxcel_dom::v7400::object::model::TypedModelHandle;
use std::io::BufReader;
use std::path::Path;

use super::options::FbxLoadOptions;
use super::raw::objects_in_stable_order;
use super::transform::{effective_axis_transform, extract_local_components, get_axis_transform};

/// Kind of FBX node visited during the parent-chain walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbxNodeKind {
    /// A Mesh node (the leaf is always one of these).
    Mesh,
    /// A Null node: Unity-exported FBX stacks these as LODGroup
    /// containers and "DummyHelper" exporter scaffolding.
    Null,
    /// A skeleton bone.
    LimbNode,
    /// A Light node.
    Light,
    /// A Camera node.
    Camera,
}

/// One link in the per-mesh FBX parent chain.
///
/// Returned by [`chain_breakdown`] in leaf-first order. The local TRS is
/// the decomposition of the node's `Lcl Translation/Rotation/Scaling`
/// (plus pivots, offsets, pre/post rotation), so any pivot-driven offset
/// shows up in `local_translation` rather than `local_rotation_quat`.
#[derive(Debug, Clone)]
pub struct FbxChainLink {
    /// Node name from the FBX `Model::name` field. Empty for unnamed nodes.
    pub node_name: String,
    /// Node kind (`Mesh`, `Null`, `LimbNode`, ...).
    pub node_kind: FbxNodeKind,
    /// Translation component of the node's local matrix.
    pub local_translation: glam::Vec3,
    /// Rotation component of the node's local matrix.
    pub local_rotation_quat: glam::Quat,
    /// Same rotation in degrees, XYZ Euler order, for readable diagnostics.
    pub local_rotation_euler_deg: glam::Vec3,
    /// Scale component of the node's local matrix.
    pub local_scale: glam::Vec3,
}

/// Full per-mesh breakdown of what the FBX loader composes when computing
/// `IoMesh::transform`.
///
/// The world matrix the loader hands downstream is:
///
/// ```text
/// effective_axis * scale * cumulative * geometric
/// ```
///
/// where `cumulative = leaf_local * parent_1_local * ... * root_local`.
/// This struct exposes each piece so diagnostic tools can show the chain
/// link-by-link instead of as a single fused matrix.
#[derive(Debug, Clone)]
pub struct FbxMeshChain {
    /// Name of the leaf Mesh node.
    pub mesh_name: String,
    /// Parent chain, leaf-first. `links[0]` is the leaf Mesh; `links[1]`
    /// is its immediate parent; etc.
    pub links: Vec<FbxChainLink>,
    /// Leaf geometric translation (does not propagate to children).
    pub geometric_translation: glam::Vec3,
    /// Leaf geometric rotation.
    pub geometric_rotation_quat: glam::Quat,
    /// Leaf geometric rotation in degrees, XYZ Euler order.
    pub geometric_rotation_euler_deg: glam::Vec3,
    /// Leaf geometric scale.
    pub geometric_scale: glam::Vec3,
    /// File-scoped axis transform (`+90 degrees` X, identity, etc.) before any
    /// per-leaf veto.
    pub file_axis_transform: glam::Mat4,
    /// Axis transform actually applied to this leaf after the veto check.
    /// Equal to `file_axis_transform` if no veto fired; identity if it did.
    pub effective_axis_transform: glam::Mat4,
    /// `true` if [`AxisPolicy::HeuristicVeto`](super::AxisPolicy::HeuristicVeto) suppressed the file-level
    /// axis transform for this leaf.
    pub axis_veto_fired: bool,
    /// Unit scale derived from `GlobalSettings.UnitScaleFactor / 100`.
    pub unit_scale: f32,
}

/// Walk an FBX file's hierarchy and return a per-link breakdown of every
/// Mesh node's parent chain.
///
/// Output mirrors what [`super::scene_from_path_with_options`] does internally
/// when computing each `IoMesh::transform`, but decomposed link by link
/// rather than fused into a single matrix. Use this when you need to see
/// what the loader is doing, e.g. to diagnose a 90 degrees rotation that's
/// being introduced somewhere in the chain.
///
/// Order of returned chains follows iteration order over the FBX
/// document (matches `scene_from_path_with_options`).
pub fn chain_breakdown(path: &Path, options: FbxLoadOptions) -> Result<Vec<FbxMeshChain>, IoError> {
    #[cfg(feature = "fbx")]
    {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let document = match AnyDocument::from_seekable_reader(reader).map_err(|error| {
            IoError::Parse(format!("FBX load failed ({}): {error:?}", path.display()))
        })? {
            AnyDocument::V7400(_, document) => document,
            _ => {
                return Err(IoError::Parse(
                    "unsupported FBX version (only binary FBX 7.4/7.5 supported)".into(),
                ));
            }
        };
        let (file_axis, unit_scale) = get_axis_transform(&document, options.axis_policy);

        let mut chains = Vec::new();
        for object in objects_in_stable_order(&document) {
            if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh_model)) = object.get_typed()
            {
                chains.push(build_chain_breakdown(
                    &mesh_model,
                    &file_axis,
                    unit_scale,
                    &options,
                ));
            }
        }
        Ok(chains)
    }
    #[cfg(not(feature = "fbx"))]
    {
        let _ = (path, options);
        Err(IoError::Parse(
            "FBX support not compiled in (enable the `fbx` feature)".into(),
        ))
    }
}

#[cfg(feature = "fbx")]
pub(super) fn build_chain_breakdown(
    mesh_model: &fbxcel_dom::v7400::object::model::MeshHandle<'_>,
    file_axis: &glam::Mat4,
    unit_scale: f32,
    options: &FbxLoadOptions,
) -> FbxMeshChain {
    use fbxcel_dom::v7400::object::model::TypedModelHandle;
    use std::ops::Deref;

    let leaf_name = mesh_model
        .name()
        .map(std::borrow::ToOwned::to_owned)
        .unwrap_or_default();
    let (leaf_local, leaf_geometric) = extract_local_components(mesh_model.deref());

    let mut links = vec![decompose_link(&leaf_name, FbxNodeKind::Mesh, &leaf_local)];
    let (gs, gr, gt) = leaf_geometric.to_scale_rotation_translation();
    let (gex, gey, gez) = gr.to_euler(glam::EulerRot::XYZ);

    let mut cumulative = leaf_local;
    let mut current: Option<TypedModelHandle<'_>> = mesh_model.parent_model();
    let mut depth_guard = 0;
    while let Some(parent) = current {
        depth_guard += 1;
        if depth_guard > 256 {
            break;
        }
        let (kind, name, parent_local) = match &parent {
            TypedModelHandle::Mesh(m) => (
                FbxNodeKind::Mesh,
                m.name().unwrap_or("").to_string(),
                extract_local_components(m.deref()).0,
            ),
            TypedModelHandle::Null(n) => (
                FbxNodeKind::Null,
                n.name().unwrap_or("").to_string(),
                extract_local_components(n.deref()).0,
            ),
            TypedModelHandle::LimbNode(n) => (
                FbxNodeKind::LimbNode,
                n.name().unwrap_or("").to_string(),
                extract_local_components(n.deref()).0,
            ),
            TypedModelHandle::Light(l) => (
                FbxNodeKind::Light,
                l.name().unwrap_or("").to_string(),
                extract_local_components(l.deref()).0,
            ),
            TypedModelHandle::Camera(c) => (
                FbxNodeKind::Camera,
                c.name().unwrap_or("").to_string(),
                extract_local_components(c.deref()).0,
            ),
            _ => break,
        };
        links.push(decompose_link(&name, kind, &parent_local));
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

    // Same decision the loader makes, from the same helper, so the diagnostic
    // cannot drift from what was actually applied.
    let (effective_axis, axis_veto_fired) =
        effective_axis_transform(options.axis_policy, file_axis, &cumulative);

    FbxMeshChain {
        mesh_name: leaf_name,
        links,
        geometric_translation: gt,
        geometric_rotation_quat: gr,
        geometric_rotation_euler_deg: glam::Vec3::new(
            gex.to_degrees(),
            gey.to_degrees(),
            gez.to_degrees(),
        ),
        geometric_scale: gs,
        file_axis_transform: *file_axis,
        effective_axis_transform: effective_axis,
        axis_veto_fired,
        unit_scale,
    }
}

#[cfg(feature = "fbx")]
pub(super) fn decompose_link(name: &str, kind: FbxNodeKind, m: &glam::Mat4) -> FbxChainLink {
    let (s, r, t) = m.to_scale_rotation_translation();
    let (ex, ey, ez) = r.to_euler(glam::EulerRot::XYZ);
    FbxChainLink {
        node_name: name.to_string(),
        node_kind: kind,
        local_translation: t,
        local_rotation_quat: r,
        local_rotation_euler_deg: glam::Vec3::new(
            ex.to_degrees(),
            ey.to_degrees(),
            ez.to_degrees(),
        ),
        local_scale: s,
    }
}
