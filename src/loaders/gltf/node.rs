//! Scene-graph walk: nodes to meshes, with their world transforms.

use crate::types::{IoMesh, SkinWeights};

use super::primitive::convert_primitive;
use super::skin::JointLookup;

/// Walk one node and its children, appending a mesh for each primitive.
///
/// `parent_world` accumulates down the tree, so each mesh comes out with its
/// world transform already composed. A node that is itself a joint becomes the
/// inherited joint for everything below it, which is how a mesh parented to a
/// bone rides that bone without being skinned.
pub(super) fn collect_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_world: glam::Mat4,
    parent_mesh_index: Option<usize>,
    joint_lookup: &JointLookup,
    parent_joint: Option<(usize, usize)>,
    out: &mut Vec<IoMesh>,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;

    // If this node is itself a joint, it overrides the inherited joint for
    // any descendant meshes. Otherwise we keep walking down with the parent's
    // joint context (so a mesh four nodes below a bone still gets attached
    // to that bone for rigid bone-parented rigs).
    let my_joint = joint_lookup.get(&node.index()).copied().or(parent_joint);

    let my_first_index = out.len();
    let mut this_node_has_mesh = false;

    if let Some(mesh) = node.mesh() {
        // glTF binds a skin to the node that references a mesh, so a real
        // skinned mesh has the skin reference on its own node. `skin indices
        // are global` and `convert_skeletons` builds skeletons in glTF skin
        // order, so we can use the glTF index directly.
        let explicit_skeleton = node.skin().map(|s| s.index());

        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if let Some(mut imported) =
                convert_primitive(&primitive, buffers, &mesh, primitive_index)
            {
                if explicit_skeleton.is_some() {
                    // Real glTF skinning: JOINTS_0 / WEIGHTS_0 already
                    // populated on the primitive. Keep the mesh transform as
                    // its scene-graph world matrix.
                    imported.skeleton_index = explicit_skeleton;
                    imported.transform = world;
                } else if let Some((skeleton_idx, joint_idx)) = my_joint {
                    // Rigid bone-parented mesh: synthesize 100%-weight
                    // skinning to the nearest joint ancestor. Bake the mesh's
                    // current world transform into the vertex data so the
                    // skinned output at bind pose equals the original world
                    // position.
                    rigidly_attach_to_joint(&mut imported, world, skeleton_idx, joint_idx);
                } else {
                    imported.transform = world;
                }

                if !this_node_has_mesh {
                    imported.parent_index = parent_mesh_index;
                    this_node_has_mesh = true;
                } else {
                    imported.parent_index = Some(my_first_index);
                }
                out.push(imported);
            }
        }
    }

    let child_parent = if this_node_has_mesh {
        Some(my_first_index)
    } else {
        parent_mesh_index
    };

    for child in node.children() {
        collect_node(
            &child,
            buffers,
            world,
            child_parent,
            joint_lookup,
            my_joint,
            out,
        );
    }
}

/// Attach every vertex of `imported` rigidly to a single joint. Used for
/// glTF rigs that animate by bone-parenting meshes rather than per-vertex
/// skinning (no JOINTS_0/WEIGHTS_0, the mesh node is a descendant of a joint
/// in the scene graph).
///
/// The mesh's bind-world transform is baked into positions and normals, the
/// scene transform is set to identity, and synthesized skin weights put
/// 100% influence on `joint_index`. At bind pose the skinning matrix is
/// identity so the rendered position equals the original world position; as
/// the joint animates, the mesh follows it.
pub(super) fn rigidly_attach_to_joint(
    imported: &mut IoMesh,
    world: glam::Mat4,
    skeleton_index: usize,
    joint_index: usize,
) {
    let world3 = glam::Mat3::from_mat4(world);
    // Normal matrix: transpose of inverse of the upper-left 3x3. For uniform
    // scale this is rotation only; for non-uniform scale it accounts for the
    // skew. If the matrix is singular (degenerate scale), fall back to the
    // raw upper-left and let `normalize_or_zero` clean up.
    let normal_mat = if world3.determinant().abs() > 1e-8 {
        world3.inverse().transpose()
    } else {
        world3
    };

    for p in &mut imported.mesh.positions {
        let v = world.transform_point3(glam::Vec3::from(*p));
        *p = v.to_array();
    }
    for n in &mut imported.mesh.normals {
        let v = normal_mat * glam::Vec3::from(*n);
        *n = v.normalize_or_zero().to_array();
    }

    let count = imported.mesh.positions.len();
    imported.mesh.skin_weights = Some(SkinWeights {
        joint_indices: vec![[joint_index as u8, 0, 0, 0]; count],
        joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; count],
    });
    imported.skeleton_index = Some(skeleton_index);
    imported.transform = glam::Mat4::IDENTITY;
}

/// Depth-first preorder over a child list, appending node indices to `order`.
///
/// Joints have to be emitted parents-first so a joint's parent index always
/// points at a joint already in the skeleton.
pub(super) fn dfs_preorder(
    node: usize,
    children: &[Vec<usize>],
    order: &mut Vec<usize>,
    visited: &mut [bool],
) {
    if visited[node] {
        return;
    }
    visited[node] = true;
    order.push(node);
    for &c in &children[node] {
        dfs_preorder(c, children, order, visited);
    }
}
