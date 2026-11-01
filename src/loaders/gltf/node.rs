//! Scene-graph walk: nodes to meshes, with their world transforms, plus the
//! node table and the punctual lights placed on nodes.

use gltf::json::validation::Checked;

use crate::error::IoError;
use crate::types::{IoMesh, IoNode, SceneLight, SceneLightKind, SkinWeights};

use super::axis::reorient_affine_mat4;
use super::primitive::convert_primitive;
use super::skin::JointLookup;

/// Walk one node and its children, appending a mesh for each primitive.
///
/// `parent_world` accumulates down the tree, so each mesh comes out with its
/// world transform already composed. A node that is itself a joint becomes the
/// inherited joint for everything below it, which is how a mesh parented to a
/// bone rides that bone without being skinned. Each mesh's index is recorded
/// against its node in `nodes`.
#[allow(clippy::too_many_arguments)]
pub(super) fn collect_node(
    document: &gltf::Document,
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_world: glam::Mat4,
    parent_mesh_index: Option<usize>,
    joint_lookup: &JointLookup,
    parent_joint: Option<(usize, usize)>,
    out: &mut Vec<IoMesh>,
    nodes: &mut [IoNode],
) -> Result<(), IoError> {
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
                convert_primitive(document, &primitive, buffers, &mesh, primitive_index)?
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
                if let Some(entry) = nodes.get_mut(node.index()) {
                    entry.meshes.push(out.len());
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
            document,
            &child,
            buffers,
            world,
            child_parent,
            joint_lookup,
            my_joint,
            out,
            nodes,
        )?;
    }
    Ok(())
}

/// Every glTF node in document order, with its parent and its local transform
/// reoriented to Z-up. Mesh indices are filled in by [`collect_node`].
pub(super) fn convert_nodes(document: &gltf::Document) -> Vec<IoNode> {
    let mut nodes: Vec<IoNode> = document
        .nodes()
        .map(|node| IoNode {
            name: node.name().map(std::borrow::ToOwned::to_owned),
            parent: None,
            transform: reorient_affine_mat4(glam::Mat4::from_cols_array_2d(
                &node.transform().matrix(),
            )),
            meshes: Vec::new(),
        })
        .collect();
    for node in document.nodes() {
        for child in node.children() {
            if let Some(entry) = nodes.get_mut(child.index()) {
                entry.parent = Some(node.index());
            }
        }
    }
    nodes
}

/// World transform of node `index`: its chain of local transforms composed
/// root first. A parent cycle in a malformed file stops after one pass over
/// the node count rather than looping.
fn node_world(nodes: &[IoNode], index: usize) -> glam::Mat4 {
    let mut world = glam::Mat4::IDENTITY;
    let mut current = Some(index);
    let mut steps = 0;
    while let Some(i) = current {
        if steps > nodes.len() {
            break;
        }
        let Some(node) = nodes.get(i) else {
            break;
        };
        world = node.transform * world;
        current = node.parent;
        steps += 1;
    }
    world
}

/// The `KHR_lights_punctual` lights placed on nodes reachable from a scene,
/// one per node that names a light.
///
/// Read from the JSON rather than through `gltf::Node::light`, which panics on a
/// light index or type the file gets wrong; the document is not validated on
/// load, so such a light is skipped instead.
pub(super) fn convert_lights(document: &gltf::Document, nodes: &[IoNode]) -> Vec<SceneLight> {
    let json = document.as_json();
    let Some(definitions) = json
        .extensions
        .as_ref()
        .and_then(|extensions| extensions.khr_lights_punctual.as_ref())
    else {
        return Vec::new();
    };

    let mut reachable = vec![false; nodes.len()];
    let mut stack: Vec<gltf::Node> = document.scenes().flat_map(|scene| scene.nodes()).collect();
    while let Some(node) = stack.pop() {
        match reachable.get_mut(node.index()) {
            Some(seen) if !*seen => *seen = true,
            _ => continue,
        }
        stack.extend(node.children());
    }

    let mut lights = Vec::new();
    for (index, node) in json.nodes.iter().enumerate() {
        if !reachable.get(index).copied().unwrap_or(false) {
            continue;
        }
        let Some(light_index) = node
            .extensions
            .as_ref()
            .and_then(|extensions| extensions.khr_lights_punctual.as_ref())
            .map(|reference| reference.light.value())
        else {
            continue;
        };
        let Some(definition) = definitions.lights.get(light_index) else {
            continue;
        };
        let Some(kind) = light_kind(definition) else {
            continue;
        };
        lights.push(SceneLight {
            name: definition.name.clone(),
            kind,
            colour: definition.color,
            intensity: definition.intensity,
            world_transform: node_world(nodes, index),
            node: Some(index),
        });
    }
    lights
}

/// The light's kind and parameters, or `None` for an unknown type. A spot light
/// missing its `spot` object takes glTF's cone defaults.
fn light_kind(
    definition: &gltf::json::extensions::scene::khr_lights_punctual::Light,
) -> Option<SceneLightKind> {
    use gltf::json::extensions::scene::khr_lights_punctual::Type;
    let range = definition.range;
    Some(match definition.type_ {
        Checked::Valid(Type::Directional) => SceneLightKind::Directional,
        Checked::Valid(Type::Point) => SceneLightKind::Point { range },
        Checked::Valid(Type::Spot) => {
            let (inner_angle, outer_angle) = definition
                .spot
                .as_ref()
                .map_or((0.0, std::f32::consts::FRAC_PI_4), |spot| {
                    (spot.inner_cone_angle, spot.outer_cone_angle)
                });
            SceneLightKind::Spot {
                range,
                inner_angle,
                outer_angle,
            }
        }
        Checked::Invalid => return None,
    })
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
    imported.skin_weights = Some(SkinWeights {
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

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::types::SceneLightKind;
    use viewport_lib_types::effects::lighting::LightKind;

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    // Nodes:
    //   0: directional light, identity transform
    //   1: point light without range, translated (1, 2, 3), child of node 5
    //   2: spot light with range, rotated -90 degrees about X (aimed straight
    //      down in glTF's Y-up frame)
    //   3: spot light without range and an empty `spot` object
    //   4: names a light index that does not exist
    //   5: parent of node 1, translated (0, 1, 0)
    //   6: carries a light but is in no scene
    const LIGHTS: &str = r#"{
  "asset": { "version": "2.0" },
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        { "name": "sun", "type": "directional", "color": [1.0, 0.5, 0.25], "intensity": 3.5 },
        { "name": "bulb", "type": "point", "intensity": 100.0 },
        { "type": "spot", "range": 10.0, "intensity": 50.0,
          "spot": { "innerConeAngle": 0.1, "outerConeAngle": 0.5 } },
        { "type": "spot", "spot": {} }
      ]
    }
  },
  "scene": 0,
  "scenes": [{ "nodes": [0, 2, 3, 4, 5] }],
  "nodes": [
    { "name": "sun_node", "extensions": { "KHR_lights_punctual": { "light": 0 } } },
    { "translation": [1.0, 2.0, 3.0], "extensions": { "KHR_lights_punctual": { "light": 1 } } },
    { "rotation": [-0.70710677, 0.0, 0.0, 0.70710677],
      "extensions": { "KHR_lights_punctual": { "light": 2 } } },
    { "extensions": { "KHR_lights_punctual": { "light": 3 } } },
    { "extensions": { "KHR_lights_punctual": { "light": 9 } } },
    { "translation": [0.0, 1.0, 0.0], "children": [1] },
    { "extensions": { "KHR_lights_punctual": { "light": 0 } } }
  ]
}"#;

    #[test]
    fn reads_punctual_lights_in_z_up() {
        let scene = scene_from_slice(LIGHTS.as_bytes(), None).unwrap();
        // Node 4's dangling index and node 6 outside every scene are skipped.
        assert_eq!(scene.lights.len(), 4);
        assert_eq!(scene.nodes.len(), 7);
        assert_eq!(scene.nodes[1].parent, Some(5));
        let by_node = |n: usize| scene.lights.iter().find(|l| l.node == Some(n)).unwrap();

        // Directional, identity node. glTF lights shine down their local -Z.
        // The Y-up to Z-up rotation maps (x, y, z) to (x, -z, y), so glTF -Z
        // = (0, 0, -1) lands on (0, 1, 0): the light shines along +Y.
        let sun = by_node(0);
        assert_eq!(sun.name.as_deref(), Some("sun"));
        assert_eq!(sun.kind, SceneLightKind::Directional);
        assert!(close3(sun.direction(), [0.0, 1.0, 0.0]));
        assert_eq!(sun.colour, [1.0, 0.5, 0.25]);
        assert_eq!(sun.intensity, 3.5);
        let source = sun.to_light_source(25.0);
        assert_eq!(source.intensity, 3.5);
        match source.kind {
            // The renderer takes the surface-to-light direction.
            LightKind::Directional { direction } => {
                assert!(close3(direction, [0.0, -1.0, 0.0]));
            }
            other => panic!("expected a directional light, got {other:?}"),
        }

        // Point light without range, under a parent: glTF world translation
        // (1, 2, 3) + (0, 1, 0) = (1, 3, 3), which is (1, -3, 3) in Z-up.
        let bulb = by_node(1);
        assert_eq!(bulb.kind, SceneLightKind::Point { range: None });
        assert_eq!(bulb.colour, [1.0, 1.0, 1.0]);
        assert_eq!(bulb.intensity, 100.0);
        assert!(close3(bulb.position(), [1.0, -3.0, 3.0]));
        match bulb.to_light_source(25.0).kind {
            LightKind::Point {
                position, range, ..
            } => {
                assert!(close3(position.as_vec3().to_array(), [1.0, -3.0, 3.0]));
                assert_eq!(range, 25.0, "an unset range takes the caller's value");
            }
            other => panic!("expected a point light, got {other:?}"),
        }

        // Spot with range, rotated -90 degrees about X: it shines down glTF -Y,
        // and glTF -Y is Z-up -Z, so it points straight down.
        let spot = by_node(2);
        assert_eq!(
            spot.kind,
            SceneLightKind::Spot {
                range: Some(10.0),
                inner_angle: 0.1,
                outer_angle: 0.5,
            }
        );
        assert!(close3(spot.direction(), [0.0, 0.0, -1.0]));
        match spot.to_light_source(25.0).kind {
            LightKind::Spot {
                direction,
                range,
                inner_angle,
                outer_angle,
                ..
            } => {
                assert!(close3(direction, [0.0, 0.0, -1.0]));
                assert_eq!(range, 10.0);
                assert_eq!((inner_angle, outer_angle), (0.1, 0.5));
            }
            other => panic!("expected a spot light, got {other:?}"),
        }

        // Spot without range and an empty `spot`: glTF's cone defaults, inner 0
        // and outer pi / 4, and glTF's default intensity of 1.
        let bare = by_node(3);
        assert_eq!(
            bare.kind,
            SceneLightKind::Spot {
                range: None,
                inner_angle: 0.0,
                outer_angle: std::f32::consts::FRAC_PI_4,
            }
        );
        assert_eq!(bare.intensity, 1.0);
    }

    #[test]
    fn a_file_without_lights_has_none() {
        let json = r#"{
  "asset": { "version": "2.0" },
  "scene": 0,
  "scenes": [{ "nodes": [0] }],
  "nodes": [{ "name": "empty" }]
}"#;
        let scene = scene_from_slice(json.as_bytes(), None).unwrap();
        assert!(scene.lights.is_empty());
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.nodes[0].name.as_deref(), Some("empty"));
    }
}
