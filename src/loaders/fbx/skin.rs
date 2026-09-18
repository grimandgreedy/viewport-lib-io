//! Skin clusters and the skeletons they bind to: bone discovery, ordering, and
//! reconciling a mesh rig against the ones already collected.

use crate::types::{Joint, Skeleton};
use fbxcel_dom::v7400::Document;
use fbxcel_dom::v7400::object::TypedObjectHandle;
use std::collections::HashMap;

use super::raw::{lookup_object, parent_model_id, read_f64_array, read_i32_array, read_mat4};

/// Skin data harvested from one mesh's deformers. `cp_influences` is indexed
/// by control-point index; each entry holds up to four `(joint, weight)`
/// pairs sorted by descending weight and normalised to sum to 1.0.
#[cfg(feature = "fbx")]
pub(super) struct ExtractedSkin {
    pub(super) skeleton: Skeleton,
    pub(super) cp_influences: Vec<Vec<(u8, f32)>>,
}

#[cfg(feature = "fbx")]
pub(super) struct BoneInfo {
    pub(super) name: String,
    pub(super) parent_id: Option<i64>,
    pub(super) transform_link: glam::Mat4,
}

#[cfg(feature = "fbx")]
pub(super) struct ClusterEntry {
    pub(super) bone_id: i64,
    pub(super) indexes: Vec<i32>,
    pub(super) weights: Vec<f64>,
}

#[cfg(feature = "fbx")]
pub(super) fn extract_skin(
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    document: &Document,
    control_point_count: usize,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
) -> Option<ExtractedSkin> {
    let mut clusters: Vec<ClusterEntry> = Vec::new();
    let mut bones: HashMap<i64, BoneInfo> = HashMap::new();

    for skin in geometry.skins() {
        for cluster in skin.clusters() {
            // The bone is on the source side of the cluster: a Model
            // (LimbNode/Null). source_objects/destination_objects are inherited
            // via Deref from ClusterHandle -> ObjectHandle.
            let bone_id_and_name = cluster
                .source_objects()
                .filter(|c| c.label().is_none())
                .filter_map(|c| c.object_handle())
                .find_map(|obj| match obj.get_typed() {
                    TypedObjectHandle::Model(m) => match m {
                        fbxcel_dom::v7400::object::model::TypedModelHandle::LimbNode(_)
                        | fbxcel_dom::v7400::object::model::TypedModelHandle::Null(_) => {
                            let raw = obj.object_id().raw();
                            let name = obj.name().unwrap_or("").to_string();
                            Some((raw, name, parent_model_id(&obj)))
                        }
                        _ => None,
                    },
                    _ => None,
                });
            let Some((bone_id, bone_name, parent_id)) = bone_id_and_name else {
                continue;
            };

            let node = cluster.node();
            let indexes = read_i32_array(&node, "Indexes").unwrap_or_default();
            let weights = read_f64_array(&node, "Weights").unwrap_or_default();
            if indexes.is_empty() || weights.is_empty() {
                continue;
            }
            let transform_link = read_mat4(&node, "TransformLink").unwrap_or(glam::Mat4::IDENTITY);

            bones.entry(bone_id).or_insert(BoneInfo {
                name: bone_name,
                parent_id,
                transform_link,
            });

            clusters.push(ClusterEntry {
                bone_id,
                indexes,
                weights,
            });
        }
    }
    if clusters.is_empty() {
        return None;
    }

    // Walk parent chains and add any missing ancestor bones so the hierarchy
    // is complete even if a joint between two used bones has no cluster.
    let mut frontier: Vec<i64> = bones.keys().copied().collect();
    while let Some(id) = frontier.pop() {
        let Some(parent_id) = bones.get(&id).and_then(|b| b.parent_id) else {
            continue;
        };
        if bones.contains_key(&parent_id) {
            continue;
        }
        if let Some(obj) = lookup_object(document, parent_id) {
            bones.insert(
                parent_id,
                BoneInfo {
                    name: obj.name().unwrap_or("").to_string(),
                    parent_id: parent_model_id(&obj),
                    transform_link: glam::Mat4::IDENTITY,
                },
            );
            frontier.push(parent_id);
        }
    }

    // Topological order: parent index < child index.
    let id_order = topo_sort(&bones);
    let id_to_idx: HashMap<i64, u8> = id_order
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i as u8))
        .collect();

    let unit_scale_mat = glam::Mat4::from_scale(glam::Vec3::splat(unit_scale));
    // `axis_transform` already lands vertices in Z-up via `get_axis_transform`.
    // Joint inverse-binds compose with the same chain so they end up in the
    // same scene space as the skinned vertices: with `C = axis * unit_scale`,
    // `world' = C * TransformLink` and `inverse_bind' = TransformLink^-1 * C^-1`.
    //
    // NOTE: this is keyed to the FILE-level transform. Under
    // `AxisPolicy::HeuristicVeto` a mesh leaf can individually veto the axis
    // transform in `extract_node_transform`; a vetoed skinned leaf paired
    // with this converted skeleton is a pre-existing inconsistency that is
    // not addressed here.
    let joints: Vec<Joint> = id_order
        .iter()
        .map(|id| {
            let info = &bones[id];
            let world = *axis_transform * unit_scale_mat * info.transform_link;
            Joint {
                name: info.name.clone(),
                parent: info.parent_id.and_then(|pid| id_to_idx.get(&pid).copied()),
                inverse_bind: world.inverse(),
            }
        })
        .collect();
    let skeleton = Skeleton {
        name: String::new(),
        joints,
    };

    // Per-control-point influences, then top-4-select and normalise.
    let mut raw: Vec<Vec<(u8, f32)>> = vec![Vec::new(); control_point_count];
    for cluster in &clusters {
        let Some(&joint_idx) = id_to_idx.get(&cluster.bone_id) else {
            continue;
        };
        for (i, &cp_i32) in cluster.indexes.iter().enumerate() {
            let cp = cp_i32 as usize;
            if cp >= raw.len() {
                continue;
            }
            let w = *cluster.weights.get(i).unwrap_or(&0.0) as f32;
            if w == 0.0 {
                continue;
            }
            raw[cp].push((joint_idx, w));
        }
    }
    for entry in &mut raw {
        entry.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entry.truncate(4);
        let sum: f32 = entry.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            for (_, w) in entry.iter_mut() {
                *w /= sum;
            }
        }
    }

    Some(ExtractedSkin {
        skeleton,
        cp_influences: raw,
    })
}

#[cfg(feature = "fbx")]
pub(super) fn topo_sort(bones: &HashMap<i64, BoneInfo>) -> Vec<i64> {
    let mut visited: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut out: Vec<i64> = Vec::new();
    // Seed the traversal in a stable order: `bones` is a `HashMap`, so iterating its keys directly
    // would make the joint order (and thus every joint index the skin weights reference) vary run
    // to run. Sorting the seeds by FBX object id makes the output deterministic while the DFS still
    // emits each parent before its children.
    let mut all_ids: Vec<i64> = bones.keys().copied().collect();
    all_ids.sort_unstable();

    fn visit(
        id: i64,
        bones: &HashMap<i64, BoneInfo>,
        visited: &mut std::collections::HashSet<i64>,
        out: &mut Vec<i64>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if let Some(p) = bones.get(&id).and_then(|b| b.parent_id)
            && bones.contains_key(&p)
        {
            visit(p, bones, visited, out);
        }
        out.push(id);
    }
    for id in all_ids {
        visit(id, bones, &mut visited, &mut out);
    }
    out
}

#[cfg(feature = "fbx")]
pub(super) fn find_or_insert_rig(skeletons: &mut Vec<Skeleton>, rig: Skeleton) -> usize {
    let rig_names: std::collections::HashSet<&str> =
        rig.joints.iter().map(|j| j.name.as_str()).collect();
    let rig_root = rig
        .joints
        .iter()
        .find(|j| j.parent.is_none())
        .map(|j| j.name.as_str());
    for (i, sk) in skeletons.iter().enumerate() {
        let sk_names: std::collections::HashSet<&str> =
            sk.joints.iter().map(|j| j.name.as_str()).collect();
        // Reuse an existing skeleton when it and this rig are the same armature,
        // i.e. one bone-name set nests inside the other and they share a root.
        //
        // The animation rig is built from *every* limb node, but a skinned mesh
        // usually binds only a SUBSET of those bones (no twist / finger / helper
        // bones), so the rig is a superset of the skin's skeleton. The rig
        // carries only placeholder identity binds, whereas the skin skeleton
        // carries the real cluster inverse-binds, so reusing the skin skeleton
        // (and retargeting the clip's name-keyed tracks onto it) is what lets a
        // clip actually drive the skinned mesh, instead of stranding the
        // animation on a second, meshless skeleton. `rig is a subset of skin` is the original
        // fully-covered case; `skin is a subset of rig` is the common subset case.
        let sk_root = sk
            .joints
            .iter()
            .find(|j| j.parent.is_none())
            .map(|j| j.name.as_str());
        let nested = rig_names.iter().all(|n| sk_names.contains(n))
            || sk_names.iter().all(|n| rig_names.contains(n));
        if nested && rig_root == sk_root && rig_root.is_some() {
            return i;
        }
    }
    let idx = skeletons.len();
    skeletons.push(rig);
    idx
}

/// Build a Skeleton from every LimbNode/Null model in the document, returning
/// the skeleton and a model-id -> joint-index map.
///
/// `conv` is the file-level conversion `C = axis_transform * unit_scale`.
/// This rig has no cluster `TransformLink` data, so each joint's FBX-space
/// bind world is taken as identity; converting all joint worlds to
/// `W' = C * W` therefore gives `inverse_bind' = (C * I)^-1 = C^-1`. This
/// keeps the fallback rig consistent with animation curves whose root-joint
/// locals are premultiplied by C in `extract_animations`.
#[cfg(feature = "fbx")]
pub(super) fn build_rig_from_limbs(
    document: &Document,
    conv: &glam::Mat4,
) -> (Skeleton, HashMap<i64, u8>) {
    use fbxcel_dom::v7400::object::model::TypedModelHandle as M;

    let mut bones: HashMap<i64, BoneInfo> = HashMap::new();
    for obj in document.objects() {
        let TypedObjectHandle::Model(m) = obj.get_typed() else {
            continue;
        };
        let is_bone = matches!(m, M::LimbNode(_) | M::Null(_));
        if !is_bone {
            continue;
        }
        let id = obj.object_id().raw();
        let name = obj.name().unwrap_or("").to_string();
        let parent = parent_model_id(&obj);
        bones.insert(
            id,
            BoneInfo {
                name,
                parent_id: parent,
                transform_link: glam::Mat4::IDENTITY,
            },
        );
    }
    if bones.is_empty() {
        return (Skeleton::default(), HashMap::new());
    }

    let id_order = topo_sort(&bones);
    let id_to_idx: HashMap<i64, u8> = id_order
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i as u8))
        .collect();

    let conv_inv = conv.inverse();
    let joints: Vec<Joint> = id_order
        .iter()
        .map(|id| {
            let info = &bones[id];
            Joint {
                name: info.name.clone(),
                parent: info.parent_id.and_then(|pid| id_to_idx.get(&pid).copied()),
                // Identity FBX bind world, converted: `inverse_bind * C^-1`.
                inverse_bind: conv_inv,
            }
        })
        .collect();
    (
        Skeleton {
            name: String::new(),
            joints,
        },
        id_to_idx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skel(bones: &[(&str, Option<u8>)]) -> Skeleton {
        Skeleton {
            name: String::new(),
            joints: bones
                .iter()
                .map(|(n, p)| Joint {
                    name: n.to_string(),
                    parent: *p,
                    inverse_bind: glam::Mat4::IDENTITY,
                })
                .collect(),
        }
    }

    /// A skinned mesh usually binds only a subset of the rig's bones. The rig
    /// (superset, same root) must reuse the skin's skeleton so the clip lands on
    /// the skeleton that actually skins the mesh, not a second, meshless one.
    #[test]
    fn rig_reuses_skin_skeleton_when_skin_is_a_subset() {
        // Skin skeleton: root + 2 bones (no finger/twist bones).
        let mut skeletons = vec![skel(&[
            ("Armature", None),
            ("Hips", Some(0)),
            ("Spine", Some(1)),
        ])];
        // Rig from all limbs: same root, plus extra helper bones.
        let rig = skel(&[
            ("Armature", None),
            ("Hips", Some(0)),
            ("Spine", Some(1)),
            ("Finger", Some(2)),
            ("Twist", Some(1)),
        ]);
        let idx = find_or_insert_rig(&mut skeletons, rig);
        assert_eq!(idx, 0, "should reuse the skin skeleton");
        assert_eq!(skeletons.len(), 1, "no second skeleton appended");
    }

    /// The original fully-covered case (skin covers rig) still reuses.
    #[test]
    fn rig_reuses_when_skin_covers_rig() {
        let mut skeletons = vec![skel(&[
            ("Armature", None),
            ("Hips", Some(0)),
            ("Spine", Some(1)),
        ])];
        let rig = skel(&[("Armature", None), ("Hips", Some(0))]);
        assert_eq!(find_or_insert_rig(&mut skeletons, rig), 0);
        assert_eq!(skeletons.len(), 1);
    }

    /// The joint order must not depend on `HashMap` iteration order: the skin weights reference
    /// joints by index, so a run-to-run reshuffle would silently mis-bind vertices (and, in a
    /// multi-submesh rig, non-deterministically pick which bones collapse). `topo_sort` must return
    /// the same order every time, with every parent still preceding its children.
    #[test]
    fn topo_sort_is_deterministic_and_orders_parents_first() {
        let make = || {
            let mut m: HashMap<i64, BoneInfo> = HashMap::new();
            // Ids deliberately unrelated to hierarchy depth, to expose any key-order dependence.
            for (id, parent) in [
                (50, None),
                (10, Some(50)),
                (30, Some(10)),
                (20, Some(50)),
                (40, Some(30)),
            ] {
                m.insert(
                    id,
                    BoneInfo {
                        name: format!("b{id}"),
                        parent_id: parent,
                        transform_link: glam::Mat4::IDENTITY,
                    },
                );
            }
            m
        };

        let first = topo_sort(&make());
        for _ in 0..32 {
            assert_eq!(topo_sort(&make()), first, "topo_sort order is not stable");
        }
        // Sorted-by-id seeding gives a fixed, reproducible order (root pulled ahead of its children
        // as each seed resolves its parent chain first).
        assert_eq!(first, vec![50, 10, 20, 30, 40]);

        let pos: HashMap<i64, usize> = first.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        for (&id, info) in &make() {
            if let Some(p) = info.parent_id {
                assert!(pos[&p] < pos[&id], "parent {p} ordered after child {id}");
            }
        }
    }

    /// A genuinely different armature (different root) is appended, not merged.
    #[test]
    fn unrelated_rig_is_appended() {
        let mut skeletons = vec![skel(&[("RigA", None), ("Bone", Some(0))])];
        let rig = skel(&[("RigB", None), ("Other", Some(0))]);
        assert_eq!(find_or_insert_rig(&mut skeletons, rig), 1);
        assert_eq!(skeletons.len(), 2);
    }
}
