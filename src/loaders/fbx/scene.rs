//! The decode itself: walk the document, build meshes and materials, and attach
//! skins, blend shapes, and animation.

use crate::error::IoError;
use crate::types::{
    AttributeData, AttributeDomain, IoMaterial, IoMesh, IoScene, MaterialTextureSlot, MorphTarget,
    Skeleton, SkinWeights, SurfaceMesh, UvTransform,
};
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::Document;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::object::TypedObjectHandle;
use fbxcel_dom::v7400::object::model::TypedModelHandle;
use std::collections::HashMap;
use std::path::Path;

use super::animation::{extract_animations, extract_morph_weight_clips};
use super::blendshape::extract_blend_shapes;
use super::geometry::{
    compute_flat_normals, fan_triangulator, pick_second_uv_channel, pick_uv_channel,
};
use super::material::convert_material;
use super::options::FbxLoadOptions;
use super::raw::objects_in_stable_order;
use super::skin::extract_skin;
use super::transform::{extract_node_transform, get_axis_transform};

/// The shared FBX decode both entry points funnel through: parse the document from a
/// seekable byte reader, then convert its meshes / materials / skeletons / animations, so
/// the path and bytes paths never diverge.
#[cfg(feature = "fbx")]
pub(super) fn build_fbx_scene(
    bytes: &[u8],
    base: Option<&Path>,
    options: FbxLoadOptions,
) -> Result<IoScene, IoError> {
    let reader = std::io::Cursor::new(bytes);

    let document = match AnyDocument::from_seekable_reader(reader)
        .map_err(|error| IoError::Parse(format!("FBX load failed: {error:?}")))?
    {
        AnyDocument::V7400(_, document) => document,
        _ => {
            return Err(IoError::Parse(
                "unsupported FBX version (only binary FBX 7.4/7.5 supported)".into(),
            ));
        }
    };

    let parent_dir = base.unwrap_or(Path::new("."));
    let (axis_transform, unit_scale) = get_axis_transform(&document, options.axis_policy);

    let mut meshes = Vec::new();
    let mut materials: Vec<IoMaterial> = Vec::new();
    let mut skeletons: Vec<Skeleton> = Vec::new();
    let mut material_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    // A texture names the UV layer it samples; a mesh decides which of its
    // layers is UV0 and which is UV1. Neither knows about the other while it is
    // being read, so the names are collected from both sides and matched up
    // once every object has been walked.
    let mut pending_uv_sets: Vec<(usize, MaterialTextureSlot, String)> = Vec::new();
    let mut uv_set_indices: HashMap<String, u32> = HashMap::new();

    for object in objects_in_stable_order(&document) {
        if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh_model)) = object.get_typed() {
            let model_name = mesh_model
                .name()
                .map(std::borrow::ToOwned::to_owned)
                .unwrap_or_else(|| format!("fbx_mesh_{}", meshes.len()));

            let node_transform =
                extract_node_transform(&mesh_model, &axis_transform, unit_scale, &options);

            let model_materials: Vec<usize> = mesh_model
                .materials()
                .map(|material_object| {
                    let material_id = material_object.object_id().raw();
                    if let Some(&index) = material_map.get(&material_id) {
                        index
                    } else {
                        let converted = convert_material(&material_object, parent_dir);
                        let index = materials.len();
                        for (slot, uv_set_name) in converted.uv_set_names {
                            pending_uv_sets.push((index, slot, uv_set_name));
                        }
                        materials.push(converted.material);
                        material_map.insert(material_id, index);
                        index
                    }
                })
                .collect();

            let geometry = match mesh_model.geometry() {
                Ok(geometry) => geometry,
                Err(_) => continue,
            };

            let polygon_vertices = match geometry.polygon_vertices() {
                Ok(vertices) => vertices,
                Err(_) => continue,
            };

            let triangle_vertices = match polygon_vertices.triangulate_each(fan_triangulator) {
                Ok(vertices) => vertices,
                Err(_) => continue,
            };

            if triangle_vertices.is_empty() {
                continue;
            }

            let mut positions = Vec::with_capacity(triangle_vertices.len());
            let mut vertex_cp_index: Vec<u32> = Vec::with_capacity(triangle_vertices.len());
            let mut positions_ok = true;
            let mut max_cp_index: u32 = 0;
            for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                let cpi = triangle_vertices.control_point_index(triangle_vertex);
                match (cpi, triangle_vertices.control_point(triangle_vertex)) {
                    (Some(cpi), Some(point)) => {
                        positions.push([point.x as f32, point.y as f32, point.z as f32]);
                        let raw = cpi.to_u32();
                        vertex_cp_index.push(raw);
                        if raw > max_cp_index {
                            max_cp_index = raw;
                        }
                    }
                    _ => {
                        positions_ok = false;
                        break;
                    }
                }
            }
            if !positions_ok || positions.is_empty() {
                continue;
            }

            let mut normals_vec: Option<Vec<[f32; 3]>> = None;
            // Collect every UV channel we can decode. Picking the
            // right one happens after the loop: a single FBX can
            // expose UV0 as a wind / vertex-shader parameter
            // channel (V values up to ~70) and UV1 as the real
            // texture coordinates. Always taking the first channel
            // would map albedo samples onto the wind data and land
            // every fragment in whatever atlas region (0, 70 mod 1)
            // ends up at: typically the transparent corner.
            let mut uv_candidates: Vec<Vec<[f32; 2]>> = Vec::new();
            // Each candidate's layer name, parallel to `uv_candidates`. A
            // texture names the layer it samples by name, and a layer that turns
            // out not to be a UV set is emitted as a named attribute, so both
            // ends need it.
            let mut uv_layer_names: Vec<String> = Vec::new();
            let mut material_indices_per_vert: Option<Vec<usize>> = None;

            // Walk EVERY layer the geometry exposes. FBX commonly
            // splits secondary UV sets, vertex-painted normals, and
            // material assignments across distinct layers: the
            // HDRP tree convention puts the wind / animation UV in
            // layer 0 and the actual texture UV in layer 1. Picking
            // only `layers().next()` strands the texture channel and
            // every albedo sample lands at the wind coordinate
            // (V ~ 70).
            for layer in geometry.layers() {
                for entry in layer.layer_element_entries() {
                    match entry.typed_layer_element() {
                        Ok(TypedLayerElementHandle::Normal(normal_handle)) => {
                            if normals_vec.is_some() {
                                continue;
                            }
                            if let Ok(normals_data) = normal_handle.normals() {
                                let mut normals = Vec::with_capacity(positions.len());
                                let mut ok = true;
                                for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                                    match normals_data.normal(&triangle_vertices, triangle_vertex) {
                                        Ok(normal) => {
                                            normals.push([
                                                normal.x as f32,
                                                normal.y as f32,
                                                normal.z as f32,
                                            ]);
                                        }
                                        Err(_) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                                if ok && normals.len() == positions.len() {
                                    normals_vec = Some(normals);
                                }
                            }
                        }
                        Ok(TypedLayerElementHandle::Uv(uv_handle)) => {
                            if let Ok(uv_data) = uv_handle.uv() {
                                let mut uvs = Vec::with_capacity(positions.len());
                                let mut ok = true;
                                for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                                    match uv_data.uv(&triangle_vertices, triangle_vertex) {
                                        // The file's own coordinates, V unflipped.
                                        // A layer emitted as a UV set is flipped
                                        // once it is chosen; a layer that turns
                                        // out to be packed per-vertex data is
                                        // not a texture coordinate and must not
                                        // be flipped at all.
                                        Ok(uv) => uvs.push([uv.x as f32, uv.y as f32]),
                                        Err(_) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                                if ok && uvs.len() == positions.len() {
                                    uv_candidates.push(uvs);
                                    uv_layer_names.push(
                                        uv_handle
                                            .name()
                                            .map(std::borrow::ToOwned::to_owned)
                                            .unwrap_or_else(|_| {
                                                format!("uv{}", uv_candidates.len() - 1)
                                            }),
                                    );
                                }
                            }
                        }
                        Ok(TypedLayerElementHandle::Material(material_handle)) => {
                            if material_indices_per_vert.is_some() {
                                continue;
                            }
                            if let Ok(material_data) = material_handle.materials() {
                                let mut material_ids = Vec::new();
                                let mut ok = true;
                                for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                                    match material_data
                                        .material_index(&triangle_vertices, triangle_vertex)
                                    {
                                        Ok(index) => material_ids.push(index.to_u32() as usize),
                                        Err(_) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                                if ok {
                                    material_indices_per_vert = Some(material_ids);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            let normals = normals_vec.unwrap_or_else(|| compute_flat_normals(&positions));

            // Skin extraction: if this geometry has any deformers, build a
            // skeleton entry and per-vertex (joint_indices, joint_weights).
            let extracted_skin = extract_skin(
                &geometry,
                &document,
                (max_cp_index as usize).saturating_add(1),
                &axis_transform,
                unit_scale,
            );
            let (skin_per_vertex, skeleton_index): (Option<SkinWeights>, Option<usize>) =
                if let Some(skin) = extracted_skin {
                    let mut ji: Vec<[u8; 4]> = Vec::with_capacity(positions.len());
                    let mut jw: Vec<[f32; 4]> = Vec::with_capacity(positions.len());
                    for &cp in &vertex_cp_index {
                        let influences = skin
                            .cp_influences
                            .get(cp as usize)
                            .map(|v| v.as_slice())
                            .unwrap_or(&[]);
                        let mut idx = [0u8; 4];
                        let mut wt = [0f32; 4];
                        for (k, (i, w)) in influences.iter().take(4).enumerate() {
                            idx[k] = *i;
                            wt[k] = *w;
                        }
                        ji.push(idx);
                        jw.push(wt);
                    }
                    let sk_idx = skeletons.len();
                    skeletons.push(skin.skeleton);
                    (
                        Some(SkinWeights {
                            joint_indices: ji,
                            joint_weights: jw,
                        }),
                        Some(sk_idx),
                    )
                } else {
                    (None, None)
                };

            // Blend shapes: harvest per-control-point deltas, then expand to
            // one delta per emitted vertex through the same control-point map
            // the skin path uses, so a target lines up with `positions`. Split
            // across sub-meshes below exactly like positions / skin.
            let blend_shapes =
                extract_blend_shapes(&geometry, (max_cp_index as usize).saturating_add(1));
            let full_morph_targets: Vec<(String, Vec<[f32; 3]>)> = blend_shapes
                .iter()
                .map(|bs| {
                    let deltas: Vec<[f32; 3]> = vertex_cp_index
                        .iter()
                        .map(|&cp| bs.cp_deltas.get(cp as usize).copied().unwrap_or([0.0; 3]))
                        .collect();
                    (bs.name.clone(), deltas)
                })
                .collect();

            // Pick the UV channel the material's base map samples.
            //
            // The rule is Unity's, and it is deliberately not a
            // value heuristic: the base map samples the **first
            // authored UV layer** (UV0). UV1 is the lightmap set;
            // wind / vertex-animation data lives in later layers and
            // the shader reads it explicitly. Value ranges are never
            // consulted, so an atlas channel that bakes an integer
            // per-card offset into V (a foliage-wind convention: the
            // integer part is the card index / wind phase, the
            // fraction the real texture coordinate, recovered by
            // repeat-wrap sampling) still reads as the albedo channel
            // it is.
            //
            // So: take the first layer that is a genuine 2D texture
            // coordinate. The only disqualifier is a channel that is
            // not a texture coordinate at all: one axis held constant
            // across the mesh, the signature of a packed per-vertex
            // scalar, in which case advance to the next layer. This
            // preserves UV0-by-order (why an asset "just works" the way
            // it does in Unity) while stepping over a genuine data
            // channel that happens to sit in layer 0.
            //
            // `VIEWPORT_FBX_LOG_UV=1` dumps every candidate's range
            // and the chosen index so callers can sanity-check the
            // pick on packs with unusual UV layouts.
            let log_uv = std::env::var_os("VIEWPORT_FBX_LOG_UV").is_some();
            let summarise = |uvs: &[[f32; 2]]| -> (f32, f32, f32, f32) {
                let (mut umin, mut umax, mut vmin, mut vmax) = (
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                );
                for uv in uvs {
                    umin = umin.min(uv[0]);
                    umax = umax.max(uv[0]);
                    vmin = vmin.min(uv[1]);
                    vmax = vmax.max(uv[1]);
                }
                (umin, umax, vmin, vmax)
            };
            if log_uv && !uv_candidates.is_empty() {
                eprintln!(
                    "VIEWPORT_FBX_LOG_UV: model '{model_name}' ({} fbx material(s)): {} UV channel(s) found (file coordinates, V not yet flipped):",
                    model_materials.len(),
                    uv_candidates.len()
                );
                for (i, uvs) in uv_candidates.iter().enumerate() {
                    let (umin, umax, vmin, vmax) = summarise(uvs);
                    let max_abs = uvs
                        .iter()
                        .fold(0.0_f32, |m, uv| m.max(uv[0].abs()).max(uv[1].abs()));
                    let area = (umax - umin).max(0.0) * (vmax - vmin).max(0.0);
                    eprintln!(
                        "  [{i}] uv=[{:.2},{:.2}]x[{:.2},{:.2}] area={:.2} max_abs={:.2} n={}",
                        umin,
                        umax,
                        vmin,
                        vmax,
                        area,
                        max_abs,
                        uvs.len()
                    );
                }
            }
            // Diagnostic override: force a specific UV channel index for
            // every mesh, to A/B which channel carries the correct albedo
            // atlas mapping on foliage with several 0..1 UV sets.
            let uv_override = std::env::var("VIEWPORT_FBX_UV_CHANNEL")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&i| i < uv_candidates.len());
            let primary: Option<usize> = if let Some(idx) = uv_override {
                if log_uv {
                    eprintln!("VIEWPORT_FBX_LOG_UV: override forcing channel {idx}");
                }
                Some(idx)
            } else if uv_candidates.len() <= 1 {
                (!uv_candidates.is_empty()).then_some(0)
            } else {
                let idx = pick_uv_channel(&uv_candidates);
                if log_uv {
                    eprintln!(
                        "VIEWPORT_FBX_LOG_UV: picked channel {idx} (first non-degenerate UV set; UV0-preferred)"
                    );
                }
                Some(idx)
            };
            // The next authored texture-coordinate layer becomes the second UV
            // set rather than being dropped. Layers `pick_uv_channel` rejects as
            // packed per-vertex scalars stay rejected here: they are not texture
            // coordinates, and routing one into UV1 would only move the problem.
            let secondary = primary.and_then(|first| pick_second_uv_channel(&uv_candidates, first));
            if log_uv {
                match secondary {
                    Some(idx) => eprintln!("VIEWPORT_FBX_LOG_UV: second UV set from channel {idx}"),
                    None => eprintln!("VIEWPORT_FBX_LOG_UV: no second UV set"),
                }
            }
            // Which layer name ended up as which set, for the textures that
            // name one. Meshes sharing a material share their layer naming in
            // practice, so the first mesh to claim a name decides it.
            for (set, index) in [(0u32, primary), (1, secondary)] {
                if let Some(name) = index.and_then(|index| uv_layer_names.get(index)) {
                    uv_set_indices.entry(name.clone()).or_insert(set);
                }
            }

            let mut uv_candidates: Vec<Option<Vec<[f32; 2]>>> =
                uv_candidates.into_iter().map(Some).collect();
            // FBX stores UVs with the V origin at the bottom (the OpenGL / Maya
            // convention); the renderer samples with V at the top (wgpu),
            // matching the glTF loader which reads tex-coords unflipped. Flip V
            // on the sets emitted as texture coordinates so an FBX atlas lands
            // the same way round as a glTF one, rather than mirrored top to
            // bottom (a face's features on the wrong geometry).
            let take_uv_set = |candidates: &mut Vec<Option<Vec<[f32; 2]>>>, index: usize| {
                candidates.get_mut(index).and_then(Option::take).map(|uvs| {
                    uvs.into_iter()
                        .map(|uv| [uv[0], 1.0 - uv[1]])
                        .collect::<Vec<_>>()
                })
            };
            let uvs_vec: Option<Vec<[f32; 2]>> =
                primary.and_then(|idx| take_uv_set(&mut uv_candidates, idx));
            let uvs1_vec: Option<Vec<[f32; 2]>> =
                secondary.and_then(|idx| take_uv_set(&mut uv_candidates, idx));

            // Whatever is left is per-vertex data the file carries in a UV
            // layer without being a texture coordinate: the packed scalars
            // `is_texture_coordinate` rejects (a wind phase in V with U held
            // constant), and any third and later set, which the mesh type does
            // not carry. Emit each under its layer name rather than dropping it.
            // Values stay as authored, V unflipped: a packed scalar is not a
            // texture coordinate and flipping it would corrupt the value. The
            // third component is padding, since the neutral vector attribute is
            // three-wide and the source is two.
            let leftover_layers: Vec<(String, Vec<[f32; 3]>)> = uv_candidates
                .into_iter()
                .enumerate()
                .filter_map(|(index, layer)| {
                    let layer = layer?;
                    let name = uv_layer_names
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| format!("uv{index}"));
                    Some((
                        name,
                        layer.into_iter().map(|uv| [uv[0], uv[1], 0.0]).collect(),
                    ))
                })
                .collect();

            if let Some(ref material_per_vertex) = material_indices_per_vert {
                if !model_materials.is_empty() && material_per_vertex.iter().any(|&m| m != 0) {
                    let num_local_materials = model_materials.len();
                    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); num_local_materials];
                    for (vertex_index, &local_material) in material_per_vertex.iter().enumerate() {
                        let clamped = local_material.min(num_local_materials - 1);
                        groups[clamped].push(vertex_index);
                    }

                    let first_mesh_index = meshes.len();
                    for (local_material_index, vertex_indices) in groups.into_iter().enumerate() {
                        if vertex_indices.is_empty() {
                            continue;
                        }

                        let sub_positions = vertex_indices.iter().map(|&i| positions[i]).collect();
                        let sub_normals = vertex_indices.iter().map(|&i| normals[i]).collect();
                        let sub_uvs: Option<Vec<[f32; 2]>> = uvs_vec
                            .as_ref()
                            .map(|uvs| vertex_indices.iter().map(|&i| uvs[i]).collect());
                        let sub_uvs1: Option<Vec<[f32; 2]>> = uvs1_vec
                            .as_ref()
                            .map(|uvs| vertex_indices.iter().map(|&i| uvs[i]).collect());
                        // Per-submesh UV extent: reveals whether a submesh
                        // (e.g. a tree's leaf cards) samples a different
                        // atlas region than the rest, or collapses onto it.
                        if log_uv {
                            if let Some(ref su) = sub_uvs {
                                let (umin, umax, vmin, vmax) = summarise(su);
                                eprintln!(
                                    "VIEWPORT_FBX_LOG_UV:   submesh '{model_name}.mat{local_material_index}' (fbx material {:?}) uv=[{:.2},{:.2}]x[{:.2},{:.2}] verts={}",
                                    model_materials.get(local_material_index),
                                    umin,
                                    umax,
                                    vmin,
                                    vmax,
                                    su.len()
                                );
                            }
                        }
                        let sub_indices: Vec<u32> = (0..vertex_indices.len() as u32).collect();

                        let sub_skin = skin_per_vertex.as_ref().map(|sw| SkinWeights {
                            joint_indices: vertex_indices
                                .iter()
                                .map(|&i| sw.joint_indices[i])
                                .collect(),
                            joint_weights: vertex_indices
                                .iter()
                                .map(|&i| sw.joint_weights[i])
                                .collect(),
                        });

                        // Slice each blend shape onto this sub-mesh's vertices,
                        // the same remap as positions. Every sub-mesh carries
                        // the full named target list (zero deltas where a face
                        // shape does not touch a body sub-mesh), so a downstream
                        // merge lines the targets up by name.
                        let sub_morphs: Vec<MorphTarget> = full_morph_targets
                            .iter()
                            .map(|(name, deltas)| MorphTarget {
                                name: name.clone(),
                                position_deltas: vertex_indices
                                    .iter()
                                    .map(|&i| deltas[i])
                                    .collect(),
                                normal_deltas: None,
                                tangent_deltas: None,
                            })
                            .collect();

                        let mut mesh_data = SurfaceMesh::default();
                        mesh_data.positions = sub_positions;
                        mesh_data.normals = sub_normals;
                        mesh_data.indices = sub_indices;
                        mesh_data.uvs = sub_uvs;
                        mesh_data.uvs1 = sub_uvs1;
                        for (name, values) in leftover_layers.iter() {
                            mesh_data.attributes.insert(
                                name.clone(),
                                AttributeData::vectors(
                                    AttributeDomain::Point,
                                    vertex_indices.iter().map(|&i| values[i]).collect(),
                                ),
                            );
                        }
                        mesh_data.skin_weights = sub_skin;
                        mesh_data.morph_targets = sub_morphs;

                        let parent_index = if meshes.len() > first_mesh_index {
                            Some(first_mesh_index)
                        } else {
                            None
                        };

                        meshes.push(IoMesh {
                            name: format!("{model_name}.mat{local_material_index}"),
                            mesh: mesh_data,
                            material_index: model_materials.get(local_material_index).copied(),
                            transform: node_transform,
                            two_sided: false,
                            parent_index,
                            skeleton_index,
                            ..IoMesh::default()
                        });
                    }
                    continue;
                }
            }

            let mut mesh_data = SurfaceMesh::default();
            mesh_data.positions = positions;
            mesh_data.normals = normals;
            mesh_data.indices = (0..mesh_data.positions.len() as u32).collect();
            mesh_data.uvs = uvs_vec;
            mesh_data.uvs1 = uvs1_vec;
            for (name, values) in leftover_layers.iter() {
                mesh_data.attributes.insert(
                    name.clone(),
                    AttributeData::vectors(AttributeDomain::Point, values.clone()),
                );
            }
            mesh_data.skin_weights = skin_per_vertex;
            mesh_data.morph_targets = full_morph_targets
                .into_iter()
                .map(|(name, position_deltas)| MorphTarget {
                    name,
                    position_deltas,
                    normal_deltas: None,
                    tangent_deltas: None,
                })
                .collect();

            // Single-submesh path: log the whole mesh's UV extent so a
            // single-material atlas mesh (trunk + leaves in one material)
            // can be inspected alongside the multi-submesh case above.
            if log_uv {
                if let Some(ref u) = mesh_data.uvs {
                    let (umin, umax, vmin, vmax) = summarise(u);
                    eprintln!(
                        "VIEWPORT_FBX_LOG_UV:   submesh '{model_name}' (single, fbx material {:?}) uv=[{:.2},{:.2}]x[{:.2},{:.2}] verts={}",
                        model_materials.first(),
                        umin,
                        umax,
                        vmin,
                        vmax,
                        u.len()
                    );
                }
            }

            meshes.push(IoMesh {
                name: model_name,
                mesh: mesh_data,
                material_index: model_materials.first().copied(),
                transform: node_transform,
                two_sided: false,
                parent_index: None,
                skeleton_index,
                ..IoMesh::default()
            });
        }
    }

    assign_hierarchy(&document, &mut meshes);

    let animations = extract_animations(&document, &mut skeletons, &axis_transform, unit_scale);

    // Blend-shape weight animation: each emitted mesh's morph targets, in the
    // order `extract_blend_shapes` produced them, correlated to their FBX
    // `BlendShapeChannel` weight curves by name (the channel name is the target
    // name). Read straight off the assembled meshes, so a submesh split needs no
    // special handling.
    // Match each textured slot's UV-set name against the layer names the meshes
    // emitted. A name that matches nothing leaves the slot on UV0: the file
    // named a layer this decode did not carry, and UV0 is the set every
    // consumer has.
    for (material_index, slot, uv_set_name) in pending_uv_sets {
        let Some(&uv_set) = uv_set_indices.get(&uv_set_name) else {
            continue;
        };
        if uv_set == 0 {
            continue;
        }
        let slot = slot.index();
        let material = &mut materials[material_index];
        match &mut material.uv_transforms[slot] {
            Some(transform) => transform.uv_set = uv_set,
            none => {
                *none = Some(UvTransform {
                    uv_set,
                    ..UvTransform::IDENTITY
                })
            }
        }
    }

    let mesh_targets: Vec<(usize, Vec<String>)> = meshes
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.mesh.morph_targets.is_empty())
        .map(|(i, m)| {
            (
                i,
                m.mesh
                    .morph_targets
                    .iter()
                    .map(|t| t.name.clone())
                    .collect(),
            )
        })
        .collect();
    let morph_animations = extract_morph_weight_clips(&document, &mesh_targets);

    Ok(IoScene {
        meshes,
        materials,
        skeletons,
        animations,
        morph_animations,
        ..IoScene::default()
    })
}

pub(super) fn assign_hierarchy(document: &Document, meshes: &mut [IoMesh]) {
    let mut id_to_mesh_index: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();

    let mut mesh_cursor = 0;
    let model_ids: Vec<(i64, String)> = objects_in_stable_order(document)
        .into_iter()
        .filter_map(|object| {
            if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh)) = object.get_typed() {
                Some((
                    object.object_id().raw(),
                    mesh.name().unwrap_or("").to_string(),
                ))
            } else {
                None
            }
        })
        .collect();

    for (object_id, model_name) in &model_ids {
        if mesh_cursor >= meshes.len() {
            break;
        }
        id_to_mesh_index.insert(*object_id, mesh_cursor);
        let material_prefix = format!("{model_name}.mat");
        mesh_cursor += 1;
        while mesh_cursor < meshes.len() && meshes[mesh_cursor].name.starts_with(&material_prefix) {
            mesh_cursor += 1;
        }
    }

    mesh_cursor = 0;
    for object in objects_in_stable_order(document) {
        if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh_model)) = object.get_typed() {
            if mesh_cursor >= meshes.len() {
                break;
            }

            let my_index = mesh_cursor;
            if let Some(parent) = mesh_model.parent_model()
                && let TypedModelHandle::Mesh(parent_mesh) = parent
            {
                let parent_id = parent_mesh.object_id().raw();
                if let Some(&parent_index) = id_to_mesh_index.get(&parent_id)
                    && parent_index != my_index
                    && meshes[my_index].parent_index.is_none()
                {
                    meshes[my_index].parent_index = Some(parent_index);
                }
            }

            let model_name = mesh_model.name().unwrap_or("");
            let material_prefix = format!("{model_name}.mat");
            mesh_cursor += 1;
            while mesh_cursor < meshes.len()
                && meshes[mesh_cursor].name.starts_with(&material_prefix)
            {
                mesh_cursor += 1;
            }
        }
    }
}
