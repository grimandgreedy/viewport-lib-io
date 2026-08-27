use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::Error as AnyhowError;
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::fbxcel;
use fbxcel_dom::v7400::Document;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::data::mesh::{PolygonVertexIndex, PolygonVertices};
use fbxcel_dom::v7400::object::TypedObjectHandle;
use fbxcel_dom::v7400::object::model::TypedModelHandle;

use crate::error::IoError;
use crate::types::{
    AlphaMode, AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler,
    AnimationTrack, AnimationTrackValues, IoMaterial, IoMesh, IoScene, Joint, MorphTarget,
    MorphWeightClip, Skeleton, SkinWeights, SurfaceMesh, TextureData, TextureSource,
};

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
    /// Apply `+90°` about X by default, with a per-leaf signed-veto check
    /// in [`extract_node_transform`]: if the cumulative parent chain
    /// already maps +Y to +Z, suppress the axis transform for that leaf.
    /// Best for mixed asset packs where some meshes bake the conversion
    /// and others don't. This is the default.
    HeuristicVeto,
    /// Honour `GlobalSettings.UpAxis` literally. `UpAxis = Y` applies
    /// `+90°` about X; `UpAxis = Z` applies identity; no per-leaf veto.
    /// Use for files you trust to declare their orientation correctly.
    HonourHeader,
    /// Always apply `+90°` about X regardless of header or chain. Use
    /// when you know raw vertices are Y-up.
    ForceYUpRaw,
    /// Never apply any axis transform. Use when you know raw vertices
    /// are already Z-up.
    PassThrough,
}

impl Default for AxisPolicy {
    fn default() -> Self {
        Self::HeuristicVeto
    }
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
    /// transform runs first (per-vertex source → consumer conversion)
    /// and the cumulative chain then places the result in the consumer
    /// frame. Use when the artist authored the prop in a non-display
    /// pose and uses metadata rotations to right it.
    PostAxis,
}

impl Default for CumulativeOrder {
    fn default() -> Self {
        Self::PreAxis
    }
}

/// Per-call overrides for [`scene_from_path_with_options`].
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

/// Kind of FBX node visited during the parent-chain walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FbxNodeKind {
    /// A Mesh node (the leaf is always one of these).
    Mesh,
    /// A Null node — Unity-exported FBX stacks these as LODGroup
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
/// where `cumulative = leaf_local · parent_1_local · ... · root_local`.
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
    /// File-scoped axis transform (`+90°` X, identity, etc.) before any
    /// per-leaf veto.
    pub file_axis_transform: glam::Mat4,
    /// Axis transform actually applied to this leaf after the veto check.
    /// Equal to `file_axis_transform` if no veto fired; identity if it did.
    pub effective_axis_transform: glam::Mat4,
    /// `true` if [`AxisPolicy::HeuristicVeto`] suppressed the file-level
    /// axis transform for this leaf.
    pub axis_veto_fired: bool,
    /// Unit scale derived from `GlobalSettings.UnitScaleFactor / 100`.
    pub unit_scale: f32,
}

/// Walk an FBX file's hierarchy and return a per-link breakdown of every
/// Mesh node's parent chain.
///
/// Output mirrors what [`scene_from_path_with_options`] does internally
/// when computing each `IoMesh::transform`, but decomposed link by link
/// rather than fused into a single matrix. Use this when you need to see
/// what the loader is doing — e.g. to diagnose a 90° rotation that's
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
        for object in document.objects() {
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
fn build_chain_breakdown(
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

    let (effective_axis, axis_veto_fired) = match options.axis_policy {
        AxisPolicy::HeuristicVeto => {
            let cum_y = cumulative.transform_vector3(glam::Vec3::Y);
            let veto = cum_y.z > 0.9 && cum_y.y.abs() < 0.5;
            if veto {
                (glam::Mat4::IDENTITY, true)
            } else {
                (*file_axis, false)
            }
        }
        AxisPolicy::HonourHeader | AxisPolicy::ForceYUpRaw | AxisPolicy::PassThrough => {
            (*file_axis, false)
        }
    };

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
fn decompose_link(name: &str, kind: FbxNodeKind, m: &glam::Mat4) -> FbxChainLink {
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

/// Decode an FBX file into a CPU-side scene using default load options.
///
/// Equivalent to [`scene_from_path_with_options`] with
/// `FbxLoadOptions::default()`.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    scene_from_path_with_options(path, FbxLoadOptions::default())
}

/// Decode an FBX file into a CPU-side scene with caller-supplied options.
///
/// See [`FbxLoadOptions`] for what's tunable and why you might want to
/// reach for it.
pub fn scene_from_path_with_options(
    path: &Path,
    options: FbxLoadOptions,
) -> Result<IoScene, IoError> {
    #[cfg(feature = "fbx")]
    {
        let bytes = std::fs::read(path)?;
        build_fbx_scene(&bytes, path.parent(), options)
    }

    #[cfg(not(feature = "fbx"))]
    {
        let _ = (path, options);
        Err(IoError::MissingFeature {
            feature: "fbx",
            context: "FBX scene decoding",
        })
    }
}

/// Decode a binary FBX (7.4/7.5) scene from in-memory bytes. The in-memory sibling of
/// [`scene_from_path`], for a mesh served from a cooked bundle (or fetched over the
/// network for a `wasm32` target) rather than read from a file. Self-contained meshes
/// decode with no base directory; see [`scene_from_bytes_with_options`] to resolve
/// external texture references.
pub fn scene_from_bytes(bytes: &[u8]) -> Result<IoScene, IoError> {
    scene_from_bytes_with_options(bytes, None, FbxLoadOptions::default())
}

/// Decode a binary FBX scene from in-memory bytes with per-call overrides. `base`
/// resolves any external texture paths the FBX references; pass `None` when textures are
/// supplied out of band (as DRAKE does, keying them as separate assets). The in-memory
/// sibling of [`scene_from_path_with_options`].
pub fn scene_from_bytes_with_options(
    bytes: &[u8],
    base: Option<&Path>,
    options: FbxLoadOptions,
) -> Result<IoScene, IoError> {
    #[cfg(feature = "fbx")]
    {
        build_fbx_scene(bytes, base, options)
    }

    #[cfg(not(feature = "fbx"))]
    {
        let _ = (bytes, base, options);
        Err(IoError::MissingFeature {
            feature: "fbx",
            context: "FBX scene decoding",
        })
    }
}

/// The shared FBX decode both entry points funnel through: parse the document from a
/// seekable byte reader, then convert its meshes / materials / skeletons / animations, so
/// the path and bytes paths never diverge.
#[cfg(feature = "fbx")]
fn build_fbx_scene(
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
    let mut materials = Vec::new();
    let mut skeletons: Vec<Skeleton> = Vec::new();
    let mut material_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    for object in document.objects() {
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
                        let material = convert_material(&material_object, parent_dir);
                        let index = materials.len();
                        materials.push(material);
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
            // right one happens after the loop — a single FBX can
            // expose UV0 as a wind / vertex-shader parameter
            // channel (V values up to ~70) and UV1 as the real
            // texture coordinates. Always taking the first channel
            // would map albedo samples onto the wind data and land
            // every fragment in whatever atlas region (0, 70 mod 1)
            // ends up at — typically the transparent corner.
            let mut uv_candidates: Vec<Vec<[f32; 2]>> = Vec::new();
            let mut material_indices_per_vert: Option<Vec<usize>> = None;

            // Walk EVERY layer the geometry exposes. FBX commonly
            // splits secondary UV sets, vertex-painted normals, and
            // material assignments across distinct layers — the
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
                                        // FBX stores UVs with the V origin at the bottom (the
                                        // OpenGL / Maya convention); the renderer samples with V
                                        // at the top (wgpu), matching the glTF loader which reads
                                        // tex-coords unflipped. Flip V so an FBX atlas lands the
                                        // same way round as a glTF one, rather than mirrored top
                                        // to bottom (a face's features on the wrong geometry).
                                        Ok(uv) => uvs.push([uv.x as f32, 1.0 - uv.y as f32]),
                                        Err(_) => {
                                            ok = false;
                                            break;
                                        }
                                    }
                                }
                                if ok && uvs.len() == positions.len() {
                                    uv_candidates.push(uvs);
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
            // not a texture coordinate at all — one axis held constant
            // across the mesh, the signature of a packed per-vertex
            // scalar — in which case advance to the next layer. This
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
                    "VIEWPORT_FBX_LOG_UV: model '{model_name}' ({} fbx material(s)): {} UV channel(s) found:",
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
            let uvs_vec: Option<Vec<[f32; 2]>> = if let Some(idx) = uv_override {
                if log_uv {
                    eprintln!("VIEWPORT_FBX_LOG_UV: override forcing channel {idx}");
                }
                uv_candidates.into_iter().nth(idx)
            } else if uv_candidates.len() <= 1 {
                uv_candidates.into_iter().next()
            } else {
                let idx = pick_uv_channel(&uv_candidates);
                if log_uv {
                    eprintln!(
                        "VIEWPORT_FBX_LOG_UV: picked channel {idx} (first non-degenerate UV set; UV0-preferred)"
                    );
                }
                uv_candidates.into_iter().nth(idx)
            };

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

/// Choose which of a mesh's several UV layers the base map samples, given the
/// per-vertex UVs of each (in authored order, so index 0 is UV0).
///
/// The rule is Unity's: the base map samples the **first authored layer** (UV0);
/// UV value ranges are never a selection signal. The one case that departs from
/// "just take UV0" is a layer that is not a texture coordinate at all — a packed
/// per-vertex scalar (a wind phase in V with U held constant), whose value on
/// one axis does not vary across the mesh. Such a layer is stepped over in
/// favour of the next. An atlas layer that bakes an integer per-card offset into
/// V (foliage wind) still varies on both axes: its raw range is large but real,
/// and repeat-wrap sampling resolves the offset, so it reads as the texture
/// coordinate it is and, being UV0, is chosen. Magnitude is deliberately not
/// consulted, only whether each axis varies at all.
///
/// Returns the chosen index; `0` when every layer is degenerate (better to keep
/// UV0's coordinates than to drop UVs). `candidates` must be non-empty.
#[cfg(feature = "fbx")]
fn pick_uv_channel(candidates: &[Vec<[f32; 2]>]) -> usize {
    // Raw span of a layer's coordinates on each axis. A genuine 2D texture
    // coordinate varies on both; a packed per-vertex scalar holds one axis
    // constant (span ~0). Wrapping is intentionally NOT applied: a value at
    // exactly 1.0 would fold onto 0.0 and make a clean [0,1] square look
    // constant, and magnitude is not a signal anyway.
    let axis_spans = |uvs: &[[f32; 2]]| -> (f32, f32) {
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
        ((umax - umin).max(0.0), (vmax - vmin).max(0.0))
    };
    // An axis that varies less than this across the whole mesh is a constant,
    // not a texture coordinate.
    const CONSTANT: f32 = 1e-4;
    candidates
        .iter()
        .position(|uvs| {
            let (uspan, vspan) = axis_spans(uvs);
            uspan > CONSTANT && vspan > CONSTANT
        })
        .unwrap_or(0)
}

fn fan_triangulator(
    polygon_vertices: &PolygonVertices<'_>,
    polygon_vertex_indices: &[PolygonVertexIndex],
    results: &mut Vec<[PolygonVertexIndex; 3]>,
) -> Result<(), AnyhowError> {
    match polygon_vertex_indices.len() {
        0..=2 => {}
        3 => {
            results.push([
                polygon_vertex_indices[0],
                polygon_vertex_indices[1],
                polygon_vertex_indices[2],
            ]);
        }
        4 => {
            let point = |i: usize| -> Option<glam::Vec3> {
                let point = polygon_vertices.control_point(polygon_vertex_indices[i])?;
                Some(glam::Vec3::new(
                    point.x as f32,
                    point.y as f32,
                    point.z as f32,
                ))
            };
            if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                (point(0), point(1), point(2), point(3))
            {
                let n1 = (p0 - p1).cross(p2 - p1);
                let n3 = (p2 - p3).cross(p0 - p3);
                if n1.dot(n3) >= 0.0 {
                    results.push([
                        polygon_vertex_indices[0],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[2],
                    ]);
                    results.push([
                        polygon_vertex_indices[2],
                        polygon_vertex_indices[3],
                        polygon_vertex_indices[0],
                    ]);
                } else {
                    results.push([
                        polygon_vertex_indices[0],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[3],
                    ]);
                    results.push([
                        polygon_vertex_indices[3],
                        polygon_vertex_indices[1],
                        polygon_vertex_indices[2],
                    ]);
                }
            } else {
                results.push([
                    polygon_vertex_indices[0],
                    polygon_vertex_indices[1],
                    polygon_vertex_indices[2],
                ]);
                results.push([
                    polygon_vertex_indices[2],
                    polygon_vertex_indices[3],
                    polygon_vertex_indices[0],
                ]);
            }
        }
        n => {
            for i in 1..n - 1 {
                results.push([
                    polygon_vertex_indices[0],
                    polygon_vertex_indices[i],
                    polygon_vertex_indices[i + 1],
                ]);
            }
        }
    }
    Ok(())
}

/// Read the FBX-space local TRS for a model node, plus its geometric
/// offset. Returns `(local, geometric)` matrices without any axis
/// transform or unit scaling applied — those are intended to be applied
/// once at the root of the walk by [`extract_node_transform`].
///
/// Works for any `ModelHandle` variant (Mesh, Null, LimbNode, ...). For
/// nodes without a `Properties70` block (rare; mostly a no-op root),
/// returns identity for both.
fn extract_local_components(
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
/// Unity-exported FBX commonly stacks Mesh → Null → Mesh chains where
/// the Null nodes carry the placement transforms (LODGroup containers,
/// "DummyHelper" exporter scaffolding). Earlier versions of this loader
/// only extracted each node's own local transform and tracked Mesh →
/// Mesh parent links separately; transforms on Null/LimbNode parents
/// silently dropped, producing detached "floating" sub-meshes in the
/// output scene. This walk picks up every ancestor type, so the world
/// position matches what a runtime engine using FBX cumulative
/// transforms would produce.
///
fn extract_node_transform(
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

    // Decide the effective axis transform for this leaf. Only the
    // signed-veto policy varies it per-leaf; the others apply
    // `axis_transform` as-is (it was already computed for the file as
    // a whole in `get_axis_transform`).
    let effective_axis = match options.axis_policy {
        AxisPolicy::HeuristicVeto => {
            // Per-leaf axis-transform veto: if the cumulative chain
            // (Lcl Rotation on the mesh or any Null/LimbNode ancestor)
            // already lands +Y on **+Z**, the scene-level Y→Z fix from
            // `get_axis_transform` would double up. The check is
            // **signed** intentionally:
            //
            // - `cumulative_y → +Z`: artist already converted Y-up to
            //   Z-up at the vertex level. Suppress our +90° X.
            // - `cumulative_y → -Z`: artist baked a rotation targeting
            //   a Y-up consumer (e.g. a Z-up authored mesh wrapped in
            //   a `-90° X` Null for Unity ingestion). Don't veto; our
            //   +90° X composes with their -90° X to identity.
            // - `cumulative_y` close to Y: standard Y-up raw content.
            //   Apply our +90° X.
            let cumulative_y_in_world = cumulative.transform_vector3(glam::Vec3::Y);
            let already_z_up = cumulative_y_in_world.z > 0.9 && cumulative_y_in_world.y.abs() < 0.5;
            if already_z_up {
                glam::Mat4::IDENTITY
            } else {
                *axis_transform
            }
        }
        AxisPolicy::HonourHeader | AxisPolicy::ForceYUpRaw | AxisPolicy::PassThrough => {
            *axis_transform
        }
    };

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

fn read_vec3_property(
    props: &fbxcel_dom::v7400::object::property::PropertiesHandle<'_>,
    name: &str,
) -> Option<glam::Vec3> {
    let property = props.get_property(name)?;
    let values = property.value_part();
    if values.len() < 3 {
        return None;
    }
    let x = attr_to_f64(&values[0])?;
    let y = attr_to_f64(&values[1])?;
    let z = attr_to_f64(&values[2])?;
    Some(glam::Vec3::new(x as f32, y as f32, z as f32))
}

fn read_int_property(
    props: &fbxcel_dom::v7400::object::property::PropertiesHandle<'_>,
    name: &str,
) -> Option<i32> {
    let property = props.get_property(name)?;
    let values = property.value_part();
    values.first().and_then(|value| match value {
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i),
        fbxcel::low::v7400::AttributeValue::I16(i) => Some(*i as i32),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i as i32),
        _ => None,
    })
}

fn attr_to_f64(value: &fbxcel::low::v7400::AttributeValue) -> Option<f64> {
    match value {
        fbxcel::low::v7400::AttributeValue::F64(f) => Some(*f),
        fbxcel::low::v7400::AttributeValue::F32(f) => Some(*f as f64),
        fbxcel::low::v7400::AttributeValue::I32(i) => Some(*i as f64),
        fbxcel::low::v7400::AttributeValue::I64(i) => Some(*i as f64),
        _ => None,
    }
}

fn euler_to_mat4(degrees: glam::Vec3, order: i32) -> glam::Mat4 {
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

/// Interpret an FBX `TransparencyFactor` into an opacity and alpha mode.
///
/// FBX transparency is exporter-ambiguous: the spec treats the factor as transparency (0 = opaque),
/// but many common writers (Unity's FBX export, Blender) store it as an *opacity* (1.0 = fully
/// opaque), and a value of exactly 1.0 is never a deliberately fully-transparent base material. So
/// both extremes map to opaque and only a strict in-between is real blending. Without this guard an
/// ordinary opaque material comes back at opacity 0 and renders invisible while still casting a
/// shadow (the shadow pass ignores alpha) — the tell-tale symptom.
fn opacity_from_transparency(transparency: f32) -> (f32, AlphaMode) {
    if transparency > 0.0 && transparency < 1.0 {
        (1.0 - transparency, AlphaMode::Blend)
    } else {
        (1.0, AlphaMode::Opaque)
    }
}

fn convert_material(
    material: &fbxcel_dom::v7400::object::material::MaterialHandle<'_>,
    parent_dir: &Path,
) -> IoMaterial {
    let props = material.properties();

    let diffuse_color = props
        .diffuse_color_or_default()
        .ok()
        .map(|color| [color.r as f32, color.g as f32, color.b as f32])
        .unwrap_or([0.7, 0.7, 0.7]);
    let diffuse_factor = props.diffuse_factor_or_default().ok().unwrap_or(1.0) as f32;
    let base_color = [
        diffuse_color[0] * diffuse_factor,
        diffuse_color[1] * diffuse_factor,
        diffuse_color[2] * diffuse_factor,
    ];

    // FBX `TransparencyFactor` is exporter-ambiguous. The spec treats it as transparency (0 =
    // opaque), but many common writers (Unity's FBX export, Blender) store it as an *opacity*
    // (1.0 = fully opaque), and a value of exactly 1.0 is never a deliberately fully-transparent
    // base material. So treat both extremes as opaque and only a strict in-between as real blending;
    // otherwise an ordinary opaque material comes back at opacity 0 and renders invisible (it still
    // casts a shadow, since the shadow pass ignores alpha, which is the tell-tale symptom).
    let transparency = props.transparency_factor_or_default().ok().unwrap_or(0.0) as f32;
    let (opacity, alpha_mode) = opacity_from_transparency(transparency);
    let shininess = props.shininess_or_default().ok().unwrap_or(20.0) as f32;
    let roughness = (1.0 - (shininess / 100.0).sqrt()).clamp(0.1, 1.0);

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| "fbx_material".into()),
        base_color,
        metallic: 0.0,
        roughness,
        emissive: [0.0, 0.0, 0.0],
        opacity,
        alpha_mode,
        double_sided: false,
        base_color_texture: material
            .diffuse_texture()
            .and_then(|texture| extract_texture(&texture, parent_dir)),
        metallic_roughness_texture: None,
        normal_map_texture: material
            .normal_map_texture()
            .and_then(|texture| extract_texture(&texture, parent_dir)),
        normal_scale: 1.0,
        ao_texture: None,
        occlusion_strength: 1.0,
        emissive_texture: None,
    }
}

fn extract_texture(
    texture: &fbxcel_dom::v7400::object::texture::TextureHandle<'_>,
    parent_dir: &Path,
) -> Option<TextureSource> {
    if let Some(clip) = texture.video_clip() {
        if let Some(content) = clip.content() {
            if !content.is_empty() {
                if let Ok(image) = image::load_from_memory(content) {
                    let rgba = image.to_rgba8();
                    let (width, height) = rgba.dimensions();
                    return Some(TextureSource::Decoded(TextureData {
                        width,
                        height,
                        rgba: rgba.into_raw(),
                    }));
                }
            }
        }

        if let Ok(relative_path) = clip.relative_filename() {
            let relative_path = relative_path.replace('\\', "/");
            let texture_path = parent_dir.join(&relative_path);
            if texture_path.exists() {
                return Some(TextureSource::File(texture_path));
            }
            if let Some(filename) = Path::new(&relative_path).file_name() {
                let fallback = parent_dir.join(filename);
                if fallback.exists() {
                    return Some(TextureSource::File(fallback));
                }
            }
        }
    }

    None
}

/// Build the axis-conversion matrix that lands FBX vertices into the
/// loader's canonical output frame: Z-up, right-handed, CCW-front.
///
/// FBX `GlobalSettings.UpAxis` encodes the source up axis as `0 = X`,
/// `1 = Y`, `2 = Z`. In principle a `UpAxis = Z` file is already in our
/// frame, but in practice many Unity-exported asset packs (Leartes Roman
/// Street, etc.) write `UpAxis = Z` while the raw vertex stream is still
/// Y-up — the file's own scene-level convention is contradicted by the
/// geometry. Some sub-meshes in the same pack carry a compensating
/// `Lcl Rotation = (90, 0, 0)` on the leaf; most don't.
///
/// The strategy used here: **always** produce a Y→Z axis transform
/// (`+90°` about X), then in [`extract_node_transform`] suppress it for
/// any leaf whose cumulative parent chain already lands +Y on +Z. That
/// covers the "exporter compensated via `Lcl Rotation`" case (tree
/// foliage) without breaking the "exporter left raw Y-up" case (roofs,
/// pots, arcs, walls).
fn get_axis_transform(document: &Document, policy: AxisPolicy) -> (glam::Mat4, f32) {
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
    //   (the loader's historical assumption — Unity exports lie about
    //   this in the header). X-up sources are vanishingly rare and we
    //   pass them through rather than guess.
    // - `HonourHeader`: trust the header. Y-up gets +90° X; Z-up gets
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

fn assign_hierarchy(document: &Document, meshes: &mut [IoMesh]) {
    let mut id_to_mesh_index: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();

    let mut mesh_cursor = 0;
    let model_ids: Vec<(i64, String)> = document
        .objects()
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
    for object in document.objects() {
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

fn compute_flat_normals(positions: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32, 0.0, 1.0]; positions.len()];
    for (i, triangle) in positions.chunks_exact(3).enumerate() {
        let v0 = glam::Vec3::from(triangle[0]);
        let v1 = glam::Vec3::from(triangle[1]);
        let v2 = glam::Vec3::from(triangle[2]);
        let n = (v1 - v0).cross(v2 - v0).normalize_or_zero();
        let base = i * 3;
        normals[base] = [n.x, n.y, n.z];
        normals[base + 1] = [n.x, n.y, n.z];
        normals[base + 2] = [n.x, n.y, n.z];
    }
    normals
}

// ---------------------------------------------------------------------------
// Skin extraction
// ---------------------------------------------------------------------------

/// Skin data harvested from one mesh's deformers. `cp_influences` is indexed
/// by control-point index; each entry holds up to four `(joint, weight)`
/// pairs sorted by descending weight and normalised to sum to 1.0.
#[cfg(feature = "fbx")]
struct ExtractedSkin {
    skeleton: Skeleton,
    cp_influences: Vec<Vec<(u8, f32)>>,
}

/// One blend shape harvested from a geometry's BlendShape deformers: a name and
/// a dense per-control-point position delta (zero where the shape leaves a
/// control point in place).
#[cfg(feature = "fbx")]
struct ExtractedBlendShape {
    name: String,
    cp_deltas: Vec<[f32; 3]>,
}

/// Read a geometry's blend shapes as dense per-control-point deltas.
///
/// FBX stores each blend shape as a `Deformer`(BlendShape) → `SubDeformer`
/// (BlendShapeChannel) → `Geometry`(Shape) chain. A Shape carries a **sparse**
/// `Indexes` (control-point indices) + `Vertices` (the position delta for each,
/// flat x/y/z) pair, which this expands into a dense array over all control
/// points. Deltas sit in the same local control-point space as the base
/// positions (the axis / unit conversion rides the mesh transform), so no
/// reorientation applies here. A channel with several in-between shapes keeps
/// the last shape's displacement per control point (the full-weight target);
/// progressive in-betweens are not blended.
///
/// Older FBX (version < 7.5, e.g. 7.1 / 7.2 as many game-character exports
/// still are) instead nests the `Shape` sub-nodes **directly inside the
/// `Geometry` node**, with no separate `BlendShape` deformer / `BlendShapeChannel`
/// objects in the connection graph — so the modern `blendshapes()` walk finds
/// nothing. When the modern chain yields no shapes, [`extract_legacy_shapes`]
/// falls back to reading those nested `Shape` sub-nodes off the raw geometry
/// tree, in the same sparse `Indexes` / `Vertices` layout.
#[cfg(feature = "fbx")]
fn extract_blend_shapes(
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    control_point_count: usize,
) -> Vec<ExtractedBlendShape> {
    let mut out: Vec<ExtractedBlendShape> = Vec::new();
    for blendshape in geometry.blendshapes() {
        for channel in blendshape.blendshape_channels() {
            let name = channel.name().unwrap_or("").to_string();
            let mut cp_deltas = vec![[0.0f32; 3]; control_point_count];
            let mut any = false;
            for shape in channel.shapes() {
                let node = shape.node();
                if accumulate_shape_deltas(&node, &mut cp_deltas) {
                    any = true;
                }
            }
            if any {
                out.push(ExtractedBlendShape { name, cp_deltas });
            }
        }
    }
    // Legacy fallback: pre-7.5 exports carry the shapes as nested `Shape` sub-nodes
    // of the geometry rather than as connected `BlendShape` deformer objects.
    if out.is_empty() {
        out = extract_legacy_shapes(geometry, control_point_count);
    }
    out
}

/// Expand one `Shape` node's sparse `Indexes` + `Vertices` position deltas into
/// `cp_deltas` (dense over all control points), returning whether it touched any.
/// Shared by the modern deformer walk and the legacy nested-shape fallback.
#[cfg(feature = "fbx")]
fn accumulate_shape_deltas(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    cp_deltas: &mut [[f32; 3]],
) -> bool {
    let indexes = read_i32_array(node, "Indexes").unwrap_or_default();
    let verts = read_f64_array(node, "Vertices").unwrap_or_default();
    expand_sparse_shape_deltas(&indexes, &verts, cp_deltas)
}

/// Scatter a Shape's sparse `Indexes` + flat `Vertices` (x/y/z per index) into
/// the dense per-control-point `cp_deltas`, returning whether it wrote anything.
/// The pure core shared by the modern deformer walk and the legacy nested-shape
/// fallback: out-of-range indices are skipped, and a truncated `Vertices` buffer
/// (fewer than `indexes.len() * 3`) is rejected wholesale rather than read past.
#[cfg(feature = "fbx")]
fn expand_sparse_shape_deltas(indexes: &[i32], verts: &[f64], cp_deltas: &mut [[f32; 3]]) -> bool {
    if indexes.is_empty() || verts.len() < indexes.len() * 3 {
        return false;
    }
    let mut any = false;
    for (k, &cp_i32) in indexes.iter().enumerate() {
        let cp = cp_i32 as usize;
        if cp >= cp_deltas.len() {
            continue;
        }
        let base = k * 3;
        cp_deltas[cp] = [
            verts[base] as f32,
            verts[base + 1] as f32,
            verts[base + 2] as f32,
        ];
        any = true;
    }
    any
}

/// Read legacy (pre-7.5) blend shapes nested as `Shape` sub-nodes inside the
/// geometry node. Each `Shape` node names the target in its first string
/// attribute and carries the same sparse `Indexes` / `Vertices` delta layout as
/// a modern Shape geometry. The ARKit facial rigs many game characters ship
/// (jawOpen / eyeBlinkLeft / …) are stored this way.
#[cfg(feature = "fbx")]
fn extract_legacy_shapes(
    geometry: &fbxcel_dom::v7400::object::geometry::MeshHandle<'_>,
    control_point_count: usize,
) -> Vec<ExtractedBlendShape> {
    let mut out: Vec<ExtractedBlendShape> = Vec::new();
    for shape in geometry.node().children_by_name("Shape") {
        let name = shape
            .attributes()
            .iter()
            .find_map(|a| match a {
                fbxcel::low::v7400::AttributeValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let mut cp_deltas = vec![[0.0f32; 3]; control_point_count];
        if accumulate_shape_deltas(&shape, &mut cp_deltas) {
            out.push(ExtractedBlendShape { name, cp_deltas });
        }
    }
    out
}

#[cfg(feature = "fbx")]
struct BoneInfo {
    name: String,
    parent_id: Option<i64>,
    transform_link: glam::Mat4,
}

#[cfg(feature = "fbx")]
struct ClusterEntry {
    bone_id: i64,
    indexes: Vec<i32>,
    weights: Vec<f64>,
}

#[cfg(feature = "fbx")]
fn extract_skin(
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
            // via Deref from ClusterHandle → ObjectHandle.
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
fn lookup_object<'a>(
    document: &'a Document,
    object_id: i64,
) -> Option<fbxcel_dom::v7400::object::ObjectHandle<'a>> {
    document
        .objects()
        .find(|o| o.object_id().raw() == object_id)
}

#[cfg(feature = "fbx")]
fn parent_model_id(obj: &fbxcel_dom::v7400::object::ObjectHandle<'_>) -> Option<i64> {
    use fbxcel_dom::v7400::object::model::TypedModelHandle as M;
    let model = match obj.get_typed() {
        TypedObjectHandle::Model(M::LimbNode(n)) => n.parent_model()?,
        TypedObjectHandle::Model(M::Null(n)) => n.parent_model()?,
        _ => return None,
    };
    // TypedModelHandle derefs to ModelHandle → ObjectHandle, so the chained
    // `**` reaches the ObjectHandle and we can read its raw id.
    let object_id_raw = match model {
        M::LimbNode(n) => (**n).object_id().raw(),
        M::Null(n) => (**n).object_id().raw(),
        M::Mesh(n) => (**n).object_id().raw(),
        M::Camera(n) => (**n).object_id().raw(),
        M::Light(n) => (**n).object_id().raw(),
        _ => return None,
    };
    Some(object_id_raw)
}

#[cfg(feature = "fbx")]
fn topo_sort(bones: &HashMap<i64, BoneInfo>) -> Vec<i64> {
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
fn read_i32_array(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<Vec<i32>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrI32(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrI64(v) => {
            Some(v.iter().map(|&i| i as i32).collect())
        }
        _ => None,
    }
}

#[cfg(feature = "fbx")]
fn read_f64_array(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<Vec<f64>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrF64(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrF32(v) => {
            Some(v.iter().map(|&f| f as f64).collect())
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Animation extraction
// ---------------------------------------------------------------------------

/// One ktime tick is 1 / 46_186_158_000 second. Conventional FBX constant.
#[cfg(feature = "fbx")]
const FBX_KTIME_PER_SECOND: f64 = 46_186_158_000.0;

#[cfg(feature = "fbx")]
fn extract_animations(
    document: &Document,
    skeletons: &mut Vec<Skeleton>,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
) -> Vec<AnimationClip> {
    let mut anim_stacks: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = Vec::new();
    for obj in document.objects() {
        let c = obj.class();
        if c == "AnimStack" || c == "AnimationStack" {
            anim_stacks.push(obj);
        }
    }
    if anim_stacks.is_empty() {
        return Vec::new();
    }

    // File-level conversion `C = axis_transform * unit_scale` — the same
    // matrix the mesh leaves and the skin inverse-binds compose with. Joint
    // WORLD transforms must become `W' = C * W`; premultiplying every ROOT
    // joint's local by C achieves that for the whole chain, so only
    // root-joint tracks are converted below. When the file-level transform
    // is identity and the unit scale is 1, all of this is a no-op.
    //
    // NOTE: skeletons and animations are file-scoped, so this is keyed to
    // the FILE-level transform from `get_axis_transform`. Under
    // `AxisPolicy::HeuristicVeto` a mesh leaf may individually veto the
    // axis transform (see `extract_node_transform`); a vetoed skinned leaf
    // paired with a converted skeleton is a pre-existing inconsistency that
    // is not addressed here.
    let conv = *axis_transform * glam::Mat4::from_scale(glam::Vec3::splat(unit_scale));
    // `axis_transform` is a pure rotation (unit scale is carried separately),
    // so its quaternion is exactly the rotation part of C.
    let conv_rot = glam::Quat::from_mat4(axis_transform).normalize();

    // Build a rig skeleton from every LimbNode/Null model in the document so
    // animation tracks have a stable joint indexing. If a skin already produced
    // a skeleton whose bone set matches, reuse it; otherwise append a new one.
    let (rig_skeleton, model_id_to_joint) = build_rig_from_limbs(document, &conv);
    if model_id_to_joint.is_empty() {
        return Vec::new();
    }

    // Find a matching existing skeleton by bone name set, otherwise append.
    let rig_skel_idx = find_or_insert_rig(skeletons, rig_skeleton);
    // Name -> idx in the final rig skeleton (post-insert), for tracks remapping.
    let rig_name_to_idx: HashMap<String, usize> = skeletons[rig_skel_idx]
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.clone(), i))
        .collect();
    // Roots of the final rig skeleton: only their local transforms need the
    // file-level conversion applied (every descendant world inherits it).
    let joint_is_root: Vec<bool> = skeletons[rig_skel_idx]
        .joints
        .iter()
        .map(|j| j.parent.is_none())
        .collect();

    // Cache per-bone PreRotation / PostRotation / RotationOrder so each
    // rotation curve sample can compose them as the FBX bone matrix does:
    //   R_local = PreR * R_anim * inverse(PostR)
    // This matches the bind-pose decomposition (which already bakes in PreR
    // via the bone's TransformLink), so at t=0 the curve agrees with bind.
    let mut bone_rot_ctx: HashMap<i64, (glam::Mat4, glam::Mat4, i32)> = HashMap::new();
    for obj in document.objects() {
        if !model_id_to_joint.contains_key(&obj.object_id().raw()) {
            continue;
        }
        let Some(props) = obj.direct_properties() else {
            continue;
        };
        let pre = read_vec3_property(&props, "PreRotation").unwrap_or(glam::Vec3::ZERO);
        let post = read_vec3_property(&props, "PostRotation").unwrap_or(glam::Vec3::ZERO);
        let order = read_int_property(&props, "RotationOrder").unwrap_or(0);
        let pre_m = euler_to_mat4(pre, 0);
        let post_inv_m = euler_to_mat4(post, 0).inverse();
        bone_rot_ctx.insert(obj.object_id().raw(), (pre_m, post_inv_m, order));
    }

    let mut clips = Vec::new();
    for stack in anim_stacks {
        let name = stack.name().unwrap_or("AnimStack").to_string();
        let mut tracks: Vec<AnimationTrack> = Vec::new();
        let mut max_t: f32 = 0.0;

        // Stack -> Layers (layers connect TO stack: layer is source, stack is dest)
        let layers: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = stack
            .source_objects()
            .filter_map(|c| c.object_handle())
            .filter(|o| {
                let c = o.class();
                c == "AnimLayer" || c == "AnimationLayer"
            })
            .collect();

        for layer in layers {
            // Layer -> CurveNodes (curvenodes connect TO layer)
            let curve_nodes: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = layer
                .source_objects()
                .filter_map(|c| c.object_handle())
                .filter(|o| {
                    let c = o.class();
                    c == "AnimCurveNode" || c == "AnimationCurveNode"
                })
                .collect();

            for cn in curve_nodes {
                // CurveNode -> Model with property label
                let target = cn.destination_objects().find_map(|c| {
                    let label = c.label()?;
                    let obj = c.object_handle()?;
                    let model_id = obj.object_id().raw();
                    let joint_idx = model_id_to_joint.get(&model_id)?;
                    let bone_name = obj.name().unwrap_or("").to_string();
                    Some((*joint_idx, bone_name, label.to_string(), model_id))
                });
                let Some((local_joint_idx, bone_name, property, model_id)) = target else {
                    continue;
                };
                let channel = match property.as_str() {
                    "Lcl Translation" => AnimationChannel::Translation,
                    "Lcl Rotation" => AnimationChannel::Rotation,
                    "Lcl Scaling" => AnimationChannel::Scale,
                    _ => continue,
                };

                // Resolve joint index in the final rig skeleton via name.
                let _ = local_joint_idx;
                let Some(&rig_joint) = rig_name_to_idx.get(&bone_name) else {
                    continue;
                };

                // CurveNode <- Curves labelled "d|X" "d|Y" "d|Z"
                let mut axis_curves: [Option<(Vec<f32>, Vec<f32>)>; 3] = [None, None, None];
                for c in cn.source_objects() {
                    let Some(label) = c.label() else { continue };
                    let axis = match label {
                        "d|X" | "d|X|X" => 0,
                        "d|Y" | "d|Y|Y" => 1,
                        "d|Z" | "d|Z|Z" => 2,
                        _ => continue,
                    };
                    let Some(obj) = c.object_handle() else {
                        continue;
                    };
                    let cc = obj.class();
                    if cc != "AnimCurve" && cc != "AnimationCurve" {
                        continue;
                    }
                    let node = obj.node();
                    let Some(times) = read_i64_array(&node, "KeyTime") else {
                        continue;
                    };
                    let Some(values) = read_f32_array(&node, "KeyValueFloat") else {
                        continue;
                    };
                    if times.is_empty() || times.len() != values.len() {
                        continue;
                    }
                    let times_s: Vec<f32> = times
                        .iter()
                        .map(|t| (*t as f64 / FBX_KTIME_PER_SECOND) as f32)
                        .collect();
                    axis_curves[axis] = Some((times_s, values));
                }
                if axis_curves.iter().all(|c| c.is_none()) {
                    continue;
                }

                // Union times across the three axes.
                let mut union_times: Vec<f32> = Vec::new();
                for c in axis_curves.iter().flatten() {
                    union_times.extend_from_slice(&c.0);
                }
                if union_times.is_empty() {
                    continue;
                }
                union_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                union_times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

                // For each axis, default-value lookup. If curve is missing,
                // hold a constant value (0 for translation/rotation, 1 for scale).
                let default = if matches!(channel, AnimationChannel::Scale) {
                    1.0
                } else {
                    0.0
                };
                let sample_axis = |a: usize, t: f32| -> f32 {
                    match &axis_curves[a] {
                        None => default,
                        Some((ts, vs)) => sample_linear(ts, vs, t),
                    }
                };

                if let Some(last) = union_times.last() {
                    if *last > max_t {
                        max_t = *last;
                    }
                }

                // Root joints carry the file-level conversion: with
                // `C = s * A` (uniform scale s, rotation A), converting the
                // root local `L0' = C * L0 = T'R'S'` decomposes per channel
                // as `t' = s * (A * t)`, `q' = q_A * q`, `s' = s * scale`
                // (scale axes stay in the joint's local frame). Non-root
                // tracks are unchanged — their worlds inherit C via the root.
                let is_root = joint_is_root.get(rig_joint).copied().unwrap_or(false);
                let values = match channel {
                    AnimationChannel::Translation => {
                        let v: Vec<glam::Vec3> = union_times
                            .iter()
                            .map(|t| {
                                let raw = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                if is_root {
                                    // C has no translation part, so this is
                                    // exactly `s * (A * raw)`.
                                    conv.transform_point3(raw)
                                } else {
                                    raw
                                }
                            })
                            .collect();
                        AnimationTrackValues::Vec3(v)
                    }
                    AnimationChannel::Scale => {
                        let v: Vec<glam::Vec3> = union_times
                            .iter()
                            .map(|t| {
                                let raw = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                if is_root { unit_scale * raw } else { raw }
                            })
                            .collect();
                        AnimationTrackValues::Vec3(v)
                    }
                    AnimationChannel::Rotation => {
                        // Compose with PreRotation / PostRotation as FBX does:
                        //   R_local = PreR * R_anim * inverse(PostR)
                        // The bind pose (derived from TransformLink) already
                        // bakes in PreR, so this composition matches it at t=0.
                        let (pre_m, post_inv_m, order) = bone_rot_ctx
                            .get(&model_id)
                            .copied()
                            .unwrap_or((glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, 0));
                        let v: Vec<glam::Quat> = union_times
                            .iter()
                            .map(|t| {
                                let deg = glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                );
                                let r_anim = euler_to_mat4(deg, order);
                                // Compose PreR * R_anim * PostR^-1 FIRST —
                                // the composed value is the joint's full
                                // local rotation, which is what the root
                                // conversion `L0' = C * L0` transforms.
                                let m = pre_m * r_anim * post_inv_m;
                                let q = glam::Quat::from_mat4(&m).normalize();
                                if is_root {
                                    (conv_rot * q).normalize()
                                } else {
                                    q
                                }
                            })
                            .collect();
                        AnimationTrackValues::Quat(v)
                    }
                };

                tracks.push(AnimationTrack {
                    joint: rig_joint,
                    channel,
                    sampler: AnimationSampler {
                        interpolation: AnimationInterpolation::Linear,
                        times: union_times,
                        values,
                    },
                });
            }
        }

        if tracks.is_empty() {
            continue;
        }
        clips.push(AnimationClip {
            name,
            duration: max_t,
            skeleton_index: rig_skel_idx,
            tracks,
        });
    }
    clips
}

#[cfg(feature = "fbx")]
fn find_or_insert_rig(skeletons: &mut Vec<Skeleton>, rig: Skeleton) -> usize {
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
        // carries the real cluster inverse-binds — so reusing the skin skeleton
        // (and retargeting the clip's name-keyed tracks onto it) is what lets a
        // clip actually drive the skinned mesh, instead of stranding the
        // animation on a second, meshless skeleton. `rig ⊆ skin` is the original
        // fully-covered case; `skin ⊆ rig` is the common subset case.
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
fn build_rig_from_limbs(document: &Document, conv: &glam::Mat4) -> (Skeleton, HashMap<i64, u8>) {
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

#[cfg(feature = "fbx")]
fn read_i64_array(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<Vec<i64>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrI64(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrI32(v) => {
            Some(v.iter().map(|&i| i as i64).collect())
        }
        _ => None,
    }
}

#[cfg(feature = "fbx")]
fn read_f32_array(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<Vec<f32>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrF32(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrF64(v) => {
            Some(v.iter().map(|&f| f as f32).collect())
        }
        _ => None,
    }
}

/// Import blend-shape **weight** animation from the FBX anim graph, the morph
/// counterpart of [`extract_animations`]. Each `BlendShapeChannel` carries a
/// `DeformPercent` property (0..100) driven by one `AnimCurve`; this walks
/// `AnimStack → AnimLayer → AnimCurveNode` (the same connection traversal the
/// bone tracks use), keeps the curve nodes whose destination property is
/// `DeformPercent`, and keys the curve by its channel's name — which is the same
/// name [`extract_blend_shapes`] gives the morph target.
///
/// `mesh_targets` is `(scene-mesh index, target names in emitted order)` for each
/// mesh that carries blend shapes. Per animation stack, one [`MorphWeightClip`]
/// is emitted for each such mesh with at least one animated target: FBX keys each
/// channel on its own times, so the mesh's animated curves are merged onto a union
/// timeline and every target sampled at each key (an unanimated target holds `0`),
/// yielding the dense row-major `[keyframe][target]` matrix the consumers expect.
#[cfg(feature = "fbx")]
fn extract_morph_weight_clips(
    document: &Document,
    mesh_targets: &[(usize, Vec<String>)],
) -> Vec<MorphWeightClip> {
    if mesh_targets.is_empty() {
        return Vec::new();
    }
    let anim_stacks: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = document
        .objects()
        .filter(|o| matches!(o.class(), "AnimStack" | "AnimationStack"))
        .collect();

    let mut out: Vec<MorphWeightClip> = Vec::new();
    for stack in anim_stacks {
        let stack_name = stack.name().unwrap_or("AnimStack").to_string();

        // Channel name -> its weight curve (times in seconds, values in 0..1).
        let mut curves: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
        for layer in stack
            .source_objects()
            .filter_map(|c| c.object_handle())
            .filter(|o| matches!(o.class(), "AnimLayer" | "AnimationLayer"))
        {
            for cn in layer
                .source_objects()
                .filter_map(|c| c.object_handle())
                .filter(|o| matches!(o.class(), "AnimCurveNode" | "AnimationCurveNode"))
            {
                // Keep only curve nodes driving a channel's DeformPercent; the
                // destination object is the BlendShapeChannel, named like the target.
                let channel_name = cn.destination_objects().find_map(|c| {
                    if c.label() != Some("DeformPercent") {
                        return None;
                    }
                    c.object_handle()?.name().map(str::to_string)
                });
                let Some(channel_name) = channel_name else {
                    continue;
                };

                for c in cn.source_objects() {
                    let Some(label) = c.label() else { continue };
                    if !label.starts_with("d|DeformPercent") {
                        continue;
                    }
                    let Some(obj) = c.object_handle() else {
                        continue;
                    };
                    if !matches!(obj.class(), "AnimCurve" | "AnimationCurve") {
                        continue;
                    }
                    let node = obj.node();
                    let Some(times) = read_i64_array(&node, "KeyTime") else {
                        continue;
                    };
                    let Some(values) = read_f32_array(&node, "KeyValueFloat") else {
                        continue;
                    };
                    if times.is_empty() || times.len() != values.len() {
                        continue;
                    }
                    let times_s: Vec<f32> = times
                        .iter()
                        .map(|t| (*t as f64 / FBX_KTIME_PER_SECOND) as f32)
                        .collect();
                    // FBX stores the weight as a percentage; the runtime wants 0..1.
                    let vals01: Vec<f32> = values.iter().map(|v| v / 100.0).collect();
                    curves.insert(channel_name.clone(), (times_s, vals01));
                    break;
                }
            }
        }
        if curves.is_empty() {
            continue;
        }

        for (mesh_index, names) in mesh_targets {
            if !names.iter().any(|n| curves.contains_key(n)) {
                continue;
            }
            // Union of every animated target's key times across this mesh.
            let mut union_times: Vec<f32> = Vec::new();
            for n in names {
                if let Some((ts, _)) = curves.get(n) {
                    union_times.extend_from_slice(ts);
                }
            }
            union_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            union_times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
            if union_times.is_empty() {
                continue;
            }

            let target_count = names.len();
            let mut weights = Vec::with_capacity(union_times.len() * target_count);
            for &t in &union_times {
                for n in names {
                    let w = match curves.get(n) {
                        Some((ts, vs)) => sample_linear(ts, vs, t),
                        None => 0.0,
                    };
                    weights.push(w);
                }
            }
            let duration = union_times.last().copied().unwrap_or(0.0);
            out.push(MorphWeightClip {
                name: stack_name.clone(),
                duration,
                mesh_index: *mesh_index,
                target_count,
                interpolation: AnimationInterpolation::Linear,
                times: union_times,
                weights,
            });
        }
    }
    out
}

#[cfg(feature = "fbx")]
fn sample_linear(times: &[f32], values: &[f32], t: f32) -> f32 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    if t >= *times.last().unwrap() {
        return *values.last().unwrap();
    }
    for i in 1..times.len() {
        if t <= times[i] {
            let span = times[i] - times[i - 1];
            if span <= 0.0 {
                return values[i];
            }
            let a = (t - times[i - 1]) / span;
            return values[i - 1] * (1.0 - a) + values[i] * a;
        }
    }
    *values.last().unwrap()
}

#[cfg(feature = "fbx")]
fn read_mat4(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<glam::Mat4> {
    let values = read_f64_array(node, name)?;
    if values.len() < 16 {
        return None;
    }
    // FBX stores matrices column-major.
    let m: [f32; 16] = std::array::from_fn(|i| values[i] as f32);
    Some(glam::Mat4::from_cols_array(&m))
}

#[cfg(all(test, feature = "fbx"))]
mod rig_reconcile_tests {
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
    /// the skeleton that actually skins the mesh — not a second, meshless one.
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

    /// The original fully-covered case (skin ⊇ rig) still reuses.
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

    /// FBX `TransparencyFactor` is exporter-ambiguous; a plain opaque material (factor 0.0 *or* the
    /// opacity-convention 1.0) must come back opaque, and only a strict in-between blends. Guards the
    /// regression where opaque characters rendered invisible (opacity 0) but still cast shadows.
    #[test]
    fn opacity_from_transparency_treats_both_extremes_as_opaque() {
        assert_eq!(opacity_from_transparency(0.0), (1.0, AlphaMode::Opaque));
        assert_eq!(opacity_from_transparency(1.0), (1.0, AlphaMode::Opaque));
        let (o, mode) = opacity_from_transparency(0.25);
        assert!((o - 0.75).abs() < 1e-6);
        assert_eq!(mode, AlphaMode::Blend);
    }

    /// A genuinely different armature (different root) is appended, not merged.
    #[test]
    fn unrelated_rig_is_appended() {
        let mut skeletons = vec![skel(&[("RigA", None), ("Bone", Some(0))])];
        let rig = skel(&[("RigB", None), ("Other", Some(0))]);
        assert_eq!(find_or_insert_rig(&mut skeletons, rig), 1);
        assert_eq!(skeletons.len(), 2);
    }

    /// A single square patch of texture coordinates on both layers: the base map
    /// samples UV0, the first authored layer, the way Unity binds it.
    #[test]
    fn uv_pick_prefers_first_layer() {
        let square = || vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_uv_channel(&[square(), square()]), 0);
    }

    /// A foliage atlas that bakes an integer per-card offset into V (V running to
    /// tens) is still a texture coordinate once wrapped, so UV0 is chosen — not
    /// rejected as a data channel. This is the RomanStreet tree's dark-branches
    /// case: the offset channel is the correct albedo.
    #[test]
    fn uv_pick_keeps_offset_atlas_on_uv0() {
        // UV0: real texcoords plus a large integer card offset in V.
        let offset_atlas = vec![
            [0.1, 12.0],
            [0.9, 12.0],
            [0.9, 12.9],
            [0.1, 12.9],
            [0.2, 47.1],
            [0.8, 47.8],
        ];
        // UV1: an ordinary lightmap-style unwrap in [0, 1].
        let lightmap = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.3, 0.7],
        ];
        assert_eq!(pick_uv_channel(&[offset_atlas, lightmap]), 0);
    }

    /// A packed per-vertex scalar in UV0 (a wind phase in V, U held constant) is
    /// not a texture coordinate — it collapses one axis even after wrapping — so
    /// the pick steps over it to the real texture layer, UV1.
    #[test]
    fn uv_pick_steps_over_packed_data_channel() {
        // UV0: U constant at 0, V an arbitrary per-vertex scalar (wind phase).
        let packed = vec![[0.0, 3.2], [0.0, 41.7], [0.0, 8.9], [0.0, 70.0]];
        // UV1: a proper texture-coordinate square.
        let texcoords = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert_eq!(pick_uv_channel(&[packed, texcoords]), 1);
    }

    /// Every layer degenerate: keep UV0's coordinates rather than drop UVs.
    #[test]
    fn uv_pick_falls_back_to_uv0_when_all_degenerate() {
        let flat_u = vec![[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
        let flat_v = vec![[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        assert_eq!(pick_uv_channel(&[flat_u, flat_v]), 0);
    }

    /// A blend shape's sparse `Indexes` + flat `Vertices` scatter into the dense
    /// per-control-point delta array, leaving untouched control points at zero.
    /// The shared core of the modern deformer walk and the legacy nested-shape
    /// fallback (the ARKit rigs pre-7.5 FBX stores inside the geometry node).
    #[test]
    fn sparse_shape_deltas_scatter_into_dense_control_points() {
        // Two of four control points move; index order need not be ascending.
        let indexes = [2, 0];
        let verts = [1.0, 2.0, 3.0, -4.0, 0.0, 0.5];
        let mut dense = vec![[0.0f32; 3]; 4];
        assert!(expand_sparse_shape_deltas(&indexes, &verts, &mut dense));
        assert_eq!(dense[2], [1.0, 2.0, 3.0]);
        assert_eq!(dense[0], [-4.0, 0.0, 0.5]);
        // Unreferenced control points stay at rest.
        assert_eq!(dense[1], [0.0, 0.0, 0.0]);
        assert_eq!(dense[3], [0.0, 0.0, 0.0]);
    }

    /// An out-of-range control-point index is skipped, and a `Vertices` buffer
    /// too short for the index count is rejected wholesale rather than read past.
    #[test]
    fn sparse_shape_deltas_reject_truncated_and_skip_out_of_range() {
        let mut dense = vec![[0.0f32; 3]; 2];
        // 9 (>= len 2) is skipped; index 1 still applies, so it wrote something.
        assert!(expand_sparse_shape_deltas(&[9, 1], &[7.0; 6], &mut dense));
        assert_eq!(dense[1], [7.0, 7.0, 7.0]);
        assert_eq!(dense[0], [0.0, 0.0, 0.0]);

        // Truncated Vertices (needs 6 floats for 2 indices, has 3) writes nothing.
        let mut dense2 = vec![[0.0f32; 3]; 2];
        assert!(!expand_sparse_shape_deltas(
            &[0, 1],
            &[1.0, 2.0, 3.0],
            &mut dense2
        ));
        assert_eq!(dense2, vec![[0.0f32; 3]; 2]);

        // Empty is a no-op.
        assert!(!expand_sparse_shape_deltas(&[], &[], &mut dense2));
    }
}
