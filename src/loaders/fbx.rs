use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use anyhow::Error as AnyhowError;
use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::fbxcel;
use fbxcel_dom::v7400::data::mesh::layer::TypedLayerElementHandle;
use fbxcel_dom::v7400::data::mesh::{PolygonVertexIndex, PolygonVertices};
use fbxcel_dom::v7400::object::model::TypedModelHandle;
use fbxcel_dom::v7400::object::TypedObjectHandle;
use fbxcel_dom::v7400::Document;

use crate::error::IoError;
use crate::types::{
    AnimationChannel, AnimationClip, AnimationInterpolation, AnimationSampler, AnimationTrack,
    AnimationTrackValues, IoMaterial, IoMesh, IoScene, Joint, Skeleton, SkinWeights, SurfaceMesh,
    TextureData, TextureSource,
};

/// Decode an FBX file into a CPU-side scene.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    #[cfg(feature = "fbx")]
    {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        let document = match AnyDocument::from_seekable_reader(reader)
            .map_err(|error| IoError::Parse(format!("FBX load failed ({}): {error:?}", path.display())))?
        {
            AnyDocument::V7400(_, document) => document,
            _ => {
                return Err(IoError::Parse(
                    "unsupported FBX version (only binary FBX 7.4/7.5 supported)".into(),
                ))
            }
        };

        let parent_dir = path.parent().unwrap_or(Path::new("."));
        let (axis_transform, unit_scale) = get_axis_transform(&document);

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

                let node_transform = extract_node_transform(&mesh_model, &axis_transform, unit_scale);

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
                let mut uvs_vec: Option<Vec<[f32; 2]>> = None;
                let mut material_indices_per_vert: Option<Vec<usize>> = None;

                if let Some(layer) = geometry.layers().next() {
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
                                                normals.push([normal.x as f32, normal.y as f32, normal.z as f32]);
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
                                if uvs_vec.is_some() {
                                    continue;
                                }
                                if let Ok(uv_data) = uv_handle.uv() {
                                    let mut uvs = Vec::with_capacity(positions.len());
                                    let mut ok = true;
                                    for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                                        match uv_data.uv(&triangle_vertices, triangle_vertex) {
                                            Ok(uv) => uvs.push([uv.x as f32, uv.y as f32]),
                                            Err(_) => {
                                                ok = false;
                                                break;
                                            }
                                        }
                                    }
                                    if ok && uvs.len() == positions.len() {
                                        uvs_vec = Some(uvs);
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
                                        match material_data.material_index(&triangle_vertices, triangle_vertex) {
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
                let (skin_per_vertex, skeleton_index): (
                    Option<SkinWeights>,
                    Option<usize>,
                ) = if let Some(skin) = extracted_skin {
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
                            let sub_uvs = uvs_vec
                                .as_ref()
                                .map(|uvs| vertex_indices.iter().map(|&i| uvs[i]).collect());
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

                            let mut mesh_data = SurfaceMesh::default();
                            mesh_data.positions = sub_positions;
                            mesh_data.normals = sub_normals;
                            mesh_data.indices = sub_indices;
                            mesh_data.uvs = sub_uvs;
                            mesh_data.skin_weights = sub_skin;

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

        let animations = extract_animations(&document, &mut skeletons);

        Ok(IoScene {
            meshes,
            materials,
            skeletons,
            animations,
            ..IoScene::default()
        })
    }

    #[cfg(not(feature = "fbx"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "fbx",
            context: "FBX scene decoding",
        })
    }
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
                Some(glam::Vec3::new(point.x as f32, point.y as f32, point.z as f32))
            };
            if let (Some(p0), Some(p1), Some(p2), Some(p3)) = (point(0), point(1), point(2), point(3)) {
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
    let geo_translation = read_vec3_property(&props, "GeometricTranslation").unwrap_or(glam::Vec3::ZERO);
    let geo_rotation_deg = read_vec3_property(&props, "GeometricRotation").unwrap_or(glam::Vec3::ZERO);
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

    let local = t * r_off * r_piv * pre_r * r * post_r_inv * r_piv_inv * s_off * s_piv * s * s_piv_inv;
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
fn extract_node_transform(
    mesh_model: &fbxcel_dom::v7400::object::model::MeshHandle<'_>,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
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

    *axis_transform * glam::Mat4::from_scale(glam::Vec3::splat(unit_scale)) * cumulative * geometric
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

    let transparency = props.transparency_factor_or_default().ok().unwrap_or(0.0) as f32;
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
        opacity: 1.0 - transparency,
        base_color_texture: material
            .diffuse_texture()
            .and_then(|texture| extract_texture(&texture, parent_dir)),
        normal_map_texture: material
            .normal_map_texture()
            .and_then(|texture| extract_texture(&texture, parent_dir)),
        ao_texture: None,
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

fn get_axis_transform(document: &Document) -> (glam::Mat4, f32) {
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
    let axis_transform = match up_axis {
        1 | 2 => glam::Mat4::IDENTITY,
        _ => glam::Mat4::IDENTITY,
    };

    (axis_transform, unit_scale)
}

fn assign_hierarchy(document: &Document, meshes: &mut [IoMesh]) {
    let mut id_to_mesh_index: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();

    let mut mesh_cursor = 0;
    let model_ids: Vec<(i64, String)> = document
        .objects()
        .filter_map(|object| {
            if let TypedObjectHandle::Model(TypedModelHandle::Mesh(mesh)) = object.get_typed() {
                Some((object.object_id().raw(), mesh.name().unwrap_or("").to_string()))
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
            while mesh_cursor < meshes.len() && meshes[mesh_cursor].name.starts_with(&material_prefix) {
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
    // Match the Y-up to Z-up reorientation applied to FBX vertex positions,
    // so joint inverse-binds land in the same scene space.
    let y_up_to_z_up_mat = glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);

    let joints: Vec<Joint> = id_order
        .iter()
        .map(|id| {
            let info = &bones[id];
            let world = y_up_to_z_up_mat * *axis_transform * unit_scale_mat * info.transform_link;
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
    let all_ids: Vec<i64> = bones.keys().copied().collect();

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
fn read_i32_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<i32>> {
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
fn read_f64_array(
    node: &fbxcel::tree::v7400::NodeHandle<'_>,
    name: &str,
) -> Option<Vec<f64>> {
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
fn extract_animations(document: &Document, skeletons: &mut Vec<Skeleton>) -> Vec<AnimationClip> {
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

    // Build a rig skeleton from every LimbNode/Null model in the document so
    // animation tracks have a stable joint indexing. If a skin already produced
    // a skeleton whose bone set matches, reuse it; otherwise append a new one.
    let (rig_skeleton, model_id_to_joint) = build_rig_from_limbs(document);
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
        let Some(props) = obj.direct_properties() else { continue };
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
            .filter(|o| { let c = o.class(); c == "AnimLayer" || c == "AnimationLayer" })
            .collect();

        for layer in layers {
            // Layer -> CurveNodes (curvenodes connect TO layer)
            let curve_nodes: Vec<fbxcel_dom::v7400::object::ObjectHandle<'_>> = layer
                .source_objects()
                .filter_map(|c| c.object_handle())
                .filter(|o| { let c = o.class(); c == "AnimCurveNode" || c == "AnimationCurveNode" })
                .collect();

            for cn in curve_nodes {
                // CurveNode -> Model with property label
                let target = cn
                    .destination_objects()
                    .find_map(|c| {
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
                    let Some(obj) = c.object_handle() else { continue };
                    let cc = obj.class();
                    if cc != "AnimCurve" && cc != "AnimationCurve" {
                        continue;
                    }
                    let node = obj.node();
                    let Some(times) = read_i64_array(&node, "KeyTime") else { continue };
                    let Some(values) = read_f32_array(&node, "KeyValueFloat") else { continue };
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

                let values = match channel {
                    AnimationChannel::Translation | AnimationChannel::Scale => {
                        let v: Vec<glam::Vec3> = union_times
                            .iter()
                            .map(|t| {
                                glam::Vec3::new(
                                    sample_axis(0, *t),
                                    sample_axis(1, *t),
                                    sample_axis(2, *t),
                                )
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
                                let m = pre_m * r_anim * post_inv_m;
                                glam::Quat::from_mat4(&m).normalize()
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
    for (i, sk) in skeletons.iter().enumerate() {
        let sk_names: std::collections::HashSet<&str> =
            sk.joints.iter().map(|j| j.name.as_str()).collect();
        // Reuse an existing skeleton iff it covers the rig fully (bone names).
        if rig_names.iter().all(|n| sk_names.contains(n)) {
            return i;
        }
    }
    let idx = skeletons.len();
    skeletons.push(rig);
    idx
}

/// Build a Skeleton from every LimbNode/Null model in the document, returning
/// the skeleton and a model-id -> joint-index map.
#[cfg(feature = "fbx")]
fn build_rig_from_limbs(document: &Document) -> (Skeleton, HashMap<i64, u8>) {
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

    let joints: Vec<Joint> = id_order
        .iter()
        .map(|id| {
            let info = &bones[id];
            Joint {
                name: info.name.clone(),
                parent: info.parent_id.and_then(|pid| id_to_idx.get(&pid).copied()),
                inverse_bind: glam::Mat4::IDENTITY,
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
        fbxcel::low::v7400::AttributeValue::ArrI32(v) => Some(v.iter().map(|&i| i as i64).collect()),
        _ => None,
    }
}

#[cfg(feature = "fbx")]
fn read_f32_array(node: &fbxcel::tree::v7400::NodeHandle<'_>, name: &str) -> Option<Vec<f32>> {
    let child = node.first_child_by_name(name)?;
    match child.attributes().first()? {
        fbxcel::low::v7400::AttributeValue::ArrF32(v) => Some(v.clone()),
        fbxcel::low::v7400::AttributeValue::ArrF64(v) => Some(v.iter().map(|&f| f as f32).collect()),
        _ => None,
    }
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
