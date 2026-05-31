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
    IoMaterial, IoMesh, IoScene, Joint, Skeleton, SkinWeights, SurfaceMesh, TextureData,
    TextureSource,
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

        Ok(IoScene {
            meshes,
            materials,
            skeletons,
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

fn extract_node_transform(
    mesh_model: &fbxcel_dom::v7400::object::model::MeshHandle<'_>,
    axis_transform: &glam::Mat4,
    unit_scale: f32,
) -> glam::Mat4 {
    let props = match mesh_model.direct_properties() {
        Some(props) => props,
        None => return *axis_transform * glam::Mat4::from_scale(glam::Vec3::splat(unit_scale)),
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
    *axis_transform * glam::Mat4::from_scale(glam::Vec3::splat(unit_scale)) * local * geometric
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
    // Match the Y-up to Z-up reorientation that drake-assets applies to FBX
    // vertex positions, so joint inverse-binds land in the same scene space.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[cfg(feature = "fbx")]
    #[test]
    fn smoke_skin_extraction_gamergirl_01() {
        let path = Path::new(
            "/Users/noah/Clone/DRAKE/crates/drake-demo/assets/GamerGirl/Base_mesh/SK_GamerGirl_01.fbx",
        );
        if !path.exists() {
            eprintln!("skipping: GamerGirl pack not present");
            return;
        }
        let scene = scene_from_path(path).expect("fbx must load");
        assert!(
            !scene.skeletons.is_empty(),
            "expected at least one skeleton extracted from skinned FBX"
        );
        let skeleton = &scene.skeletons[0];
        assert!(
            !skeleton.joints.is_empty(),
            "skeleton must contain at least one joint"
        );
        let skinned_count = scene
            .meshes
            .iter()
            .filter(|m| m.skeleton_index.is_some() && m.mesh.skin_weights.is_some())
            .count();
        assert!(
            skinned_count > 0,
            "expected at least one mesh with bound skin weights"
        );
        // Sanity-check the weights of one skinned mesh.
        if let Some(mesh) = scene
            .meshes
            .iter()
            .find(|m| m.mesh.skin_weights.is_some())
        {
            let sw = mesh.mesh.skin_weights.as_ref().unwrap();
            assert_eq!(sw.joint_indices.len(), mesh.mesh.positions.len());
            assert_eq!(sw.joint_weights.len(), mesh.mesh.positions.len());
            // Find at least one vertex with non-zero weight.
            let has_weight = sw
                .joint_weights
                .iter()
                .any(|w| w.iter().any(|&v| v > 0.0));
            assert!(has_weight, "all vertex skin weights are zero");
        }
    }
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
