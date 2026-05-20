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
use crate::types::{IoMaterial, IoMesh, IoScene, SurfaceMesh, TextureData, TextureSource};

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
                let mut positions_ok = true;
                for triangle_vertex in triangle_vertices.triangle_vertex_indices() {
                    match triangle_vertices.control_point(triangle_vertex) {
                        Some(point) => positions.push([point.x as f32, point.y as f32, point.z as f32]),
                        None => {
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

                            let mut mesh_data = SurfaceMesh::default();
                            mesh_data.positions = sub_positions;
                            mesh_data.normals = sub_normals;
                            mesh_data.indices = sub_indices;
                            mesh_data.uvs = sub_uvs;

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

                meshes.push(IoMesh {
                    name: model_name,
                    mesh: mesh_data,
                    material_index: model_materials.first().copied(),
                    transform: node_transform,
                    two_sided: false,
                    parent_index: None,
                    ..IoMesh::default()
                });
            }
        }

        assign_hierarchy(&document, &mut meshes);

        Ok(IoScene {
            meshes,
            materials,
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
