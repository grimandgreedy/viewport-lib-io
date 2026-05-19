use std::path::Path;

use crate::error::IoError;
use crate::types::{IoMaterial, IoMesh, IoScene, TextureData, TextureSource};

/// Decode a glTF or GLB file into a CPU-side scene.
pub fn scene_from_path(path: &Path) -> Result<IoScene, IoError> {
    #[cfg(feature = "gltf")]
    {
        if !path.exists() {
            return Err(IoError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("file not found: {}", path.display()),
            )));
        }

        let (document, buffers, images) = gltf::import(path)
            .map_err(|error| IoError::Parse(format!("glTF load failed ({}): {error:?}", path.display())))?;
        let parent_dir = path.parent().unwrap_or(Path::new("."));

        let materials = document
            .materials()
            .map(|material| convert_material(&material, &images, parent_dir))
            .collect();

        let mut meshes = Vec::new();
        for scene in document.scenes() {
            for node in scene.nodes() {
                collect_node(&node, &buffers, glam::Mat4::IDENTITY, None, &mut meshes);
            }
        }

        Ok(IoScene {
            meshes,
            materials,
            ..IoScene::default()
        })
    }

    #[cfg(not(feature = "gltf"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "gltf",
            context: "glTF scene decoding",
        })
    }
}

fn collect_node(
    node: &gltf::Node,
    buffers: &[gltf::buffer::Data],
    parent_world: glam::Mat4,
    parent_mesh_index: Option<usize>,
    out: &mut Vec<IoMesh>,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;

    let my_first_index = out.len();
    let mut this_node_has_mesh = false;

    if let Some(mesh) = node.mesh() {
        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if let Some(mut imported) = convert_primitive(&primitive, buffers, &mesh, primitive_index) {
                imported.transform = world;
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
        collect_node(&child, buffers, world, child_parent, out);
    }
}

fn convert_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    mesh: &gltf::Mesh,
    primitive_index: usize,
) -> Option<IoMesh> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
    if positions.is_empty() {
        return None;
    }

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let normals = reader
        .read_normals()
        .map(|iter| iter.collect())
        .unwrap_or_else(|| compute_vertex_normals(&positions, &indices));

    let uvs = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().collect());

    let tangents = reader.read_tangents().map(|iter| iter.collect());

    let material_index = primitive.material().index();

    let base_name = mesh
        .name()
        .map(std::borrow::ToOwned::to_owned)
        .unwrap_or_else(|| format!("mesh_{}", mesh.index()));
    let name = if mesh.primitives().len() > 1 {
        format!("{base_name}.{primitive_index}")
    } else {
        base_name
    };

    let mut mesh_data = viewport_lib::MeshData::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    mesh_data.uvs = uvs;
    mesh_data.tangents = tangents;

    Some(IoMesh {
        name,
        mesh_data,
        material_index,
        transform: glam::Mat4::IDENTITY,
        two_sided: primitive.material().double_sided(),
        parent_index: None,
        ..IoMesh::default()
    })
}

fn convert_material(
    material: &gltf::Material,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> IoMaterial {
    let pbr = material.pbr_metallic_roughness();
    let base_color_factor = pbr.base_color_factor();
    let base_color = [base_color_factor[0], base_color_factor[1], base_color_factor[2]];

    let base_color_texture = pbr
        .base_color_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
    let normal_map_texture = material
        .normal_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));
    let ao_texture = material
        .occlusion_texture()
        .and_then(|info| image_to_texture_source(&info.texture(), images, parent_dir));

    IoMaterial {
        name: material
            .name()
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_else(|| format!("material_{}", material.index().unwrap_or(0))),
        base_color,
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        opacity: base_color_factor[3],
        base_color_texture,
        normal_map_texture,
        ao_texture,
    }
}

fn image_to_texture_source(
    texture: &gltf::Texture,
    images: &[gltf::image::Data],
    parent_dir: &Path,
) -> Option<TextureSource> {
    let source = texture.source();
    let index = source.index();

    if let Some(data) = images.get(index) {
        return Some(TextureSource::Decoded(TextureData {
            width: data.width,
            height: data.height,
            rgba: to_rgba8(data),
        }));
    }

    match source.source() {
        gltf::image::Source::Uri { uri, .. } if !uri.starts_with("data:") => {
            Some(TextureSource::File(parent_dir.join(uri)))
        }
        _ => None,
    }
}

fn to_rgba8(data: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;

    match data.format {
        Format::R8G8B8A8 => data.pixels.clone(),
        Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 3 * 4);
            for rgb in data.pixels.chunks_exact(3) {
                rgba.extend_from_slice(rgb);
                rgba.push(255);
            }
            rgba
        }
        Format::R8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() * 4);
            for &r in &data.pixels {
                rgba.extend_from_slice(&[r, r, r, 255]);
            }
            rgba
        }
        Format::R8G8 => {
            let mut rgba = Vec::with_capacity(data.pixels.len() / 2 * 4);
            for rg in data.pixels.chunks_exact(2) {
                rgba.extend_from_slice(&[rg[0], rg[1], 0, 255]);
            }
            rgba
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            let bytes_per_pixel = match data.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                Format::R16G16B16A16 => 4,
                _ => unreachable!(),
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * bytes_per_pixel * 2;
                let mut channels = [0u8; 4];
                for c in 0..bytes_per_pixel {
                    let lo = data.pixels.get(base + c * 2).copied().unwrap_or(0);
                    let hi = data.pixels.get(base + c * 2 + 1).copied().unwrap_or(0);
                    channels[c] = (u16::from_le_bytes([lo, hi]) >> 8) as u8;
                }
                match bytes_per_pixel {
                    1 => {
                        channels[1] = channels[0];
                        channels[2] = channels[0];
                        channels[3] = 255;
                    }
                    2 => {
                        channels[2] = 0;
                        channels[3] = 255;
                    }
                    3 => channels[3] = 255,
                    _ => {}
                }
                rgba.extend_from_slice(&channels);
            }
            rgba
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(data.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            let pixel_count = (data.width * data.height) as usize;
            let mut rgba = Vec::with_capacity(pixel_count * 4);
            for i in 0..pixel_count {
                let base = i * channels * 4;
                let mut out = [0u8; 4];
                for c in 0..channels.min(4) {
                    let bytes = [
                        data.pixels.get(base + c * 4).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 1).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 2).copied().unwrap_or(0),
                        data.pixels.get(base + c * 4 + 3).copied().unwrap_or(0),
                    ];
                    let value = f32::from_le_bytes(bytes);
                    out[c] = (value.clamp(0.0, 1.0) * 255.0) as u8;
                }
                if channels < 4 {
                    out[3] = 255;
                }
                rgba.extend_from_slice(&out);
            }
            rgba
        }
    }
}

fn compute_vertex_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let (i0, i1, i2) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        let v0 = glam::Vec3::from_array(positions[i0]);
        let v1 = glam::Vec3::from_array(positions[i1]);
        let v2 = glam::Vec3::from_array(positions[i2]);
        let n = (v1 - v0).cross(v2 - v0);
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }

    normals
        .into_iter()
        .map(|n| {
            let n = n.normalize_or_zero();
            [n.x, n.y, n.z]
        })
        .collect()
}
