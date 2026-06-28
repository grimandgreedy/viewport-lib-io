use std::collections::HashMap;
use std::path::Path;

use crate::error::IoError;
use crate::types::SurfaceMesh;

/// Decode a Gmsh `.msh` file into neutral surface mesh data.
pub fn mesh_from_path(path: &Path) -> Result<SurfaceMesh, IoError> {
    let text = std::fs::read_to_string(path)?;
    load_msh(&text)
}

/// Nodes indexed by their Gmsh tag. Tags are arbitrary integers, not dense indices.
type NodeMap = HashMap<u64, [f64; 3]>;

fn load_msh(text: &str) -> Result<SurfaceMesh, IoError> {
    let version = parse_version(text)?;

    let nodes = if version < 4.0 {
        parse_nodes_v2(text)?
    } else {
        parse_nodes_v4(text)?
    };

    if nodes.is_empty() {
        return Err(IoError::Parse("no nodes found in msh file".into()));
    }

    let (surface_tris, surface_quads, volume_tets) = if version < 4.0 {
        parse_elements_v2(text)?
    } else {
        parse_elements_v4(text)?
    };

    let triangles: Vec<[u64; 3]> = if !surface_tris.is_empty() || !surface_quads.is_empty() {
        let mut tris = surface_tris;
        for quad in surface_quads {
            tris.push([quad[0], quad[1], quad[2]]);
            tris.push([quad[0], quad[2], quad[3]]);
        }
        tris
    } else if !volume_tets.is_empty() {
        extract_tet_boundary(&volume_tets)
    } else {
        return Err(IoError::Parse(
            "no renderable elements found in msh file".into(),
        ));
    };

    let mut tag_to_index: HashMap<u64, u32> = HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for triangle in &triangles {
        for &tag in triangle {
            let position = nodes.get(&tag).ok_or_else(|| {
                IoError::Parse(format!("element references undeclared node tag {tag}"))
            })?;
            let index = if let Some(&index) = tag_to_index.get(&tag) {
                index
            } else {
                let index = positions.len() as u32;
                positions.push([position[0] as f32, position[1] as f32, position[2] as f32]);
                tag_to_index.insert(tag, index);
                index
            };
            indices.push(index);
        }
    }

    let normals = compute_smooth_normals(&positions, &indices);

    let mut mesh_data = SurfaceMesh::default();
    mesh_data.positions = positions;
    mesh_data.normals = normals;
    mesh_data.indices = indices;
    Ok(mesh_data)
}

fn parse_version(text: &str) -> Result<f32, IoError> {
    let mut in_section = false;
    for line in text.lines() {
        match line.trim() {
            "$MeshFormat" => in_section = true,
            "$EndMeshFormat" => break,
            value if in_section => {
                return value
                    .split_whitespace()
                    .next()
                    .and_then(|version| version.parse().ok())
                    .ok_or_else(|| IoError::Parse("cannot read msh version number".into()));
            }
            _ => {}
        }
    }
    Err(IoError::Parse("missing $MeshFormat section".into()))
}

fn parse_nodes_v2(text: &str) -> Result<NodeMap, IoError> {
    let mut nodes = NodeMap::new();
    let mut in_section = false;
    let mut total = 0usize;
    let mut read = 0usize;

    for line in text.lines() {
        match line.trim() {
            "$Nodes" => in_section = true,
            "$EndNodes" => break,
            _ if !in_section => {}
            value if total == 0 => total = value.parse().unwrap_or(0),
            value => {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if parts.len() >= 4
                    && let (Ok(tag), Ok(x), Ok(y), Ok(z)) = (
                        parts[0].parse::<u64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                    )
                {
                    nodes.insert(tag, [x, y, z]);
                    read += 1;
                    if read >= total {
                        break;
                    }
                }
            }
        }
    }

    Ok(nodes)
}

fn parse_elements_v2(text: &str) -> Result<(Vec<[u64; 3]>, Vec<[u64; 4]>, Vec<[u64; 4]>), IoError> {
    let mut triangles = Vec::new();
    let mut quads = Vec::new();
    let mut tets = Vec::new();

    let mut in_section = false;
    let mut total = 0usize;
    let mut read = 0usize;

    for line in text.lines() {
        match line.trim() {
            "$Elements" => in_section = true,
            "$EndElements" => break,
            _ if !in_section => {}
            value if total == 0 => total = value.parse().unwrap_or(0),
            value => {
                let parts: Vec<u64> = value
                    .split_whitespace()
                    .filter_map(|token| token.parse().ok())
                    .collect();

                if parts.len() >= 3 {
                    let element_type = parts[1];
                    let num_tags = parts[2] as usize;
                    let node_start = 3 + num_tags;
                    push_element(
                        element_type,
                        &parts,
                        node_start,
                        &mut triangles,
                        &mut quads,
                        &mut tets,
                    );
                }

                read += 1;
                if read >= total {
                    break;
                }
            }
        }
    }

    Ok((triangles, quads, tets))
}

fn parse_nodes_v4(text: &str) -> Result<NodeMap, IoError> {
    let mut nodes = NodeMap::new();
    let mut lines = text.lines();

    for line in lines.by_ref() {
        if line.trim() == "$Nodes" {
            break;
        }
    }

    let header = match lines.next() {
        Some(line) => line,
        None => return Ok(nodes),
    };
    let num_blocks: usize = header
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    'blocks: for _ in 0..num_blocks {
        let block_header = loop {
            match lines.next() {
                Some(line) if line.trim() == "$EndNodes" => return Ok(nodes),
                Some(line) if !line.trim().is_empty() => break line,
                Some(_) => {}
                None => return Ok(nodes),
            }
        };

        let parts: Vec<&str> = block_header.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        let num_in_block: usize = parts[3].parse().unwrap_or(0);
        let mut tags: Vec<u64> = Vec::with_capacity(num_in_block);
        for _ in 0..num_in_block {
            match lines.next() {
                Some(line) => {
                    if let Ok(tag) = line.trim().parse::<u64>() {
                        tags.push(tag);
                    }
                }
                None => break 'blocks,
            }
        }

        for tag in tags {
            match lines.next() {
                Some(line) => {
                    let coords: Vec<&str> = line.split_whitespace().collect();
                    if coords.len() >= 3 {
                        let x = coords[0].parse().unwrap_or(0.0);
                        let y = coords[1].parse().unwrap_or(0.0);
                        let z = coords[2].parse().unwrap_or(0.0);
                        nodes.insert(tag, [x, y, z]);
                    }
                }
                None => break 'blocks,
            }
        }
    }

    Ok(nodes)
}

fn parse_elements_v4(text: &str) -> Result<(Vec<[u64; 3]>, Vec<[u64; 4]>, Vec<[u64; 4]>), IoError> {
    let mut triangles = Vec::new();
    let mut quads = Vec::new();
    let mut tets = Vec::new();

    let mut lines = text.lines();
    for line in lines.by_ref() {
        if line.trim() == "$Elements" {
            break;
        }
    }

    let header = match lines.next() {
        Some(line) => line,
        None => return Ok((triangles, quads, tets)),
    };
    let num_blocks: usize = header
        .split_whitespace()
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);

    'blocks: for _ in 0..num_blocks {
        let block_header = loop {
            match lines.next() {
                Some(line) if line.trim() == "$EndElements" => return Ok((triangles, quads, tets)),
                Some(line) if !line.trim().is_empty() => break line,
                Some(_) => {}
                None => return Ok((triangles, quads, tets)),
            }
        };

        let parts: Vec<&str> = block_header.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        let element_type: u64 = parts[2].parse().unwrap_or(0);
        let num_in_block: usize = parts[3].parse().unwrap_or(0);

        for _ in 0..num_in_block {
            let line = match lines.next() {
                Some(line) => line,
                None => break 'blocks,
            };
            let parts: Vec<u64> = line
                .split_whitespace()
                .filter_map(|value| value.parse().ok())
                .collect();
            if !parts.is_empty() {
                push_element(
                    element_type,
                    &parts,
                    1,
                    &mut triangles,
                    &mut quads,
                    &mut tets,
                );
            }
        }
    }

    Ok((triangles, quads, tets))
}

fn push_element(
    element_type: u64,
    parts: &[u64],
    node_start: usize,
    triangles: &mut Vec<[u64; 3]>,
    quads: &mut Vec<[u64; 4]>,
    tets: &mut Vec<[u64; 4]>,
) {
    let n = node_start;
    match element_type {
        2 | 9 | 21 if parts.len() >= n + 3 => {
            triangles.push([parts[n], parts[n + 1], parts[n + 2]]);
        }
        3 | 10 | 16 if parts.len() >= n + 4 => {
            quads.push([parts[n], parts[n + 1], parts[n + 2], parts[n + 3]]);
        }
        4 | 11 if parts.len() >= n + 4 => {
            tets.push([parts[n], parts[n + 1], parts[n + 2], parts[n + 3]]);
        }
        _ => {}
    }
}

fn extract_tet_boundary(tets: &[[u64; 4]]) -> Vec<[u64; 3]> {
    let mut face_map: HashMap<[u64; 3], (u8, [u64; 3])> = HashMap::new();

    for &[a, b, c, d] in tets {
        let faces = [[a, c, b], [a, b, d], [b, c, d], [a, d, c]];
        for face in faces {
            let mut key = face;
            key.sort_unstable();
            let entry = face_map.entry(key).or_insert((0, face));
            entry.0 += 1;
        }
    }

    face_map
        .into_values()
        .filter(|(count, _)| *count == 1)
        .map(|(_, face)| face)
        .collect()
}

fn compute_smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![glam::Vec3::ZERO; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let a = glam::Vec3::from(positions[triangle[0] as usize]);
        let b = glam::Vec3::from(positions[triangle[1] as usize]);
        let c = glam::Vec3::from(positions[triangle[2] as usize]);
        let face_normal = (b - a).cross(c - a);
        for &vertex in triangle {
            normals[vertex as usize] += face_normal;
        }
    }

    normals
        .iter()
        .map(|normal| {
            let normal = normal.normalize_or_zero();
            [normal.x, normal.y, normal.z]
        })
        .collect()
}
