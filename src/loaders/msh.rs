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

#[cfg(test)]
mod tests {
    use super::*;

    /// Gmsh 2.2: nodes in one flat list, elements carrying a variable tag count
    /// before their node list.
    fn v2_quad() -> String {
        String::from(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
             $Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n\
             $Elements\n2\n1 2 2 0 1 1 2 3\n2 2 2 0 1 1 3 4\n$EndElements\n",
        )
    }

    /// Gmsh 4.1: the same quad, described through entity blocks with the node
    /// tags and their coordinates on separate runs of lines.
    fn v4_quad() -> String {
        String::from(
            "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
             $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n\
             0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
             $Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n",
        )
    }

    #[test]
    fn the_version_line_decides_which_parser_runs() {
        assert_eq!(parse_version(&v2_quad()).unwrap(), 2.2);
        assert_eq!(parse_version(&v4_quad()).unwrap(), 4.1);
        assert!(parse_version("$Nodes\n0\n$EndNodes\n").is_err());
        assert!(parse_version("$MeshFormat\nnot-a-number\n$EndMeshFormat\n").is_err());
    }

    /// The two format versions lay the same mesh out completely differently, so
    /// decoding both and comparing checks each parser against an independent
    /// description rather than against a transcript.
    #[test]
    fn both_format_versions_decode_to_the_same_mesh() {
        let v2 = load_msh(&v2_quad()).expect("decode 2.2");
        let v4 = load_msh(&v4_quad()).expect("decode 4.1");

        assert_eq!(v2.positions, v4.positions);
        assert_eq!(v2.indices, v4.indices);
        assert_eq!(v2.positions.len(), 4, "the shared corner is emitted once");
        assert_eq!(v2.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    /// Node tags are arbitrary integers, not positions in an array. A file
    /// numbering its nodes in the thousands decodes to the same dense mesh.
    #[test]
    fn sparse_node_tags_are_remapped_to_dense_indices() {
        let text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                    $Nodes\n3\n1000 0 0 0\n2000 1 0 0\n3000 1 1 0\n$EndNodes\n\
                    $Elements\n1\n1 2 2 0 1 1000 2000 3000\n$EndElements\n";

        let mesh = load_msh(text).expect("decode sparse tags");
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.positions[1], [1.0, 0.0, 0.0]);
    }

    /// A quad element becomes two triangles across its 0-2 diagonal.
    #[test]
    fn a_quad_element_splits_into_two_triangles() {
        let text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                    $Nodes\n4\n1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n$EndNodes\n\
                    $Elements\n1\n1 3 2 0 1 1 2 3 4\n$EndElements\n";

        let mesh = load_msh(text).expect("decode quad element");
        assert_eq!(mesh.indices, vec![0, 1, 2, 0, 2, 3]);
    }

    /// With no surface elements the loader falls back to the boundary of the
    /// volume. Two tetrahedra glued on one face have eight faces between them,
    /// of which six are boundary.
    #[test]
    fn a_tet_only_mesh_falls_back_to_its_boundary() {
        let faces = extract_tet_boundary(&[[1, 2, 3, 4], [1, 2, 3, 5]]);
        assert_eq!(faces.len(), 6);

        let mut keys: Vec<[u64; 3]> = faces
            .into_iter()
            .map(|mut f| {
                f.sort_unstable();
                f
            })
            .collect();
        keys.sort_unstable();
        assert!(
            !keys.contains(&[1, 2, 3]),
            "the glued face is interior, not boundary"
        );
    }

    /// An element naming a node the file never declared is a parse error, not a
    /// panic or a silently dropped triangle.
    #[test]
    fn an_undeclared_node_tag_is_an_error() {
        let text = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                    $Nodes\n1\n1 0 0 0\n$EndNodes\n\
                    $Elements\n1\n1 2 2 0 1 1 7 9\n$EndElements\n";

        let error = load_msh(text).expect_err("undeclared tag");
        assert!(
            format!("{error}").contains("undeclared node tag"),
            "the message names the problem: {error}"
        );
    }

    /// A file with nodes but nothing renderable, and a file with no nodes at
    /// all, both report rather than returning an empty mesh.
    #[test]
    fn a_mesh_with_nothing_to_draw_is_an_error() {
        let no_elements = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                           $Nodes\n1\n1 0 0 0\n$EndNodes\n\
                           $Elements\n0\n$EndElements\n";
        assert!(load_msh(no_elements).is_err());

        let no_nodes = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
                        $Nodes\n0\n$EndNodes\n$Elements\n0\n$EndElements\n";
        assert!(load_msh(no_nodes).is_err());
    }

    /// Smooth normals are unit length and, for a flat quad in the XY plane,
    /// all point the same way.
    #[test]
    fn smooth_normals_are_unit_length() {
        let mesh = load_msh(&v2_quad()).expect("decode 2.2");
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        for normal in &mesh.normals {
            let length = glam::Vec3::from(*normal).length();
            assert!((length - 1.0).abs() < 1e-5, "unit normal, got {length}");
            assert!(normal[2] > 0.99, "a +Z quad faces +Z, got {normal:?}");
        }
    }
}
