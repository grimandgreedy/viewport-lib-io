//! Per-point and per-cell fields, read off a VTK attribute set into the neutral
//! attribute types, plus the normals a surface needs when the file carries none.

use std::collections::HashMap;

use vtkio::model::{Attribute, Attributes};

pub(super) fn extract_attributes(
    attrs: &Attributes,
    num_points: usize,
    num_cells: usize,
    detect_halfedges: bool,
) -> (
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<f32>>,
) {
    let mut point_data = HashMap::new();
    let mut cell_data = HashMap::new();
    let mut edge_data = HashMap::new();
    let halfedge_len = if detect_halfedges { 3 * num_cells } else { 0 };

    for attr in &attrs.point {
        extract_attribute_into(
            attr,
            num_points,
            halfedge_len,
            &mut point_data,
            &mut edge_data,
        );
    }
    for attr in &attrs.cell {
        extract_attribute_into(
            attr,
            num_cells,
            halfedge_len,
            &mut cell_data,
            &mut edge_data,
        );
    }

    (point_data, cell_data, edge_data)
}

pub(super) fn extract_attribute_into(
    attr: &Attribute,
    expected_len: usize,
    halfedge_len: usize,
    out: &mut HashMap<String, Vec<f32>>,
    edge_out: &mut HashMap<String, Vec<f32>>,
) {
    match attr {
        Attribute::DataArray(arr) => {
            let num_comp = arr.elem.num_comp() as usize;
            let Some(floats) = arr.data.clone().cast_into::<f32>() else {
                return;
            };
            let n_tuples = if num_comp > 0 {
                floats.len() / num_comp
            } else {
                0
            };
            if n_tuples == expected_len {
                emit_scalar_arrays(&arr.name, num_comp, floats, expected_len, out);
            } else if halfedge_len > 0 && num_comp == 1 && floats.len() == halfedge_len {
                edge_out.insert(arr.name.clone(), floats);
            }
        }
        Attribute::Field { data_array, .. } => {
            for field in data_array {
                let num_comp = field.elem as usize;
                let Some(floats) = field.data.clone().cast_into::<f32>() else {
                    continue;
                };
                let n_tuples = if num_comp > 0 {
                    floats.len() / num_comp
                } else {
                    0
                };
                if n_tuples == expected_len {
                    emit_scalar_arrays(&field.name, num_comp, floats, expected_len, out);
                } else if halfedge_len > 0 && num_comp == 1 && floats.len() == halfedge_len {
                    edge_out.insert(field.name.clone(), floats);
                }
            }
        }
    }
}

pub(super) fn emit_scalar_arrays(
    name: &str,
    num_comp: usize,
    floats: Vec<f32>,
    expected_len: usize,
    out: &mut HashMap<String, Vec<f32>>,
) {
    if num_comp == 0 || floats.len() != num_comp * expected_len {
        return;
    }
    if num_comp == 1 {
        out.insert(name.to_string(), floats);
        return;
    }
    if num_comp == 3 {
        let mut xs = Vec::with_capacity(expected_len);
        let mut ys = Vec::with_capacity(expected_len);
        let mut zs = Vec::with_capacity(expected_len);
        let mut mags = Vec::with_capacity(expected_len);
        for chunk in floats.chunks_exact(3) {
            let (x, y, z) = (chunk[0], chunk[1], chunk[2]);
            xs.push(x);
            ys.push(y);
            zs.push(z);
            mags.push((x * x + y * y + z * z).sqrt());
        }
        out.insert(format!("{name}_x"), xs);
        out.insert(format!("{name}_y"), ys);
        out.insert(format!("{name}_z"), zs);
        out.insert(format!("{name}_mag"), mags);
        return;
    }
    let mut mags = Vec::with_capacity(expected_len);
    for chunk in floats.chunks_exact(num_comp) {
        mags.push(chunk.iter().map(|value| value * value).sum::<f32>().sqrt());
    }
    out.insert(format!("{name}_mag"), mags);
}

pub(super) fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = glam::Vec3::from(positions[i0]);
        let p1 = glam::Vec3::from(positions[i1]);
        let p2 = glam::Vec3::from(positions[i2]);
        let normal = (p1 - p0).cross(p2 - p0);
        for index in [i0, i1, i2] {
            normals[index][0] += normal.x;
            normals[index][1] += normal.y;
            normals[index][2] += normal.z;
        }
    }
    for normal in &mut normals {
        *normal = glam::Vec3::from(*normal).normalize_or_zero().into();
    }
    normals
}
