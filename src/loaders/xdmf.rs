use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use hdf5::File as Hdf5File;
use quick_xml::Reader;
use quick_xml::events::Event;

use crate::{error::IoError, types::IoDataSet};
use super::common::{Dataset, VolumeGeometry, VolumeGrid};
use super::error::ReadError;
use super::pvd::{PvdSeries, TimestepEntry};

/// Decode an XDMF file into one or more scientific datasets.
pub fn datasets_from_path(path: &Path) -> Result<Vec<IoDataSet>, IoError> {
    read(path)
        .map(|datasets| datasets.into_iter().map(Dataset::into_io_dataset).collect())
        .map_err(Into::into)
}

#[derive(Clone, Debug, Default)]
struct XmlNode {
    name: String,
    attrs: HashMap<String, String>,
    text: String,
    children: Vec<XmlNode>,
}

/// Read an XDMF file and return one or more internal scientific datasets.
pub fn read(path: &Path) -> Result<Vec<Dataset>, ReadError> {
    let doc = parse_document(path)?;
    let domain = find_domain(&doc)?;
    let grid = first_readable_grid(domain)
        .ok_or_else(|| ReadError::Xdmf("No readable Grid found in XDMF document".to_string()))?;
    datasets_from_grid(grid, path)
}

/// Read time-series metadata from an XDMF file.
pub fn read_series(path: &Path) -> Result<PvdSeries, ReadError> {
    let doc = parse_document(path)?;
    let domain = find_domain(&doc)?;

    let mut timesteps = Vec::new();
    let top_grids = grid_children(domain);
    if top_grids.is_empty() {
        return Err(ReadError::Xdmf(
            "XDMF Domain contains no Grid elements".to_string(),
        ));
    }

    for (idx, grid) in top_grids.iter().enumerate() {
        if is_temporal_collection(grid) {
            for (child_idx, child) in grid_children(grid).iter().enumerate() {
                let time = grid_time(child).unwrap_or(child_idx as f64);
                timesteps.push(TimestepEntry {
                    time,
                    file: path.to_path_buf(),
                    selector: Some(format!("{idx}/{child_idx}")),
                });
            }
            if !timesteps.is_empty() {
                timesteps.sort_by(|a, b| a.time.total_cmp(&b.time));
                return Ok(PvdSeries { timesteps });
            }
        }
    }

    Ok(PvdSeries {
        timesteps: vec![TimestepEntry {
            time: grid_time(top_grids[0]).unwrap_or(0.0),
            file: path.to_path_buf(),
            selector: Some("0".to_string()),
        }],
    })
}

/// Read datasets from a specific selected XDMF grid.
pub fn read_selected_grid(path: &Path, selector: &str) -> Result<Vec<Dataset>, ReadError> {
    let doc = parse_document(path)?;
    let domain = find_domain(&doc)?;
    let grid = select_grid(domain, selector)
        .ok_or_else(|| ReadError::Xdmf(format!("Invalid XDMF Grid selector: {selector}")))?;
    datasets_from_grid(grid, path)
}

fn parse_document(path: &Path) -> Result<XmlNode, ReadError> {
    let text = fs::read_to_string(path)?;
    parse_xml(&text)
}

fn parse_xml(text: &str) -> Result<XmlNode, ReadError> {
    let mut reader = Reader::from_str(text);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut root: Option<XmlNode> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let mut attrs = HashMap::new();
                for attr in e.attributes() {
                    let attr =
                        attr.map_err(|err| ReadError::Xdmf(format!("XML attribute error: {err}")))?;
                    let key = String::from_utf8_lossy(attr.key.local_name().as_ref()).into_owned();
                    let value = attr
                        .decode_and_unescape_value(reader.decoder())
                        .map_err(|err| {
                            ReadError::Xdmf(format!("XML attribute decode error: {err}"))
                        })?
                        .into_owned();
                    attrs.insert(key, value);
                }
                stack.push(XmlNode {
                    name: String::from_utf8_lossy(e.local_name().as_ref()).into_owned(),
                    attrs,
                    text: String::new(),
                    children: Vec::new(),
                });
            }
            Ok(Event::Empty(e)) => {
                let mut attrs = HashMap::new();
                for attr in e.attributes() {
                    let attr =
                        attr.map_err(|err| ReadError::Xdmf(format!("XML attribute error: {err}")))?;
                    let key = String::from_utf8_lossy(attr.key.local_name().as_ref()).into_owned();
                    let value = attr
                        .decode_and_unescape_value(reader.decoder())
                        .map_err(|err| {
                            ReadError::Xdmf(format!("XML attribute decode error: {err}"))
                        })?
                        .into_owned();
                    attrs.insert(key, value);
                }
                let node = XmlNode {
                    name: String::from_utf8_lossy(e.local_name().as_ref()).into_owned(),
                    attrs,
                    text: String::new(),
                    children: Vec::new(),
                };
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    root = Some(node);
                }
            }
            Ok(Event::Text(e)) => {
                if let Some(node) = stack.last_mut() {
                    let text = e
                        .unescape()
                        .map_err(|err| ReadError::Xdmf(format!("XML text decode error: {err}")))?;
                    if !node.text.is_empty() {
                        node.text.push(' ');
                    }
                    node.text.push_str(text.trim());
                }
            }
            Ok(Event::CData(e)) => {
                if let Some(node) = stack.last_mut() {
                    let text = String::from_utf8_lossy(e.as_ref());
                    if !node.text.is_empty() {
                        node.text.push(' ');
                    }
                    node.text.push_str(text.trim());
                }
            }
            Ok(Event::End(_)) => {
                let node = stack
                    .pop()
                    .ok_or_else(|| ReadError::Xdmf("Malformed XML stack".to_string()))?;
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    root = Some(node);
                }
            }
            Ok(Event::Eof) => break,
            Ok(Event::Decl(_))
            | Ok(Event::Comment(_))
            | Ok(Event::PI(_))
            | Ok(Event::DocType(_)) => {}
            Err(err) => return Err(ReadError::Xdmf(format!("XML parse error: {err}"))),
        }
        buf.clear();
    }

    root.ok_or_else(|| ReadError::Xdmf("Empty XML document".to_string()))
}

fn find_domain(doc: &XmlNode) -> Result<&XmlNode, ReadError> {
    if doc.name == "Domain" {
        return Ok(doc);
    }
    doc.children
        .iter()
        .find(|child| child.name == "Domain")
        .ok_or_else(|| ReadError::Xdmf("XDMF document missing Domain element".to_string()))
}

fn grid_children(node: &XmlNode) -> Vec<&XmlNode> {
    node.children
        .iter()
        .filter(|child| child.name == "Grid")
        .collect()
}

fn is_temporal_collection(grid: &XmlNode) -> bool {
    grid.attrs
        .get("GridType")
        .map(|v| v.eq_ignore_ascii_case("Collection"))
        .unwrap_or(false)
        && grid
            .attrs
            .get("CollectionType")
            .map(|v| v.eq_ignore_ascii_case("Temporal"))
            .unwrap_or(false)
}

fn first_readable_grid(grid_parent: &XmlNode) -> Option<&XmlNode> {
    for grid in grid_children(grid_parent) {
        if is_temporal_collection(grid) {
            if let Some(child) = grid_children(grid).into_iter().next() {
                return Some(child);
            }
        } else {
            return Some(grid);
        }
    }
    None
}

fn select_grid<'a>(domain: &'a XmlNode, selector: &str) -> Option<&'a XmlNode> {
    let mut current = domain;
    for segment in selector.split('/') {
        let idx: usize = segment.parse().ok()?;
        let grids = grid_children(current);
        current = *grids.get(idx)?;
    }
    Some(current)
}

fn grid_time(grid: &XmlNode) -> Option<f64> {
    grid.children
        .iter()
        .find(|child| child.name == "Time")
        .and_then(|time| time.attrs.get("Value"))
        .and_then(|value| value.parse::<f64>().ok())
}

fn datasets_from_grid(grid: &XmlNode, source: &Path) -> Result<Vec<Dataset>, ReadError> {
    if grid
        .attrs
        .get("GridType")
        .map(|v| v.eq_ignore_ascii_case("Collection"))
        .unwrap_or(false)
    {
        let mut out = Vec::new();
        for child in grid_children(grid) {
            out.extend(datasets_from_grid(child, source)?);
        }
        if out.is_empty() {
            return Err(ReadError::Empty);
        }
        return Ok(out);
    }

    Ok(vec![parse_structured_grid(grid, source)?])
}

fn parse_structured_grid(grid: &XmlNode, source: &Path) -> Result<Dataset, ReadError> {
    let topology = child_named(grid, "Topology")
        .ok_or_else(|| ReadError::Xdmf("Grid missing Topology".to_string()))?;
    let geometry = child_named(grid, "Geometry")
        .ok_or_else(|| ReadError::Xdmf("Grid missing Geometry".to_string()))?;

    let topology_type = topology
        .attrs
        .get("TopologyType")
        .or_else(|| topology.attrs.get("Type"))
        .ok_or_else(|| ReadError::Xdmf("Topology missing TopologyType".to_string()))?;

    match topology_type.as_str() {
        "3DCoRectMesh" | "2DCoRectMesh" => parse_corect_grid(grid, topology, geometry, source),
        "3DRectMesh" | "2DRectMesh" => parse_rect_grid(grid, topology, geometry, source),
        other => Err(ReadError::UnsupportedFormat(format!(
            "Unsupported XDMF topology type: {other}"
        ))),
    }
}

fn parse_corect_grid(
    grid: &XmlNode,
    topology: &XmlNode,
    geometry: &XmlNode,
    source: &Path,
) -> Result<Dataset, ReadError> {
    let dims = parse_topology_dims(topology)?;
    let data_items = geometry
        .children
        .iter()
        .filter(|child| child.name == "DataItem")
        .collect::<Vec<_>>();
    if data_items.len() < 2 {
        return Err(ReadError::Xdmf(
            "CoRectMesh Geometry must contain origin and spacing DataItems".to_string(),
        ));
    }

    let origin_vals = read_data_item_f32(data_items[0], source)?;
    let spacing_vals = read_data_item_f32(data_items[1], source)?;
    let origin = vec_to_xyz(&origin_vals);
    let spacing = vec_to_xyz(&spacing_vals);

    let (point_data, cell_data) = parse_attributes(grid, source)?;
    let is_2d = dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1;

    if is_2d {
        let positions = image_data_positions(
            dims[0] as usize,
            dims[1] as usize,
            dims[2] as usize,
            origin,
            spacing,
        );
        let indices =
            structured_surface_indices(dims[0] as usize, dims[1] as usize, dims[2] as usize);
        if indices.is_empty() || positions.is_empty() {
            return Err(ReadError::Empty);
        }
        let normals = compute_normals(&positions, &indices);
        Ok(Dataset {
            positions,
            indices,
            normals,
            point_data,
            cell_data,
            edge_data: HashMap::new(),
            sparse_volume: None,
            volume: None,
            volume_mesh: None,
        })
    } else {
        Ok(Dataset {
            positions: Vec::new(),
            indices: Vec::new(),
            normals: Vec::new(),
            point_data: point_data.clone(),
            cell_data: cell_data.clone(),
            edge_data: HashMap::new(),
            sparse_volume: None,
            volume: Some(VolumeGrid {
                dims,
                geometry: VolumeGeometry::Uniform { origin, spacing },
                point_data,
                cell_data,
            }),
            volume_mesh: None,
        })
    }
}

fn parse_rect_grid(
    grid: &XmlNode,
    topology: &XmlNode,
    geometry: &XmlNode,
    source: &Path,
) -> Result<Dataset, ReadError> {
    let dims = parse_topology_dims(topology)?;
    let geometry_type = geometry
        .attrs
        .get("GeometryType")
        .or_else(|| geometry.attrs.get("Type"))
        .cloned()
        .unwrap_or_default();
    if geometry_type != "VXVYVZ" && geometry_type != "VXVY" {
        return Err(ReadError::UnsupportedFormat(format!(
            "Unsupported XDMF rectilinear geometry type: {geometry_type}"
        )));
    }

    let data_items = geometry
        .children
        .iter()
        .filter(|child| child.name == "DataItem")
        .collect::<Vec<_>>();
    if data_items.len() < 2 {
        return Err(ReadError::Xdmf(
            "RectMesh Geometry must contain coordinate DataItems".to_string(),
        ));
    }

    let xs = read_data_item_f32(data_items[0], source)?;
    let ys = read_data_item_f32(data_items[1], source)?;
    let zs = if data_items.len() >= 3 {
        read_data_item_f32(data_items[2], source)?
    } else {
        vec![0.0]
    };

    let (point_data, cell_data) = parse_attributes(grid, source)?;
    let positions = rectilinear_positions(&xs, &ys, &zs);
    let indices = structured_surface_indices(dims[0] as usize, dims[1] as usize, dims[2] as usize);
    if indices.is_empty() || positions.is_empty() {
        return Err(ReadError::Empty);
    }
    let normals = compute_normals(&positions, &indices);
    let volume = if dims[0] > 1 && dims[1] > 1 && dims[2] > 1 {
        Some(VolumeGrid {
            dims,
            geometry: VolumeGeometry::Rectilinear {
                xs: xs.clone(),
                ys: ys.clone(),
                zs: zs.clone(),
            },
            point_data: point_data.clone(),
            cell_data: cell_data.clone(),
        })
    } else {
        None
    };

    Ok(Dataset {
        positions,
        indices,
        normals,
        point_data,
        cell_data,
        edge_data: HashMap::new(),
        sparse_volume: None,
        volume,
        volume_mesh: None,
    })
}

fn parse_attributes(
    grid: &XmlNode,
    source: &Path,
) -> Result<(HashMap<String, Vec<f32>>, HashMap<String, Vec<f32>>), ReadError> {
    let mut point_data = HashMap::new();
    let mut cell_data = HashMap::new();

    for attr in grid
        .children
        .iter()
        .filter(|child| child.name == "Attribute")
    {
        let name = attr
            .attrs
            .get("Name")
            .cloned()
            .unwrap_or_else(|| "unnamed".to_string());
        let center = attr
            .attrs
            .get("Center")
            .map(|v| v.to_ascii_lowercase())
            .unwrap_or_else(|| "node".to_string());
        let data_item = child_named(attr, "DataItem")
            .ok_or_else(|| ReadError::Xdmf(format!("Attribute {name} missing DataItem")))?;
        let values = read_data_item_f32(data_item, source)?;
        match center.as_str() {
            "node" => {
                point_data.insert(name, values);
            }
            "cell" => {
                cell_data.insert(name, values);
            }
            other => {
                return Err(ReadError::UnsupportedFormat(format!(
                    "Unsupported XDMF Attribute Center: {other}"
                )));
            }
        }
    }

    Ok((point_data, cell_data))
}

fn child_named<'a>(node: &'a XmlNode, name: &str) -> Option<&'a XmlNode> {
    node.children.iter().find(|child| child.name == name)
}

fn parse_topology_dims(topology: &XmlNode) -> Result<[u32; 3], ReadError> {
    let raw = topology
        .attrs
        .get("Dimensions")
        .or_else(|| topology.attrs.get("NumberOfElements"))
        .ok_or_else(|| {
            ReadError::Xdmf("Topology missing Dimensions/NumberOfElements".to_string())
        })?;
    let dims = raw
        .split_whitespace()
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| ReadError::Xdmf(format!("Invalid topology dimension value: {s}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match dims.as_slice() {
        [ny, nx] => Ok([*nx, *ny, 1]),
        [nz, ny, nx] => Ok([*nx, *ny, *nz]),
        _ => Err(ReadError::UnsupportedFormat(format!(
            "Unsupported XDMF structured dimensions: {raw}"
        ))),
    }
}

fn read_data_item_f32(data_item: &XmlNode, source: &Path) -> Result<Vec<f32>, ReadError> {
    let format = data_item
        .attrs
        .get("Format")
        .map(|v| v.as_str())
        .unwrap_or("XML");
    match format {
        "XML" => data_item
            .text
            .split_whitespace()
            .map(|s| {
                s.parse::<f32>()
                    .map_err(|_| ReadError::Xdmf(format!("Invalid XML numeric value: {s}")))
            })
            .collect(),
        "HDF" => read_hdf_data_item_f32(data_item, source),
        other => Err(ReadError::UnsupportedFormat(format!(
            "Unsupported XDMF DataItem format: {other}"
        ))),
    }
}

fn read_hdf_data_item_f32(data_item: &XmlNode, source: &Path) -> Result<Vec<f32>, ReadError> {
    let locator = data_item.text.trim();
    let (file_part, dataset_part) = locator
        .split_once(':')
        .ok_or_else(|| ReadError::Xdmf(format!("Invalid HDF DataItem locator: {locator}")))?;
    let h5_path = resolve_relative(source, file_part);
    let file = Hdf5File::open(&h5_path)
        .map_err(|err| ReadError::Hdf5(format!("Failed to open {}: {err}", h5_path.display())))?;
    let dataset = file.dataset(dataset_part).map_err(|err| {
        ReadError::Hdf5(format!(
            "Failed to open dataset {dataset_part} in {}: {err}",
            h5_path.display()
        ))
    })?;

    if let Ok(values) = dataset.read_raw::<f32>() {
        return Ok(values);
    }
    if let Ok(values) = dataset.read_raw::<f64>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = dataset.read_raw::<i32>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = dataset.read_raw::<u32>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = dataset.read_raw::<i64>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }
    if let Ok(values) = dataset.read_raw::<u64>() {
        return Ok(values.into_iter().map(|v| v as f32).collect());
    }

    Err(ReadError::UnsupportedFormat(format!(
        "Unsupported HDF5 dataset type for locator {locator}"
    )))
}

fn resolve_relative(source: &Path, target: &str) -> PathBuf {
    let path = Path::new(target);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        source.parent().unwrap_or(Path::new(".")).join(path)
    }
}

fn vec_to_xyz(values: &[f32]) -> [f32; 3] {
    [
        *values.first().unwrap_or(&0.0),
        *values.get(1).unwrap_or(&0.0),
        *values.get(2).unwrap_or(&0.0),
    ]
}

fn rectilinear_positions(xs: &[f32], ys: &[f32], zs: &[f32]) -> Vec<[f32; 3]> {
    // When zs has only one value the data is flat in z (x-y plane in file space).
    // Remap to x-z plane: file y → world z, file z → world y.
    let flat_z = zs.len() <= 1;
    let mut positions = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &fz in zs {
        for &fy in ys {
            for &x in xs {
                let (y, z) = if flat_z { (fz, fy) } else { (fy, fz) };
                positions.push([x, y, z]);
            }
        }
    }
    positions
}

fn image_data_positions(
    ni: usize,
    nj: usize,
    nk: usize,
    origin: [f32; 3],
    spacing: [f32; 3],
) -> Vec<[f32; 3]> {
    // When nk=1 the data is flat in z (x-y plane in file space).
    // Remap to x-z plane: j→z, k→y.
    let flat_z = nk <= 1;
    let mut positions = Vec::with_capacity(ni * nj * nk.max(1));
    for iz in 0..nk.max(1) {
        for iy in 0..nj.max(1) {
            for ix in 0..ni.max(1) {
                let x = origin[0] + spacing[0] * ix as f32;
                let (y, z) = if flat_z {
                    (origin[2] + spacing[2] * iz as f32, origin[1] + spacing[1] * iy as f32)
                } else {
                    (origin[1] + spacing[1] * iy as f32, origin[2] + spacing[2] * iz as f32)
                };
                positions.push([x, y, z]);
            }
        }
    }
    positions
}

fn structured_surface_indices(ni: usize, nj: usize, nk: usize) -> Vec<u32> {
    if ni >= 2 && nj >= 2 && nk <= 1 {
        return quad_grid_indices(ni, nj);
    }
    if ni >= 2 && nk >= 2 && nj <= 1 {
        return quad_grid_indices(ni, nk);
    }
    if nj >= 2 && nk >= 2 && ni <= 1 {
        return quad_grid_indices(nj, nk);
    }
    if ni < 2 || nj < 2 || nk < 2 {
        return Vec::new();
    }

    let idx = |i: usize, j: usize, k: usize| -> u32 { (i + j * ni + k * ni * nj) as u32 };
    let mut indices = Vec::new();

    for j in 0..(nj - 1) {
        for i in 0..(ni - 1) {
            let v00 = idx(i, j, 0);
            let v10 = idx(i + 1, j, 0);
            let v01 = idx(i, j + 1, 0);
            let v11 = idx(i + 1, j + 1, 0);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);

            let v00 = idx(i, j, nk - 1);
            let v10 = idx(i + 1, j, nk - 1);
            let v01 = idx(i, j + 1, nk - 1);
            let v11 = idx(i + 1, j + 1, nk - 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }
    for k in 0..(nk - 1) {
        for i in 0..(ni - 1) {
            let v00 = idx(i, 0, k);
            let v10 = idx(i + 1, 0, k);
            let v01 = idx(i, 0, k + 1);
            let v11 = idx(i + 1, 0, k + 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);

            let v00 = idx(i, nj - 1, k);
            let v10 = idx(i + 1, nj - 1, k);
            let v01 = idx(i, nj - 1, k + 1);
            let v11 = idx(i + 1, nj - 1, k + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
        }
    }
    for k in 0..(nk - 1) {
        for j in 0..(nj - 1) {
            let v00 = idx(0, j, k);
            let v10 = idx(0, j + 1, k);
            let v01 = idx(0, j, k + 1);
            let v11 = idx(0, j + 1, k + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);

            let v00 = idx(ni - 1, j, k);
            let v10 = idx(ni - 1, j + 1, k);
            let v01 = idx(ni - 1, j, k + 1);
            let v11 = idx(ni - 1, j + 1, k + 1);
            indices.extend_from_slice(&[v00, v01, v10, v10, v01, v11]);
        }
    }

    indices
}

fn quad_grid_indices(nx: usize, ny: usize) -> Vec<u32> {
    let idx = |i: usize, j: usize| -> u32 { (i + j * nx) as u32 };
    let mut indices = Vec::with_capacity((nx - 1) * (ny - 1) * 6);
    for j in 0..(ny - 1) {
        for i in 0..(nx - 1) {
            let v00 = idx(i, j);
            let v10 = idx(i + 1, j);
            let v01 = idx(i, j + 1);
            let v11 = idx(i + 1, j + 1);
            indices.extend_from_slice(&[v00, v10, v01, v10, v11, v01]);
        }
    }
    indices
}

fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let a = glam::Vec3::from_array(positions[tri[0] as usize]);
        let b = glam::Vec3::from_array(positions[tri[1] as usize]);
        let c = glam::Vec3::from_array(positions[tri[2] as usize]);
        let n = (b - a).cross(c - a);
        for &idx in tri {
            let acc = glam::Vec3::from_array(normals[idx as usize]) + n;
            normals[idx as usize] = acc.to_array();
        }
    }
    for n in &mut normals {
        *n = glam::Vec3::from_array(*n).normalize_or_zero().to_array();
    }
    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_temporal_selector_paths() {
        let xml = r#"
            <Xdmf Version="3.0">
              <Domain>
                <Grid Name="Series" GridType="Collection" CollectionType="Temporal">
                  <Grid Name="Step0"><Time Value="0.0"/></Grid>
                  <Grid Name="Step1"><Time Value="1.5"/></Grid>
                </Grid>
              </Domain>
            </Xdmf>
        "#;
        let doc = parse_xml(xml).unwrap();
        let domain = find_domain(&doc).unwrap();
        let mut timesteps = Vec::new();
        for (idx, grid) in grid_children(domain).iter().enumerate() {
            if is_temporal_collection(grid) {
                for (child_idx, child) in grid_children(grid).iter().enumerate() {
                    timesteps.push(TimestepEntry {
                        time: grid_time(child).unwrap_or_default(),
                        file: PathBuf::new(),
                        selector: Some(format!("{idx}/{child_idx}")),
                    });
                }
            }
        }
        assert_eq!(timesteps.len(), 2);
        assert_eq!(timesteps[0].selector.as_deref(), Some("0/0"));
        assert_eq!(timesteps[1].selector.as_deref(), Some("0/1"));
        assert_eq!(timesteps[1].time, 1.5);
    }

    #[test]
    fn reads_corect_xdmf_with_hdf_attribute() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("brimcast-xdmf-{stamp}"));
        std::fs::create_dir_all(&dir).unwrap();
        let h5_path = dir.join("volume.h5");
        let xdmf_path = dir.join("volume.xdmf");

        let file = Hdf5File::create(&h5_path).unwrap();
        let ds = file
            .new_dataset::<f32>()
            .shape([8])
            .create("pressure")
            .unwrap();
        ds.write(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();

        let xml = r#"
            <Xdmf Version="3.0">
              <Domain>
                <Grid Name="Volume">
                  <Topology TopologyType="3DCoRectMesh" Dimensions="2 2 2"/>
                  <Geometry GeometryType="ORIGIN_DXDYDZ">
                    <DataItem Dimensions="3" Format="XML">0 0 0</DataItem>
                    <DataItem Dimensions="3" Format="XML">1 1 1</DataItem>
                  </Geometry>
                  <Attribute Name="pressure" Center="Node">
                    <DataItem Dimensions="8" Format="HDF">volume.h5:/pressure</DataItem>
                  </Attribute>
                </Grid>
              </Domain>
            </Xdmf>
        "#;
        std::fs::write(&xdmf_path, xml).unwrap();

        let datasets = read(&xdmf_path).unwrap();
        assert_eq!(datasets.len(), 1);
        let dataset = &datasets[0];
        assert!(dataset.volume.is_some());
        let volume = dataset.volume.as_ref().unwrap();
        assert_eq!(volume.dims, [2, 2, 2]);
        assert_eq!(volume.point_data.get("pressure").unwrap().len(), 8);
        assert_eq!(volume.point_data.get("pressure").unwrap()[5], 5.0);

        let _ = std::fs::remove_file(&xdmf_path);
        let _ = std::fs::remove_file(&h5_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
