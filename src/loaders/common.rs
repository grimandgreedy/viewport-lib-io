use std::collections::HashMap;

use crate::types::{
    AttributeData, AttributeDomain, DecodedDataSet, PointSet, SparseGrid, StructuredVolume,
    SurfaceMesh, VolumeGridGeometry, VolumeMesh,
};

/// Structured scalar volume data for scientific dataset loaders.
#[derive(Clone, Debug)]
pub struct VolumeGrid {
    /// Point dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Grid geometry.
    pub geometry: VolumeGeometry,
    /// Point-centred scalar/vector fields keyed by attribute name.
    pub point_data: HashMap<String, Vec<f32>>,
    /// Cell-centred scalar/vector fields keyed by attribute name.
    pub cell_data: HashMap<String, Vec<f32>>,
}

/// Geometry model for a structured volume.
#[derive(Clone, Debug)]
pub enum VolumeGeometry {
    /// Uniform voxel spacing.
    Uniform { origin: [f32; 3], spacing: [f32; 3] },
    /// Axis-aligned grid with variable spacing along each axis.
    Rectilinear {
        xs: Vec<f32>,
        ys: Vec<f32>,
        zs: Vec<f32>,
    },
}

impl VolumeGrid {
    /// World-space bounds of the volume.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        match &self.geometry {
            VolumeGeometry::Uniform { origin, spacing } => {
                let max = [
                    origin[0] + spacing[0] * (self.dims[0].saturating_sub(1)) as f32,
                    origin[1] + spacing[1] * (self.dims[1].saturating_sub(1)) as f32,
                    origin[2] + spacing[2] * (self.dims[2].saturating_sub(1)) as f32,
                ];
                (*origin, max)
            }
            VolumeGeometry::Rectilinear { xs, ys, zs } => {
                let min = [
                    *xs.first().unwrap_or(&0.0),
                    *ys.first().unwrap_or(&0.0),
                    *zs.first().unwrap_or(&0.0),
                ];
                let max = [
                    *xs.last().unwrap_or(&0.0),
                    *ys.last().unwrap_or(&0.0),
                    *zs.last().unwrap_or(&0.0),
                ];
                (min, max)
            }
        }
    }

    /// Produce a uniform sample volume for the named scalar attribute.
    pub fn resample_scalar_uniform(
        &self,
        name: &str,
    ) -> Option<(Vec<f32>, [u32; 3], [f32; 3], [f32; 3])> {
        let point_scalars = self.point_scalar_values(name)?;
        match &self.geometry {
            VolumeGeometry::Uniform { origin, spacing } => {
                Some((point_scalars, self.dims, *origin, *spacing))
            }
            VolumeGeometry::Rectilinear { xs, ys, zs } => {
                let dims = self.dims;
                if dims[0] < 2 || dims[1] < 2 || dims[2] < 2 {
                    return None;
                }
                let origin = [
                    *xs.first().unwrap_or(&0.0),
                    *ys.first().unwrap_or(&0.0),
                    *zs.first().unwrap_or(&0.0),
                ];
                let spacing = [
                    (*xs.last().unwrap_or(&origin[0]) - origin[0]) / (dims[0] - 1) as f32,
                    (*ys.last().unwrap_or(&origin[1]) - origin[1]) / (dims[1] - 1) as f32,
                    (*zs.last().unwrap_or(&origin[2]) - origin[2]) / (dims[2] - 1) as f32,
                ];
                let mut out = Vec::with_capacity(point_scalars.len());
                for iz in 0..dims[2] {
                    let z = origin[2] + spacing[2] * iz as f32;
                    for iy in 0..dims[1] {
                        let y = origin[1] + spacing[1] * iy as f32;
                        for ix in 0..dims[0] {
                            let x = origin[0] + spacing[0] * ix as f32;
                            out.push(sample_rectilinear_point_field(
                                dims,
                                xs,
                                ys,
                                zs,
                                &point_scalars,
                                [x, y, z],
                            ));
                        }
                    }
                }
                Some((out, dims, origin, spacing))
            }
        }
    }

    /// Return scalar values for a named attribute, checking point_data then cell_data.
    pub fn scalar_values(&self, name: &str) -> Option<&[f32]> {
        if let Some(v) = self.point_data.get(name) {
            return Some(v);
        }
        self.cell_data.get(name).map(|v| v.as_slice())
    }

    pub fn point_scalar_values(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(values) = self.point_data.get(name) {
            return Some(values.clone());
        }
        let cell_values = self.cell_data.get(name)?;
        cell_to_point_structured(self.dims, cell_values)
    }
}

/// Internal scientific dataset model used by several source loaders.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Vertex positions in world space.
    pub positions: Vec<[f32; 3]>,
    /// Triangle index list.
    pub indices: Vec<u32>,
    /// Per-vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Point-centred scalar fields.
    pub point_data: HashMap<String, Vec<f32>>,
    /// Cell-centred scalar fields.
    pub cell_data: HashMap<String, Vec<f32>>,
    /// Per-halfedge scalar fields.
    pub edge_data: HashMap<String, Vec<f32>>,
    /// Sparse voxel grid representation.
    pub sparse_volume: Option<Box<SparseGrid>>,
    /// Optional structured volume representation.
    pub volume: Option<VolumeGrid>,
    /// Optional unstructured volume mesh.
    pub volume_mesh: Option<Box<VolumeMesh>>,
}

impl Dataset {
    /// Return scalar values for a named attribute, checking point_data, cell_data, then volume.
    pub fn scalar_values(&self, name: &str) -> Option<&[f32]> {
        if let Some(v) = self.point_data.get(name) {
            return Some(v);
        }
        if let Some(v) = self.cell_data.get(name) {
            return Some(v);
        }
        if let Some(v) = self.edge_data.get(name) {
            return Some(v);
        }
        if let Some(vol) = &self.volume {
            if let Some(v) = vol.point_data.get(name) {
                return Some(v);
            }
            if let Some(v) = vol.cell_data.get(name) {
                return Some(v);
            }
        }
        None
    }

    fn to_surface_mesh(&self) -> SurfaceMesh {
        let mut attributes = HashMap::new();
        for (name, values) in &self.point_data {
            attributes.insert(
                name.clone(),
                AttributeData::scalars(AttributeDomain::Point, values.clone()),
            );
        }
        for (name, values) in &self.cell_data {
            attributes.insert(
                name.clone(),
                AttributeData::scalars(AttributeDomain::Cell, values.clone()),
            );
        }
        for (name, values) in &self.edge_data {
            attributes.insert(
                name.clone(),
                AttributeData::scalars(AttributeDomain::Halfedge, values.clone()),
            );
        }
        SurfaceMesh {
            positions: self.positions.clone(),
            normals: self.normals.clone(),
            indices: self.indices.clone(),
            uvs: None,
            uvs1: None,
            tangents: None,
            colours: None,
            attributes,
            skin_weights: None,
            morph_targets: Vec::new(),
        }
    }

    pub fn into_io_dataset(self) -> DecodedDataSet {
        let has_surface_mesh = !self.positions.is_empty() && !self.indices.is_empty();
        let has_point_set = !self.positions.is_empty() && self.indices.is_empty();

        let surface_mesh = if has_surface_mesh {
            Some(self.to_surface_mesh())
        } else {
            None
        };

        let point_set = if has_point_set {
            let scalar_attributes = self.point_data.clone();
            let scalars = if scalar_attributes.len() == 1 {
                scalar_attributes
                    .values()
                    .next()
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            Some(PointSet {
                name: String::new(),
                positions: self.positions.clone(),
                colors: Vec::new(),
                scalars,
                scalar_attributes,
            })
        } else {
            None
        };

        let volume = self.volume.map(|volume| StructuredVolume {
            name: String::new(),
            dims: volume.dims,
            geometry: match volume.geometry {
                VolumeGeometry::Uniform { origin, spacing } => {
                    VolumeGridGeometry::Uniform { origin, spacing }
                }
                VolumeGeometry::Rectilinear { xs, ys, zs } => {
                    VolumeGridGeometry::Rectilinear { xs, ys, zs }
                }
            },
            point_fields: volume.point_data,
            cell_fields: volume.cell_data,
        });

        DecodedDataSet {
            name: String::new(),
            surface_mesh,
            point_set,
            volume,
            sparse_grid: self.sparse_volume,
            volume_mesh: self.volume_mesh,
        }
    }
}

fn cell_to_point_structured(dims: [u32; 3], cell_values: &[f32]) -> Option<Vec<f32>> {
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return None;
    }
    let cx = nx.saturating_sub(1) as usize;
    let cy = ny.saturating_sub(1) as usize;
    let cz = nz.saturating_sub(1) as usize;
    let expected = cx.checked_mul(cy)?.checked_mul(cz)?;
    if cell_values.len() != expected {
        return None;
    }

    let point_count = (nx as usize)
        .checked_mul(ny as usize)?
        .checked_mul(nz as usize)?;
    let mut sums = vec![0.0f32; point_count];
    let mut counts = vec![0u32; point_count];

    let point_index = |ix: usize, iy: usize, iz: usize| -> usize {
        ix + iy * nx as usize + iz * nx as usize * ny as usize
    };
    let cell_index = |ix: usize, iy: usize, iz: usize| -> usize { ix + iy * cx + iz * cx * cy };

    for iz in 0..cz {
        for iy in 0..cy {
            for ix in 0..cx {
                let value = cell_values[cell_index(ix, iy, iz)];
                for dz in 0..=1 {
                    for dy in 0..=1 {
                        for dx in 0..=1 {
                            let pi = point_index(ix + dx, iy + dy, iz + dz);
                            sums[pi] += value;
                            counts[pi] += 1;
                        }
                    }
                }
            }
        }
    }

    for (sum, count) in sums.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *sum /= *count as f32;
        }
    }
    Some(sums)
}

fn sample_rectilinear_point_field(
    dims: [u32; 3],
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    scalars: &[f32],
    pos: [f32; 3],
) -> f32 {
    let [nx, ny, nz] = dims.map(|d| d as usize);
    let locate = |coords: &[f32], value: f32| -> (usize, usize, f32) {
        if coords.len() < 2 {
            return (0, 0, 0.0);
        }
        if value <= coords[0] {
            return (0, 1, 0.0);
        }
        if value >= coords[coords.len() - 1] {
            let hi = coords.len() - 1;
            return (hi - 1, hi, 1.0);
        }
        let mut hi = 1usize;
        while hi < coords.len() && coords[hi] < value {
            hi += 1;
        }
        let lo = hi - 1;
        let denom = (coords[hi] - coords[lo]).max(f32::EPSILON);
        let t = (value - coords[lo]) / denom;
        (lo, hi, t.clamp(0.0, 1.0))
    };
    let idx = |ix: usize, iy: usize, iz: usize| -> usize { ix + iy * nx + iz * nx * ny };

    let (x0, x1, tx) = locate(xs, pos[0]);
    let (y0, y1, ty) = locate(ys, pos[1]);
    let (z0, z1, tz) = locate(zs, pos[2]);
    let c000 = scalars[idx(x0, y0, z0)];
    let c100 = scalars[idx(x1, y0, z0)];
    let c010 = scalars[idx(x0, y1, z0)];
    let c110 = scalars[idx(x1, y1, z0)];
    let c001 = scalars[idx(x0, y0, z1)];
    let c101 = scalars[idx(x1, y0, z1)];
    let c011 = scalars[idx(x0, y1, z1)];
    let c111 = scalars[idx(x1, y1, z1)];

    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    let _ = nz;
    c0 * (1.0 - tz) + c1 * tz
}
