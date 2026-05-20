use std::collections::HashMap;
use std::path::PathBuf;

/// Sentinel used to pad unused slots in fixed-width volumetric cell connectivity.
pub const CELL_SENTINEL: u32 = u32::MAX;

/// CPU-side RGBA8 image data.
#[derive(Clone, Debug)]
pub struct RasterImageData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixel data.
    pub rgba: Vec<u8>,
}

/// CPU-side RGBA32F image data.
#[derive(Clone, Debug)]
pub struct HdrImageData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA32F pixel data.
    pub rgba: Vec<f32>,
}

/// Where a scene material's texture content comes from.
#[derive(Clone, Debug)]
pub enum TextureSource {
    /// Resolve texture bytes from a file path at the consumer layer.
    File(PathBuf),
    /// Use already-decoded pixels directly.
    Decoded(RasterImageData),
}

/// Material data extracted from a scene file.
#[derive(Clone, Debug)]
pub struct MaterialData {
    /// Material name from the source scene.
    pub name: String,
    /// Base colour in linear space.
    pub base_color: [f32; 3],
    /// Metallic factor.
    pub metallic: f32,
    /// Roughness factor.
    pub roughness: f32,
    /// Opacity factor from the source file.
    pub opacity: f32,
    /// Base colour texture, if present.
    pub base_color_texture: Option<TextureSource>,
    /// Normal map, if present.
    pub normal_map_texture: Option<TextureSource>,
    /// Ambient-occlusion texture, if present.
    pub ao_texture: Option<TextureSource>,
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [0.7, 0.7, 0.7],
            metallic: 0.0,
            roughness: 0.5,
            opacity: 1.0,
            base_color_texture: None,
            normal_map_texture: None,
            ao_texture: None,
        }
    }
}

/// Domain on which an attribute is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributeDomain {
    /// One value per point or vertex.
    Point,
    /// One value per cell.
    Cell,
    /// One value per face.
    Face,
    /// One value per directed edge.
    Edge,
    /// One value per halfedge/corner-edge.
    Halfedge,
    /// One value per face corner.
    Corner,
}

/// Attribute payload values.
#[derive(Clone, Debug)]
pub enum AttributeValues {
    /// Scalar values.
    Scalars(Vec<f32>),
    /// RGBA colours.
    Colors(Vec<[f32; 4]>),
    /// 3D vectors.
    Vectors(Vec<[f32; 3]>),
}

/// Named attribute data on a mesh-like topology.
#[derive(Clone, Debug)]
pub struct AttributeData {
    /// Topological domain the values are defined on.
    pub domain: AttributeDomain,
    /// Attribute payload.
    pub values: AttributeValues,
}

impl AttributeData {
    /// Construct scalar attribute data.
    pub fn scalars(domain: AttributeDomain, values: Vec<f32>) -> Self {
        Self {
            domain,
            values: AttributeValues::Scalars(values),
        }
    }

    /// Construct colour attribute data.
    pub fn colors(domain: AttributeDomain, values: Vec<[f32; 4]>) -> Self {
        Self {
            domain,
            values: AttributeValues::Colors(values),
        }
    }

    /// Construct vector attribute data.
    pub fn vectors(domain: AttributeDomain, values: Vec<[f32; 3]>) -> Self {
        Self {
            domain,
            values: AttributeValues::Vectors(values),
        }
    }
}

/// Per-vertex joint influence data for CPU skinning workflows.
#[derive(Clone, Debug)]
pub struct SkinWeights {
    /// Joint indices for each vertex: 4 per vertex, parallel to positions.
    pub joint_indices: Vec<[u8; 4]>,
    /// Blend weights for each vertex: 4 per vertex.
    pub joint_weights: Vec<[f32; 4]>,
}

/// Source-agnostic surface mesh data.
#[derive(Clone, Debug, Default)]
pub struct SurfaceMesh {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Triangle index list.
    pub indices: Vec<u32>,
    /// Optional per-vertex UV coordinates.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Optional per-vertex tangents.
    pub tangents: Option<Vec<[f32; 4]>>,
    /// Named attributes on this mesh.
    pub attributes: HashMap<String, AttributeData>,
    /// Optional skinning weights.
    pub skin_weights: Option<SkinWeights>,
}

/// A mesh entry inside a loaded scene.
#[derive(Clone, Debug)]
pub struct SceneMesh {
    /// Mesh name.
    pub name: String,
    /// Mesh geometry and attributes.
    pub mesh: SurfaceMesh,
    /// Index into `SceneData::materials`.
    pub material_index: Option<usize>,
    /// Local transform carried through from the source format.
    pub transform: glam::Mat4,
    /// Whether the source format explicitly marked the mesh double-sided.
    pub two_sided: bool,
    /// Parent mesh index for scene hierarchy reconstruction.
    pub parent_index: Option<usize>,
    /// Human-readable names for imported vertex attributes.
    pub vertex_attribute_names: Vec<String>,
    /// Optional importer-specific tags.
    pub metadata: HashMap<String, String>,
}

impl Default for SceneMesh {
    fn default() -> Self {
        Self {
            name: String::new(),
            mesh: SurfaceMesh::default(),
            material_index: None,
            transform: glam::Mat4::IDENTITY,
            two_sided: false,
            parent_index: None,
            vertex_attribute_names: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// A generic point set with optional colours and scalar attributes.
#[derive(Clone, Debug, Default)]
pub struct PointSet {
    /// Point-set name.
    pub name: String,
    /// Point positions.
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point colours.
    pub colors: Vec<[f32; 4]>,
    /// Optional primary scalar values.
    pub scalars: Vec<f32>,
    /// Named scalar attributes carried with the points.
    pub scalar_attributes: HashMap<String, Vec<f32>>,
}

/// Spherical-harmonic degree for gaussian splat colour data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShDegree {
    /// Degree-0 SH; three coefficients `[r, g, b]` per splat.
    Zero,
    /// Degree-1 SH.
    One,
    /// Degree-3 SH.
    Three,
}

impl ShDegree {
    /// Number of RGB coefficients per splat.
    pub fn coeff_count(self) -> usize {
        match self {
            Self::Zero => 3,
            Self::One => 12,
            Self::Three => 48,
        }
    }
}

/// Source-agnostic gaussian splat input data.
#[derive(Clone, Debug)]
pub struct GaussianSplatSet {
    /// Object-space center positions, one per splat.
    pub positions: Vec<[f32; 3]>,
    /// Scale per splat.
    pub scales: Vec<[f32; 3]>,
    /// Rotation per splat as `[x, y, z, w]`.
    pub rotations: Vec<[f32; 4]>,
    /// Opacity per splat.
    pub opacities: Vec<f32>,
    /// Packed SH coefficients.
    pub sh_coefficients: Vec<f32>,
    /// SH degree for this set.
    pub sh_degree: ShDegree,
}

impl Default for GaussianSplatSet {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: ShDegree::Zero,
        }
    }
}

impl PointSet {
    /// Create a degree-0 gaussian splat set from points.
    pub fn to_gaussian_splats(
        &self,
        default_scale: [f32; 3],
        default_opacity: f32,
    ) -> GaussianSplatSet {
        let count = self.positions.len();
        let mut sh_coefficients = Vec::with_capacity(count * 3);
        for index in 0..count {
            let rgb = self
                .colors
                .get(index)
                .copied()
                .map(|color| [color[0], color[1], color[2]])
                .unwrap_or([1.0, 1.0, 1.0]);
            sh_coefficients.extend_from_slice(&rgb);
        }

        GaussianSplatSet {
            positions: self.positions.clone(),
            scales: vec![default_scale; count],
            rotations: vec![[0.0, 0.0, 0.0, 1.0]; count],
            opacities: vec![default_opacity; count],
            sh_coefficients,
            sh_degree: ShDegree::Zero,
        }
    }
}

/// Geometry model for a structured volume.
#[derive(Clone, Debug)]
pub enum VolumeGridGeometry {
    /// Uniform voxel spacing.
    Uniform {
        /// World-space origin of the first sample.
        origin: [f32; 3],
        /// Uniform spacing between samples on each axis.
        spacing: [f32; 3],
    },
    /// Axis-aligned grid with variable spacing along each axis.
    Rectilinear {
        /// Sample coordinates along X.
        xs: Vec<f32>,
        /// Sample coordinates along Y.
        ys: Vec<f32>,
        /// Sample coordinates along Z.
        zs: Vec<f32>,
    },
}

impl Default for VolumeGridGeometry {
    fn default() -> Self {
        Self::Uniform {
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }
}

/// Dense structured volume data and named fields.
#[derive(Clone, Debug, Default)]
pub struct StructuredVolume {
    /// Volume name or field-set label.
    pub name: String,
    /// Point dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Geometry description for the grid.
    pub geometry: VolumeGridGeometry,
    /// Named point-centred scalar fields.
    pub point_fields: HashMap<String, Vec<f32>>,
    /// Named cell-centred scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
}

impl StructuredVolume {
    /// World-space axis-aligned bounds of the volume.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        match &self.geometry {
            VolumeGridGeometry::Uniform { origin, spacing } => {
                let max = [
                    origin[0] + spacing[0] * self.dims[0].saturating_sub(1) as f32,
                    origin[1] + spacing[1] * self.dims[1].saturating_sub(1) as f32,
                    origin[2] + spacing[2] * self.dims[2].saturating_sub(1) as f32,
                ];
                (*origin, max)
            }
            VolumeGridGeometry::Rectilinear { xs, ys, zs } => {
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

    /// Return scalar values for a named field, checking point fields then cell fields.
    pub fn scalar_values(&self, name: &str) -> Option<&[f32]> {
        if let Some(values) = self.point_fields.get(name) {
            return Some(values);
        }
        self.cell_fields.get(name).map(Vec::as_slice)
    }
}

/// Sparse regular grid data.
#[derive(Clone, Debug, Default)]
pub struct SparseGrid {
    /// World-space position of the `[0, 0, 0]` corner of cell `[0, 0, 0]`.
    pub origin: [f32; 3],
    /// Side length of one cubic cell in world units.
    pub cell_size: f32,
    /// Grid indices `[i, j, k]` of occupied cells.
    pub active_cells: Vec<[u32; 3]>,
    /// Named per-cell scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
    /// Named per-node scalar fields.
    pub node_fields: HashMap<String, Vec<f32>>,
    /// Named per-cell RGBA colours.
    pub cell_colors: HashMap<String, Vec<[f32; 4]>>,
}

/// Unstructured volumetric mesh data.
#[derive(Clone, Debug, Default)]
pub struct VolumeMesh {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Cell connectivity; unused slots are padded with [`CELL_SENTINEL`].
    pub cells: Vec<[u32; 8]>,
    /// Named per-cell scalar fields.
    pub cell_fields: HashMap<String, Vec<f32>>,
    /// Named per-cell RGBA colours.
    pub cell_colors: HashMap<String, Vec<[f32; 4]>>,
}

/// A decoded scientific/container dataset that may expose several representations.
#[derive(Clone, Debug, Default)]
pub struct DecodedDataSet {
    /// Dataset name.
    pub name: String,
    /// Optional extracted surface mesh.
    pub surface_mesh: Option<SurfaceMesh>,
    /// Optional point-set representation.
    pub point_set: Option<PointSet>,
    /// Optional dense structured volume.
    pub volume: Option<StructuredVolume>,
    /// Optional sparse regular grid.
    pub sparse_grid: Option<Box<SparseGrid>>,
    /// Optional unstructured volume mesh.
    pub volume_mesh: Option<Box<VolumeMesh>>,
}

impl DecodedDataSet {
    /// Return the dataset as a surface mesh when available.
    pub fn as_surface_mesh(&self) -> Option<&SurfaceMesh> {
        self.surface_mesh.as_ref()
    }

    /// Return the dataset as a point set when available.
    pub fn as_point_set(&self) -> Option<&PointSet> {
        self.point_set.as_ref()
    }

    /// Return the dataset as a structured volume when available.
    pub fn as_structured_volume(&self) -> Option<&StructuredVolume> {
        self.volume.as_ref()
    }

    /// Return the dataset as a sparse grid when available.
    pub fn as_sparse_grid(&self) -> Option<&SparseGrid> {
        self.sparse_grid.as_deref()
    }

    /// Return the dataset as a volume mesh when available.
    pub fn as_volume_mesh(&self) -> Option<&VolumeMesh> {
        self.volume_mesh.as_deref()
    }
}

/// A decoded multi-object scene.
#[derive(Clone, Debug, Default)]
pub struct SceneData {
    /// Meshes in the scene.
    pub meshes: Vec<SceneMesh>,
    /// Materials referenced by meshes.
    pub materials: Vec<MaterialData>,
    /// Point sets carried by the scene.
    pub point_sets: Vec<PointSet>,
}

pub(crate) type TextureData = RasterImageData;
pub(crate) type HdrTextureData = HdrImageData;
pub(crate) type IoMaterial = MaterialData;
pub(crate) type IoMesh = SceneMesh;
pub(crate) type IoPointCloud = PointSet;
pub(crate) type IoVolumeGeometry = VolumeGridGeometry;
pub(crate) type IoVolume = StructuredVolume;
pub(crate) type IoSparseVolume = SparseGrid;
pub(crate) type IoVolumeMesh = VolumeMesh;
pub(crate) type IoDataSet = DecodedDataSet;
pub(crate) type IoScene = SceneData;
