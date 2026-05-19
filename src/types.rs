use std::collections::HashMap;
use std::path::PathBuf;

/// CPU-side texture data ready for `viewport-lib` texture upload calls.
#[derive(Clone, Debug)]
pub struct TextureData {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixel data.
    pub rgba: Vec<u8>,
}

/// CPU-side HDR environment image data ready for environment upload calls.
#[derive(Clone, Debug)]
pub struct HdrTextureData {
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
    Decoded(TextureData),
}

/// CPU-side material data extracted from a scene file.
#[derive(Clone, Debug)]
pub struct IoMaterial {
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

impl Default for IoMaterial {
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

/// CPU-side mesh entry for multi-mesh scene formats.
pub struct IoMesh {
    /// Mesh name.
    pub name: String,
    /// Mesh data already shaped for `viewport-lib`.
    pub mesh_data: viewport_lib::MeshData,
    /// Index into `IoScene::materials`.
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

impl Default for IoMesh {
    fn default() -> Self {
        Self {
            name: String::new(),
            mesh_data: viewport_lib::MeshData::default(),
            material_index: None,
            transform: glam::Mat4::IDENTITY,
            two_sided: false,
            parent_index: None,
            vertex_attribute_names: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// CPU-side point cloud entry for scene formats that carry them.
#[derive(Clone, Debug, Default)]
pub struct IoPointCloud {
    /// Point cloud name.
    pub name: String,
    /// Point positions.
    pub positions: Vec<[f32; 3]>,
    /// Optional colours.
    pub colors: Vec<[f32; 4]>,
    /// Optional scalar values.
    pub scalars: Vec<f32>,
    /// Named scalar attributes carried with the point cloud.
    pub scalar_attributes: HashMap<String, Vec<f32>>,
}

/// Geometry model for a structured volume.
#[derive(Clone, Debug)]
pub enum IoVolumeGeometry {
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

impl Default for IoVolumeGeometry {
    fn default() -> Self {
        Self::Uniform {
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }
}

/// CPU-side dense structured volume data for `viewport-lib` volume uploads.
#[derive(Clone, Debug, Default)]
pub struct IoVolume {
    /// Volume name or field-set label.
    pub name: String,
    /// Point dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// Geometry description for the grid.
    pub geometry: IoVolumeGeometry,
    /// Named scalar fields in x-fastest order.
    pub scalar_fields: HashMap<String, Vec<f32>>,
}

impl IoVolume {
    /// World-space axis-aligned bounds of the volume.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        match &self.geometry {
            IoVolumeGeometry::Uniform { origin, spacing } => {
                let max = [
                    origin[0] + spacing[0] * self.dims[0].saturating_sub(1) as f32,
                    origin[1] + spacing[1] * self.dims[1].saturating_sub(1) as f32,
                    origin[2] + spacing[2] * self.dims[2].saturating_sub(1) as f32,
                ];
                (*origin, max)
            }
            IoVolumeGeometry::Rectilinear { xs, ys, zs } => {
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
}

/// Clone/debug wrapper for `viewport-lib` sparse voxel grid data.
pub struct IoSparseVolume(pub viewport_lib::SparseVolumeGridData);

impl Clone for IoSparseVolume {
    fn clone(&self) -> Self {
        let mut out = viewport_lib::SparseVolumeGridData::default();
        out.origin = self.0.origin;
        out.cell_size = self.0.cell_size;
        out.active_cells = self.0.active_cells.clone();
        out.cell_scalars = self.0.cell_scalars.clone();
        out.node_scalars = self.0.node_scalars.clone();
        out.cell_colours = self.0.cell_colours.clone();
        Self(out)
    }
}

impl std::fmt::Debug for IoSparseVolume {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoSparseVolume")
            .field("origin", &self.0.origin)
            .field("cell_size", &self.0.cell_size)
            .field("active_cells_count", &self.0.active_cells.len())
            .finish()
    }
}

/// Clone/debug wrapper for `viewport-lib` volume-mesh data.
pub struct IoVolumeMesh(pub viewport_lib::VolumeMeshData);

impl Clone for IoVolumeMesh {
    fn clone(&self) -> Self {
        let mut out = viewport_lib::VolumeMeshData::default();
        out.positions = self.0.positions.clone();
        out.cells = self.0.cells.clone();
        out.cell_scalars = self.0.cell_scalars.clone();
        out.cell_colours = self.0.cell_colours.clone();
        Self(out)
    }
}

impl std::fmt::Debug for IoVolumeMesh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoVolumeMesh")
            .field("positions_count", &self.0.positions.len())
            .field("cells_count", &self.0.cells.len())
            .finish()
    }
}

/// Scientific/container dataset result that may span multiple viewport families.
#[derive(Default)]
pub struct IoDataSet {
    /// Dataset name.
    pub name: String,
    /// Optional extracted surface mesh.
    pub surface_mesh: Option<viewport_lib::MeshData>,
    /// Optional point-cloud representation.
    pub point_cloud: Option<IoPointCloud>,
    /// Optional dense structured volume.
    pub volume: Option<IoVolume>,
    /// Optional sparse regular volume grid.
    pub sparse_volume: Option<Box<IoSparseVolume>>,
    /// Optional unstructured volume mesh.
    pub volume_mesh: Option<Box<IoVolumeMesh>>,
}

/// CPU-side scene result for multi-object formats.
#[derive(Default)]
pub struct IoScene {
    /// Meshes in the scene.
    pub meshes: Vec<IoMesh>,
    /// Materials referenced by meshes.
    pub materials: Vec<IoMaterial>,
    /// Point clouds carried by the scene.
    pub point_clouds: Vec<IoPointCloud>,
}
