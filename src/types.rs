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
