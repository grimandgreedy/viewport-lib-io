//! Neutral baked-lightmap data.
//!
//! A [`LightmapData`] is a baked lightmap atlas in a GPU-agnostic form: linear
//! HDR pixels in a chosen encoding, sized to an atlas, that a renderer turns
//! into a UV1-sampled texture. It is the interchange type between whatever
//! produces a lightmap (an offline bake, or a future lightmap importer here in
//! this crate) and whatever consumes it, so neither side depends on the other.
//!
//! The encoding is an open enum so richer bases (spherical gaussians, SH-L1) can
//! be added without changing the container.

/// A baked lightmap atlas in a neutral, GPU-agnostic form.
#[derive(Clone, Debug)]
pub struct LightmapData {
    /// Atlas width in texels.
    pub width: u32,
    /// Atlas height in texels.
    pub height: u32,
    /// How the per-texel lighting is stored.
    pub encoding: LightmapEncoding,
}

/// How a [`LightmapData`]'s per-texel lighting is stored. All pixel buffers are
/// row-major, `width * height` long, with the atlas origin at the top-left. `w`
/// in a radiance texel carries coverage: 1.0 on a baked texel, 0.0 on an empty
/// one.
#[derive(Clone, Debug)]
pub enum LightmapEncoding {
    /// One linear RGB radiance value per texel. Normals do not respond to the
    /// baked light in this encoding.
    NonDirectional {
        /// Baked incident radiance (`rgb`) and coverage (`w`).
        radiance: Vec<[f32; 4]>,
    },
    /// A radiance atlas plus a per-texel dominant light direction, so normal
    /// mapping responds to the baked light. `direction` holds the unit dominant
    /// direction in `xyz` (world space) and a directionality factor in `w`
    /// (`0` = fully ambient, `1` = a single direction).
    DominantDirection {
        /// Baked incident radiance (`rgb`) and coverage (`w`).
        radiance: Vec<[f32; 4]>,
        /// Dominant direction (`xyz`) and directionality (`w`).
        direction: Vec<[f32; 4]>,
    },
}

impl LightmapData {
    /// Number of texels in the atlas (`width * height`).
    pub fn texel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// The baked radiance atlas (`rgb` + coverage in `w`), present in every
    /// encoding.
    pub fn radiance(&self) -> &[[f32; 4]] {
        match &self.encoding {
            LightmapEncoding::NonDirectional { radiance } => radiance,
            LightmapEncoding::DominantDirection { radiance, .. } => radiance,
        }
    }

    /// The dominant-direction atlas, if this lightmap carries one.
    pub fn direction(&self) -> Option<&[[f32; 4]]> {
        match &self.encoding {
            LightmapEncoding::NonDirectional { .. } => None,
            LightmapEncoding::DominantDirection { direction, .. } => Some(direction),
        }
    }

    /// Whether every pixel buffer matches `width * height`. A consumer should
    /// check this before uploading; a producer should return only valid data.
    pub fn is_well_formed(&self) -> bool {
        let n = self.texel_count();
        match &self.encoding {
            LightmapEncoding::NonDirectional { radiance } => radiance.len() == n,
            LightmapEncoding::DominantDirection { radiance, direction } => {
                radiance.len() == n && direction.len() == n
            }
        }
    }
}
