//! Public load options.

/// Per-call options for [`super::sprite_sheet_from_path_with_options`] and
/// [`super::sprite_sheet_from_bytes_with_options`], matching the switches of
/// Aseprite's own sprite sheet export.
///
/// They apply to documents only. A JSON sheet already says how it was packed,
/// and is read as it is whatever these say. The default decodes a document
/// exactly as [`super::sprite_sheet_from_path`] does, as a default export
/// would: untrimmed, one atlas cell per frame, no gutter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsepriteLoadOptions {
    /// Crop each frame to its non-transparent pixels, recording where the crop
    /// sat in [`SpriteFrame::source_offset`]. A fully transparent frame gets an
    /// empty rectangle and keeps its place and duration in the animation.
    ///
    /// [`SpriteFrame::source_offset`]: crate::types::SpriteFrame::source_offset
    pub trim: bool,
    /// Let frames whose pixels are identical (after any trim) share one atlas
    /// rectangle. Each keeps its own name and duration.
    pub merge_duplicates: bool,
    /// Transparent pixels between packed frames, so a sampler filtering
    /// linearly does not bleed one frame into the next.
    pub gutter: u32,
}

impl AsepriteLoadOptions {
    /// Set whether frames are cropped to their non-transparent pixels.
    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }

    /// Set whether identical frames share one atlas rectangle.
    pub fn with_merge_duplicates(mut self, merge: bool) -> Self {
        self.merge_duplicates = merge;
        self
    }

    /// Set the transparent pixels between packed frames.
    pub fn with_gutter(mut self, gutter: u32) -> Self {
        self.gutter = gutter;
        self
    }
}
