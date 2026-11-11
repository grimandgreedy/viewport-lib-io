//! Flattening a frame's layers into one straight-alpha sRGB image.
//!
//! Blend modes follow the published compositing definitions: the separable
//! modes per channel, the hue, saturation, colour and luminosity modes through
//! luminosity and saturation, and Aseprite's addition, subtract and divide.
//! The blended colour is mixed with the source by the backdrop's alpha and then
//! composited source-over, in sRGB values from `0` to `1`, rounded to 8 bits
//! at the end.

use super::raw::{ColourDepth, Document, LayerKind};

/// A blend mode, by Aseprite's id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Blend {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColourDodge,
    ColourBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Colour,
    Luminosity,
    Addition,
    Subtract,
    Divide,
}

impl Blend {
    /// The mode for an id, normal for one this reader does not know.
    pub(super) fn from_id(id: u16) -> Self {
        use Blend::*;
        const ALL: [Blend; 19] = [
            Normal,
            Multiply,
            Screen,
            Overlay,
            Darken,
            Lighten,
            ColourDodge,
            ColourBurn,
            HardLight,
            SoftLight,
            Difference,
            Exclusion,
            Hue,
            Saturation,
            Colour,
            Luminosity,
            Addition,
            Subtract,
            Divide,
        ];
        ALL.get(id as usize).copied().unwrap_or(Normal)
    }
}

/// One separable blend of backdrop `b` and source `s`, each `0..=1`.
fn separable(mode: Blend, b: f32, s: f32) -> f32 {
    let multiply = |b: f32, s: f32| b * s;
    let screen = |b: f32, s: f32| b + s - b * s;
    let hard_light = |b: f32, s: f32| {
        if s <= 0.5 {
            multiply(b, 2.0 * s)
        } else {
            screen(b, 2.0 * s - 1.0)
        }
    };
    match mode {
        Blend::Multiply => multiply(b, s),
        Blend::Screen => screen(b, s),
        Blend::Overlay => hard_light(s, b),
        Blend::Darken => b.min(s),
        Blend::Lighten => b.max(s),
        Blend::ColourDodge => {
            if b == 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (b / (1.0 - s)).min(1.0)
            }
        }
        Blend::ColourBurn => {
            if b >= 1.0 {
                1.0
            } else if s == 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - b) / s).min(1.0)
            }
        }
        Blend::HardLight => hard_light(b, s),
        Blend::SoftLight => {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 {
                    ((16.0 * b - 12.0) * b + 4.0) * b
                } else {
                    b.sqrt()
                };
                b + (2.0 * s - 1.0) * (d - b)
            }
        }
        Blend::Difference => (b - s).abs(),
        Blend::Exclusion => b + s - 2.0 * b * s,
        Blend::Addition => (b + s).min(1.0),
        Blend::Subtract => (b - s).max(0.0),
        Blend::Divide => {
            if s == 0.0 {
                if b == 0.0 { 0.0 } else { 1.0 }
            } else {
                (b / s).min(1.0)
            }
        }
        _ => s,
    }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_colour(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut c = c;
    if n < 0.0 {
        c = c.map(|v| l + (v - l) * l / (l - n));
    }
    if x > 1.0 {
        c = c.map(|v| l + (v - l) * (1.0 - l) / (x - l));
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_colour(c.map(|v| v + d))
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| c[*a].total_cmp(&c[*b]));
    let [min, mid, max] = order;
    let mut out = [0.0; 3];
    if c[max] > c[min] {
        out[mid] = (c[mid] - c[min]) * s / (c[max] - c[min]);
        out[max] = s;
    }
    out
}

/// The blended colour of backdrop `b` and source `s` under `mode`.
fn blend_colour(mode: Blend, b: [f32; 3], s: [f32; 3]) -> [f32; 3] {
    match mode {
        Blend::Hue => set_lum(set_sat(s, sat(b)), lum(b)),
        Blend::Saturation => set_lum(set_sat(b, sat(s)), lum(b)),
        Blend::Colour => set_lum(s, lum(b)),
        Blend::Luminosity => set_lum(b, lum(s)),
        _ => [0, 1, 2].map(|i| separable(mode, b[i], s[i])),
    }
}

/// Composite `source` over `backdrop` under `mode`, the source's alpha scaled
/// by `opacity`. Both pixels are straight-alpha sRGB.
pub(super) fn blend_pixel(
    mode: Blend,
    backdrop: [u8; 4],
    source: [u8; 4],
    opacity: f32,
) -> [u8; 4] {
    let unit = |v: u8| v as f32 / 255.0;
    let a_s = unit(source[3]) * opacity;
    if a_s <= 0.0 {
        return backdrop;
    }
    let a_b = unit(backdrop[3]);
    let b = [unit(backdrop[0]), unit(backdrop[1]), unit(backdrop[2])];
    let s = [unit(source[0]), unit(source[1]), unit(source[2])];
    let blended = blend_colour(mode, b, s);
    let a_o = a_s + a_b * (1.0 - a_s);
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut out = [0, 0, 0, byte(a_o)];
    for i in 0..3 {
        // Where the backdrop is transparent the source shows as itself.
        let mixed = (1.0 - a_b) * s[i] + a_b * blended[i];
        out[i] = byte((a_s * mixed + a_b * (1.0 - a_s) * b[i]) / a_o);
    }
    out
}

/// A cel's pixels as straight-alpha RGBA.
///
/// Greyscale expands to grey with its alpha. An indexed pixel takes its
/// palette entry, except that the transparent index is transparent on every
/// layer but the background, which draws it in its palette colour.
pub(super) fn to_rgba(
    depth: ColourDepth,
    pixels: &[u8],
    palette: &[[u8; 4]],
    transparent_index: u8,
    background: bool,
) -> Vec<u8> {
    match depth {
        ColourDepth::Rgba => pixels.to_vec(),
        ColourDepth::Greyscale => pixels
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        ColourDepth::Indexed => pixels
            .iter()
            .flat_map(|i| {
                if *i == transparent_index && !background {
                    [0; 4]
                } else {
                    palette.get(*i as usize).copied().unwrap_or([0; 4])
                }
            })
            .collect(),
    }
}

/// A layer as compositing sees it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LayerInfo {
    pub(super) parent: Option<usize>,
    pub(super) kind: LayerKind,
    pub(super) visible: bool,
    pub(super) reference: bool,
    pub(super) blend: Blend,
    /// `0..=1`.
    pub(super) opacity: f32,
}

/// Each layer's parent: the nearest earlier layer one nesting level up.
pub(super) fn parents(levels: &[u16]) -> Vec<Option<usize>> {
    levels
        .iter()
        .enumerate()
        .map(|(i, level)| {
            if *level == 0 {
                return None;
            }
            (0..i).rev().find(|j| levels[*j] + 1 == *level)
        })
        .collect()
}

/// A cel ready to composite.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CelImage {
    pub(super) layer: usize,
    pub(super) position: [i32; 2],
    /// `0..=1`.
    pub(super) opacity: f32,
    pub(super) z_index: i16,
    pub(super) size: [u32; 2],
    /// Straight-alpha RGBA, row by row.
    pub(super) rgba: Vec<u8>,
}

/// The order a frame's siblings draw in, bottom first: a cel goes at its
/// layer's index plus its z-index, and on a tie the lower z-index goes first.
/// A group sits at its own index. Each item is `(layer, cel)`, the cel `None`
/// for a group.
pub(super) fn draw_order(items: &mut [(usize, Option<i16>)]) {
    items.sort_by_key(|(layer, z)| {
        let z = z.unwrap_or(0) as i64;
        (*layer as i64 + z, z)
    });
}

/// Whether a layer and every group around it is shown. Reference layers and
/// tilemap layers never draw.
fn drawn(layers: &[LayerInfo], index: usize) -> bool {
    let layer = &layers[index];
    if !layer.visible || layer.reference || layer.kind == LayerKind::Tilemap {
        return false;
    }
    layer.parent.is_none_or(|p| drawn(layers, p))
}

/// Flatten one frame onto a transparent canvas.
///
/// With `isolate_groups`, each group composites its children on their own and
/// then blends the result with its own mode and opacity. Without it a group's
/// mode and opacity are not valid, and its children draw straight onto what is
/// below.
pub(super) fn flatten(
    canvas: [u32; 2],
    layers: &[LayerInfo],
    cels: &[CelImage],
    isolate_groups: bool,
) -> Vec<u8> {
    let mut out = vec![0u8; canvas[0] as usize * canvas[1] as usize * 4];
    draw_children(None, canvas, layers, cels, isolate_groups, &mut out);
    out
}

fn draw_children(
    group: Option<usize>,
    canvas: [u32; 2],
    layers: &[LayerInfo],
    cels: &[CelImage],
    isolate_groups: bool,
    out: &mut [u8],
) {
    let mut items: Vec<(usize, Option<i16>)> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        if layer.parent == group && layer.kind == LayerKind::Group {
            items.push((i, None));
        }
    }
    let mut cel_of = Vec::new();
    for (c, cel) in cels.iter().enumerate() {
        if layers.get(cel.layer).is_some_and(|l| l.parent == group) {
            items.push((cel.layer, Some(cel.z_index)));
            cel_of.push((cel.layer, cel.z_index, c));
        }
    }
    draw_order(&mut items);

    for (layer, z) in items {
        if !drawn(layers, layer) {
            continue;
        }
        let info = &layers[layer];
        match z {
            None if isolate_groups => {
                let mut own = vec![0u8; out.len()];
                draw_children(Some(layer), canvas, layers, cels, isolate_groups, &mut own);
                for (dst, src) in out.chunks_exact_mut(4).zip(own.chunks_exact(4)) {
                    let src = [src[0], src[1], src[2], src[3]];
                    let b = [dst[0], dst[1], dst[2], dst[3]];
                    dst.copy_from_slice(&blend_pixel(info.blend, b, src, info.opacity));
                }
            }
            None => draw_children(Some(layer), canvas, layers, cels, isolate_groups, out),
            Some(z) => {
                let Some((_, _, c)) = cel_of.iter().find(|(l, cz, _)| *l == layer && *cz == z)
                else {
                    continue;
                };
                draw_cel(&cels[*c], info, canvas, out);
            }
        }
    }
}

fn draw_cel(cel: &CelImage, layer: &LayerInfo, canvas: [u32; 2], out: &mut [u8]) {
    let opacity = cel.opacity * layer.opacity;
    for y in 0..cel.size[1] as i64 {
        let cy = cel.position[1] as i64 + y;
        if cy < 0 || cy >= canvas[1] as i64 {
            continue;
        }
        for x in 0..cel.size[0] as i64 {
            let cx = cel.position[0] as i64 + x;
            if cx < 0 || cx >= canvas[0] as i64 {
                continue;
            }
            let s = ((y * cel.size[0] as i64 + x) * 4) as usize;
            let d = ((cy * canvas[0] as i64 + cx) * 4) as usize;
            let src = [
                cel.rgba[s],
                cel.rgba[s + 1],
                cel.rgba[s + 2],
                cel.rgba[s + 3],
            ];
            let dst = [out[d], out[d + 1], out[d + 2], out[d + 3]];
            out[d..d + 4].copy_from_slice(&blend_pixel(layer.blend, dst, src, opacity));
        }
    }
}

/// How compositing sees each of a document's layers.
pub(super) fn layer_infos(doc: &Document) -> Vec<LayerInfo> {
    let levels: Vec<u16> = doc.layers.iter().map(|l| l.child_level).collect();
    doc.layers
        .iter()
        .zip(parents(&levels))
        .map(|(l, parent)| LayerInfo {
            parent,
            kind: l.kind,
            visible: l.visible,
            reference: l.reference,
            blend: Blend::from_id(l.blend),
            opacity: l.opacity as f32 / 255.0,
        })
        .collect()
}

/// One frame's cels as straight-alpha RGBA.
pub(super) fn cel_images(doc: &Document, frame: usize) -> Vec<CelImage> {
    doc.frames[frame]
        .cels
        .iter()
        .map(|c| CelImage {
            layer: c.layer,
            position: c.position,
            opacity: c.opacity as f32 / 255.0,
            z_index: c.z_index,
            size: c.size,
            rgba: to_rgba(
                doc.depth,
                &c.pixels,
                &doc.palette,
                doc.transparent_index,
                doc.layers[c.layer].background,
            ),
        })
        .collect()
}

/// Every frame of a document flattened at the canvas size, in frame order.
pub(super) fn frame_images(doc: &Document) -> Vec<Vec<u8>> {
    let layers = layer_infos(doc);
    (0..doc.frames.len())
        .map(|f| flatten(doc.size, &layers, &cel_images(doc, f), doc.groups_isolated))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(got: [u8; 4], want: [u8; 4], what: &str) {
        for i in 0..4 {
            assert!(
                (got[i] as i32 - want[i] as i32).abs() <= 1,
                "{what}: got {got:?}, want {want:?}"
            );
        }
    }

    #[test]
    fn every_separable_mode_matches_its_formula() {
        // Backdrop 0.6, source 0.2 on every channel, both opaque.
        let (b, s) = ([153, 153, 153, 255], [51, 51, 51, 255]);
        let cases = [
            (Blend::Normal, 51),
            (Blend::Multiply, 31),
            (Blend::Screen, 173),
            (Blend::Overlay, 92),
            (Blend::Darken, 51),
            (Blend::Lighten, 153),
            (Blend::ColourDodge, 191),
            (Blend::ColourBurn, 0),
            (Blend::HardLight, 61),
            (Blend::SoftLight, 116),
            (Blend::Difference, 102),
            (Blend::Exclusion, 143),
            (Blend::Addition, 204),
            (Blend::Subtract, 102),
            (Blend::Divide, 255),
        ];
        for (mode, v) in cases {
            assert_near(
                blend_pixel(mode, b, s, 1.0),
                [v, v, v, 255],
                &format!("{mode:?}"),
            );
        }
    }

    #[test]
    fn the_hsl_modes_match_their_formulas() {
        // A reddish backdrop under a grey source.
        let (b, s) = ([153, 51, 51, 255], [51, 51, 51, 255]);
        assert_near(
            blend_pixel(Blend::Luminosity, b, s, 1.0),
            [122, 20, 20, 255],
            "luminosity",
        );
        assert_near(
            blend_pixel(Blend::Colour, b, s, 1.0),
            [82, 82, 82, 255],
            "colour",
        );
        assert_near(
            blend_pixel(Blend::Saturation, b, s, 1.0),
            [82, 82, 82, 255],
            "saturation",
        );
        assert_near(blend_pixel(Blend::Hue, b, s, 1.0), [82, 82, 82, 255], "hue");
    }

    #[test]
    fn ids_map_in_order_and_unknown_ones_are_normal() {
        assert_eq!(Blend::from_id(0), Blend::Normal);
        assert_eq!(Blend::from_id(12), Blend::Hue);
        assert_eq!(Blend::from_id(18), Blend::Divide);
        assert_eq!(Blend::from_id(40), Blend::Normal);
    }

    #[test]
    fn alpha_composites_source_over() {
        // Half opacity over opaque white: halfway.
        assert_near(
            blend_pixel(Blend::Normal, [255, 255, 255, 255], [0, 0, 0, 255], 0.5),
            [128, 128, 128, 255],
            "half black over white",
        );
        // Anything over nothing is itself, whatever the mode.
        assert_near(
            blend_pixel(Blend::Multiply, [0; 4], [200, 100, 50, 128], 1.0),
            [200, 100, 50, 128],
            "over transparent",
        );
        assert_eq!(
            blend_pixel(Blend::Normal, [1, 2, 3, 4], [9, 9, 9, 0], 1.0),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn colour_depths_expand_to_rgba() {
        assert_eq!(
            to_rgba(ColourDepth::Greyscale, &[10, 200], &[], 0, false),
            [10, 10, 10, 200]
        );
        let palette = [[0, 0, 0, 255], [255, 0, 0, 255]];
        assert_eq!(
            to_rgba(ColourDepth::Indexed, &[0, 1], &palette, 0, false),
            [0, 0, 0, 0, 255, 0, 0, 255]
        );
        assert_eq!(
            to_rgba(ColourDepth::Indexed, &[0], &palette, 0, true),
            [0, 0, 0, 255],
            "the background draws the transparent index"
        );
    }

    #[test]
    fn parents_are_the_nearest_layer_one_level_up() {
        assert_eq!(
            parents(&[0, 1, 2, 1, 0, 0]),
            [None, Some(0), Some(1), Some(0), None, None]
        );
    }

    #[test]
    fn z_index_moves_a_cel_among_its_siblings() {
        let mut items = vec![(0, Some(0)), (1, Some(0)), (2, Some(-2)), (3, None)];
        draw_order(&mut items);
        assert_eq!(
            items,
            [(2, Some(-2)), (0, Some(0)), (1, Some(0)), (3, None)]
        );
        // On a tie the lower z-index is further back.
        let mut tie = vec![(0, Some(1)), (1, Some(0))];
        draw_order(&mut tie);
        assert_eq!(tie, [(1, Some(0)), (0, Some(1))]);
    }

    fn layer(parent: Option<usize>, kind: LayerKind) -> LayerInfo {
        LayerInfo {
            parent,
            kind,
            visible: true,
            reference: false,
            blend: Blend::Normal,
            opacity: 1.0,
        }
    }

    fn cel(layer: usize, rgba: [u8; 4]) -> CelImage {
        CelImage {
            layer,
            position: [0, 0],
            opacity: 1.0,
            z_index: 0,
            size: [1, 1],
            rgba: rgba.to_vec(),
        }
    }

    #[test]
    fn hidden_reference_and_tilemap_layers_do_not_draw() {
        let mut layers = vec![
            layer(None, LayerKind::Image),
            layer(None, LayerKind::Image),
            layer(None, LayerKind::Group),
            layer(Some(2), LayerKind::Image),
            layer(None, LayerKind::Tilemap),
        ];
        let cels = [
            cel(0, [10, 0, 0, 255]),
            cel(1, [20, 0, 0, 255]),
            cel(3, [30, 0, 0, 255]),
            cel(4, [40, 0, 0, 255]),
        ];
        assert_eq!(
            flatten([1, 1], &layers, &cels, false),
            [30, 0, 0, 255],
            "the tilemap is skipped and the group's child is on top"
        );

        layers[2].visible = false;
        assert_eq!(
            flatten([1, 1], &layers, &cels, false),
            [20, 0, 0, 255],
            "a hidden group hides its child"
        );
        layers[1].reference = true;
        assert_eq!(
            flatten([1, 1], &layers, &cels, false),
            [10, 0, 0, 255],
            "a reference layer is a guide"
        );
    }

    #[test]
    fn an_isolated_group_blends_as_one_layer() {
        let mut layers = vec![
            layer(None, LayerKind::Image),
            layer(None, LayerKind::Group),
            layer(Some(1), LayerKind::Image),
            layer(Some(1), LayerKind::Image),
        ];
        layers[1].opacity = 0.5;
        let cels = [
            cel(0, [255, 255, 255, 255]),
            cel(2, [0, 0, 0, 255]),
            cel(3, [0, 0, 0, 255]),
        ];
        // Isolated: the two black cels make one opaque black layer, then half
        // of it shows over white.
        assert_near(
            flatten([1, 1], &layers, &cels, true).try_into().unwrap(),
            [128, 128, 128, 255],
            "isolated",
        );
        // Passed through: the group's opacity is not valid, so black covers.
        assert_eq!(flatten([1, 1], &layers, &cels, false), [0, 0, 0, 255]);
    }

    #[test]
    fn cels_clip_to_the_canvas_and_multiply_their_opacity() {
        let mut layers = vec![layer(None, LayerKind::Image)];
        layers[0].opacity = 0.5;
        let mut c = cel(0, [0, 0, 0, 255]);
        c.size = [2, 1];
        c.rgba = vec![0, 0, 0, 255, 0, 0, 0, 255];
        c.position = [1, 0];
        c.opacity = 0.5;
        let out = flatten([2, 1], &layers, &[c], false);
        assert_eq!(out[..4], [0, 0, 0, 0], "left of the cel");
        assert_near(
            out[4..].try_into().unwrap(),
            [0, 0, 0, 64],
            "a quarter, cut at the edge",
        );
    }
}
