use std::path::Path;

use viewport_lib_types::data::texture::TextureData;

use crate::error::IoError;
#[cfg(feature = "svg")]
use crate::types::{SpreadMethod, VectorArt, VectorDashes, VectorPaint, VectorShape, VectorStroke};
#[cfg(feature = "svg")]
use viewport_lib_types::colour::Colour;
#[cfg(feature = "svg")]
use viewport_lib_types::overlay::vector::{FillRule, PathSegment, SubPath};
use viewport_lib_types::overlay::{GradientStop, LineCap, LineJoin};

/// Rasterise an SVG file into an 8-bit RGBA texture labelled sRGB colour, at
/// the canvas size in pixels.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    let bytes = std::fs::read(path)?;
    rasterise(&bytes, path.parent())
}

/// Rasterise an SVG into an 8-bit RGBA texture labelled sRGB colour, from bytes
/// already read. The in-memory sibling of [`texture_from_path`].
///
/// With no file there is no directory to resolve external references against,
/// so an `<image>` that points at a file beside the SVG is left out of the
/// raster; one embedded as a data URI still draws.
pub fn texture_from_bytes(bytes: &[u8]) -> Result<TextureData, IoError> {
    rasterise(bytes, None)
}

/// Rasterise SVG bytes at the canvas size, resolving external references
/// against `resources_dir` when there is one.
fn rasterise(bytes: &[u8], resources_dir: Option<&Path>) -> Result<TextureData, IoError> {
    #[cfg(feature = "svg")]
    {
        let mut options = resvg::usvg::Options::default();
        options.resources_dir = resources_dir.map(Path::to_path_buf);

        let tree = resvg::usvg::Tree::from_data(bytes, &options)
            .map_err(|error| IoError::Parse(format!("failed to parse SVG: {error}")))?;
        let size = tree.size().to_int_size();
        let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())
            .ok_or_else(|| IoError::Parse("failed to allocate SVG raster image".into()))?;

        resvg::render(
            &tree,
            resvg::tiny_skia::Transform::identity(),
            &mut pixmap.as_mut(),
        );

        let mut rgba = Vec::with_capacity(pixmap.pixels().len() * 4);
        for pixel in pixmap.pixels() {
            let color = pixel.demultiply();
            rgba.extend_from_slice(&[color.red(), color.green(), color.blue(), color.alpha()]);
        }

        Ok(TextureData::srgb(size.width(), size.height(), rgba))
    }

    #[cfg(not(feature = "svg"))]
    {
        let _ = (bytes, resources_dir);
        Err(IoError::MissingFeature {
            feature: "svg",
            context: "SVG texture decoding",
        })
    }
}

/// Load an SVG file as neutral vector paths instead of a rasterized texture.
///
/// Walks the parsed path tree and emits one [`VectorShape`] per painted path,
/// with its subpaths (curves preserved, not flattened), fill rule, and resolved
/// fill and stroke, plus its element id and the ids of the groups
/// enclosing it. Coordinates are baked into the drawing's user-unit space
/// (each path's absolute transform is applied), so the output places directly
/// without further transform bookkeeping. This is the SVG canvas frame: X to
/// the right, Y *down* from the top-left origin, not the crate's Z-up scene
/// convention, which covers 3D geometry only. Feed the result into
/// `viewport_lib::OverlayShapeItem::vector` to draw it.
///
/// Solid colours and linear and radial gradients are resolved into a
/// [`VectorPaint`], gradients keeping their stops, spread and geometry with a
/// transform into the art's coordinates. Pattern paints leave
/// [`VectorShape::fill`] or [`VectorShape::stroke`] as `None` (the geometry is
/// still emitted). Enclosing group opacity is folded into every alpha.
/// Invisible paths (`visibility="hidden"`) are skipped, matching what
/// [`texture_from_path`] rasterizes.
///
/// Stroke width and dash lengths are scaled by the transform baked into the
/// geometry, so they stay in step with the contours; see
/// [`VectorStroke::width`] for the non-uniform case. Caps, joins and the mitre
/// limit are carried as written, except that `miter-clip` becomes a mitre.
/// Group clip paths, masks, and filters are ignored. Use [`texture_from_path`] instead when
/// a baked raster image is wanted.
pub fn vector_from_path(path: &Path) -> Result<VectorArt, IoError> {
    #[cfg(feature = "svg")]
    {
        let bytes = std::fs::read(path)?;
        let mut options = resvg::usvg::Options::default();
        options.resources_dir = path.parent().map(std::path::Path::to_path_buf);
        vector_from_data(&bytes, &options)
    }

    #[cfg(not(feature = "svg"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "svg",
            context: "SVG vector-path decoding",
        })
    }
}

/// Load neutral vector paths from in-memory SVG bytes.
///
/// The in-memory sibling of [`vector_from_path`], for art served from a cooked
/// bundle (or fetched over the network for a `wasm32` target) rather than read
/// from a file. Output is identical to the path form: same shapes, same
/// coordinates, same paint.
///
/// Without a file there is no directory to resolve external references against,
/// but that costs nothing here: the nodes that reference external files are
/// images, which this loader skips, and text, which needs a font database fed to
/// `usvg` either way. Anything the loader does emit is self-contained in the
/// bytes.
pub fn vector_from_bytes(bytes: &[u8]) -> Result<VectorArt, IoError> {
    #[cfg(feature = "svg")]
    {
        vector_from_data(bytes, &resvg::usvg::Options::default())
    }

    #[cfg(not(feature = "svg"))]
    {
        let _ = bytes;
        Err(IoError::MissingFeature {
            feature: "svg",
            context: "SVG vector-path decoding",
        })
    }
}

/// Parse SVG bytes and walk the resulting tree into neutral vector shapes.
#[cfg(feature = "svg")]
fn vector_from_data(bytes: &[u8], options: &resvg::usvg::Options) -> Result<VectorArt, IoError> {
    let tree = resvg::usvg::Tree::from_data(bytes, options)
        .map_err(|error| IoError::Parse(format!("failed to parse SVG: {error}")))?;

    let size = tree.size();
    let mut shapes = Vec::new();
    let root = tree.root();
    collect_shapes(root, root.opacity().get(), &mut Vec::new(), &mut shapes);

    Ok(VectorArt {
        shapes,
        size: [size.width(), size.height()],
    })
}

/// Recurse the node tree, appending a `VectorShape` for each painted path.
///
/// `opacity` is the product of the enclosing groups' opacities, folded into
/// each shape's fill alpha since the neutral output has no group nesting.
/// `groups` is the stack of enclosing group ids, for the same reason.
#[cfg(feature = "svg")]
fn collect_shapes(
    group: &resvg::usvg::Group,
    opacity: f32,
    groups: &mut Vec<String>,
    out: &mut Vec<VectorShape>,
) {
    for node in group.children() {
        match node {
            resvg::usvg::Node::Group(child) => {
                let named = !child.id().is_empty();
                if named {
                    groups.push(child.id().to_string());
                }
                collect_shapes(child, opacity * child.opacity().get(), groups, out);
                if named {
                    groups.pop();
                }
            }
            resvg::usvg::Node::Path(path) => {
                if let Some(shape) = path_to_shape(path, opacity, groups) {
                    out.push(shape);
                }
            }
            // Images and text are not vector fills; text needs a font DB fed to
            // usvg to resolve to outlines, which this loader does not do.
            resvg::usvg::Node::Image(_) | resvg::usvg::Node::Text(_) => {}
        }
    }
}

/// Convert one usvg path into a `VectorShape` in absolute coordinates.
#[cfg(feature = "svg")]
fn path_to_shape(path: &resvg::usvg::Path, opacity: f32, groups: &[String]) -> Option<VectorShape> {
    // A hidden path is not painted, so it is not part of the drawing.
    if !path.is_visible() {
        return None;
    }

    // Bake the absolute transform into the geometry so the output needs no
    // further transform bookkeeping.
    let data = path.data().clone().transform(path.abs_transform())?;
    let subpaths = segments_to_subpaths(&data);
    if subpaths.is_empty() {
        return None;
    }

    let (fill_rule, fill) = match path.fill() {
        Some(fill) => {
            let rule = match fill.rule() {
                resvg::usvg::FillRule::EvenOdd => FillRule::EvenOdd,
                resvg::usvg::FillRule::NonZero => FillRule::NonZero,
            };
            (
                rule,
                resolve_paint(
                    fill.paint(),
                    fill.opacity().get() * opacity,
                    path.abs_transform(),
                ),
            )
        }
        None => (FillRule::NonZero, None),
    };

    let stroke = path.stroke().and_then(|stroke| {
        let paint = resolve_paint(
            stroke.paint(),
            stroke.opacity().get() * opacity,
            path.abs_transform(),
        )?;
        // usvg reports stroke lengths in the path's own units, so they need the
        // same transform the geometry just had baked in. A non-uniform scale
        // has no single right answer; average the two factors.
        let scale = transform_scale(path.abs_transform());
        Some(VectorStroke {
            paint,
            width: stroke.width().get() * scale,
            cap: match stroke.linecap() {
                resvg::usvg::LineCap::Butt => LineCap::Butt,
                resvg::usvg::LineCap::Round => LineCap::Round,
                resvg::usvg::LineCap::Square => LineCap::Square,
            },
            join: match stroke.linejoin() {
                // `miter-clip` cuts the corner at the limit rather than
                // bevelling it; a mitre is the nearest join that is carried.
                resvg::usvg::LineJoin::Miter | resvg::usvg::LineJoin::MiterClip => LineJoin::Mitre,
                resvg::usvg::LineJoin::Round => LineJoin::Round,
                resvg::usvg::LineJoin::Bevel => LineJoin::Bevel,
            },
            mitre_limit: stroke.miterlimit().get(),
            // usvg has already repeated an odd list and dropped one that sums
            // to zero.
            dashes: stroke.dasharray().map(|array| VectorDashes {
                array: array.iter().map(|length| length * scale).collect(),
                offset: stroke.dashoffset() * scale,
            }),
        })
    });

    Some(VectorShape {
        subpaths,
        fill_rule,
        fill,
        stroke,
        id: Some(path.id())
            .filter(|id| !id.is_empty())
            .map(str::to_string),
        groups: groups.to_vec(),
    })
}

/// Resolve a paint at `alpha`, with `transform` the one baked into the shape's
/// geometry. SVG colours are sRGB. Patterns are not resolved and yield `None`.
fn resolve_paint(
    paint: &resvg::usvg::Paint,
    alpha: f32,
    transform: resvg::tiny_skia::Transform,
) -> Option<VectorPaint> {
    use resvg::usvg::Paint;

    match paint {
        Paint::Color(c) => Some(VectorPaint::Solid(srgb(*c, alpha))),
        // usvg has already turned bounding-box units into user space, folding
        // the box into the gradient's transform.
        Paint::LinearGradient(g) => Some(VectorPaint::LinearGradient {
            start: [g.x1(), g.y1()],
            end: [g.x2(), g.y2()],
            stops: gradient_stops(g.stops(), alpha),
            spread: spread_method(g.spread_method()),
            transform: affine(transform.pre_concat(g.transform())),
        }),
        Paint::RadialGradient(g) => Some(VectorPaint::RadialGradient {
            centre: [g.cx(), g.cy()],
            radius: g.r().get(),
            focal: [g.fx(), g.fy()],
            stops: gradient_stops(g.stops(), alpha),
            spread: spread_method(g.spread_method()),
            transform: affine(transform.pre_concat(g.transform())),
        }),
        Paint::Pattern(_) => None,
    }
}

fn srgb(c: resvg::usvg::Color, alpha: f32) -> Colour {
    Colour::srgb_u8(c.red, c.green, c.blue, 255).with_alpha(alpha)
}

/// Gradient stops with each stop's opacity multiplied by `alpha`. usvg has
/// already clamped the offsets and sorted them.
fn gradient_stops(stops: &[resvg::usvg::Stop], alpha: f32) -> Vec<GradientStop> {
    stops
        .iter()
        .map(|stop| {
            GradientStop::new(
                stop.offset().get(),
                srgb(stop.color(), stop.opacity().get() * alpha),
            )
        })
        .collect()
}

fn spread_method(spread: resvg::usvg::SpreadMethod) -> SpreadMethod {
    match spread {
        resvg::usvg::SpreadMethod::Pad => SpreadMethod::Pad,
        resvg::usvg::SpreadMethod::Reflect => SpreadMethod::Reflect,
        resvg::usvg::SpreadMethod::Repeat => SpreadMethod::Repeat,
    }
}

fn affine(t: resvg::tiny_skia::Transform) -> glam::Affine2 {
    glam::Affine2::from_cols_array(&[t.sx, t.ky, t.kx, t.sy, t.tx, t.ty])
}

/// The scalar a length in a transform's source space multiplies by. `get_scale`
/// takes the magnitude of each axis, so rotation and skew are handled; the two
/// factors are averaged since one scalar cannot express a non-uniform scale.
#[cfg(feature = "svg")]
fn transform_scale(transform: resvg::tiny_skia::Transform) -> f32 {
    let (x, y) = transform.get_scale();
    (x + y) / 2.0
}

/// Split a tiny-skia path into subpaths, preserving line and Bezier segments.
#[cfg(feature = "svg")]
fn segments_to_subpaths(data: &resvg::tiny_skia::Path) -> Vec<SubPath> {
    use resvg::tiny_skia::PathSegment as Ts;

    let mut out = Vec::new();
    let mut current: Option<SubPath> = None;

    for segment in data.segments() {
        match segment {
            Ts::MoveTo(p) => {
                if let Some(sub) = current.take() {
                    out.push(sub);
                }
                current = Some(SubPath::new([p.x, p.y]));
            }
            Ts::LineTo(p) => {
                if let Some(sub) = current.as_mut() {
                    sub.segments.push(PathSegment::Line { to: [p.x, p.y] });
                }
            }
            Ts::QuadTo(c, p) => {
                if let Some(sub) = current.as_mut() {
                    sub.segments.push(PathSegment::Quad {
                        ctrl: [c.x, c.y],
                        to: [p.x, p.y],
                    });
                }
            }
            Ts::CubicTo(c1, c2, p) => {
                if let Some(sub) = current.as_mut() {
                    sub.segments.push(PathSegment::Cubic {
                        ctrl1: [c1.x, c1.y],
                        ctrl2: [c2.x, c2.y],
                        to: [p.x, p.y],
                    });
                }
            }
            Ts::Close => {
                if let Some(sub) = current.as_mut() {
                    sub.closed = true;
                }
            }
        }
    }
    if let Some(sub) = current.take() {
        out.push(sub);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(paint: &VectorPaint) -> [f32; 4] {
        match paint {
            VectorPaint::Solid(colour) => colour.to_linear_rgba(),
            other => panic!("expected a solid paint, got {other:?}"),
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "viewport_lib_io_{name}_{}_{}.svg",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn decodes_svg_rgba_pixels() {
        let path = temp_path("svg_decode");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="2" height="1">
  <rect x="0" y="0" width="1" height="1" fill="#ff0000"/>
  <rect x="1" y="0" width="1" height="1" fill="#00ff00"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let image = texture_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(
            image.colour_space(),
            viewport_lib_types::colour::ColourSpace::Srgb
        );
        assert!(matches!(
            image.payload(),
            viewport_lib_types::data::texture::TexturePayload::Rgba8(bytes)
                if bytes == &[255, 0, 0, 255, 0, 255, 0, 255]
        ));
    }

    #[test]
    fn invalid_svg_returns_parse_error() {
        let path = temp_path("invalid_svg");
        std::fs::write(&path, "<svg").unwrap();

        let err = texture_from_path(&path).unwrap_err();
        let _ = std::fs::remove_file(&path);

        match err {
            IoError::Parse(message) => {
                assert!(message.contains("failed to parse SVG"));
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn vector_path_with_hole_keeps_two_subpaths() {
        let path = temp_path("svg_vector_hole");
        // One filled path with an outer square and an inner square, even-odd.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path d="M0 0 H10 V10 H0 Z M2 2 H8 V8 H2 Z" fill="#ff0000" fill-rule="evenodd"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.size, [10.0, 10.0]);
        assert_eq!(art.shapes.len(), 1, "expected a single filled path");
        let shape = &art.shapes[0];
        assert_eq!(shape.subpaths.len(), 2, "outer contour plus the hole");
        assert_eq!(shape.fill_rule, FillRule::EvenOdd);
        let fill = rgba(
            shape
                .fill
                .as_ref()
                .expect("solid fill resolves to a colour"),
        );
        assert!(
            fill[0] > 0.9 && fill[1] < 0.1 && fill[2] < 0.1,
            "red fill: {fill:?}"
        );
    }

    #[test]
    fn vector_curve_survives_unflattened() {
        let path = temp_path("svg_vector_curve");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path d="M0 0 C 0 5, 5 10, 10 10 Z" fill="#00ff00"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 1);
        let has_cubic = art.shapes[0]
            .subpaths
            .iter()
            .flat_map(|s| &s.segments)
            .any(|seg| matches!(seg, PathSegment::Cubic { .. }));
        assert!(has_cubic, "cubic segment should survive without flattening");
    }

    #[test]
    fn hidden_paths_are_skipped() {
        let path = temp_path("svg_vector_hidden");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect x="0" y="0" width="4" height="4" fill="#ff0000" visibility="hidden"/>
  <rect x="5" y="5" width="4" height="4" fill="#00ff00"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 1, "the hidden rect is not painted");
        let fill = rgba(
            art.shapes[0]
                .fill
                .as_ref()
                .expect("solid fill resolves to a colour"),
        );
        assert!(
            fill[1] > 0.9 && fill[0] < 0.1,
            "green rect survives: {fill:?}"
        );
    }

    #[test]
    fn group_opacity_folds_into_fill_alpha() {
        let path = temp_path("svg_vector_group_opacity");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <g opacity="0.5">
    <g opacity="0.5">
      <rect x="0" y="0" width="4" height="4" fill="#ff0000"/>
    </g>
  </g>
  <rect x="5" y="5" width="4" height="4" fill="#ff0000" fill-opacity="0.5"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 2);
        let nested = rgba(art.shapes[0].fill.as_ref().expect("solid fill"))[3];
        assert!(
            (nested - 0.25).abs() < 1e-5,
            "nested group opacity multiplies: {nested}"
        );
        let own = rgba(art.shapes[1].fill.as_ref().expect("solid fill"))[3];
        assert!((own - 0.5).abs() < 1e-5, "fill-opacity is unchanged: {own}");
    }

    /// `currentColor` is the ordinary way to author a tintable icon, and usvg resolves it against
    /// the inherited `color` at parse. Worth pinning: the neutral output carries a literal either
    /// way, so nothing downstream can vary it per use, and art with no `color` in scope resolves
    /// to the CSS default of black rather than to nothing, which reads as a missing icon on a dark
    /// surface.
    #[test]
    fn current_colour_resolves_against_the_inherited_colour() {
        let path = temp_path("svg_current_colour");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" color="#ff0000">
  <path d="M0 0 L10 10" fill="none" stroke="currentColor" stroke-width="2"/>
  <rect x="0" y="0" width="4" height="4" fill="currentColor"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 2);
        let stroke = art.shapes[0]
            .stroke
            .as_ref()
            .expect("currentColor resolves to a solid stroke");
        assert!(
            rgba(&stroke.paint)[0] > 0.9 && rgba(&stroke.paint)[1] < 0.1,
            "the stroke took the root colour: {:?}",
            stroke.paint
        );
        let fill = rgba(
            art.shapes[1]
                .fill
                .as_ref()
                .expect("currentColor resolves to a fill"),
        );
        assert!(
            fill[0] > 0.9 && fill[1] < 0.1,
            "the fill took the root colour: {fill:?}"
        );
    }

    /// The other half: with no `color` anywhere in scope, `currentColor` is black, not absent.
    /// Art that looks tintable therefore imports as black art, which is why a caller that wants
    /// one drawing in several colours tints it rather than relying on this.
    #[test]
    fn current_colour_without_a_colour_in_scope_resolves_to_black() {
        let path = temp_path("svg_current_colour_default");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path d="M0 0 L10 10" fill="none" stroke="currentColor" stroke-width="2"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let stroke = art.shapes[0]
            .stroke
            .as_ref()
            .expect("a stroke is still resolved");
        assert!(
            rgba(&stroke.paint)[..3].iter().all(|c| *c < 1e-5),
            "black, not absent: {:?}",
            stroke.paint
        );
    }

    #[test]
    fn stroke_only_path_resolves_a_stroke_and_no_fill() {
        let path = temp_path("svg_stroke_only");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path d="M0 0 L10 10" fill="none" stroke="#ff0000" stroke-width="2"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 1);
        let shape = &art.shapes[0];
        assert!(shape.fill.is_none(), "no fill: stroke-only art");
        let stroke = shape
            .stroke
            .as_ref()
            .expect("solid stroke resolves to a colour");
        assert!(
            rgba(&stroke.paint)[0] > 0.9 && rgba(&stroke.paint)[1] < 0.1,
            "red stroke: {:?}",
            stroke.paint
        );
        assert!((stroke.width - 2.0).abs() < 1e-5, "width: {}", stroke.width);
        assert!(
            !shape.subpaths[0].closed,
            "an open line stays open, so a consumer strokes rather than fills it"
        );
    }

    #[test]
    fn path_with_both_paints_resolves_both() {
        let path = temp_path("svg_fill_and_stroke");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <rect x="1" y="1" width="8" height="8" fill="#00ff00" stroke="#0000ff" stroke-width="1"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 1);
        let shape = &art.shapes[0];
        let fill = rgba(shape.fill.as_ref().expect("solid fill"));
        assert!(fill[1] > 0.9, "green fill: {fill:?}");
        let stroke = shape.stroke.as_ref().expect("solid stroke");
        assert!(
            rgba(&stroke.paint)[2] > 0.9,
            "blue stroke: {:?}",
            stroke.paint
        );
    }

    #[test]
    fn stroke_width_scales_with_the_baked_transform() {
        let path = temp_path("svg_stroke_width_scale");
        // usvg reports stroke width in the path's own units, so the width has
        // to follow the transform that the geometry has baked into it.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">
  <g transform="scale(2)">
    <path d="M0 0 L10 0" fill="none" stroke="#000000" stroke-width="3"/>
  </g>
  <g transform="scale(2,4)">
    <path d="M0 5 L10 5" fill="none" stroke="#000000" stroke-width="3"/>
  </g>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(art.shapes.len(), 2);
        let uniform = art.shapes[0].stroke.as_ref().expect("solid stroke").width;
        assert!((uniform - 6.0).abs() < 1e-5, "2x scale: {uniform}");
        // The geometry also lands at twice the authored x, so the two agree.
        assert_eq!(art.shapes[0].subpaths[0].segments.len(), 1);
        assert!(matches!(
            art.shapes[0].subpaths[0].segments[0],
            PathSegment::Line { to } if (to[0] - 20.0).abs() < 1e-5
        ));

        let non_uniform = art.shapes[1].stroke.as_ref().expect("solid stroke").width;
        assert!(
            (non_uniform - 9.0).abs() < 1e-5,
            "average of 2x and 4x: {non_uniform}"
        );
    }

    fn linear_fill(shape: &VectorShape) -> (Vec<GradientStop>, f32) {
        match shape
            .fill
            .as_ref()
            .map(|f| f.to_overlay_fill(shape_bounds(shape)))
        {
            Some(viewport_lib_types::overlay::OverlayFill::LinearGradient { stops, angle }) => {
                (stops, angle)
            }
            other => panic!("expected a linear gradient fill, got {other:?}"),
        }
    }

    /// The bounding box of a shape's end points, which for the rectangles here
    /// is the rectangle.
    fn shape_bounds(shape: &VectorShape) -> [f32; 4] {
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for sub in &shape.subpaths {
            let ends = std::iter::once(sub.start).chain(sub.segments.iter().map(|seg| match seg {
                PathSegment::Line { to }
                | PathSegment::Quad { to, .. }
                | PathSegment::Cubic { to, .. } => *to,
            }));
            for p in ends {
                b = [
                    b[0].min(p[0]),
                    b[1].min(p[1]),
                    b[2].max(p[0]),
                    b[3].max(p[1]),
                ];
            }
        }
        b
    }

    fn assert_near(got: f32, want: f32, what: &str) {
        assert!((got - want).abs() < 1e-4, "{what}: {got}, want {want}");
    }

    #[test]
    fn a_bounding_box_gradient_fill_converts_to_an_angle_across_the_shape() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <defs>
    <linearGradient id="across"><stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/></linearGradient>
    <linearGradient id="down" x2="0" y2="1"><stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/></linearGradient>
  </defs>
  <rect x="10" y="20" width="60" height="30" fill="url(#across)"/>
  <rect x="10" y="60" width="60" height="30" fill="url(#down)"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let (stops, angle) = linear_fill(&art.shapes[0]);
        assert_near(angle, 0.0, "left to right");
        assert_eq!(stops.len(), 2, "{stops:?}");
        assert_near(stops[0].position, 0.0, "first stop");
        assert_near(stops[1].position, 1.0, "last stop");
        assert!(stops[0].colour.to_linear_rgba()[0] > 0.99, "{stops:?}");
        assert!(stops[1].colour.to_linear_rgba()[2] > 0.99, "{stops:?}");

        let (_, angle) = linear_fill(&art.shapes[1]);
        assert_near(angle, std::f32::consts::FRAC_PI_2, "top to bottom");
    }

    #[test]
    fn a_user_space_gradient_follows_the_transform_baked_into_the_geometry() {
        // The gradient spans the rectangle in the path's own units; the group
        // moves and scales both, so it still spans it on the canvas.
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <defs>
    <linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="10" y2="0">
      <stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/>
    </linearGradient>
  </defs>
  <g transform="translate(20 5) scale(2)">
    <rect x="0" y="0" width="10" height="10" fill="url(#g)"/>
  </g>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let shape = &art.shapes[0];
        assert_eq!(shape_bounds(shape), [20.0, 5.0, 40.0, 25.0]);
        let (stops, angle) = linear_fill(shape);
        assert_near(angle, 0.0, "left to right");
        assert_near(stops[0].position, 0.0, "first stop");
        assert_near(stops.last().unwrap().position, 1.0, "last stop");
        assert_eq!(
            stops.len(),
            2,
            "the gradient spans the rectangle: {stops:?}"
        );
    }

    #[test]
    fn a_radial_gradient_keeps_its_geometry_spread_and_stop_opacity() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <defs>
    <radialGradient id="r" cx="0.5" cy="0.5" r="0.5" fx="0.25" fy="0.5" spreadMethod="reflect">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="1" stop-color="#000000" stop-opacity="0.5"/>
    </radialGradient>
  </defs>
  <g opacity="0.5">
    <rect x="0" y="0" width="40" height="40" fill="url(#r)"/>
  </g>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let Some(VectorPaint::RadialGradient {
            centre,
            radius,
            focal,
            stops,
            spread,
            transform,
        }) = &art.shapes[0].fill
        else {
            panic!("expected a radial gradient, got {:?}", art.shapes[0].fill);
        };
        assert_eq!(*spread, SpreadMethod::Reflect);
        // In bounding-box units, with the box folded into the transform.
        let to_canvas = |p: [f32; 2]| transform.transform_point2(p.into());
        assert!((to_canvas(*centre) - glam::vec2(20.0, 20.0)).length() < 1e-4);
        assert!((to_canvas(*focal) - glam::vec2(10.0, 20.0)).length() < 1e-4);
        assert_near(
            radius * transform.matrix2.x_axis.length(),
            20.0,
            "radius on the canvas",
        );
        assert_near(stops[0].colour.alpha(), 0.5, "group opacity");
        assert_near(
            stops[1].colour.alpha(),
            0.25,
            "stop opacity times group opacity",
        );
    }

    #[test]
    fn gradient_stroke_is_carried() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <defs>
    <linearGradient id="g"><stop offset="0" stop-color="#ff0000"/><stop offset="1" stop-color="#0000ff"/></linearGradient>
  </defs>
  <path d="M0 0 L10 10" fill="none" stroke="url(#g)" stroke-width="2"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let stroke = art.shapes[0]
            .stroke
            .as_ref()
            .expect("the gradient stroke is resolved");
        let VectorPaint::LinearGradient { stops, .. } = &stroke.paint else {
            panic!("expected a linear gradient stroke, got {:?}", stroke.paint);
        };
        assert_eq!(stops.len(), 2);
        // The overlay strokes in one colour: the gradient's average.
        let mean = stroke.to_overlay_stroke(1.0).colour.to_linear_rgba();
        assert!(mean[0] > 0.4 && mean[2] > 0.4 && mean[1] < 1e-6, "{mean:?}");
    }

    #[test]
    fn pattern_paint_is_not_resolved() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <defs>
    <pattern id="p" width="2" height="2" patternUnits="userSpaceOnUse"><rect width="1" height="1" fill="#000"/></pattern>
  </defs>
  <rect x="0" y="0" width="10" height="10" fill="url(#p)"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        assert_eq!(art.shapes.len(), 1, "the geometry is still emitted");
        assert!(art.shapes[0].fill.is_none());
    }

    #[test]
    fn caps_joins_and_dashes_round_trip_scaled_with_the_width() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <g transform="scale(2)">
    <path d="M0 0 L10 0 L10 10" fill="none" stroke="#000" stroke-width="1"
          stroke-linecap="round" stroke-linejoin="bevel" stroke-miterlimit="7"
          stroke-dasharray="4 2" stroke-dashoffset="1"/>
    <path d="M0 20 L10 20" fill="none" stroke="#000" stroke-dasharray="3 1 2"
          stroke-linejoin="miter-clip" stroke-linecap="square"/>
    <path d="M0 30 L10 30" fill="none" stroke="#000"/>
  </g>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let styled = art.shapes[0].stroke.as_ref().unwrap();
        assert_eq!(styled.cap, LineCap::Round);
        assert_eq!(styled.join, LineJoin::Bevel);
        assert_eq!(styled.mitre_limit, 7.0, "a ratio, so it does not scale");
        assert_eq!(
            styled.dashes,
            Some(VectorDashes {
                array: vec![8.0, 4.0],
                offset: 2.0
            }),
            "dash lengths scale with the width"
        );

        let odd = art.shapes[1].stroke.as_ref().unwrap();
        assert_eq!(odd.cap, LineCap::Square);
        assert_eq!(
            odd.join,
            LineJoin::Mitre,
            "miter-clip falls back to a mitre"
        );
        assert_eq!(
            odd.dashes.as_ref().map(|d| d.array.clone()),
            Some(vec![6.0, 2.0, 4.0, 6.0, 2.0, 4.0]),
            "an odd list is repeated to an even one"
        );

        let plain = art.shapes[2].stroke.as_ref().unwrap();
        assert_eq!(plain.cap, LineCap::Butt);
        assert_eq!(plain.join, LineJoin::Mitre);
        assert_eq!(plain.mitre_limit, 4.0, "the SVG default");
        assert_eq!(plain.dashes, None);
    }

    #[test]
    fn group_opacity_folds_into_stroke_alpha() {
        let path = temp_path("svg_stroke_group_opacity");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <g opacity="0.5">
    <path d="M0 0 L10 10" fill="none" stroke="#ff0000" stroke-width="2" stroke-opacity="0.5"/>
  </g>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let art = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let stroke = art.shapes[0].stroke.as_ref().expect("solid stroke");
        assert!(
            (rgba(&stroke.paint)[3] - 0.25).abs() < 1e-5,
            "group opacity times stroke-opacity: {}",
            rgba(&stroke.paint)[3]
        );
    }

    #[test]
    fn element_ids_survive_on_shapes() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path id="NSW" d="M0 0 H4 V4 H0 Z" fill="#ff0000"/>
  <rect id="" x="5" y="5" width="4" height="4" fill="#ff0000"/>
  <circle id="dot" cx="5" cy="5" r="1" fill="#0000ff"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        let ids: Vec<_> = art.shapes.iter().map(|s| s.id.as_deref()).collect();
        assert_eq!(ids, [Some("NSW"), None, Some("dot")], "empty id is None");
        assert!(art.shapes.iter().all(|s| s.groups.is_empty()));
    }

    #[test]
    fn enclosing_group_ids_list_outermost_first_and_skip_unnamed() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <g id="layer1">
    <g transform="translate(1 0)">
      <g id="room-12">
        <rect x="0" y="0" width="2" height="2" fill="#ff0000"/>
        <rect x="3" y="0" width="2" height="2" fill="#ff0000"/>
      </g>
    </g>
    <rect id="loose" x="0" y="5" width="2" height="2" fill="#00ff00"/>
  </g>
  <rect x="8" y="8" width="1" height="1" fill="#0000ff"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        assert_eq!(art.shapes.len(), 4);
        assert_eq!(art.shapes[0].groups, ["layer1", "room-12"]);
        assert_eq!(art.shapes[1].groups, ["layer1", "room-12"]);
        assert_eq!(
            art.shapes[2].groups,
            ["layer1"],
            "popped back out of room-12"
        );
        assert!(art.shapes[3].groups.is_empty(), "top level has no groups");
    }

    /// usvg strips ids from the content a `<use>` copies in, so a clone has no id of its own and
    /// the `<use>` element's id, which lands on its group, is the only key it has.
    #[test]
    fn use_clones_have_no_id_and_carry_the_use_id_as_a_group() {
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink" width="10" height="10">
  <defs><rect id="seat" width="1" height="1" fill="#ff0000"/></defs>
  <use id="seat-a1" xlink:href="#seat" x="0"/>
  <use id="seat-a2" xlink:href="#seat" x="2"/>
</svg>"##;

        let art = vector_from_bytes(svg.as_bytes()).unwrap();

        assert_eq!(art.shapes.len(), 2);
        for shape in &art.shapes {
            assert_eq!(
                shape.id, None,
                "the referenced element's id is not repeated"
            );
        }
        assert_eq!(
            art.shapes[0].groups.last().map(String::as_str),
            Some("seat-a1")
        );
        assert_eq!(
            art.shapes[1].groups.last().map(String::as_str),
            Some("seat-a2")
        );
    }

    #[test]
    fn bytes_and_path_decode_to_the_same_art() {
        let path = temp_path("svg_vector_bytes");
        let svg = r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">
  <path d="M0 0 H10 V10 H0 Z M2 2 H8 V8 H2 Z" fill="#ff0000" fill-rule="evenodd"/>
  <path d="M0 0 C 0 5, 5 10, 10 10" fill="none" stroke="#0000ff" stroke-width="2"/>
</svg>"##;

        std::fs::write(&path, svg).unwrap();
        let from_path = vector_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let from_bytes = vector_from_bytes(svg.as_bytes()).unwrap();

        assert_eq!(from_bytes, from_path, "same art whichever way it is read");
        assert_eq!(from_bytes.shapes.len(), 2);
        assert!(from_bytes.shapes[1].stroke.is_some(), "stroke survives");
    }

    #[test]
    fn invalid_svg_bytes_return_a_parse_error() {
        let err = vector_from_bytes(b"<svg").unwrap_err();
        match err {
            IoError::Parse(message) => assert!(message.contains("failed to parse SVG")),
            other => panic!("expected Parse error, got {other:?}"),
        }
    }
}
