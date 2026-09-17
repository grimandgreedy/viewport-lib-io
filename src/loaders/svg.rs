use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;
#[cfg(feature = "svg")]
use crate::types::{FillRule, PathSegment, SubPath, VectorArt, VectorShape};

/// Decode an SVG file into RGBA8 pixels.
pub fn texture_from_path(path: &Path) -> Result<TextureData, IoError> {
    #[cfg(feature = "svg")]
    {
        let bytes = std::fs::read(path)?;
        let mut options = resvg::usvg::Options::default();
        options.resources_dir = path.parent().map(std::path::Path::to_path_buf);

        let tree = resvg::usvg::Tree::from_data(&bytes, &options)
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

        Ok(TextureData {
            width: size.width(),
            height: size.height(),
            rgba,
        })
    }

    #[cfg(not(feature = "svg"))]
    {
        let _ = path;
        Err(IoError::MissingFeature {
            feature: "svg",
            context: "SVG texture decoding",
        })
    }
}

/// Load an SVG file as neutral vector paths instead of a rasterized texture.
///
/// Walks the parsed path tree and emits one [`VectorShape`] per filled path,
/// with its subpaths (curves preserved, not flattened), fill rule, and resolved
/// solid fill colour. Coordinates are baked into the drawing's user-unit space
/// (each path's absolute transform is applied), so the output places directly
/// without further transform bookkeeping. This is the SVG canvas frame: X to
/// the right, Y *down* from the top-left origin, not the crate's Z-up scene
/// convention, which covers 3D geometry only. Feed the result into
/// `viewport_lib::OverlayShapeItem::vector` to draw it.
///
/// Only solid fills are resolved to a colour; gradient and pattern fills leave
/// [`VectorShape::fill`] as `None` (the geometry is still emitted). Enclosing
/// group opacity is folded into the fill alpha. Invisible paths
/// (`visibility="hidden"`) are skipped, matching what [`texture_from_path`]
/// rasterizes.
///
/// Strokes are not carried: the output models filled area only, so a
/// stroke-only path yields a shape with no fill, and stroke paint and width are
/// dropped. Group clip paths, masks, and filters are ignored. Use
/// [`texture_from_path`] instead when a baked raster image is wanted.
pub fn vector_from_path(path: &Path) -> Result<VectorArt, IoError> {
    #[cfg(feature = "svg")]
    {
        let bytes = std::fs::read(path)?;
        let mut options = resvg::usvg::Options::default();
        options.resources_dir = path.parent().map(std::path::Path::to_path_buf);

        let tree = resvg::usvg::Tree::from_data(&bytes, &options)
            .map_err(|error| IoError::Parse(format!("failed to parse SVG: {error}")))?;

        let size = tree.size();
        let mut shapes = Vec::new();
        let root = tree.root();
        collect_shapes(root, root.opacity().get(), &mut shapes);

        Ok(VectorArt {
            shapes,
            size: [size.width(), size.height()],
        })
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

/// Recurse the node tree, appending a `VectorShape` for each filled path.
///
/// `opacity` is the product of the enclosing groups' opacities, folded into
/// each shape's fill alpha since the neutral output has no group nesting.
#[cfg(feature = "svg")]
fn collect_shapes(group: &resvg::usvg::Group, opacity: f32, out: &mut Vec<VectorShape>) {
    for node in group.children() {
        match node {
            resvg::usvg::Node::Group(child) => {
                collect_shapes(child, opacity * child.opacity().get(), out)
            }
            resvg::usvg::Node::Path(path) => {
                if let Some(shape) = path_to_shape(path, opacity) {
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
fn path_to_shape(path: &resvg::usvg::Path, opacity: f32) -> Option<VectorShape> {
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
            let colour = match fill.paint() {
                resvg::usvg::Paint::Color(c) => Some([
                    srgb_to_linear(c.red),
                    srgb_to_linear(c.green),
                    srgb_to_linear(c.blue),
                    fill.opacity().get() * opacity,
                ]),
                // Gradients and patterns are not reduced to a single colour.
                _ => None,
            };
            (rule, colour)
        }
        None => (FillRule::NonZero, None),
    };

    Some(VectorShape {
        subpaths,
        fill_rule,
        fill,
    })
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
                current = Some(SubPath {
                    start: [p.x, p.y],
                    segments: Vec::new(),
                    closed: false,
                });
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

/// Convert an 8-bit sRGB channel to linear `[0, 1]`, matching how overlay fills
/// interpret their colours.
#[cfg(feature = "svg")]
fn srgb_to_linear(c: u8) -> f32 {
    let s = c as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[cfg(feature = "svg")]
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

        assert_eq!(image.width, 2);
        assert_eq!(image.height, 1);
        assert_eq!(image.rgba, vec![255, 0, 0, 255, 0, 255, 0, 255]);
    }

    #[cfg(feature = "svg")]
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

    #[cfg(feature = "svg")]
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
        let fill = shape.fill.expect("solid fill resolves to a colour");
        assert!(
            fill[0] > 0.9 && fill[1] < 0.1 && fill[2] < 0.1,
            "red fill: {fill:?}"
        );
    }

    #[cfg(feature = "svg")]
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

    #[cfg(feature = "svg")]
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
        let fill = art.shapes[0].fill.expect("solid fill resolves to a colour");
        assert!(
            fill[1] > 0.9 && fill[0] < 0.1,
            "green rect survives: {fill:?}"
        );
    }

    #[cfg(feature = "svg")]
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
        let nested = art.shapes[0].fill.expect("solid fill")[3];
        assert!(
            (nested - 0.25).abs() < 1e-5,
            "nested group opacity multiplies: {nested}"
        );
        let own = art.shapes[1].fill.expect("solid fill")[3];
        assert!((own - 0.5).abs() < 1e-5, "fill-opacity is unchanged: {own}");
    }
}
