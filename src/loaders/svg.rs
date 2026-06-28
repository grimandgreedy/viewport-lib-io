use std::path::Path;

use crate::error::IoError;
use crate::types::TextureData;

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
}
